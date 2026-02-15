"""
训练因果IK模型（窗口化版本）

适用于 SFU 数据集：
- 输入：10帧历史，每帧14维（7维位姿 + 7维关节角度）
- 输出：下一帧的7维关节角度

核心改进:
1. 时序编码器 - 提取历史帧的时序特征
2. 图神经网络（GNN）- 学习关节间因果耦合
3. 物理约束损失 - FK一致性
4. 关节耦合损失 - 确保因果依赖

用法:
    python train_causal_ik.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import sys
import time
import os

sys.path.insert(0, '/home/wsy/Desktop/casual')

from dataset_generalized import create_windowed_dataloaders
from causal_ik_model import PhysicsAwareCausalIK, PhysicsAwareLoss, PositionBasedLoss, PhysicsAwareCausalIKWithHistory
import pinocchio as pin

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("causal_train2.log"), logging.StreamHandler()]
)


class Config:
    # 数据配置
    data_path = "/home/wsy/Desktop/casual/merged_training_data.npz"  # SFU窗口化数据集
    input_dim = 7    # 7维位姿（位置+姿态）
    output_dim = 7   # 7维关节角度
    num_frames = 10  # 历史窗口大小

    # 训练参数
    batch_size = 512
    epochs = 500
    initial_lr = 1e-3
    min_lr = 1e-6
    patience = 7
    factor = 0.5
    cooldown = 5

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 模型配置
    hidden_dim = 512
    num_gnn_layers = 1  # GNN层数

    # 物理损失权重
    ik_weight = 1.0       # IK损失权重
    fk_weight = 0.5       # FK一致性损失权重
    continuity_weight = 1.0  # 连续性约束权重    # 连续性损失权重（防止预测突变）
    coupling_weight = 0.0  # 关节耦合损失权重（暂时关闭）

    # 数据增强
    use_augmentation = False
    augmentation_level = 'moderate'

    # 使用 PositionBasedLoss（直接优化位置误差）
    use_position_loss = True


def mixup_data(x, y, alpha=0.2):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def load_robot_model():
    """加载Pinocchio机器人模型（用于FK计算）"""
    urdf_path = "/home/wsy/Desktop/casual/unitree_g1/g1_custom_collision_29dof.urdf"

    try:
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()

        # 左臂关节ID
        left_arm_joints = [16, 17, 18, 19, 20, 21, 22]
        wrist_frame_id = 57 
        logging.info(f"✓ 成功加载Pinocchio模型")
        logging.info(f"  URDF: {urdf_path}")
        logging.info(f"  左臂关节 ID: {left_arm_joints}")
        logging.info(f"  手腕 frame ID: {wrist_frame_id}")

        return model, data, left_arm_joints, wrist_frame_id
    except Exception as e:
        logging.error(f"✗ 加载Pinocchio模型失败: {e}")
        return None, None, None, None


def forward_kinematics_batch(pin_model, pin_data, joint_ids, wrist_frame_id, joint_angles):
    """
    批量计算正向运动学

    Args:
        joint_angles: [batch, 7] 关节角度
        wrist_frame_id: 手腕frame ID

    Returns:
        wrist_positions: [batch, 3] 手腕位置
    """
    batch_size = joint_angles.shape[0]
    device = joint_angles.device

    # 转换为numpy（Pinocchio需要）- 需要先detach
    joint_angles_np = joint_angles.detach().cpu().numpy()

    wrist_positions = []
    for i in range(batch_size):
        q = np.zeros(pin_model.nq, dtype=np.float64)
        for j, joint_id in enumerate(joint_ids[:7]):
            q[joint_id] = joint_angles_np[i, j]

        # 计算FK - 只需要framesForwardKinematics
        pin.framesForwardKinematics(pin_model, pin_data, q)

        wrist_pos = pin_data.oMf[wrist_frame_id].translation.copy()
        wrist_positions.append(wrist_pos)

    wrist_positions = np.array(wrist_positions)

    # 转回tensor
    return torch.from_numpy(wrist_positions).to(device)


def train():
    config = Config()
    # print("config.hidden_dim:",config.hidden_dim,"use_mixup",config.use_mixup,"use_velocity",config.use_velocity,)

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = 'causal_ik'

    logging.info("=" * 70)
    logging.info("训练因果IK模型")
    logging.info("=" * 70)
    logging.info("核心改进:")
    logging.info("  1. 图神经网络（GNN）- 学习关节间因果耦合")
    logging.info("  2. 物理约束损失 - FK一致性")
    logging.info("  3. 关节耦合损失 - 因果依赖")
    logging.info("  4. Mixup数据增强")

    # 加载数据集
    logging.info("\n加载数据集...")
    train_loader, val_loader = create_windowed_dataloaders(config.data_path, config)

    # 创建因果IK模型（带历史窗口）
    logging.info(f"\n创建模型: {model_name}")
    model = PhysicsAwareCausalIKWithHistory(
        num_gnn_layers=config.num_gnn_layers,
        frame_dim=config.input_dim,  # 14维（7位姿+7角度）
        num_frames=config.num_frames,  # 10帧
        output_dim=config.output_dim,  # 7维关节角度
        hidden_dim=config.hidden_dim
    )
    model = model.to(config.device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"模型参数量: {total_params:.2f}M")

    # 加载Pinocchio模型（用于FK损失）
    logging.info("\n加载Pinocchio模型（用于物理约束）...")
    pin_model, pin_data, joint_ids, wrist_frame_id = load_robot_model()

    if pin_model is None:
        logging.warning("无法加载Pinocchio模型，将不使用FK损失")
        config.fk_weight = 0

    # 损失函数
    criterion = nn.MSELoss()

    # 创建损失函数
    if config.use_position_loss:
        # 使用位置误差为主损失
        logging.info("✓ 使用 PositionBasedLoss（直接优化位置误差）")
        loss_fn = PositionBasedLoss(
            position_weight=config.fk_weight,
            joint_weight=config.ik_weight,
            coupling_weight=config.coupling_weight
        )
    # else:
    #     # 使用物理感知损失（IK为主）
    #     logging.info("✓ 使用 PhysicsAwareLoss（IK为主）")
    #     loss_fn = PhysicsAwareLoss(
    #         ik_weight=config.ik_weight,
    #         fk_weight=config.fk_weight,
    #         coupling_weight=config.coupling_weight
    #     )

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.initial_lr, weight_decay=1e-4)

    # 学习率调度器 - 使用 CosineAnnealingWarmRestarts 获得更好的收敛
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,        # 每50个epoch重启一次
        T_mult=2,      # 每次重启周期翻倍
        eta_min=config.min_lr
    )

    # 断点路径（根据损失类型使用不同文件名）
    loss_type = "position" if config.use_position_loss else "ik"
    checkpoint_path = f"/home/wsy/Desktop/casual/{model_name}_{loss_type}_small_gap_074.pth"

    # 加载断点
    best_val_loss = float("inf")
    last_improvement_epoch = 0
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        logging.info("加载断点...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        logging.info(f"从第{start_epoch} epoch继续")

    # 训练循环
    logging.info(f"\n开始训练（{config.epochs} epochs）...")

    for epoch in range(start_epoch, config.epochs):
        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        train_ik_loss = 0.0
        train_fk_loss = 0.0
        train_coupling_loss = 0.0
        train_continuity_loss = 0.0  # 连续性损失累积
        train_preds = []
        train_trues = []
        start_time = time.time()

        mixup_count = 0
        normal_count = 0

        for batch_X, batch_y, batch_last_angle in train_loader:
            # batch_X: [batch, 10, 7]（位姿）, batch_y: [batch, 7]（角度）
            # batch_last_angle: [batch, 7] - 输入窗口最后一帧的角度
            batch_X, batch_y, batch_last_angle = batch_X.to(config.device), batch_y.to(config.device), batch_last_angle.to(config.device)

            # 扁平化时序输入: [batch, 10, 7] -> [batch, 70]
            batch_size = batch_X.shape[0]
            batch_X_flat = batch_X.view(batch_size, -1)  # [batch, 10*7]

            optimizer.zero_grad()

            # 前向传播（返回中间特征用于耦合损失）
            joint_angles, nodes = model(batch_X_flat, return_intermediates=True)

            # 计算损失
            if config.use_position_loss and pin_model is not None:
                # ====== 使用 PositionBasedLoss（直接优化位置误差） ======
                # 1. 用 Pinocchio 计算预测角度的手腕位置
                pred_wrist_pos = forward_kinematics_batch(
                    pin_model, pin_data, joint_ids, wrist_frame_id, joint_angles
                )

                # 2. 用 Pinocchio 计算真实角度的手腕位置（关键修改！）
                true_wrist_pos_fk = forward_kinematics_batch(
                    pin_model, pin_data, joint_ids, wrist_frame_id, batch_y
                )

                # 3. 计算位置误差为主损失
                total_loss, loss_dict = loss_fn(
                    pred_wrist_pos, true_wrist_pos_fk,  # 都用 Pinocchio 计算
                    joint_angles, batch_y, nodes
                )

                # 记录损失
                batch_pos_loss = loss_dict['position_loss']
                batch_joint_loss = loss_dict['joint_loss']
                batch_fk_loss = batch_pos_loss  # 位置损失

            # else:
                # ====== 使用 PhysicsAwareLoss（IK 为主） ======
                # 准备批次数据
                # batch_data = {
                #     'wrist_pose_features': batch_X,
                #     'target_joint_angles': batch_y,
                #     'target_wrist_pos': batch_X[:, :3]
                # }

                # # 计算基础物理感知损失（IK + Coupling）
                # total_loss, loss_dict = loss_fn(model, batch_data)

                # # 计算真实的FK损失（使用Pinocchio）
                # batch_fk_loss = 0.0
                # if pin_model is not None and config.fk_weight > 0:

                #     pred_wrist_pos = forward_kinematics_batch(
                #         pin_model, pin_data, joint_ids, wrist_frame_id, joint_angles
                #     )

                #     true_wrist_pos = forward_kinematics_batch(
                #         pin_model, pin_data, joint_ids, wrist_frame_id, batch_y
                #     )


                #     fk_loss = criterion(pred_wrist_pos, true_wrist_pos)
                #     total_loss = total_loss + config.fk_weight * fk_loss
                #     batch_fk_loss = fk_loss.item()

            # 连续性损失：预测角度与最后一帧角度的差异（防止突变）
            continuity_loss = torch.mean((joint_angles - batch_last_angle) ** 2)
            total_loss = total_loss + config.continuity_weight * continuity_loss

            # 累积连续性损失
            train_continuity_loss += continuity_loss.item() * batch_X.size(0)

            normal_count += batch_X.size(0)

            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += total_loss.item() * batch_X.size(0)

            if config.use_position_loss:
                # PositionBasedLoss 的日志
                train_ik_loss += batch_joint_loss * batch_X.size(0)
                train_fk_loss += batch_pos_loss * batch_X.size(0)
                train_coupling_loss += 0.0
            else:
                # PhysicsAwareLoss 的日志
                train_ik_loss += loss_dict['ik_loss'] * batch_X.size(0)
                train_fk_loss += batch_fk_loss * batch_X.size(0)
                train_coupling_loss += loss_dict['coupling_loss'] * batch_X.size(0)

            if len(train_preds) < 10000:
                train_preds.append(joint_angles.cpu().detach().numpy())
                train_trues.append(batch_y.cpu().detach().numpy())

        # 计算训练指标
        train_preds = np.vstack(train_preds)
        train_trues = np.vstack(train_trues)
        train_r2 = 1 - (np.sum((train_trues - train_preds) ** 2) /
                      (np.sum((train_trues - np.mean(train_trues)) ** 2) + 1e-8))
        train_loss = train_loss / len(train_loader.dataset)
        train_time = time.time() - start_time

        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_trues = []

        with torch.no_grad():
            val_fk_loss = 0.0
            for batch_X, batch_y, batch_last_angle in val_loader:
                # batch_X: [batch, 10, 7]（位姿）, batch_y: [batch, 7]（角度）
                # batch_last_angle: [batch, 7] - 最后一帧的角度
                batch_X, batch_y, batch_last_angle = batch_X.to(config.device), batch_y.to(config.device), batch_last_angle.to(config.device)

                # 扁平化时序输入: [batch, 10, 7] -> [batch, 70]
                batch_size = batch_X.shape[0]
                batch_X_flat = batch_X.view(batch_size, -1)  # [batch, 10*7]

                # 前向传播
                joint_angles, nodes = model(batch_X_flat, return_intermediates=True)

                # 计算损失（与训练阶段一致）
                if config.use_position_loss and pin_model is not None:
                    # ====== 使用 PositionBasedLoss ======
                    pred_wrist_pos = forward_kinematics_batch(
                        pin_model, pin_data, joint_ids, wrist_frame_id, joint_angles
                    )
                    # 从真实角度计算手腕位置（修改：都用pinocchio FK计算）
                    true_wrist_pos = forward_kinematics_batch(
                        pin_model, pin_data, joint_ids, wrist_frame_id, batch_y
                    )

                    total_loss, loss_dict = loss_fn(
                        pred_wrist_pos, true_wrist_pos,
                        joint_angles, batch_y, nodes
                    )
                    val_fk_loss += loss_dict['position_loss'] * batch_X.size(0)

                else:
                    # ====== 使用 PhysicsAwareLoss ======
                    batch_data = {
                        'wrist_pose_features': batch_X,
                        'target_joint_angles': batch_y,
                        'target_wrist_pos': batch_X[:, :3]
                    }

                    # 计算基础损失
                    total_loss, loss_dict = loss_fn(model, batch_data)

                    # 计算真实的FK损失（使用Pinocchio）
                    if pin_model is not None and config.fk_weight > 0:
                        pred_wrist_pos = forward_kinematics_batch(
                            pin_model, pin_data, joint_ids, wrist_frame_id, joint_angles
                        )
                        # 从真实角度计算手腕位置（修改：都用pinocchio FK计算）
                        true_wrist_pos = forward_kinematics_batch(
                            pin_model, pin_data, joint_ids, wrist_frame_id, batch_y
                        )
                        fk_loss = criterion(pred_wrist_pos, true_wrist_pos)
                        total_loss = total_loss + config.fk_weight * fk_loss
                        val_fk_loss += fk_loss.item() * batch_X.size(0)

                # 连续性损失：与训练阶段一致
                continuity_loss = torch.mean((joint_angles - batch_last_angle) ** 2)
                total_loss = total_loss + config.continuity_weight * continuity_loss

                val_loss += total_loss.item() * batch_X.size(0)

                val_preds.append(joint_angles.cpu().numpy())
                val_trues.append(batch_y.cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_trues = np.vstack(val_trues)
        val_r2 = 1 - (np.sum((val_trues - val_preds) ** 2) /
                      (np.sum((val_trues - np.mean(val_trues)) ** 2) + 1e-8))
        val_loss = val_loss / len(val_loader.dataset)

        # 学习率调度 (CosineAnnealingWarmRestarts 不需要参数)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 日志
        if config.use_position_loss:
            # PositionBasedLoss 日志（显示位置误差）
            logging.info(
                f"Epoch [{epoch}/{config.epochs}] | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"Train R²: {train_r2:.6f} | Val R²: {val_r2:.6f} | "
                f"GAP :{train_continuity_loss/len(train_loader.dataset):.6f} | "  # 连续性损失平均值
                f"Pos: {train_fk_loss/len(train_loader.dataset):.6f} | "  # 位置误差（m²）
                f"Joint: {train_ik_loss/len(train_loader.dataset):.4f} | "  # 关节误差
                f"LR: {current_lr:.6f} | Time: {train_time:.1f}s"
            )
        else:
            # PhysicsAwareLoss 日志
            logging.info(
                f"Epoch [{epoch}/{config.epochs}] | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f} | "
                f"IK: {train_ik_loss/len(train_loader.dataset):.4f} | "
                f"FK: {train_fk_loss/len(train_loader.dataset):.4f} | "
                f"Coup: {train_coupling_loss/len(train_loader.dataset):.4f} | "
                f"LR: {current_lr:.6f} | Time: {train_time:.1f}s | "
                f"Mixup: {mixup_count/(mixup_count+normal_count)*100:.0f}%"
            )

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            last_improvement_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            logging.info(f"  >>> 保存最优模型（验证损失：{val_loss:.4f}）")

        # # 早停检查
        # if epoch - last_improvement_epoch >= 20 and epoch >= 50:
        #     logging.info(f"早停：连续{epoch - last_improvement_epoch}个epoch验证损失未改善")
        #     break

    logging.info("\n训练完成！")
    logging.info(f"最优验证损失: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
