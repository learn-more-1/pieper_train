"""
训练基于隐式神经场的IK模型（2003版本 - 历史位姿）

核心特性:
1. 使用10帧历史位姿作为输入（从X的前7维提取）
2. 避免自回归误差累积
3. 捕捉运动趋势和动态信息
4. 时序卷积编码器处理历史位姿序列
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

sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/2101')

from model import ImplicitIKWithHistory, ImplicitIKWithHistoryEnsemble, compute_dataset_statistics, NormalizationLayer
from dataset_generalized import create_windowed_dataloaders

sys.path.insert(0, '/home/bonuli/Pieper/pieper1101')
try:
    from gpu_fk_wrapper import SimpleGPUFK
except ImportError:
    SimpleGPUFK = None

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("implicit_ik_train.log"), logging.StreamHandler()]
)


class Config:
    # 数据配置
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
    num_joints = 7
    num_frames = 10  # 历史帧数

    # 训练参数
    batch_size = 1024
    epochs = 150
    initial_lr = 1e-3
    min_lr = 1e-6
    early_stop_patience = 15    # 早停：验证损失15个epoch不下降则停止
    lr_patience = 5             # 学习率降低：验证损失5个epoch不下降则降低LR
    lr_factor = 0.5             # 学习率降低因子
    cooldown = 2                # 学习率降低后的冷却期

    # 设备
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # ========== 2003 配置 - 历史位姿版本 ==========
    pose_dim = 7
    joint_dim = 7
    hidden_dim = 1200       # 隐藏层维度
    temporal_hidden = 256   # 时序编码器隐藏层
    num_freqs = 15          # 位置编码频率
    num_models = 8          # 集成模型数量
    use_ensemble = True

    # 损失权重
    ik_weight = 1.0
    fk_weight = 1.0
    consistency_weight = 0.1

    # 每个关节的损失权重
    joint_loss_weights = torch.tensor([
        1.6,   # J0 (Shoulder Pitch)
        2.0,   # J1 (Shoulder Roll)
        1.6,   # J2 (Shoulder Yaw)
        2.5,   # J3 (Elbow)
        1.0,   # J4 (Forearm Roll)
        0.7,   # J5 (Wrist Yaw)
        0.6,   # J6 (Wrist Pitch)
    ])
    # ================================================

    num_workers = 8
    pin_memory = True

    urdf_path = "/home/bonuli/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
    use_pinocchio_fk = True


def load_pinocchio_model(urdf_path="/home/bonuli/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"):
    try:
        from pinocchio_fk import PinocchioFK
        pinocchio_fk = PinocchioFK(urdf_path)
        logging.info(f"✓ 成功加载Pinocchio FK模型")
        return pinocchio_fk
    except ImportError as e:
        logging.warning(f"Pinocchio导入失败: {e}")
        return None
    except Exception as e:
        logging.error(f"✗ 加载Pinocchio FK模型失败: {e}")
        return None


def forward_kinematics_with_pose(gpu_fk, joint_angles):
    if gpu_fk is None:
        return None, None
    positions = gpu_fk.forward(joint_angles)
    return positions, None


def weighted_mse_loss(pred, target, weights):
    squared_error = (pred - target) ** 2
    weighted_error = squared_error * weights.unsqueeze(0)
    return torch.mean(weighted_error)


def train():
    config = Config()

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = 'implicit_ik'

    logging.info("=" * 70)
    logging.info("训练基于隐式神经场的IK模型（2003 - 历史位姿版本）")
    logging.info("=" * 70)
    logging.info("核心特性:")
    logging.info("  1. 使用10帧历史位姿（从X提取）")
    logging.info("  2. 时序卷积编码器处理运动轨迹")
    logging.info("  3. 避免自回归误差累积")
    logging.info(f"  4. 集成模型: {config.num_models}个")
    logging.info(f"  5. 隐藏层: {config.hidden_dim}")

    # 打印关节权重
    joint_names = ['Shoulder Pitch', 'Shoulder Roll', 'Shoulder Yaw',
                   'Elbow', 'Forearm Roll', 'Wrist Yaw', 'Wrist Pitch']
    logging.info("\n关节损失权重:")
    for i, (name, weight) in enumerate(zip(joint_names, config.joint_loss_weights)):
        logging.info(f"  J{i} ({name:15s}): {weight:.2f}")

    joint_weights = config.joint_loss_weights.to(config.device)

    # 加载数据集
    logging.info("\n加载数据集...")
    train_loader, val_loader = create_windowed_dataloaders(config.data_path, config)

    # 计算统计信息
    logging.info("\n计算数据集统计信息...")
    pose_mean, pose_std, joint_mean, joint_std = compute_dataset_statistics(
        train_loader, config.pose_dim, config.joint_dim
    )

    norm_layer = NormalizationLayer(pose_mean, pose_std, joint_mean, joint_std)
    norm_layer = norm_layer.to(config.device)

    # 创建模型
    logging.info(f"\n创建模型: {model_name}")

    if config.use_ensemble:
        model = ImplicitIKWithHistoryEnsemble(
            pose_dim=config.pose_dim,
            joint_dim=config.joint_dim,
            hidden_dim=config.hidden_dim,
            temporal_hidden=config.temporal_hidden,
            num_freqs=config.num_freqs,
            num_frames=config.num_frames,
            num_models=config.num_models
        )
    else:
        model = ImplicitIKWithHistory(
            pose_dim=config.pose_dim,
            joint_dim=config.joint_dim,
            hidden_dim=config.hidden_dim,
            temporal_hidden=config.temporal_hidden,
            num_freqs=config.num_freqs,
            num_frames=config.num_frames
        )

    model = model.to(config.device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"模型参数量: {total_params:.2f}M")

    # 加载FK模型
    logging.info("\n加载FK模型...")
    gpu_fk = None
    if config.use_pinocchio_fk:
        gpu_fk = load_pinocchio_model(config.urdf_path)

    if gpu_fk is None:
        logging.info("使用SimpleGPU FK...")
        try:
            gpu_fk_obj = SimpleGPUFK()
            gpu_fk = gpu_fk_obj
        except:
            gpu_fk = None

    if gpu_fk is None:
        logging.warning("FK模型加载失败，将不使用FK损失")
        config.fk_weight = 0.0
        config.consistency_weight = 0.0
    else:
        logging.info(f"✓ FK模型加载成功")

    mse_criterion = nn.MSELoss()

    # 增加正则化：weight_decay从1e-4提高到1e-3
    optimizer = optim.AdamW(model.parameters(), lr=config.initial_lr, weight_decay=1e-3)

    # 使用ReduceLROnPlateau：根据验证损失动态调整学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',              # 验证损失越小越好
        factor=config.lr_factor, # LR降低因子
        patience=config.lr_patience,  # 多少个epoch不下降才降低LR
        min_lr=config.min_lr,
        cooldown=config.cooldown,  # 降低LR后的冷却期
        # verbose=True             # 打印LR调整信息
    )

    checkpoint_path = f"/home/bonuli/Pieper/2101/{model_name}_2101.pth"

    best_val_loss = float("inf")
    start_epoch = 0
    epochs_no_improve = 0  # 早停计数器
    if os.path.exists(checkpoint_path):
        logging.info("加载断点...")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        logging.info(f"从第{start_epoch} epoch继续")

    logging.info(f"\n开始训练（{config.epochs} epochs）...")

    for epoch in range(start_epoch, config.epochs):
        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        train_ik_loss = 0.0
        train_fk_loss = 0.0
        train_preds = []
        train_trues = []
        start_time = time.time()

        for batch_X, batch_y, batch_last_angle in train_loader:
            batch_X, batch_y, batch_last_angle = (
                batch_X.to(config.device, non_blocking=True),
                batch_y.to(config.device, non_blocking=True),
                batch_last_angle.to(config.device, non_blocking=True)
            )

            batch_size = batch_X.shape[0]

            # ========== 2003关键：从X提取历史位姿 ==========
            # batch_X: [batch, 10, 14] = [batch, 10, pose_dim(7) + joint_dim(7)]
            # 提取历史位姿（前7维）
            history_poses = batch_X[:, :, :config.pose_dim]  # [batch, 10, 7]

            # 提取目标位姿和目标角度
            if batch_y.shape[1] == 14:
                target_pose = batch_y[:, :config.pose_dim]
                target_angles = batch_y[:, config.pose_dim:]
            else:
                target_angles = batch_y
                # 如果y没有位姿，用最后一帧X的位姿作为目标
                target_pose = batch_X[:, -1, :config.pose_dim]

            optimizer.zero_grad()

            # 归一化
            target_pose_norm = norm_layer.normalize_pose(target_pose)
            history_poses_norm = norm_layer.normalize_history_poses(history_poses)

            # 前向传播
            pred_joint_angles = model(target_pose_norm, history_poses_norm)

            # 反归一化
            pred_joint_angles_denorm = norm_layer.denormalize_joint(pred_joint_angles)

            # IK损失
            ik_loss = weighted_mse_loss(pred_joint_angles_denorm, target_angles, joint_weights)

            # FK损失
            fk_loss = 0.0
            if config.fk_weight > 0 and gpu_fk is not None:
                pred_position, _ = forward_kinematics_with_pose(gpu_fk, pred_joint_angles_denorm)
                target_position_from_fk, _ = forward_kinematics_with_pose(gpu_fk, target_angles)

                if pred_position is not None and target_position_from_fk is not None:
                    fk_loss = mse_criterion(pred_position, target_position_from_fk)

            consistency_loss = 0.0
            if config.consistency_weight > 0:
                consistency_loss = torch.mean((pred_joint_angles_denorm - torch.clamp(
                    pred_joint_angles_denorm, -np.pi, np.pi
                )) ** 2)

            total_loss = (
                config.ik_weight * ik_loss +
                config.fk_weight * fk_loss +
                config.consistency_weight * consistency_loss
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * batch_size
            train_ik_loss += ik_loss.item() * batch_size
            train_fk_loss += fk_loss.item() * batch_size

            if len(train_preds) < 10000:
                train_preds.append(pred_joint_angles_denorm.cpu().detach().numpy())
                train_trues.append(target_angles.cpu().detach().numpy())

        train_preds = np.vstack(train_preds)
        train_trues = np.vstack(train_trues)
        train_r2 = 1 - (np.sum((train_trues - train_preds) ** 2) /
                      (np.sum((train_trues - np.mean(train_trues)) ** 2) + 1e-8))
        train_loss = train_loss / len(train_loader.dataset)
        train_time = time.time() - start_time

        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0.0
        val_ik_loss = 0.0
        val_fk_loss = 0.0
        val_preds = []
        val_trues = []

        with torch.no_grad():
            for batch_X, batch_y, batch_last_angle in val_loader:
                batch_X, batch_y, batch_last_angle = (
                    batch_X.to(config.device, non_blocking=True),
                    batch_y.to(config.device, non_blocking=True),
                    batch_last_angle.to(config.device, non_blocking=True)
                )

                batch_size = batch_X.shape[0]

                # 提取历史位姿
                history_poses = batch_X[:, :, :config.pose_dim]

                if batch_y.shape[1] == 14:
                    target_pose = batch_y[:, :config.pose_dim]
                    target_angles = batch_y[:, config.pose_dim:]
                else:
                    target_angles = batch_y
                    target_pose = batch_X[:, -1, :config.pose_dim]

                target_pose_norm = norm_layer.normalize_pose(target_pose)
                history_poses_norm = norm_layer.normalize_history_poses(history_poses)

                pred_joint_angles = model(target_pose_norm, history_poses_norm)
                pred_joint_angles_denorm = norm_layer.denormalize_joint(pred_joint_angles)

                ik_loss = weighted_mse_loss(pred_joint_angles_denorm, target_angles, joint_weights)

                fk_loss = 0.0
                if config.fk_weight > 0 and gpu_fk is not None:
                    pred_position, _ = forward_kinematics_with_pose(gpu_fk, pred_joint_angles_denorm)
                    target_position_from_fk, _ = forward_kinematics_with_pose(gpu_fk, target_angles)

                    if pred_position is not None and target_position_from_fk is not None:
                        fk_loss = mse_criterion(pred_position, target_position_from_fk)

                consistency_loss = 0.0

                total_loss = (
                    config.ik_weight * ik_loss +
                    config.fk_weight * fk_loss +
                    config.consistency_weight * consistency_loss
                )

                val_loss += total_loss.item() * batch_size
                val_ik_loss += ik_loss.item() * batch_size
                val_fk_loss += fk_loss.item() * batch_size

                val_preds.append(pred_joint_angles_denorm.cpu().numpy())
                val_trues.append(target_angles.cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_trues = np.vstack(val_trues)
        val_r2 = 1 - (np.sum((val_trues - val_preds) ** 2) /
                      (np.sum((val_trues - np.mean(val_trues)) ** 2) + 1e-8))
        val_loss = val_loss / len(val_loader.dataset)

        # 学习率调度：传入验证损失
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            is_best = True
        else:
            epochs_no_improve += 1
            is_best = False

        # 打印早停信息
        if epochs_no_improve > 0:
            logging.info(f"  验证损失未改善: {epochs_no_improve}/{config.early_stop_patience}")

        # 检查是否需要早停
        if epochs_no_improve >= config.early_stop_patience:
            logging.info(f"\n早停触发！验证损失连续{config.early_stop_patience}个epoch未改善")
            logging.info(f"停止训练，最优验证损失: {best_val_loss:.6f}")
            break

        logging.info(
            f"Epoch [{epoch+1}/{config.epochs}] | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Train R²: {train_r2:.6f} | Val R²: {val_r2:.6f} | "
            f"IK: {train_ik_loss/len(train_loader.dataset):.4f} | "
            f"FK: {train_fk_loss/len(train_loader.dataset):.6f} | "
            f"LR: {current_lr:.6f} | Time: {train_time:.1f}s"
        )

        # 保存最优模型
        if is_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'pose_mean': pose_mean,
                'pose_std': pose_std,
                'joint_mean': joint_mean,
                'joint_std': joint_std,
                'config': {
                    'pose_dim': config.pose_dim,
                    'joint_dim': config.joint_dim,
                    'hidden_dim': config.hidden_dim,
                    'temporal_hidden': config.temporal_hidden,
                    'num_freqs': config.num_freqs,
                    'num_frames': config.num_frames,
                    'num_models': config.num_models,
                    'use_ensemble': config.use_ensemble,
                }
            }, checkpoint_path)
            logging.info(f"  >>> 保存最优模型（验证损失：{val_loss:.6f}）")

    logging.info("\n训练完成！")
    logging.info(f"最优验证损失: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
