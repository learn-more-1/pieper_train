"""
训练基于隐式神经场的IK模型（带关节权重版本）

核心思想:
1. 使用隐式神经场学习 位姿->关节角 的直接映射
2. 无需历史帧信息
3. 支持端到端训练，实时推理
4. 带关节权重：根据每个关节的预测误差调整损失权重
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
sys.path.insert(0, '/home/bonuli/Pieper/2001')

from model import ImplicitIK, ImplicitIKEnsemble, compute_dataset_statistics, NormalizationLayer
from dataset_generalized import create_windowed_dataloaders

# 复制GPU FK wrapper（如果没有的话）
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

    # 训练参数
    batch_size = 1024
    epochs = 150
    initial_lr = 1e-3
    min_lr = 1e-6
    patience = 10
    factor = 0.5

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 模型配置
    pose_dim = 7
    joint_dim = 7
    hidden_dim = 1000
    num_freqs = 10          # 位置编码频率数量
    use_condition = False   # 是否使用当前关节角作为条件
    use_ensemble = True    # 是否使用集成模型
    num_models = 5          # 集成模型数量

    # 损失权重
    ik_weight = 1.0         # 关节角度损失
    fk_weight = 0.5         # 位置误差损失（使用Pinocchio FK时建议保持此值）
    consistency_weight = 0.1  # 一致性损失（预测角度的FK应该接近目标位姿）

    # ========== 新增：每个关节的损失权重 ==========
    # 根据GRAB测试结果，误差大的关节给更高权重
    # 测试结果: J0:16.30°, J1:20.32°, J2:16.08°, J3:25.44°, J4:10.53°, J5:7.43°, J6:6.12°
    # 权重设计：误差越大，权重越高 (按比例缩放)
    joint_loss_weights = torch.tensor([
        1.6,   # J0 (Shoulder Pitch): 16.30°
        2.0,   # J1 (Shoulder Roll):  20.32° (最大)
        1.6,   # J2 (Shoulder Yaw):   16.08°
        2.5,   # J3 (Elbow):          25.44° (最大)
        1.0,   # J4 (Forearm Roll):   10.53° (基准)
        0.7,   # J5 (Wrist Yaw):       7.43°
        0.6,   # J6 (Wrist Pitch):    6.12° (最小)
    ])
    # ================================================

    # 数据加载优化
    num_workers = 8
    pin_memory = True

    # FK配置
    urdf_path = "/home/bonuli/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
    use_pinocchio_fk = True  # 是否使用Pinocchio FK（更精确但较慢，推荐设为True）


def load_robot_model():
    """加载GPU FK模型"""
    if SimpleGPUFK is None:
        logging.warning("SimpleGPUFK不可用")
        return None, None

    try:
        gpu_fk = SimpleGPUFK()
        left_arm_joints = [16, 17, 18, 19, 20, 21, 22]
        logging.info(f"✓ 成功加载GPU FK模型 (SimpleGPUFK)")
        logging.info(f"  左臂关节 ID: {left_arm_joints}")
        return gpu_fk, left_arm_joints
    except Exception as e:
        logging.error(f"✗ 加载GPU FK模型失败: {e}")
        return None, None


# 可选：加载Pinocchio FK（更精确但速度较慢）
def load_pinocchio_model(urdf_path="/home/bonuli/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"):
    """
    加载Pinocchio FK模型（可选，更精确）

    Args:
        urdf_path: URDF文件路径

    Returns:
        PinocchioFK实例或None
    """
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
    """
    GPU加速的批量FK计算（仅位置）

    Args:
        gpu_fk: GPU FK模型
        joint_angles: [batch, 7] 关节角度

    Returns:
        positions: [batch, 3] 手腕位置
        orientations: [batch, 4] None
    """
    if gpu_fk is None:
        return None, None
    positions = gpu_fk.forward(joint_angles)
    return positions, None


def weighted_mse_loss(pred, target, weights):
    """
    带权重的MSE损失

    Args:
        pred: [batch, num_joints] 预测值
        target: [batch, num_joints] 目标值
        weights: [num_joints] 每个关节的权重

    Returns:
        loss: 标量损失
    """
    # 计算每个样本的平方误差 [batch, num_joints]
    squared_error = (pred - target) ** 2
    # 应用权重并求平均
    weighted_error = squared_error * weights.unsqueeze(0)  # 广播
    return torch.mean(weighted_error)


def train():
    config = Config()

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = 'implicit_ik'

    logging.info("=" * 70)
    logging.info("训练基于隐式神经场的IK模型（带关节权重版本）")
    logging.info("=" * 70)
    logging.info("核心特性:")
    logging.info("  1. 隐式神经场：位姿 -> 关节角直接映射")
    logging.info("  2. 位置编码：多尺度正弦/余弦特征")
    logging.info("  3. 无需历史帧信息")
    logging.info(f"  4. 使用条件: {config.use_condition}")
    logging.info(f"  5. 使用集成: {config.use_ensemble}")
    logging.info("  6. 带关节权重：针对误差大的关节增加损失权重")

    # 打印关节权重
    joint_names = ['Shoulder Pitch', 'Shoulder Roll', 'Shoulder Yaw',
                   'Elbow', 'Forearm Roll', 'Wrist Yaw', 'Wrist Pitch']
    logging.info("\n关节损失权重:")
    for i, (name, weight) in enumerate(zip(joint_names, config.joint_loss_weights)):
        logging.info(f"  J{i} ({name:15s}): {weight:.2f}")

    # 将权重移到设备
    joint_weights = config.joint_loss_weights.to(config.device)

    # 加载数据集
    logging.info("\n加载数据集...")
    train_loader, val_loader = create_windowed_dataloaders(config.data_path, config)

    # 计算数据集统计信息（用于归一化）
    logging.info("\n计算数据集统计信息...")
    pose_mean, pose_std, joint_mean, joint_std = compute_dataset_statistics(
        train_loader, config.pose_dim, config.joint_dim
    )

    # 创建归一化层
    norm_layer = NormalizationLayer(pose_mean, pose_std, joint_mean, joint_std)
    norm_layer = norm_layer.to(config.device)

    # 创建模型
    logging.info(f"\n创建模型: {model_name}")

    if config.use_ensemble:
        model = ImplicitIKEnsemble(
            pose_dim=config.pose_dim,
            joint_dim=config.joint_dim,
            hidden_dim=config.hidden_dim,
            num_models=config.num_models,
            num_freqs=config.num_freqs
        )
    else:
        model = ImplicitIK(
            pose_dim=config.pose_dim,
            joint_dim=config.joint_dim,
            hidden_dim=config.hidden_dim,
            use_fourier=False,
            num_freqs=config.num_freqs,
            use_condition=config.use_condition
        )

    model = model.to(config.device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"模型参数量: {total_params:.2f}M")

    # 加载FK模型（用于FK损失）
    logging.info("\n加载FK模型（用于FK损失）...")

    # 首选：Pinocchio FK（精确，推荐用于训练）
    gpu_fk = None
    if config.use_pinocchio_fk:
        logging.info("尝试加载Pinocchio FK...")
        gpu_fk = load_pinocchio_model(config.urdf_path)

    # 备选：SimpleGPU FK（快速但可能不够精确）
    if gpu_fk is None:
        logging.info("使用SimpleGPU FK（注意：连杆参数可能不够精确，建议使用Pinocchio FK）...")
        gpu_fk, joint_ids = load_robot_model()

    if gpu_fk is None:
        logging.warning("FK模型加载失败，将不使用FK损失")
        config.fk_weight = 0.0
        config.consistency_weight = 0.0
    else:
        logging.info(f"✓ FK模型加载成功，FK权重: {config.fk_weight}")

    # 损失函数
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.initial_lr, weight_decay=1e-4)

    # 学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=70,
        T_mult=2,
        eta_min=config.min_lr
    )

    # 断点路径
    checkpoint_path = f"/home/bonuli/Pieper/2001/{model_name}_2001.pth"

    # 加载断点
    best_val_loss = float("inf")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        logging.info("加载断点...")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
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
        train_consistency_loss = 0.0
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

            # 判断y的格式并提取目标位姿和角度
            if batch_y.shape[1] == 14:
                target_pose = batch_y[:, :7]   # [batch, 7] 目标位姿
                target_angles = batch_y[:, 7:]  # [batch, 7] 目标关节角
                # 使用目标位姿的位置部分
                target_position = target_pose[:, :3]
            else:
                target_angles = batch_y  # [batch, 7]
                # 使用FK计算目标位置
                target_position, _ = forward_kinematics_with_pose(gpu_fk, target_angles)
                if target_position is None:
                    target_position = torch.zeros(batch_size, 3, device=config.device)

                # 构造假的目标位姿（用于训练）
                target_position_xy = target_position[:, :2]
                target_position_z = target_position[:, 2:3]
                target_pose = torch.cat([
                    target_position,
                    torch.zeros(batch_size, 4, device=config.device)  # 假四元数
                ], dim=1)

            optimizer.zero_grad()

            # 归一化目标位姿
            target_pose_norm = norm_layer.normalize_pose(target_pose)

            # 前向传播
            if config.use_condition:
                # 使用当前关节角作为条件
                current_angles = batch_last_angle
                pred_joint_angles = model(target_pose_norm, current_angles)
            else:
                pred_joint_angles = model(target_pose_norm)

            # 反归一化预测的关节角
            pred_joint_angles_denorm = norm_layer.denormalize_joint(pred_joint_angles)

            # ========== 使用带权重的IK损失 ==========
            ik_loss = weighted_mse_loss(pred_joint_angles_denorm, target_angles, joint_weights)
            # =========================================

            # 2. FK损失：预测角度的末端位置 vs 目标角度的末端位置
            fk_loss = 0.0
            consistency_loss = 0.0

            if config.fk_weight > 0:
                # 用预测角度计算FK位置
                pred_position, _ = forward_kinematics_with_pose(gpu_fk, pred_joint_angles_denorm)
                # 用目标角度计算FK位置
                target_position_from_fk, _ = forward_kinematics_with_pose(gpu_fk, target_angles)

                if pred_position is not None and target_position_from_fk is not None:
                    # FK损失：预测位置 vs 从目标角度计算的位置
                    fk_loss = mse_criterion(pred_position, target_position_from_fk)

            # 3. 一致性损失（可选）
            if config.consistency_weight > 0:
                # 确保预测的关节角在合理范围内
                consistency_loss = torch.mean((pred_joint_angles_denorm - torch.clamp(
                    pred_joint_angles_denorm, -np.pi, np.pi
                )) ** 2)

            # 总损失
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
            train_consistency_loss += consistency_loss.item() * batch_size

            if len(train_preds) < 10000:
                train_preds.append(pred_joint_angles_denorm.cpu().detach().numpy())
                train_trues.append(target_angles.cpu().detach().numpy())

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

                # 判断y的格式并提取目标位姿和角度
                if batch_y.shape[1] == 14:
                    target_pose = batch_y[:, :7]
                    target_angles = batch_y[:, 7:]
                    target_position = target_pose[:, :3]
                else:
                    target_angles = batch_y
                    target_position, _ = forward_kinematics_with_pose(gpu_fk, target_angles)
                    if target_position is None:
                        target_position = torch.zeros(batch_size, 3, device=config.device)

                    target_pose = torch.cat([
                        target_position,
                        torch.zeros(batch_size, 4, device=config.device)
                    ], dim=1)

                # 归一化目标位姿
                target_pose_norm = norm_layer.normalize_pose(target_pose)

                # 前向传播
                if config.use_condition:
                    current_angles = batch_last_angle
                    pred_joint_angles = model(target_pose_norm, current_angles)
                else:
                    pred_joint_angles = model(target_pose_norm)

                pred_joint_angles_denorm = norm_layer.denormalize_joint(pred_joint_angles)

                # ========== 使用带权重的IK损失 ==========
                ik_loss = weighted_mse_loss(pred_joint_angles_denorm, target_angles, joint_weights)
                # =========================================

                fk_loss = 0.0
                if config.fk_weight > 0:
                    # 用预测角度计算FK位置
                    pred_position, _ = forward_kinematics_with_pose(gpu_fk, pred_joint_angles_denorm)
                    # 用目标角度计算FK位置
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

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 日志
        logging.info(
            f"Epoch [{epoch}/{config.epochs}] | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Train R²: {train_r2:.6f} | Val R²: {val_r2:.6f} | "
            f"IK: {train_ik_loss/len(train_loader.dataset):.4f} | "
            f"FK: {train_fk_loss/len(train_loader.dataset):.6f} | "
            f"Cons: {train_consistency_loss/len(train_loader.dataset):.6f} | "
            f"LR: {current_lr:.6f} | Time: {train_time:.1f}s"
        )

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
                    'num_freqs': config.num_freqs,
                    'use_condition': config.use_condition,
                    'use_ensemble': config.use_ensemble,
                    'num_models': config.num_models,
                }
            }, checkpoint_path)
            logging.info(f"  >>> 保存最优模型（验证损失：{val_loss:.6f}）")

    logging.info("\n训练完成！")
    logging.info(f"最优验证损失: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
