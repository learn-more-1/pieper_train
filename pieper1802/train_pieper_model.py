"""
训练 Pieper 准则因果IK模型

核心思想:
1. Pieper 准则：不同关节对末端位姿的影响程度不同
2. Shoulder 的 rpy 对末端位置影响大
3. Shoulder 和 wrist 的 rp 对 wrist yaw 影响大
4. 使用 Attention 动态学习影响权重
5. 分别对末端位置和姿态做 FiLM 调制

数据流:
- 输入: 历史关节角度 [batch, 10, 7]
- 预处理: 提取最后一帧作为当前关节角度
- FK计算: 从目标关节角度计算末端位姿(位置+四元数)
- 模型: (当前关节角度, 目标末端位姿) -> 预测关节角度
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
from causal_ik_model_pieper2 import PieperCausalIK
sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from dataset_generalized import create_windowed_dataloaders
from gpu_fk_wrapper import SimpleGPUFK  # GPU加速FK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("pieper_train.log"), logging.StreamHandler()]
)


class Config:
    # 数据配置
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
    num_joints = 7
    num_frames = 10

    # 训练参数
    batch_size = 512
    epochs = 300
    initial_lr = 1e-3
    min_lr = 1e-6
    patience = 10
    factor = 0.5

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 模型配置
    joint_dim = 7
    hidden_dim = 256
    num_layers = 2

    # 损失权重
    ik_weight = 1.0       # 关节角度损失
    fk_weight = 0.5       # 位置误差损失
    continuity_weight = 1.0  # 连续性损失

    # 数据加载优化（提升GPU利用率）
    num_workers = 4       # 多进程数据加载
    pin_memory = True     # 加速CPU->GPU传输


def load_robot_model():
    """加载GPU FK模型"""
    try:
        gpu_fk = SimpleGPUFK()
        left_arm_joints = [16, 17, 18, 19, 20, 21, 22]
        logging.info(f"✓ 成功加载GPU FK模型")
        logging.info(f"  左臂关节 ID: {left_arm_joints}")
        return gpu_fk, left_arm_joints
    except Exception as e:
        logging.error(f"✗ 加载GPU FK模型失败: {e}")
        return None, None


def forward_kinematics_with_pose(gpu_fk, joint_angles):
    """
    GPU加速的批量FK计算（仅位置）

    Args:
        gpu_fk: GPU FK模型
        joint_angles: [batch, 7] 关节角度

    Returns:
        positions: [batch, 3] 手腕位置
        orientations: [batch, 4] None (保持接口兼容)
    """
    positions = gpu_fk.forward(joint_angles)
    return positions, None


def train():
    config = Config()

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = 'pieper_causal_ik'

    logging.info("=" * 70)
    logging.info("训练 Pieper 准则因果IK模型")
    logging.info("=" * 70)
    logging.info("核心特性:")
    logging.info("  1. Attention学习关节对末端位姿的影响权重")
    logging.info("  2. 分别对位置和姿态做FiLM调制")
    logging.info("  3. GNN学习关节间因果耦合")

    # 加载数据集
    logging.info("\n加载数据集...")
    train_loader, val_loader = create_windowed_dataloaders(config.data_path, config)

    # 创建Pieper模型
    logging.info(f"\n创建模型: {model_name}")
    model = PieperCausalIK(
        num_joints=config.joint_dim,
        num_frames=config.num_frames,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers
    )
    model = model.to(config.device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"模型参数量: {total_params:.2f}M")

    # 加载GPU FK模型（用于FK计算）
    logging.info("\n加载GPU FK模型（用于计算末端位姿）...")
    gpu_fk, joint_ids = load_robot_model()

    if gpu_fk is None:
        logging.error("无法加载GPU FK模型，退出")
        return

    # 损失函数
    mse_criterion = nn.MSELoss()

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
    checkpoint_path = f"/home/bonuli/Pieper/pieper1802/{model_name}_1101.pth"

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
        train_continuity_loss = 0.0
        train_preds = []
        train_trues = []
        start_time = time.time()

        for batch_X, batch_y, batch_last_angle in train_loader:
            # batch_X: [batch, 10, 7] 机器人历史角度
            # batch_y: [batch, 7] or [batch, 14] - 机器人角度，或(人位姿+机器人角度)
            # batch_last_angle: [batch, 7] 历史窗口最后一帧的角度
            # 使用non_async加速CPU->GPU传输
            batch_X, batch_y, batch_last_angle = (
                batch_X.to(config.device, non_blocking=True),
                batch_y.to(config.device, non_blocking=True),
                batch_last_angle.to(config.device, non_blocking=True)
            )

            batch_size = batch_X.shape[0]

            # 1. 使用历史窗口（10帧）
            history_frames = batch_X  # [batch, 10, 7]

            # 2. 判断y的格式并提取目标位姿和角度
            if batch_y.shape[1] == 14:
                # y包含位姿信息: 前7维人位姿，后7维机器人角度
                target_pose = batch_y[:, :7]   # [batch, 7] 人目标位姿
                target_angles = batch_y[:, 7:]  # [batch, 7] 机器人目标角度
                # 从人位姿提取末端位置和姿态
                target_position = target_pose[:, :3]      # [batch, 3]
                target_orientation = target_pose[:, 3:7]  # [batch, 4] 四元数
            else:
                # y只包含机器人角度，需要用FK计算位姿
                target_angles = batch_y  # [batch, 7]
                # 从机器人目标角度计算末端位姿（使用FK）
                target_position, target_orientation = forward_kinematics_with_pose(gpu_fk, target_angles)
                # GPU FK只返回位置，姿态使用None（模型内部会处理）
                target_orientation = None

            optimizer.zero_grad()

            # 3. 前向传播
            pred_joint_angles, info = model(
                history_frames,
                target_position,
                target_orientation
            )

            # 4. 计算损失
            # IK损失: 预测角度 vs 目标角度
            ik_loss = mse_criterion(pred_joint_angles, target_angles)

            # FK损失: 预测角度的末端位置 vs 目标末端位置
            pred_position, _ = forward_kinematics_with_pose(gpu_fk, pred_joint_angles)

            fk_position, _ = forward_kinematics_with_pose(gpu_fk, target_angles)
            fk_loss = mse_criterion(pred_position, fk_position)

            # 连续性损失: 预测角度与当前角度的差异（使用历史最后一帧）
            continuity_loss = torch.mean((pred_joint_angles - batch_last_angle) ** 2)

            # 总损失
            total_loss = (
                config.ik_weight * ik_loss +
                config.fk_weight * fk_loss +
                config.continuity_weight * continuity_loss
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * batch_size
            train_ik_loss += ik_loss.item() * batch_size
            train_fk_loss += fk_loss.item() * batch_size
            train_continuity_loss += continuity_loss.item() * batch_size

            if len(train_preds) < 10000:
                train_preds.append(pred_joint_angles.cpu().detach().numpy())
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
                # 使用non_blocking加速CPU->GPU传输
                batch_X, batch_y, batch_last_angle = (
                    batch_X.to(config.device, non_blocking=True),
                    batch_y.to(config.device, non_blocking=True),
                    batch_last_angle.to(config.device, non_blocking=True)
                )

                batch_size = batch_X.shape[0]

                # 1. 使用历史窗口（10帧）
                history_frames = batch_X  # [batch, 10, 7]

                # 2. 判断y的格式并提取目标位姿和角度
                if batch_y.shape[1] == 14:
                    # y包含位姿信息
                    target_pose = batch_y[:, :7]
                    target_angles = batch_y[:, 7:]
                    target_position = target_pose[:, :3]
                    target_orientation = target_pose[:, 3:7]
                else:
                    # y只包含机器人角度，需要用FK计算位姿
                    target_angles = batch_y
                    target_position, target_orientation = forward_kinematics_with_pose(gpu_fk, target_angles)
                    # GPU FK只返回位置，姿态使用None
                    target_orientation = None

                # 3. 前向传播
                pred_joint_angles, info = model(
                    history_frames,
                    target_position,
                    target_orientation
                )

                # 4. 计算损失
                ik_loss = mse_criterion(pred_joint_angles, target_angles)

                pred_position, _ = forward_kinematics_with_pose(gpu_fk, pred_joint_angles)
                fk_loss = mse_criterion(pred_position, target_position)

                continuity_loss = torch.mean((pred_joint_angles - batch_last_angle) ** 2)

                total_loss = (
                    config.ik_weight * ik_loss +
                    config.fk_weight * fk_loss +
                    config.continuity_weight * continuity_loss
                )

                val_loss += total_loss.item() * batch_size
                val_ik_loss += ik_loss.item() * batch_size
                val_fk_loss += fk_loss.item() * batch_size

                val_preds.append(pred_joint_angles.cpu().numpy())
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
            f"GAP: {train_continuity_loss/len(train_loader.dataset):.4f} | "
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
            }, checkpoint_path)
            logging.info(f"  >>> 保存最优模型（验证损失：{val_loss:.6f}）")

    logging.info("\n训练完成！")
    logging.info(f"最优验证损失: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
