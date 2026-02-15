"""
GPU利用率优化方案

当前问题：GPU利用率只有15%，说明GPU大部分时间在等待CPU准备数据

优化方案（按优先级排序）：
1. 多进程数据加载 (num_workers > 0)
2. 使用pin_memory加速CPU->GPU传输
3. 混合精度训练 (FP16)
4. torch.compile() 加速模型
5. 增大batch size

预期加速：3-5x
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
from causal_ik_model_pieper import PieperCausalIK
sys.path.insert(0, '/home/wsy/Desktop/casual')
sys.path.insert(0, '/home/wsy/Desktop/casual/pieper_NN')

from dataset_generalized import create_windowed_dataloaders
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("pieper_train_optimized.log"), logging.StreamHandler()]
)


class Config:
    # 数据配置
    data_path = "/home/wsy/Desktop/casual/merged_training_data.npz"
    num_joints = 7
    num_frames = 10

    # 训练参数（优化版）
    batch_size = 1024  # 增大batch size（从512）
    epochs = 300
    initial_lr = 2e-3  # 相应增大学习率
    min_lr = 1e-6
    patience = 10
    factor = 0.5

    # 设备
    device = torch.device("cuda:0")

    # 模型配置
    joint_dim = 7
    hidden_dim = 256
    num_layers = 2

    # 损失权重
    ik_weight = 1.0
    fk_weight = 0.5
    continuity_weight = 1.0

    # 数据加载优化
    num_workers = 4  # 多进程数据加载
    pin_memory = True  # 加速CPU->GPU传输
    prefetch_factor = 2  # 预取batch

    # 混合精度训练
    use_amp = True  # 自动混合精度（FP16）


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
    """GPU加速的批量FK计算"""
    positions = gpu_fk.forward(joint_angles)
    return positions, None


def train():
    config = Config()

    model_name = 'pieper_causalik_optimized'

    logging.info("=" * 70)
    logging.info("训练 Pieper 准则因果IK模型（GPU优化版）")
    logging.info("=" * 70)
    logging.info("优化措施:")
    logging.info("  1. num_workers=4 (多进程数据加载)")
    logging.info("  2. pin_memory=True (加速数据传输)")
    logging.info("  3. batch_size=1024 (增大batch)")
    logging.info("  4. 混合精度训练 (FP16)")
    logging.info("  5. torch.compile() (模型编译)")

    # 加载数据集（优化版DataLoader）
    logging.info("\n加载数据集...")
    train_loader, val_loader = create_windowed_dataloaders(
        config.data_path, config,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    # 创建模型
    logging.info(f"\n创建模型: {model_name}")
    model = PieperCausalIK(
        num_joints=config.joint_dim,
        num_frames=config.num_frames,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers
    )
    model = model.to(config.device)

    # 编译模型（PyTorch 2.0+）
    try:
        logging.info("编译模型（torch.compile）...")
        model = torch.compile(model, mode='max-autotune')
        logging.info("✓ 模型编译成功")
    except:
        logging.warning("⚠ torch.compile失败，使用原始模型")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"模型参数量: {total_params:.2f}M")

    # 加载GPU FK
    logging.info("\n加载GPU FK模型...")
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

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

    # 断点路径
    checkpoint_path = f"/home/wsy/Desktop/casual/pieper_NN/{model_name}.pth"

    # 加载断点
    best_val_loss = float("inf")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        logging.info("加载断点...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        if scaler and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
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
            # 数据移到GPU（使用pin_memory加速）
            batch_X = batch_X.to(config.device, non_blocking=True)
            batch_y = batch_y.to(config.device, non_blocking=True)
            batch_last_angle = batch_last_angle.to(config.device, non_blocking=True)

            history_frames = batch_X  # [batch, 10, 7]
            batch_size = batch_X.shape[0]

            # 判断y的格式
            if batch_y.shape[1] == 14:
                target_angles = batch_y[:, 7:]
            else:
                target_angles = batch_y

            optimizer.zero_grad()

            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                # 前向传播
                pred_joint_angles, info = model(
                    history_frames,
                    torch.randn(batch_size, 3).to(config.device),  # 临时placeholder
                    None  # 不使用姿态
                )

                # FK损失
                pred_position, _ = forward_kinematics_with_pose(gpu_fk, pred_joint_angles)
                fk_position, _ = forward_kinematics_with_pose(gpu_fk, target_angles)

                # 计算损失
                ik_loss = mse_criterion(pred_joint_angles, target_angles)
                fk_loss = mse_criterion(pred_position, fk_position)
                continuity_loss = torch.mean((pred_joint_angles - batch_last_angle) ** 2)

                total_loss = (
                    config.ik_weight * ik_loss +
                    config.fk_weight * fk_loss +
                    config.continuity_weight * continuity_loss
                )

            # 反向传播（混合精度）
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
        val_preds = []
        val_trues = []

        with torch.no_grad():
            for batch_X, batch_y, batch_last_angle in val_loader:
                batch_X = batch_X.to(config.device, non_blocking=True)
                batch_y = batch_y.to(config.device, non_blocking=True)
                batch_last_angle = batch_last_angle.to(config.device, non_blocking=True)

                history_frames = batch_X
                batch_size = batch_X.shape[0]

                if batch_y.shape[1] == 14:
                    target_angles = batch_y[:, 7:]
                else:
                    target_angles = batch_y

                with torch.cuda.amp.autocast(enabled=config.use_amp):
                    pred_joint_angles, _ = model(
                        history_frames,
                        torch.randn(batch_size, 3).to(config.device),
                        None
                    )

                    pred_position, _ = forward_kinematics_with_pose(gpu_fk, pred_joint_angles)
                    fk_position, _ = forward_kinematics_with_pose(gpu_fk, target_angles)

                    ik_loss = mse_criterion(pred_joint_angles, target_angles)
                    fk_loss = mse_criterion(pred_position, fk_position)
                    continuity_loss = torch.mean((pred_joint_angles - batch_last_angle) ** 2)

                    total_loss = (
                        config.ik_weight * ik_loss +
                        config.fk_weight * fk_loss +
                        config.continuity_weight * continuity_loss
                    )

                val_loss += total_loss.item() * batch_size

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
            f"FK: {train_fk_loss/len(train_loader.dataset):.4f} | "
            f"GAP: {train_continuity_loss/len(train_loader.dataset):.4f} | "
            f"LR: {current_lr:.6f} | Time: {train_time:.1f}s"
        )

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            if scaler is not None:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            torch.save(save_dict, checkpoint_path)
            logging.info(f"  >>> 保存最优模型（验证损失：{val_loss:.6f}）")

    logging.info("\n训练完成！")
    logging.info(f"最优验证损失: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
