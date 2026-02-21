"""
训练简化的IK模型 - 无需历史帧

核心思想：
- 用可学习的 joint_coupling_prototype 记住关节间的耦合关系
- 推理时只需输入目标位姿，无需历史角度
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
sys.path.insert(0, '/home/bonuli/pieper/casual')
sys.path.insert(0, '/home/bonuli/Pieper/casual/pieper1101')

from causal_ik_model_pieper_simple import SimplifiedCausalIK
from dataset_generalized import create_windowed_dataloaders
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("train_simplified.log"), logging.StreamHandler()]
)


class Config:
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
    batch_size = 512
    epochs = 100
    lr = 1e-3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 模型配置
    num_joints = 7
    hidden_dim = 256

    # 损失权重
    ik_weight = 1.0
    fk_weight = 0.3

    # 保存路径
    checkpoint_path = "/home/bonuli/Pieper/pieper1702/simplified_causal_ik.pth"


def load_robot_model():
    try:
        gpu_fk = SimpleGPUFK()
        logging.info(f"✓ 成功加载GPU FK模型")
        return gpu_fk
    except Exception as e:
        logging.error(f"✗ 加载GPU FK模型失败: {e}")
        return None


def train():
    config = Config()

    logging.info("=" * 70)
    logging.info("训练简化的IK模型（无需历史帧）")
    logging.info("=" * 70)
    logging.info("核心特性:")
    logging.info("  1. coupling_prototype [7, 256] 学习关节间耦合关系")
    logging.info("  2. 推理时只需输入目标位姿，无需历史角度")
    logging.info("  3. 参数量少，训练快速")

    # 加载数据
    logging.info("\n加载数据集...")
    train_loader, val_loader = create_windowed_dataloaders(config.data_path, config)

    # 创建模型
    logging.info(f"\n创建模型...")
    model = SimplifiedCausalIK(
        num_joints=config.num_joints,
        hidden_dim=config.hidden_dim
    ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    coupling_params = sum(p.numel() for p in model.coupling_embedding.parameters()) / 1e3
    logging.info(f"总参数量: {total_params:.2f}M")
    logging.info(f"耦合嵌入参数: {coupling_params:.2f}K")

    # 加载FK
    logging.info("\n加载GPU FK模型...")
    gpu_fk = load_robot_model()
    if gpu_fk is None:
        return

    # 损失函数
    mse_criterion = nn.MSELoss()

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

    # 学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # 训练循环
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        train_ik_loss = 0.0
        train_fk_loss = 0.0
        start_time = time.time()

        for batch_X, batch_y, batch_last_angle in train_loader:
            # 注意：简化模型不使用 batch_X（历史帧）
            batch_y = batch_y.to(config.device)
            batch_size = batch_y.shape[0]

            # 提取目标角度和位姿
            if batch_y.shape[1] == 14:
                target_pose = batch_y[:, :7]
                target_angles = batch_y[:, 7:]
                target_position = target_pose[:, :3]
                target_orientation = target_pose[:, 3:7]
            else:
                target_angles = batch_y
                target_position = gpu_fk.forward(target_angles)
                target_orientation = None

            optimizer.zero_grad()

            # 前向传播（只需要目标位姿，不需要历史！）
            pred_angles, info = model(target_position, target_orientation)

            # 计算损失
            ik_loss = mse_criterion(pred_angles, target_angles)

            pred_position = gpu_fk.forward(pred_angles)
            target_position_fk = gpu_fk.forward(target_angles)
            fk_loss = mse_criterion(pred_position, target_position_fk)

            total_loss = config.ik_weight * ik_loss + config.fk_weight * fk_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * batch_size
            train_ik_loss += ik_loss.item() * batch_size
            train_fk_loss += fk_loss.item() * batch_size

        train_loss = train_loss / len(train_loader.dataset)
        train_time = time.time() - start_time

        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0.0
        val_ik_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y, batch_last_angle in val_loader:
                batch_y = batch_y.to(config.device)
                batch_size = batch_y.shape[0]

                if batch_y.shape[1] == 14:
                    target_pose = batch_y[:, :7]
                    target_angles = batch_y[:, 7:]
                    target_position = target_pose[:, :3]
                    target_orientation = target_pose[:, 3:7]
                else:
                    target_angles = batch_y
                    target_position = gpu_fk.forward(target_angles)
                    target_orientation = None

                pred_angles, info = model(target_position, target_orientation)

                ik_loss = mse_criterion(pred_angles, target_angles)
                val_loss += ik_loss.item() * batch_size
                val_ik_loss += ik_loss.item() * batch_size

        val_loss = val_loss / len(val_loader.dataset)
        val_ik_loss = val_ik_loss / len(val_loader.dataset)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        logging.info(
            f"Epoch [{epoch}/{config.epochs}] | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"IK: {train_ik_loss/len(train_loader.dataset):.4f} | "
            f"FK: {train_fk_loss/len(train_loader.dataset):.6f} | "
            f"LR: {current_lr:.6f} | Time: {train_time:.1f}s"
        )

        # 每10个epoch打印一次耦合原型
        if epoch % 10 == 0:
            coupling = model.coupling_embedding.coupling_prototype.data.cpu()
            logging.info(f"  coupling_prototype norm: {coupling.norm(dim=-1).numpy()}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, config.checkpoint_path)
            logging.info(f"  >>> 保存最优模型")

    logging.info(f"\n训练完成！最优验证损失: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
