"""
快速训练简化模型 - 使用 GRAB 数据集

用于快速验证模型效果
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import sys
import os
sys.path.insert(0, '/home/bonuli/pieper/casual')
sys.path.insert(0, '/home/bonuli/Pieper/casual/pieper1101')

from causal_ik_model_pieper_simple import SimplifiedCausalIK
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)


class QuickConfig:
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"
    batch_size = 512
    epochs = 30
    lr = 1e-3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 快速训练：只用前 N 个样本
    max_samples = 50000  # 可以根据需要调整

    # 模型配置
    num_joints = 7
    hidden_dim = 256

    # 损失权重
    ik_weight = 1.0
    fk_weight = 0.3

    # 保存路径
    checkpoint_path = "/home/bonuli/Pieper/pieper1101/simplified_causal_ik_GRAB.pth"


def load_GRAB_data(data_path, max_samples=None):
    """加载 GRAB 数据"""
    logging.info(f"加载数据: {data_path}")
    data = np.load(data_path)

    y = data['y'].astype(np.float32)  # [N, 14]

    if max_samples is not None and max_samples < len(y):
        y = y[:max_samples]
        logging.info(f"  使用前 {max_samples} 个样本")

    logging.info(f"  数据形状: {y.shape}")
    return y


def train():
    config = QuickConfig()

    logging.info("=" * 70)
    logging.info("快速训练：简化 IK 模型（GRAB 数据）")
    logging.info("=" * 70)

    # 加载数据
    y = load_GRAB_data(config.data_path, config.max_samples)

    # 划分训练集和验证集
    split = int(0.9 * len(y))
    y_train = y[:split]
    y_val = y[split:]

    logging.info(f"  训练集: {len(y_train)}")
    logging.info(f"  验证集: {len(y_val)}")

    # 创建模型
    logging.info(f"\n创建模型...")
    model = SimplifiedCausalIK(
        num_joints=config.num_joints,
        hidden_dim=config.hidden_dim
    ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"  参数量: {total_params:.2f}M")

    # 加载 FK
    try:
        gpu_fk = SimpleGPUFK()
        logging.info(f"  ✓ GPU FK 加载成功")
    except:
        logging.error(f"  ✗ GPU FK 加载失败")
        return

    # 损失函数
    mse_criterion = nn.MSELoss()

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # 训练循环
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_ik_loss = 0.0
        train_fk_loss = 0.0

        # 训练
        n_batches = (len(y_train) + config.batch_size - 1) // config.batch_size

        for i in range(n_batches):
            start_idx = i * config.batch_size
            end_idx = min((i + 1) * config.batch_size, len(y_train))

            batch_y = torch.from_numpy(y_train[start_idx:end_idx]).to(config.device)

            # 提取数据
            target_pose = batch_y[:, :7]
            target_angles = batch_y[:, 7:]
            target_position = target_pose[:, :3]
            target_orientation = target_pose[:, 3:7]

            optimizer.zero_grad()

            # 前向传播
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

            train_loss += total_loss.item() * len(batch_y)
            train_ik_loss += ik_loss.item() * len(batch_y)
            train_fk_loss += fk_loss.item() * len(batch_y)

        train_loss = train_loss / len(y_train)

        # 验证
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            n_batches = (len(y_val) + config.batch_size - 1) // config.batch_size

            for i in range(n_batches):
                start_idx = i * config.batch_size
                end_idx = min((i + 1) * config.batch_size, len(y_val))

                batch_y = torch.from_numpy(y_val[start_idx:end_idx]).to(config.device)

                target_pose = batch_y[:, :7]
                target_angles = batch_y[:, 7:]
                target_position = target_pose[:, :3]
                target_orientation = target_pose[:, 3:7]

                pred_angles, info = model(target_position, target_orientation)

                ik_loss = mse_criterion(pred_angles, target_angles)
                val_loss += ik_loss.item() * len(batch_y)

        val_loss = val_loss / len(y_val)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        logging.info(
            f"Epoch [{epoch}/{config.epochs}] | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"IK: {train_ik_loss/len(y_train):.4f} | "
            f"FK: {train_fk_loss/len(y_train):.6f} | "
            f"LR: {current_lr:.6f}"
        )

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
            }, config.checkpoint_path)
            logging.info(f"  >>> 保存")

    logging.info(f"\n训练完成！最优验证损失: {best_val_loss:.6f}")
    logging.info(f"模型保存到: {config.checkpoint_path}")


if __name__ == "__main__":
    train()
