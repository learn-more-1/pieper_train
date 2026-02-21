"""
训练 Transformer IK 模型

特点：
1. 基于 Transformer 的无历史 IK 模型
2. Self-Attention 学习关节间关系
3. Cross-Attention 注入末端位姿
4. 余弦退火 + 早停
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import time
import sys

sys.path.insert(0, '/home/bonuli/Pieper/pieper1702')

from causal_ik_model_transformer import TransformerIK
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train_transformer_ik.log")]
)


class TransformerConfig:
    # 数据路径
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"

    # 训练参数
    batch_size = 2048
    epochs = 100
    lr = 2e-4
    weight_decay = 1e-4
    warmup_epochs = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 模型参数
    num_joints = 7
    hidden_dim = 256
    num_layers = 6
    num_heads = 8
    ff_dim = 512

    # 损失权重
    ik_weight = 1.0
    fk_weight = 0.2

    # 早停
    patience = 15
    min_delta = 1e-5

    # 保存路径
    checkpoint_path = "/home/bonuli/Pieper/pieper1702/transformer_ik_best.pth"


class WarmupCosineScheduler:
    """预热 + 余弦退火学习率调度"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


def load_data(data_path, device=None, preload=False):
    """加载数据"""
    logging.info(f"加载数据: {data_path}")
    start = time.time()

    data = np.load(data_path)
    y = data['y'].astype(np.float32)

    split_idx = int(len(y) * 0.9)
    train_data = y[:split_idx]
    val_data = y[split_idx:]

    if preload and device is not None and device.type == 'cuda':
        logging.info(f"  预加载到GPU...")
        train_data = torch.from_numpy(train_data).to(device)
        val_data = torch.from_numpy(val_data).to(device)
        logging.info(f"  ✓ 数据已预加载到GPU")

    elapsed = time.time() - start
    logging.info(f"  训练集: {len(train_data)}")
    logging.info(f"  验证集: {len(val_data)}")
    logging.info(f"  加载时间: {elapsed:.2f}s")

    return train_data, val_data


def train_epoch(model, train_data, config, gpu_fk, optimizer, scheduler, epoch):
    """训练一个epoch"""
    model.train()
    train_loss = 0.0
    train_ik_loss = 0.0
    train_fk_loss = 0.0

    n_batches = (len(train_data) + config.batch_size - 1) // config.batch_size

    for i in range(n_batches):
        start_idx = i * config.batch_size
        end_idx = min((i + 1) * config.batch_size, len(train_data))

        if isinstance(train_data, torch.Tensor):
            batch_y = train_data[start_idx:end_idx]
        else:
            batch_y = torch.from_numpy(train_data[start_idx:end_idx]).to(config.device)

        target_pose = batch_y[:, :7]
        target_angles = batch_y[:, 7:]
        target_position = target_pose[:, :3]
        target_orientation = target_pose[:, 3:7]

        optimizer.zero_grad()

        pred_angles, _ = model(target_position, target_orientation)

        ik_loss = F.mse_loss(pred_angles, target_angles)

        pred_position = gpu_fk.forward(pred_angles)
        target_position_fk = gpu_fk.forward(target_angles)
        fk_loss = F.mse_loss(pred_position, target_position_fk)

        total_loss = config.ik_weight * ik_loss + config.fk_weight * fk_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = len(batch_y)
        train_loss += total_loss.item() * batch_size
        train_ik_loss += ik_loss.item() * batch_size
        train_fk_loss += fk_loss.item() * batch_size

        if (i + 1) % 100 == 0:
            progress = (i + 1) / n_batches * 100
            logging.info(
                f"  Progress: {progress:.1f}% | "
                f"Loss: {total_loss.item():.6f} | "
                f"IK: {ik_loss.item():.4f} | FK: {fk_loss.item():.6f}"
            )

    train_loss = train_loss / len(train_data)
    train_ik_loss = train_ik_loss / len(train_data)
    train_fk_loss = train_fk_loss / len(train_data)

    return train_loss, train_ik_loss, train_fk_loss


def validate(model, val_data, config, gpu_fk):
    """验证"""
    model.eval()
    val_mae = 0.0
    val_mse = 0.0

    n_batches = (len(val_data) + config.batch_size - 1) // config.batch_size

    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * config.batch_size
            end_idx = min((i + 1) * config.batch_size, len(val_data))

            if isinstance(val_data, torch.Tensor):
                batch_y = val_data[start_idx:end_idx]
            else:
                batch_y = torch.from_numpy(val_data[start_idx:end_idx]).to(config.device)

            target_pose = batch_y[:, :7]
            target_angles = batch_y[:, 7:]
            target_position = target_pose[:, :3]
            target_orientation = target_pose[:, 3:7]

            pred_angles, _ = model(target_position, target_orientation)

            mae = F.l1_loss(pred_angles, target_angles)
            mse = F.mse_loss(pred_angles, target_angles)

            val_mae += mae.item() * len(batch_y)
            val_mse += mse.item() * len(batch_y)

    val_mae = val_mae / len(val_data)
    val_mse = val_mse / len(val_data)

    return val_mae, val_mse


def train():
    config = TransformerConfig()

    logging.info("=" * 70)
    logging.info("训练 Transformer IK 模型")
    logging.info("=" * 70)
    logging.info(f"  架构: {config.num_layers} 层 Transformer, {config.num_heads} 头注意力")

    # 加载数据
    train_data, val_data = load_data(
        config.data_path,
        device=config.device,
        preload=True
    )

    # 创建模型
    logging.info(f"\n创建模型...")
    model = TransformerIK(
        num_joints=config.num_joints,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim
    ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"  参数量: {total_params:.2f}M")

    # 加载 FK
    try:
        gpu_fk = SimpleGPUFK()
        logging.info(f"\n✓ GPU FK 加载成功")
    except:
        logging.error(f"\n✗ GPU FK 加载失败")
        return

    # 优化器和学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.epochs,
        min_lr=1e-6
    )

    # 早停
    best_val_mae = float("inf")
    patience_counter = 0

    n_train_batches = (len(train_data) + config.batch_size - 1) // config.batch_size
    logging.info(f"\n开始训练...")
    logging.info(f"  batches/epoch: {n_train_batches}")

    for epoch in range(config.epochs):
        epoch_start = time.time()
        current_lr = scheduler.step(epoch)

        logging.info(f"\nEpoch [{epoch}/{config.epochs}] | LR: {current_lr:.6f}")

        # 训练
        train_loss, train_ik_loss, train_fk_loss = train_epoch(
            model, train_data, config, gpu_fk, optimizer, scheduler, epoch
        )

        # 验证
        val_mae, val_mse = validate(model, val_data, config, gpu_fk)

        epoch_time = time.time() - epoch_start

        logging.info(
            f"\n{'='*60}\n"
            f"Epoch [{epoch}/{config.epochs}] | Time: {epoch_time:.1f}s\n"
            f"  Train Loss: {train_loss:.6f}\n"
            f"  IK: {train_ik_loss:.4f} | FK: {train_fk_loss:.6f}\n"
            f"  Val MAE: {val_mae:.6f} | Val MSE: {val_mse:.6f}\n"
            f"{'='*60}"
        )

        # 保存最优模型
        if val_mae < best_val_mae - config.min_delta:
            best_val_mae = val_mae
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_mae': best_val_mae,
                'config': {
                    'num_joints': config.num_joints,
                    'hidden_dim': config.hidden_dim,
                    'num_layers': config.num_layers,
                    'num_heads': config.num_heads,
                    'ff_dim': config.ff_dim
                }
            }, config.checkpoint_path)

            logging.info(f"  >>> 保存最优模型 (val_mae: {val_mae:.6f})")
        else:
            patience_counter += 1
            logging.info(f"  无改善 ({patience_counter}/{config.patience})")

        # 早停
        if patience_counter >= config.patience:
            logging.info(f"\n早停触发！停止训练。")
            break

    logging.info(f"\n训练完成！")
    logging.info(f"  最优验证 MAE: {best_val_mae:.6f}")
    logging.info(f"  模型保存到: {config.checkpoint_path}")


if __name__ == "__main__":
    train()
