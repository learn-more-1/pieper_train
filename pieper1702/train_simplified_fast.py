"""
快速训练简化模型 - 直接加载 GRAB 数据，不使用 DataLoader

优化：
1. 直接从 .npz 加载，不使用 create_windowed_dataloaders
2. 不创建历史窗口
3. 使用 pin_memory 和 non_blocking 加速传输
4. 减少验证频率
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
import sys
sys.path.insert(0, '/home/bonuli/pieper/casual')
sys.path.insert(0, '/home/bonuli/Pieper/casual/pieper1101')

from causal_ik_model_pieper_simple import SimplifiedCausalIK
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)


class FastConfig:
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
    batch_size = 1024  # 增大批次提高GPU利用率
    epochs = 50
    lr = 1e-3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据预加载到GPU
    preload_to_gpu = True

    # 快速训练选项
    max_samples = None  # None 表示全部数据，或设置如 500000
    val_split = 0.1
    val_every = 2  # 每2个epoch验证一次

    # 模型配置
    num_joints = 7
    hidden_dim = 256

    # 损失权重
    ik_weight = 1.0
    fk_weight = 0.2

    # 保存路径
    checkpoint_path = "/home/bonuli/Pieper/pieper1702/simplified_causal_ik_GRAB2.pth"


def load_data_fast(data_path, max_samples=None, val_split=0.1, device=None, preload=False):
    """
    快速加载数据，可选预加载到GPU

    Returns:
        train_data: tensor or array
        val_data: tensor or array
    """
    logging.info(f"快速加载数据: {data_path}")
    start = time.time()

    data = np.load(data_path)
    y = data['y'].astype(np.float32)  # [N, 14]

    if max_samples is not None and max_samples < len(y):
        y = y[:max_samples]
        logging.info(f"  使用前 {max_samples} 个样本")

    # 划分训练/验证
    split_idx = int(len(y) * (1 - val_split))
    train_data = y[:split_idx]
    val_data = y[split_idx:]

    # 预加载到GPU
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


def train_fast():
    config = FastConfig()

    logging.info("=" * 70)
    logging.info("快速训练：简化 IK 模型")
    logging.info("=" * 70)
    logging.info(f"  批次大小: {config.batch_size}")
    logging.info(f"  验证频率: 每 {config.val_every} epoch")

    # 加载数据
    train_data, val_data = load_data_fast(
        config.data_path,
        config.max_samples,
        config.val_split,
        config.device,
        config.preload_to_gpu
    )

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
        logging.info(f"  ✓ GPU FK")
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
    n_batches = (len(train_data) + config.batch_size - 1) // config.batch_size

    logging.info(f"\n开始训练...")
    logging.info(f"  batches/epoch: {n_batches}")

    for epoch in range(config.epochs):
        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        train_ik_loss = 0.0
        train_fk_loss = 0.0
        epoch_start = time.time()

        for i in range(n_batches):
            start_idx = i * config.batch_size
            end_idx = min((i + 1) * config.batch_size, len(train_data))

            # 数据已经预加载到GPU，直接切片
            if isinstance(train_data, torch.Tensor):
                batch_y = train_data[start_idx:end_idx]
            else:
                batch_y = torch.from_numpy(train_data[start_idx:end_idx]).to(config.device, non_blocking=True)
            batch_size = len(batch_y)

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

            train_loss += total_loss.item() * batch_size
            train_ik_loss += ik_loss.item() * batch_size
            train_fk_loss += fk_loss.item() * batch_size

            # 进度显示
            if (i + 1) % 100 == 0 or (i + 1) == n_batches:
                progress = (i + 1) / n_batches * 100
                logging.info(
                    f"  Epoch [{epoch}/{config.epochs}] "
                    f"Progress: {progress:.1f}% ({i+1}/{n_batches}) | "
                    f"Loss: {total_loss.item():.6f}"
                )

        train_loss = train_loss / len(train_data)
        epoch_time = time.time() - epoch_start

        # ==================== 验证阶段 ====================
        if epoch % config.val_every == 0 or epoch == config.epochs - 1:
            model.eval()
            val_loss = 0.0
            val_ik_loss = 0.0

            n_val_batches = (len(val_data) + config.batch_size - 1) // config.batch_size

            with torch.no_grad():
                for i in range(n_val_batches):
                    start_idx = i * config.batch_size
                    end_idx = min((i + 1) * config.batch_size, len(val_data))

                    if isinstance(val_data, torch.Tensor):
                        batch_y = val_data[start_idx:end_idx]
                    else:
                        batch_y = torch.from_numpy(val_data[start_idx:end_idx]).to(config.device, non_blocking=True)

                    target_pose = batch_y[:, :7]
                    target_angles = batch_y[:, 7:]
                    target_position = target_pose[:, :3]
                    target_orientation = target_pose[:, 3:7]

                    pred_angles, info = model(target_position, target_orientation)

                    ik_loss = mse_criterion(pred_angles, target_angles)
                    val_loss += ik_loss.item() * len(batch_y)
                    val_ik_loss += ik_loss.item() * len(batch_y)

            val_loss = val_loss / len(val_data)
            val_ik_loss = val_ik_loss / len(val_data)

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            logging.info(
                f"\n{'='*60}\n"
                f"Epoch [{epoch}/{config.epochs}] | Time: {epoch_time:.1f}s\n"
                f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}\n"
                f"  IK: {train_ik_loss/len(train_data):.4f} | FK: {train_fk_loss/len(train_data):.6f}\n"
                f"  LR: {current_lr:.6f}\n"
                f"{'='*60}"
            )

            # 保存最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                }, config.checkpoint_path)
                logging.info(f"  >>> 保存最优模型 (val_loss: {val_loss:.6f})")

    logging.info(f"\n训练完成！")
    logging.info(f"  最优验证损失: {best_val_loss:.6f}")
    logging.info(f"  模型保存到: {config.checkpoint_path}")


if __name__ == "__main__":
    train_fast()
