"""
方案2: 在 GRAB 数据上微调教师模型

让教师模型先适应 GRAB 数据，再用于蒸馏
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import sys
sys.path.insert(0, '/home/wsy/Desktop/casual')
sys.path.insert(0, '/home/wsy/Desktop/casual/pieper1101')

from causal_ik_model_pieper2 import PieperCausalIK
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("finetune_teacher.log")]
)


class FinetuneConfig:
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"
    pretrained_path = "/home/wsy/Desktop/casual/pieper_NN/pieper_causal_ik_1101.pth"

    batch_size = 512
    epochs = 20  # 微调不需要太多epoch
    lr = 5e-5  # 微调用更小的学习率

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 微调选项
    max_samples = 200000  # 用部分数据微调
    val_split = 0.1

    # 损失权重
    ik_weight = 1.0
    fk_weight = 0.2

    # 保存路径
    checkpoint_path = "/home/wsy/Desktop/casual/pieper_NN/pieper_causal_ik_GRAB_teacher.pth"


def load_data(data_path, max_samples=None, val_split=0.1):
    """加载数据"""
    logging.info(f"加载数据: {data_path}")
    data = np.load(data_path)
    y = data['y'].astype(np.float32)

    if max_samples is not None and max_samples < len(y):
        y = y[:max_samples]
        logging.info(f"  使用前 {max_samples} 个样本")

    split_idx = int(len(y) * (1 - val_split))
    train_data = y[:split_idx]
    val_data = y[split_idx:]

    logging.info(f"  训练集: {len(train_data)}")
    logging.info(f"  验证集: {len(val_data)}")

    return train_data, val_data


def finetune_teacher():
    config = FinetuneConfig()

    logging.info("=" * 70)
    logging.info("微调教师模型：适应 GRAB 数据")
    logging.info("=" * 70)

    # 加载数据
    train_data, val_data = load_data(
        config.data_path,
        config.max_samples,
        config.val_split
    )

    # 加载预训练模型
    logging.info(f"\n加载预训练模型: {config.pretrained_path}")
    teacher = PieperCausalIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        num_layers=2
    ).to(config.device)

    checkpoint = torch.load(config.pretrained_path, map_location=config.device)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"  ✓ 成功加载预训练权重")
    logging.info(f"  原始验证损失: {checkpoint.get('best_val_loss', 'N/A')}")

    # 加载 FK
    try:
        gpu_fk = SimpleGPUFK()
        logging.info(f"  ✓ GPU FK")
    except:
        logging.error(f"  ✗ GPU FK 加载失败")
        return

    # 优化器（只微调最后几层，可以降低学习率）
    # 选项1: 微调所有参数
    optimizer = optim.AdamW(teacher.parameters(), lr=config.lr, weight_decay=1e-4)

    # 选项2: 只微调部分参数（uncomment if needed）
    # for name, param in teacher.named_parameters():
    #     if 'temporal_encoder' not in name:  # 冻结时序编码器
    #         param.requires_grad = False

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # 训练循环
    best_val_loss = float("inf")
    n_batches = (len(train_data) + config.batch_size - 1) // config.batch_size

    logging.info(f"\n开始微调...")
    logging.info(f"  batches/epoch: {n_batches}")

    for epoch in range(config.epochs):
        teacher.train()
        train_loss = 0.0
        train_ik_loss = 0.0
        train_fk_loss = 0.0

        for i in range(n_batches):
            start_idx = i * config.batch_size
            end_idx = min((i + 1) * config.batch_size, len(train_data))

            batch_y = torch.from_numpy(train_data[start_idx:end_idx]).to(config.device)
            batch_size = len(batch_y)

            target_pose = batch_y[:, :7]
            target_angles = batch_y[:, 7:]
            target_position = target_pose[:, :3]
            target_orientation = target_pose[:, 3:7]

            # 用真实角度作为历史
            history_frames = target_angles.unsqueeze(1).repeat(1, 10, 1)

            optimizer.zero_grad()

            # 前向传播
            pred_angles, _ = teacher(history_frames, target_position, target_orientation)

            # 计算损失
            ik_loss = nn.functional.mse_loss(pred_angles, target_angles)

            pred_position = gpu_fk.forward(pred_angles)
            target_position_fk = gpu_fk.forward(target_angles)
            fk_loss = nn.functional.mse_loss(pred_position, target_position_fk)

            total_loss = config.ik_weight * ik_loss + config.fk_weight * fk_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * batch_size
            train_ik_loss += ik_loss.item() * batch_size
            train_fk_loss += fk_loss.item() * batch_size

            if (i + 1) % 50 == 0 or (i + 1) == n_batches:
                progress = (i + 1) / n_batches * 100
                logging.info(
                    f"  Epoch [{epoch}/{config.epochs}] "
                    f"Progress: {progress:.1f}% | Loss: {total_loss.item():.6f}"
                )

        train_loss = train_loss / len(train_data)

        # 验证
        teacher.eval()
        val_loss = 0.0

        n_val_batches = (len(val_data) + config.batch_size - 1) // config.batch_size

        with torch.no_grad():
            for i in range(n_val_batches):
                start_idx = i * config.batch_size
                end_idx = min((i + 1) * config.batch_size, len(val_data))

                batch_y = torch.from_numpy(val_data[start_idx:end_idx]).to(config.device)

                target_pose = batch_y[:, :7]
                target_angles = batch_y[:, 7:]
                target_position = target_pose[:, :3]
                target_orientation = target_pose[:, 3:7]

                history_frames = target_angles.unsqueeze(1).repeat(1, 10, 1)

                pred_angles, _ = teacher(history_frames, target_position, target_orientation)

                ik_loss = nn.functional.mse_loss(pred_angles, target_angles)
                val_loss += ik_loss.item() * len(batch_y)

        val_loss = val_loss / len(val_data)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        logging.info(
            f"\nEpoch [{epoch}/{config.epochs}] | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.6f}"
        )

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': teacher.state_dict(),
                'best_val_loss': best_val_loss,
            }, config.checkpoint_path)
            logging.info(f"  >>> 保存微调后的教师模型")

    logging.info(f"\n微调完成！")
    logging.info(f"  最优验证损失: {best_val_loss:.6f}")
    logging.info(f"  模型保存到: {config.checkpoint_path}")

    logging.info(f"\n下一步：使用微调后的教师模型进行蒸馏")
    logging.info(f"  修改 train_distillation.py 中的 teacher_path 为:")
    logging.info(f"  teacher_path = '{config.checkpoint_path}'")


if __name__ == "__main__":
    finetune_teacher()
