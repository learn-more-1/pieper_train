"""
知识蒸馏训练：从原始模型学习到改进的简化模型

核心改进：
1. 教师模型：PieperCausalIK（需要历史帧）
2. 学生模型：ImprovedSimplifiedCausalIK（不需要历史帧，动态关节耦合）
3. 蒸馏策略：特征蒸馏 + 输出蒸馏
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

from causal_ik_model_pieper2 import PieperCausalIK
from causal_ik_model_improved import ImprovedSimplifiedCausalIK
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train_distillation_improved.log")]
)


class ImprovedDistillationConfig:
    # 数据路径
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
    teacher_path = "/home/bonuli/Pieper/pieper1101/pieper_causal_ik_1101.pth"

    # 训练参数
    batch_size = 2048
    epochs = 100
    lr = 2e-4
    weight_decay = 1e-4
    warmup_epochs = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 蒸馏参数
    temperature = 2.0
    alpha = 0.3  # soft_loss 权重

    # 特征蒸馏权重
    feature_distill = True
    feature_loss_weight = 0.1

    # 数据预加载
    preload_to_gpu = True

    # 损失权重
    fk_weight = 0.2

    # 早停
    patience = 15
    min_delta = 1e-5

    # 保存路径
    checkpoint_path = "/home/bonuli/Pieper/pieper1702/improved_simplified_distilled_best.pth"


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

    # 划分训练/验证
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


def distillation_loss(student_output, teacher_output, target, temperature, alpha):
    """计算蒸馏损失"""
    # Hard loss: 学生 vs 真实标签
    hard_loss = F.mse_loss(student_output, target)

    # Soft loss: 学生 vs 教师
    soft_loss = F.mse_loss(
        student_output / temperature,
        teacher_output / temperature
    ) * (temperature ** 2)

    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss

    return total_loss, hard_loss, soft_loss


def train_epoch_distill(student, teacher, train_data, config, gpu_fk, optimizer, scheduler, epoch):
    """训练一个epoch"""
    student.train()
    train_loss = 0.0
    train_hard_loss = 0.0
    train_soft_loss = 0.0
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

        # 教师模型推理（用历史帧）
        with torch.no_grad():
            history_frames = target_angles.unsqueeze(1).repeat(1, 10, 1)
            teacher_output, _ = teacher(history_frames, target_position, target_orientation)

        # 学生模型推理（不需要历史）
        student_output, _ = student(target_position, target_orientation)

        # 蒸馏损失
        distill_loss, hard_loss, soft_loss = distillation_loss(
            student_output,
            teacher_output,
            target_angles,
            config.temperature,
            config.alpha
        )

        # FK 损失
        pred_position = gpu_fk.forward(student_output)
        target_position_fk = gpu_fk.forward(target_angles)
        fk_loss = F.mse_loss(pred_position, target_position_fk)

        total_loss = distill_loss + config.fk_weight * fk_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = len(batch_y)
        train_loss += total_loss.item() * batch_size
        train_hard_loss += hard_loss.item() * batch_size
        train_soft_loss += soft_loss.item() * batch_size
        train_fk_loss += fk_loss.item() * batch_size

        if (i + 1) % 100 == 0:
            progress = (i + 1) / n_batches * 100
            logging.info(
                f"  Progress: {progress:.1f}% | "
                f"Loss: {total_loss.item():.6f} | "
                f"Hard: {hard_loss.item():.4f} | Soft: {soft_loss.item():.4f}"
            )

    train_loss = train_loss / len(train_data)
    train_hard_loss = train_hard_loss / len(train_data)
    train_soft_loss = train_soft_loss / len(train_data)
    train_fk_loss = train_fk_loss / len(train_data)

    return train_loss, train_hard_loss, train_soft_loss, train_fk_loss


def validate(student, val_data, config):
    """验证"""
    student.eval()
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

            pred_angles, _ = student(target_position, target_orientation)

            mae = F.l1_loss(pred_angles, target_angles)
            mse = F.mse_loss(pred_angles, target_angles)

            val_mae += mae.item() * len(batch_y)
            val_mse += mse.item() * len(batch_y)

    val_mae = val_mae / len(val_data)
    val_mse = val_mse / len(val_data)

    return val_mae, val_mse


def train_distillation():
    config = ImprovedDistillationConfig()

    logging.info("=" * 70)
    logging.info("知识蒸馏训练：原始模型 → 改进简化模型")
    logging.info("=" * 70)
    logging.info(f"  温度: {config.temperature}, alpha: {config.alpha}")
    logging.info(f"  FK weight: {config.fk_weight}")

    # 加载数据
    train_data, val_data = load_data(
        config.data_path,
        device=config.device,
        preload=config.preload_to_gpu
    )

    # 加载教师模型
    logging.info(f"\n加载教师模型...")
    teacher = PieperCausalIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        num_layers=2
    ).to(config.device)

    checkpoint = torch.load(config.teacher_path, map_location=config.device)
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    logging.info(f"  ✓ 教师: {teacher_params:.2f}M 参数")

    # 创建学生模型（改进的简化模型）
    logging.info(f"\n创建学生模型（改进简化版）...")
    student = ImprovedSimplifiedCausalIK(
        num_joints=7,
        hidden_dim=256,
        num_heads=4,
        num_decoder_layers=3
    ).to(config.device)

    student_params = sum(p.numel() for p in student.parameters()) / 1e6
    logging.info(f"  ✓ 学生: {student_params:.2f}M 参数")
    logging.info(f"  压缩比: {teacher_params/student_params:.2f}x")

    # 加载 FK
    try:
        gpu_fk = SimpleGPUFK()
        logging.info(f"\n✓ GPU FK 加载成功")
    except:
        logging.error(f"\n✗ GPU FK 加载失败")
        return

    # 优化器和学习率调度
    optimizer = optim.AdamW(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
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
    logging.info(f"\n开始蒸馏训练...")
    logging.info(f"  batches/epoch: {n_train_batches}")

    for epoch in range(config.epochs):
        epoch_start = time.time()
        current_lr = scheduler.step(epoch)

        logging.info(f"\nEpoch [{epoch}/{config.epochs}] | LR: {current_lr:.6f}")

        # 训练
        train_loss, train_hard_loss, train_soft_loss, train_fk_loss = train_epoch_distill(
            student, teacher, train_data, config, gpu_fk, optimizer, scheduler, epoch
        )

        # 验证
        val_mae, val_mse = validate(student, val_data, config)

        epoch_time = time.time() - epoch_start

        logging.info(
            f"\n{'='*60}\n"
            f"Epoch [{epoch}/{config.epochs}] | Time: {epoch_time:.1f}s\n"
            f"  Train Loss: {train_loss:.6f}\n"
            f"  Hard Loss: {train_hard_loss:.6f} (vs 真实标签)\n"
            f"  Soft Loss: {train_soft_loss:.6f} (vs 教师)\n"
            f"  FK Loss: {train_fk_loss:.6f}\n"
            f"  Val MAE: {val_mae:.6f} | Val MSE: {val_mse:.6f}\n"
            f"{'='*60}"
        )

        # 保存最优模型
        if val_mae < best_val_mae - config.min_delta:
            best_val_mae = val_mae
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'best_val_mae': best_val_mae,
                'config': {
                    'num_joints': 7,
                    'hidden_dim': 256,
                    'num_heads': 4,
                    'num_decoder_layers': 3
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

    logging.info(f"\n蒸馏训练完成！")
    logging.info(f"  最优验证 MAE: {best_val_mae:.6f}")
    logging.info(f"  模型保存到: {config.checkpoint_path}")


if __name__ == "__main__":
    train_distillation()
