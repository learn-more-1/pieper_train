"""
知识蒸馏训练：从原始模型学习到简化模型（使用 ACCAD_CMU 数据）

核心思想：
1. 教师模型：PieperCausalIK（需要历史帧，在 ACCAD_CMU 上训练好）
2. 学生模型：SimplifiedCausalIK（不需要历史帧，待训练）
3. 蒸馏目标：让学生模型的输出接近教师模型
4. 数据集：使用 ACCAD_CMU（与教师训练数据一致，避免领域不匹配）

损失函数：
- hard_loss: 学生预测 vs 真实标签（关节角度）
- soft_loss: 学生预测 vs 教师预测（知识蒸馏）
- fk_loss: 末端位置误差
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import time
import sys
sys.path.insert(0, '/home/bonuli/pieper/casual')
sys.path.insert(0, '/home/bonuli/Pieper/casual/pieper1101')

from causal_ik_model_pieper_simple import SimplifiedCausalIK
from causal_ik_model_pieper2 import PieperCausalIK
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)


class DistillationConfig:
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
    teacher_path = "/home/bonuli/Pieper/pieper1101/pieper_causal_ik_1101.pth"

    batch_size = 2048  # 增大批次降低梯度方差
    epochs = 300
    lr = 5e-4  # 降低学习率提高稳定性（和 GRAB 训练一致）

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据预加载到GPU内存
    preload_to_gpu = True

    # 蒸馏参数（使用更稳定的配置）
    temperature = 3.0  # 降低温度提高稳定性
    alpha = 0.5        # 平衡 soft 和 hard loss        # soft_loss 权重，(1-alpha) 是 hard_loss 权重（超参数搜索最佳值）

    # 快速训练选项
    max_samples = None  # None 表示全部数据
    val_split = 0.1
    val_every = 2

    # 损失权重
    fk_weight = 0.2

    # 保存路径
    checkpoint_path = "/home/bonuli/Pieper/pieper1703/simplified_causal_ik_distilled_accad.pth"


def load_data_fast(data_path, max_samples=None, val_split=0.1, device=None, preload=False):
    """
    快速加载数据，可选预加载到GPU

    Args:
        preload: 是否预加载到GPU（提高利用率但占用显存）
    """
    logging.info(f"加载数据: {data_path}")
    start = time.time()

    data = np.load(data_path)
    y = data['y'].astype(np.float32)

    if max_samples is not None and max_samples < len(y):
        y = y[:max_samples]
        logging.info(f"  使用前 {max_samples} 个样本")

    split_idx = int(len(y) * (1 - val_split))
    train_data = y[:split_idx]
    val_data = y[split_idx:]

    # 预加载到GPU
    if preload and device is not None and device.type == 'cuda':
        logging.info(f"  预加载到GPU...")
        train_data = torch.from_numpy(train_data).to(device)
        val_data = torch.from_numpy(val_data).to(device)
        logging.info(f"  ✓ 数据已预加载到GPU")

    logging.info(f"  训练集: {len(train_data)}")
    logging.info(f"  验证集: {len(val_data)}")
    logging.info(f"  加载时间: {time.time() - start:.2f}s")

    return train_data, val_data


def distillation_loss(student_output, teacher_output, target, temperature, alpha):
    """
    计算蒸馏损失

    Args:
        student_output: [batch, 7] 学生模型输出
        teacher_output: [batch, 7] 教师模型输出（detach）
        target: [batch, 7] 真实标签
        temperature: 蒸馏温度
        alpha: soft_loss 权重

    Returns:
        loss: 总损失
    """
    # Hard loss: 学生 vs 真实标签
    hard_loss = F.mse_loss(student_output, target)

    # Soft loss: 学生 vs 教师（使用logits的软标签）
    # 对于回归任务，使用温度缩放的 MSE
    soft_loss = F.mse_loss(
        student_output / temperature,
        teacher_output / temperature
    ) * (temperature ** 2)

    # 组合损失
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss

    return total_loss, hard_loss, soft_loss


def train_distillation():
    config = DistillationConfig()

    logging.info("=" * 70)
    logging.info("知识蒸馏训练：原始模型 → 简化模型")
    logging.info("=" * 70)
    logging.info(f"  教师模型: {config.teacher_path}")
    logging.info(f"  温度: {config.temperature}, alpha: {config.alpha}")

    # 加载数据
    train_data, val_data = load_data_fast(
        config.data_path,
        config.max_samples,
        config.val_split,
        config.device,
        config.preload_to_gpu
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
    logging.info(f"  ✓ 教师模型加载成功")

    for param in teacher.parameters():
        param.requires_grad = False  # 冻结教师模型

    # 创建学生模型
    logging.info(f"\n创建学生模型...")
    student = SimplifiedCausalIK(
        num_joints=7,
        hidden_dim=256
    ).to(config.device)

    total_params = sum(p.numel() for p in student.parameters()) / 1e6
    teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    logging.info(f"  教师: {teacher_params:.2f}M 参数")
    logging.info(f"  学生: {total_params:.2f}M 参数")
    logging.info(f"  压缩比: {teacher_params/total_params:.2f}x")

    # 加载 FK
    try:
        gpu_fk = SimpleGPUFK()
        logging.info(f"  ✓ GPU FK")
    except:
        logging.error(f"  ✗ GPU FK 加载失败")
        return

    # 优化器
    optimizer = optim.AdamW(student.parameters(), lr=config.lr, weight_decay=1e-4)
    # 余弦退火重启：每50轮完成一个周期，然后重启学习率
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,        # 每50轮完成一个周期
        T_mult=1,      # 每个周期长度保持不变
        eta_min=1e-6   # 最小学习率
    )

    # 训练循环
    best_val_loss = float("inf")
    n_batches = (len(train_data) + config.batch_size - 1) // config.batch_size

    logging.info(f"\n开始蒸馏训练...")
    logging.info(f"  batches/epoch: {n_batches}")

    for epoch in range(config.epochs):
        student.train()
        train_loss = 0.0
        train_hard_loss = 0.0
        train_soft_loss = 0.0
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

            # 教师模型推理（用历史帧）
            with torch.no_grad():
                # 用真实角度作为历史（教师需要历史）
                history_frames = target_angles.unsqueeze(1).repeat(1, 10, 1)
                teacher_output, _ = teacher(history_frames, target_position, target_orientation)

            # 学生模型推理（不需要历史）
            student_output, info = student(target_position, target_orientation)

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

            # 总损失
            total_loss = distill_loss + config.fk_weight * fk_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * batch_size
            train_hard_loss += hard_loss.item() * batch_size
            train_soft_loss += soft_loss.item() * batch_size
            train_fk_loss += fk_loss.item() * batch_size

        train_loss = train_loss / len(train_data)
        epoch_time = time.time() - epoch_start

        # 验证
        if epoch % config.val_every == 0 or epoch == config.epochs - 1:
            student.eval()
            val_loss = 0.0
            val_mae = 0.0

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

                    student_output, info = student(target_position, target_orientation)

                    mae = F.l1_loss(student_output, target_angles)
                    val_loss += mae.item() * len(batch_y)
                    val_mae += mae.item() * len(batch_y)

            val_loss = val_loss / len(val_data)
            val_mae = val_mae / len(val_data)

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            logging.info(
                f"\n{'='*60}\n"
                f"Epoch [{epoch}/{config.epochs}] | Time: {epoch_time:.1f}s\n"
                f"  Train Loss: {train_loss:.6f}\n"
                f"  Hard Loss: {train_hard_loss/len(train_data):.6f} (vs 真实标签)\n"
                f"  Soft Loss: {train_soft_loss/len(train_data):.6f} (vs 教师模型)\n"
                f"  FK Loss: {train_fk_loss/len(train_data):.6f}\n"
                f"  Val MAE: {val_mae:.6f}\n"
                f"  LR: {current_lr:.6f}\n"
                f"{'='*60}"
            )

            # 保存最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student.state_dict(),
                    'best_val_loss': best_val_loss,
                }, config.checkpoint_path)
                logging.info(f"  >>> 保存最优模型 (val_mae: {val_mae:.6f})")

    logging.info(f"\n蒸馏训练完成！")
    logging.info(f"  最优验证 MAE: {best_val_loss:.6f}")
    logging.info(f"  模型保存到: {config.checkpoint_path}")


if __name__ == "__main__":
    train_distillation()
