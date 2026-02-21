"""
优化蒸馏超参数

搜索最佳的 temperature 和 alpha 组合
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import itertools
import sys
import os
sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1101')

from causal_ik_model_pieper_simple import SimplifiedCausalIK
from causal_ik_model_pieper2 import PieperCausalIK
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("distillation_optimization.log")]
)


class SearchConfig:
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"
    teacher_path = "/home/bonuli/Pieper/pieper1101/pieper_causal_ik_1101.pth"

    batch_size = 1024
    epochs = 10  # 快速验证，用少量epoch
    lr = 1e-3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 超参数搜索空间
    temperatures = [1.0, 2.0, 3.0, 5.0, 10.0]
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

    # 快速训练选项
    max_samples = 100000  # 用部分数据快速搜索
    val_split = 0.2

    # 保存路径
    results_path = "/home/bonuli/Pieper/pieper1702/distillation_search_results.txt"


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


def distillation_loss_fn(student_output, teacher_output, target, temperature, alpha, fk_weight=0.2):
    """计算蒸馏损失"""
    # Hard loss
    hard_loss = nn.functional.mse_loss(student_output, target)

    # Soft loss
    soft_loss = nn.functional.mse_loss(
        student_output / temperature,
        teacher_output / temperature
    ) * (temperature ** 2)

    # 组合
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss

    return total_loss, hard_loss, soft_loss


def train_one_config(train_data, val_data, teacher, temperature, alpha, config):
    """训练一个配置"""
    device = config.device

    # 创建学生模型
    student = SimplifiedCausalIK(num_joints=7, hidden_dim=256).to(device)

    optimizer = optim.AdamW(student.parameters(), lr=config.lr, weight_decay=1e-4)

    # 训练几个epoch
    n_batches = (len(train_data) + config.batch_size - 1) // config.batch_size

    for epoch in range(config.epochs):
        student.train()
        train_loss = 0.0

        for i in range(n_batches):
            start_idx = i * config.batch_size
            end_idx = min((i + 1) * config.batch_size, len(train_data))

            batch_y = torch.from_numpy(train_data[start_idx:end_idx]).to(device)
            batch_size = len(batch_y)

            target_pose = batch_y[:, :7]
            target_angles = batch_y[:, 7:]
            target_position = target_pose[:, :3]
            target_orientation = target_pose[:, 3:7]

            optimizer.zero_grad()

            # 教师推理
            with torch.no_grad():
                history_frames = target_angles.unsqueeze(1).repeat(1, 10, 1)
                teacher_output, _ = teacher(history_frames, target_position, target_orientation)

            # 学生推理
            student_output, _ = student(target_position, target_orientation)

            # 蒸馏损失
            total_loss, _, _ = distillation_loss_fn(
                student_output, teacher_output, target_angles,
                temperature, alpha
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * batch_size

        train_loss = train_loss / len(train_data)

    # 验证
    student.eval()
    val_loss = 0.0
    val_mae = 0.0

    n_val_batches = (len(val_data) + config.batch_size - 1) // config.batch_size

    with torch.no_grad():
        for i in range(n_val_batches):
            start_idx = i * config.batch_size
            end_idx = min((i + 1) * config.batch_size, len(val_data))

            batch_y = torch.from_numpy(val_data[start_idx:end_idx]).to(device)

            target_pose = batch_y[:, :7]
            target_angles = batch_y[:, 7:]
            target_position = target_pose[:, :3]
            target_orientation = target_pose[:, 3:7]

            student_output, _ = student(target_position, target_orientation)

            mae = nn.functional.l1_loss(student_output, target_angles)
            val_mae += mae.item() * len(batch_y)

    val_mae = val_mae / len(val_data)

    return val_mae


def search_hyperparameters():
    config = SearchConfig()

    logging.info("=" * 70)
    logging.info("蒸馏超参数搜索")
    logging.info("=" * 70)
    logging.info(f"温度搜索空间: {config.temperatures}")
    logging.info(f"Alpha搜索空间: {config.alphas}")
    logging.info(f"总配置数: {len(config.temperatures) * len(config.alphas)}")

    # 加载数据
    train_data, val_data = load_data(
        config.data_path,
        config.max_samples,
        config.val_split
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

    logging.info(f"  ✓ 教师模型加载成功")

    # 搜索
    results = []
    total_configs = len(config.temperatures) * len(config.alphas)

    logging.info(f"\n开始搜索...")
    best_mae = float("inf")
    best_config = None

    for idx, (temp, alpha) in enumerate(itertools.product(config.temperatures, config.alphas)):
        logging.info(f"\n[{idx+1}/{total_configs}] T={temp}, α={alpha}")

        val_mae = train_one_config(train_data, val_data, teacher, temp, alpha, config)

        logging.info(f"  验证 MAE: {val_mae:.6f}")

        results.append({
            'temperature': temp,
            'alpha': alpha,
            'val_mae': val_mae
        })

        if val_mae < best_mae:
            best_mae = val_mae
            best_config = (temp, alpha)
            logging.info(f"  >>> 新的最佳配置！T={temp}, α={alpha}, MAE={val_mae:.6f}")

    # 排序结果
    results.sort(key=lambda x: x['val_mae'])

    # 保存结果
    with open(config.results_path, 'w') as f:
        f.write("蒸馏超参数搜索结果\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"最佳配置: T={best_config[0]}, α={best_config[1]}, MAE={best_mae:.6f}\n\n")
        f.write("所有配置（按MAE排序）:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'排名':<5} {'温度':<8} {'Alpha':<8} {'验证MAE':<12}\n")
        f.write("-" * 70 + "\n")

        for i, r in enumerate(results):
            f.write(f"{i+1:<5} {r['temperature']:<8.2f} {r['alpha']:<8.2f} {r['val_mae']:<12.6f}\n")

    logging.info(f"\n{'='*70}")
    logging.info(f"搜索完成！")
    logging.info(f"最佳配置: T={best_config[0]}, α={best_config[1]}")
    logging.info(f"最佳 MAE: {best_mae:.6f}")
    logging.info(f"结果已保存到: {config.results_path}")

    # 打印Top 5
    logging.info(f"\nTop 5 配置:")
    logging.info(f"{'-'*60}")
    for i, r in enumerate(results[:5]):
        logging.info(f"{i+1}. T={r['temperature']:.2f}, α={r['alpha']:.2f}, MAE={r['val_mae']:.6f}")


if __name__ == "__main__":
    search_hyperparameters()
