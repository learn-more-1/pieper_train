"""
微调训练脚本：方案3 - 适应无历史数据场景

核心思想：
1. 加载预训练模型
2. 训练时随机将部分batch的历史设为None，模拟推理场景
3. 逐步增加None历史的比例（curriculum learning）
4. 让 default_history 参数学习到合适的默认值
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
sys.path.insert(0, '/home/wsy/Desktop/casual')
sys.path.insert(0, '/home/wsy/Desktop/casual/pieper1101')

from causal_ik_model_pieper2 import PieperCausalIK
from dataset_generalized import create_windowed_dataloaders
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("finetune_no_history.log"), logging.StreamHandler()]
)


class FinetuneConfig:
    # 数据配置
    data_path = "/home/wsy/Desktop/casual/merged_training_data.npz"
    num_joints = 7
    num_frames = 10

    # 微调参数
    batch_size = 512
    epochs = 50  # 微调不需要太多epoch
    lr = 5e-4    # 微调使用较小学习率

    # None历史比例（curriculum learning）
    # 从0开始，逐渐增加到1.0
    initial_none_prob = 0.0
    final_none_prob = 1.0
    none_prob_warmup_epochs = 20  # 在前20个epoch逐渐增加

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 模型配置
    joint_dim = 7
    hidden_dim = 256
    num_layers = 2

    # 损失权重
    ik_weight = 1.0
    fk_weight = 0.5
    continuity_weight = 1.0

    # 预训练模型路径
    pretrained_path = "/home/wsy/Desktop/casual/pieper_NN/pieper_causal_ik_1101.pth"


def get_none_history_prob(epoch, config):
    """
    计算当前epoch的None历史概率（curriculum learning）

    从 initial_none_prob 线性增加到 final_none_prob
    """
    if epoch >= config.none_prob_warmup_epochs:
        return config.final_none_prob

    progress = epoch / config.none_prob_warmup_epochs
    return config.initial_none_prob + progress * (
        config.final_none_prob - config.initial_none_prob
    )


def load_robot_model():
    """加载GPU FK模型"""
    try:
        gpu_fk = SimpleGPUFK()
        logging.info(f"✓ 成功加载GPU FK模型")
        return gpu_fk
    except Exception as e:
        logging.error(f"✗ 加载GPU FK模型失败: {e}")
        return None


def forward_kinematics_with_pose(gpu_fk, joint_angles):
    """
    GPU加速的批量FK计算

    Args:
        gpu_fk: GPU FK模型
        joint_angles: [batch, 7] 关节角度

    Returns:
        positions: [batch, 3] 手腕位置
    """
    positions = gpu_fk.forward(joint_angles)
    return positions, None


def finetune():
    config = FinetuneConfig()

    logging.info("=" * 70)
    logging.info("微调训练：适应无历史数据场景（方案3）")
    logging.info("=" * 70)
    logging.info("核心特性:")
    logging.info("  1. 加载预训练模型")
    logging.info("  2. Curriculum Learning: 逐步增加None历史比例")
    logging.info("  3. 训练 default_history 参数")
    logging.info(f"  4. None历史概率: {config.initial_none_prob} → {config.final_none_prob}")

    # 加载数据集
    logging.info("\n加载数据集...")
    train_loader, val_loader = create_windowed_dataloaders(config.data_path, config)

    # 加载预训练模型
    logging.info(f"\n加载预训练模型: {config.pretrained_path}")
    model = PieperCausalIK(
        num_joints=config.joint_dim,
        num_frames=config.num_frames,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers
    ).to(config.device)

    # 加载预训练权重
    if os.path.exists(config.pretrained_path):
        checkpoint = torch.load(config.pretrained_path, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logging.info(f"✓ 成功加载预训练权重")
        logging.info(f"  原始验证损失: {checkpoint.get('best_val_loss', 'N/A')}")
    else:
        logging.warning(f"⚠ 预训练权重不存在，使用随机初始化")

    # 检查 default_history 参数
    logging.info(f"\ndefault_history 参数:")
    logging.info(f"  shape: {model.default_history.shape}")
    logging.info(f"  值: {model.default_history.data[0, 0, :].cpu().numpy()}")

    # 加载GPU FK模型
    logging.info("\n加载GPU FK模型...")
    gpu_fk = load_robot_model()
    if gpu_fk is None:
        logging.error("无法加载GPU FK模型，退出")
        return

    # 损失函数
    mse_criterion = nn.MSELoss()

    # 优化器（只微调 default_history 和最后的输出层，可以降低学习率）
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

    # 学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # 微调保存路径
    finetune_path = "/home/wsy/Desktop/casual/pieper_NN/pieper_causal_ik_no_history.pth"

    # 训练循环
    logging.info(f"\n开始微调（{config.epochs} epochs）...")

    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        # 计算当前epoch的None历史概率
        none_prob = get_none_history_prob(epoch, config)

        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        train_ik_loss = 0.0
        train_fk_loss = 0.0
        train_with_history_count = 0
        train_without_history_count = 0
        start_time = time.time()

        for batch_X, batch_y, batch_last_angle in train_loader:
            batch_X, batch_y, batch_last_angle = (
                batch_X.to(config.device, non_blocking=True),
                batch_y.to(config.device, non_blocking=True),
                batch_last_angle.to(config.device, non_blocking=True)
            )

            batch_size = batch_X.shape[0]

            # 核心修改：随机决定是否使用历史帧
            use_none_history = torch.rand(1).item() < none_prob

            if use_none_history:
                # 不使用历史帧（模拟推理场景）
                history_frames = None
                train_without_history_count += batch_size
            else:
                # 使用历史帧（正常训练）
                history_frames = batch_X
                train_with_history_count += batch_size

            # 判断y的格式并提取目标位姿和角度
            if batch_y.shape[1] == 14:
                target_pose = batch_y[:, :7]
                target_angles = batch_y[:, 7:]
                target_position = target_pose[:, :3]
                target_orientation = target_pose[:, 3:7]
            else:
                target_angles = batch_y
                target_position, target_orientation = forward_kinematics_with_pose(gpu_fk, target_angles)
                target_orientation = None

            optimizer.zero_grad()

            # 前向传播
            pred_joint_angles, info = model(
                history_frames,
                target_position,
                target_orientation
            )

            # 计算损失
            ik_loss = mse_criterion(pred_joint_angles, target_angles)

            pred_position, _ = forward_kinematics_with_pose(gpu_fk, pred_joint_angles)
            fk_position, _ = forward_kinematics_with_pose(gpu_fk, target_angles)
            fk_loss = mse_criterion(pred_position, fk_position)

            continuity_loss = torch.mean((pred_joint_angles - batch_last_angle) ** 2)

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

        train_loss = train_loss / len(train_loader.dataset)
        train_time = time.time() - start_time

        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0.0
        val_loss_with_history = 0.0
        val_loss_without_history = 0.0
        val_with_history_count = 0
        val_without_history_count = 0

        with torch.no_grad():
            for batch_X, batch_y, batch_last_angle in val_loader:
                batch_X, batch_y, batch_last_angle = (
                    batch_X.to(config.device, non_blocking=True),
                    batch_y.to(config.device, non_blocking=True),
                    batch_last_angle.to(config.device, non_blocking=True)
                )

                batch_size = batch_X.shape[0]

                # 判断y的格式并提取目标位姿和角度
                if batch_y.shape[1] == 14:
                    target_pose = batch_y[:, :7]
                    target_angles = batch_y[:, 7:]
                    target_position = target_pose[:, :3]
                    target_orientation = target_pose[:, 3:7]
                else:
                    target_angles = batch_y
                    target_position, target_orientation = forward_kinematics_with_pose(gpu_fk, target_angles)
                    target_orientation = None

                # 分别测试有历史和无历史的情况
                # 1. 有历史
                pred_with_history, _ = model(batch_X, target_position, target_orientation)
                ik_loss_with = mse_criterion(pred_with_history, target_angles)
                val_loss_with_history += ik_loss_with.item() * batch_size
                val_with_history_count += batch_size

                # 2. 无历史
                pred_without_history, _ = model(None, target_position, target_orientation)
                ik_loss_without = mse_criterion(pred_without_history, target_angles)
                val_loss_without_history += ik_loss_without.item() * batch_size
                val_without_history_count += batch_size

                # 总损失（使用训练时的None概率加权）
                total_loss = (1 - none_prob) * ik_loss_with + none_prob * ik_loss_without
                val_loss += total_loss.item() * batch_size

        val_loss = val_loss / len(val_loader.dataset)
        val_loss_with_history = val_loss_with_history / val_with_history_count if val_with_history_count > 0 else 0
        val_loss_without_history = val_loss_without_history / val_without_history_count if val_without_history_count > 0 else 0

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 日志
        logging.info(
            f"Epoch [{epoch}/{config.epochs}] | "
            f"None Prob: {none_prob:.2f} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Val (with hist): {val_loss_with_history:.6f} | "
            f"Val (no hist): {val_loss_without_history:.6f} | "
            f"IK: {train_ik_loss/len(train_loader.dataset):.4f} | "
            f"FK: {train_fk_loss/len(train_loader.dataset):.6f} | "
            f"LR: {current_lr:.6f} | Time: {train_time:.1f}s"
        )

        # 每5个epoch打印一次 default_history 的值
        if epoch % 5 == 0:
            logging.info(f"  default_history[0,0]: {model.default_history.data[0, 0, :].cpu().numpy()}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'val_loss_with_history': val_loss_with_history,
                'val_loss_without_history': val_loss_without_history,
            }, finetune_path)
            logging.info(f"  >>> 保存最优模型（验证损失：{val_loss:.6f}）")

    logging.info("\n微调完成！")
    logging.info(f"最优验证损失: {best_val_loss:.6f}")
    logging.info(f"\n最终 default_history 值:")
    logging.info(f"  [0, 0]: {model.default_history.data[0, 0, :].cpu().numpy()}")
    logging.info(f"  [0, 5]: {model.default_history.data[0, 5, :].cpu().numpy()}")


if __name__ == "__main__":
    finetune()
