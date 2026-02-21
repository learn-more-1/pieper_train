"""
微调原始模型的 default_history 参数

在 GRAB 数据上训练 default_history，让无历史时也能表现好
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import sys
sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1101')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1702')  # 使用修改后的模型

from causal_ik_model_pieper2 import PieperCausalIK  # 使用 pieper1702 的版本（支持 default_history）
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("finetune_default_history.log")]
)


class FinetuneConfig:
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"
    pretrained_path = "/home/bonuli/Pieper/pieper1101/pieper_causal_ik_1101.pth"

    batch_size = 2048
    epochs = 30
    lr = 1e-4  # 只微调 default_history，用小学习率

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Curriculum learning: 逐渐增加无历史的比例
    initial_none_prob = 0.0
    final_none_prob = 1.0
    warmup_epochs = 20

    # 损失权重
    ik_weight = 1.0
    fk_weight = 0.2

    # 保存路径
    checkpoint_path = "/home/bonuli/Pieper/pieper1101/pieper_causal_ik_GRAB_adapted.pth"


def load_data(data_path):
    """加载数据"""
    logging.info(f"加载数据: {data_path}")
    data = np.load(data_path)
    y = data['y'].astype(np.float32)

    split_idx = int(len(y) * 0.9)
    train_data = y[:split_idx]
    val_data = y[split_idx:]

    logging.info(f"  训练集: {len(train_data)}")
    logging.info(f"  验证集: {len(val_data)}")

    return train_data, val_data


def get_none_prob(epoch, config):
    """计算当前 epoch 使用无历史的概率"""
    if epoch >= config.warmup_epochs:
        return config.final_none_prob
    progress = epoch / config.warmup_epochs
    return config.initial_none_prob + progress * (
        config.final_none_prob - config.initial_none_prob
    )


def finetune():
    config = FinetuneConfig()

    logging.info("=" * 70)
    logging.info("微调原始模型：适应无历史场景")
    logging.info("=" * 70)

    # 加载数据
    train_data, val_data = load_data(config.data_path)

    # 加载模型
    logging.info(f"\n加载预训练模型: {config.pretrained_path}")
    model = PieperCausalIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        num_layers=2
    ).to(config.device)

    checkpoint = torch.load(config.pretrained_path, map_location=config.device)

    # 检查模型是否有 default_history 参数
    model_state = model.state_dict()
    has_default_history = 'default_history' in model_state

    if not has_default_history:
        logging.info(f"  ⚠ 模型没有 default_history 参数，添加...")
        # 添加 default_history 参数
        model.register_buffer(
            'default_history',
            torch.zeros(1, 10, 7, device=config.device)
        )

    # 加载预训练权重（允许部分加载）
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing:
        logging.info(f"  缺少的参数: {missing}")
    if unexpected:
        logging.info(f"  多余的参数: {unexpected}")

    # 检查 default_history 是否需要梯度
    if hasattr(model, 'default_history'):
        if isinstance(model.default_history, nn.Parameter):
            logging.info(f"  ✓ default_history 是 Parameter，可以训练")
        else:
            # 将 buffer 转换为 Parameter
            logging.info(f"  将 default_history 从 buffer 转换为 Parameter...")
            del model.default_history
            model.register_buffer(
                'default_history',
                torch.zeros(1, 10, 7, device=config.device, requires_grad=True)
            )
            # 重新加载权重（保持其他参数不变）
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    logging.info(f"  ✓ 成功加载预训练权重")

    # 冻结除 default_history 外的所有参数
    trainable_count = 0
    for name, param in model.named_parameters():
        if 'default_history' in name:
            param.requires_grad = True
            trainable_count += param.numel()
            logging.info(f"  训练参数: {name} ({param.numel()} 个)")
        else:
            param.requires_grad = False

    if trainable_count == 0:
        logging.error(f"  ✗ 没有可训练的参数！模型可能不支持 default_history")
        logging.info(f"  可用参数: {[n for n, _ in model.named_parameters()]}")
        return

    logging.info(f"  总训练参数: {trainable_count}")

    # 只优化 default_history
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

    # 加载 FK
    try:
        gpu_fk = SimpleGPUFK()
        logging.info(f"  ✓ GPU FK")
    except:
        logging.error(f"  ✗ GPU FK 加载失败")
        return

    # 训练循环
    best_val_loss = float("inf")
    n_batches = (len(train_data) + config.batch_size - 1) // config.batch_size

    logging.info(f"\n开始微调...")
    logging.info(f"  batches/epoch: {n_batches}")
    logging.info(f"  curriculum: {config.initial_none_prob} → {config.final_none_prob}")

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_with_hist = 0
        train_without_hist = 0
        none_prob = get_none_prob(epoch, config)

        for i in range(n_batches):
            start_idx = i * config.batch_size
            end_idx = min((i + 1) * config.batch_size, len(train_data))

            batch_y = torch.from_numpy(train_data[start_idx:end_idx]).to(config.device)

            target_pose = batch_y[:, :7]
            target_angles = batch_y[:, 7:]
            target_position = target_pose[:, :3]
            target_orientation = target_pose[:, 3:7]

            # Curriculum: 随机决定是否使用历史
            use_none_history = torch.rand(1).item() < none_prob

            if use_none_history:
                history_frames = None
                train_without_hist += len(batch_y)
            else:
                history_frames = target_angles.unsqueeze(1).repeat(1, 10, 1)
                train_with_hist += len(batch_y)

            optimizer.zero_grad()

            pred_angles, _ = model(history_frames, target_position, target_orientation)

            # 计算损失
            ik_loss = nn.functional.mse_loss(pred_angles, target_angles)

            pred_position = gpu_fk.forward(pred_angles)
            target_position_fk = gpu_fk.forward(target_angles)
            fk_loss = nn.functional.mse_loss(pred_position, target_position_fk)

            total_loss = config.ik_weight * ik_loss + config.fk_weight * fk_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * len(batch_y)

            if (i + 1) % 100 == 0:
                progress = (i + 1) / n_batches * 100
                logging.info(
                    f"  Epoch [{epoch}/{config.epochs}] "
                    f"Progress: {progress:.1f}% | Loss: {total_loss.item():.6f} | "
                    f"None Prob: {none_prob:.2f}"
                )

        train_loss = train_loss / len(train_data)

        # 验证（分别测试有历史和无历史）
        model.eval()
        val_loss_with = 0.0
        val_loss_without = 0.0

        with torch.no_grad():
            # 测试有历史
            batch_y = torch.from_numpy(val_data[:config.batch_size]).to(config.device)
            target_pose = batch_y[:, :7]
            target_angles = batch_y[:, 7:]
            target_position = target_pose[:, :3]
            target_orientation = target_pose[:, 3:7]

            history = target_angles.unsqueeze(1).repeat(1, 10, 1)
            pred, _ = model(history, target_position, target_orientation)
            val_loss_with = nn.functional.mse_loss(pred, target_angles).item()

            # 测试无历史
            pred, _ = model(None, target_position, target_orientation)
            val_loss_without = nn.functional.mse_loss(pred, target_angles).item()

        logging.info(
            f"\nEpoch [{epoch}/{config.epochs}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val (with hist): {val_loss_with:.6f} | "
            f"Val (no hist): {val_loss_without:.6f} | "
            f"None Prob: {none_prob:.2f}"
        )

        # 保存最优模型（基于无历史性能）
        if val_loss_without < best_val_loss:
            best_val_loss = val_loss_without
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
            }, config.checkpoint_path)
            logging.info(f"  >>> 保存 (no_hist: {val_loss_without:.6f})")

    logging.info(f"\n微调完成！")
    logging.info(f"  最优验证损失（无历史）: {best_val_loss:.6f}")
    logging.info(f"  模型保存到: {config.checkpoint_path}")

    # 测试最终效果
    logging.info(f"\n最终效果测试:")
    batch_y = torch.from_numpy(val_data[:1000]).to(config.device)
    target_pose = batch_y[:, :7]
    target_angles = batch_y[:, 7:]
    target_position = target_pose[:, :3]
    target_orientation = target_pose[:, 3:7]

    model.eval()
    with torch.no_grad():
        # 有历史
        history = target_angles.unsqueeze(1).repeat(1, 10, 1)
        pred_with, _ = model(history, target_position, target_orientation)
        mae_with = nn.functional.l1_loss(pred_with, target_angles).item()

        # 无历史
        pred_without, _ = model(None, target_position, target_orientation)
        mae_without = nn.functional.l1_loss(pred_without, target_angles).item()

    logging.info(f"  有历史 MAE: {mae_with:.6f}")
    logging.info(f"  无历史 MAE: {mae_without:.6f}")


if __name__ == "__main__":
    finetune()
