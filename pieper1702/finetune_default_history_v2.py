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
import os

# 清空之前的导入，确保使用正确的模型
for mod in list(sys.modules.keys()):
    if 'causal_ik_model_pieper' in mod:
        del sys.modules[mod]

# 设置正确的路径
sys.path.insert(0, '/home/wsy/Desktop/casual/pieper1101')
sys.path.insert(1, '/home/bonuli/Pieper/pieper1702')

# 先导入 gpu_fk_wrapper
from gpu_fk_wrapper import SimpleGPUFK

# 然后导入模型（pieper1702 版本会覆盖 pieper1101 版本）
from causal_ik_model_pieper2 import PieperCausalIK

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
    lr = 1e-4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    initial_none_prob = 0.0
    final_none_prob = 0.5  # 保持在 0.5，确保持续训练 default_history
    warmup_epochs = 15  # 延长 warmup 期

    ik_weight = 1.0
    fk_weight = 0.2

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

    train_data, val_data = load_data(config.data_path)

    logging.info(f"\n加载预训练模型: {config.pretrained_path}")
    model = PieperCausalIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        num_layers=2
    ).to(config.device)

    checkpoint = torch.load(config.pretrained_path, map_location=config.device)

    # 检查并添加 default_history 参数
    if not hasattr(model, 'default_history'):
        logging.info(f"  添加 default_history 参数...")
        # 使用 register_parameter 正确注册参数
        default_history_param = nn.Parameter(torch.zeros(1, 10, 7, device=config.device))
        model.register_parameter('default_history', default_history_param)
    else:
        if not isinstance(model.default_history, nn.Parameter):
            logging.info(f"  转换 default_history 为 Parameter...")
            default_history_param = nn.Parameter(torch.zeros(1, 10, 7, device=config.device))
            model.register_parameter('default_history', default_history_param)

    # 加载权重（default_history 会被跳过，因为没有在 checkpoint 中）
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    logging.info(f"  ✓ 成功加载预训练权重")

    # 调试：确认模型文件和 forward 方法
    logging.info(f"  模型类型: {type(model)}")
    logging.info(f"  模型文件: {model.__class__.__module__}")
    import inspect
    forward_source = inspect.getsource(model.forward)
    logging.info(f"  forward 方法支持 None: {'if history_frames is None' in forward_source}")

    # 测试 forward 是否支持 None
    try:
        test_pos = torch.randn(1, 3, device=config.device)
        test_ori = torch.randn(1, 4, device=config.device)
        with torch.no_grad():
            test_pred, _ = model(None, test_pos, test_ori)
        logging.info(f"  ✓ forward 方法支持 None 输入")
    except Exception as e:
        logging.error(f"  ✗ forward 方法不支持 None: {e}")
        return

    # 只训练 default_history
    trainable_count = 0
    for name, param in model.named_parameters():
        if 'default_history' in name:
            param.requires_grad = True
            trainable_count += param.numel()
            logging.info(f"  训练参数: {name} ({param.numel()} 个)")
        else:
            param.requires_grad = False

    if trainable_count == 0:
        logging.error(f"  ✗ 没有 default_history 参数！")
        return

    logging.info(f"  总训练参数: {trainable_count}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

    try:
        gpu_fk = SimpleGPUFK()
        logging.info(f"  ✓ GPU FK")
    except:
        logging.error(f"  ✗ GPU FK 加载失败")
        return

    best_val_loss = float("inf")
    n_batches = (len(train_data) + config.batch_size - 1) // config.batch_size

    logging.info(f"\n开始微调...")
    logging.info(f"  batches/epoch: {n_batches}")
    logging.info(f"  curriculum: {config.initial_none_prob} → {config.final_none_prob}")

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        none_prob = get_none_prob(epoch, config)

        # 早期全部使用 None 历史，确保 default_history 得到训练
        # 前 10 个 epochs 使用 100% None 历史
        if epoch < 10:
            actual_none_prob = 1.0
        else:
            actual_none_prob = none_prob

        logging.info(f"  Epoch [{epoch}/{config.epochs}] | None Prob: {none_prob:.2f} (actual: {actual_none_prob:.2f})")

        for i in range(n_batches):
            start_idx = i * config.batch_size
            end_idx = min((i + 1) * config.batch_size, len(train_data))

            batch_y = torch.from_numpy(train_data[start_idx:end_idx]).to(config.device)

            target_pose = batch_y[:, :7]
            target_angles = batch_y[:, 7:]
            target_position = target_pose[:, :3]
            target_orientation = target_pose[:, 3:7]

            use_none_history = torch.rand(1).item() < actual_none_prob

            if use_none_history:
                history_frames = None
            else:
                history_frames = target_angles.unsqueeze(1).repeat(1, 10, 1)

            optimizer.zero_grad()

            pred_angles, _ = model(history_frames, target_position, target_orientation)

            ik_loss = nn.functional.mse_loss(pred_angles, target_angles)

            pred_position = gpu_fk.forward(pred_angles)
            target_position_fk = gpu_fk.forward(target_angles)
            fk_loss = nn.functional.mse_loss(pred_position, target_position_fk)

            total_loss = config.ik_weight * ik_loss + config.fk_weight * fk_loss

            # 只在 loss 有梯度时才进行反向传播
            # 当 history_frames 不是 None 时，default_history 不参与计算，loss 没有梯度
            if total_loss.requires_grad:
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

        # 验证
        model.eval()
        val_loss_with = 0.0
        val_loss_without = 0.0

        with torch.no_grad():
            batch_y = torch.from_numpy(val_data[:config.batch_size]).to(config.device)
            target_pose = batch_y[:, :7]
            target_angles = batch_y[:, 7:]
            target_position = target_pose[:, :3]
            target_orientation = target_pose[:, 3:7]

            # 有历史
            history = target_angles.unsqueeze(1).repeat(1, 10, 1)
            pred, _ = model(history, target_position, target_orientation)
            val_loss_with = nn.functional.mse_loss(pred, target_angles).item()

            # 无历史
            pred, _ = model(None, target_position, target_orientation)
            val_loss_without = nn.functional.mse_loss(pred, target_angles).item()

        logging.info(
            f"\nEpoch [{epoch}/{config.epochs}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val (with hist): {val_loss_with:.6f} | "
            f"Val (no hist): {val_loss_without:.6f} | "
            f"None Prob: {none_prob:.2f}"
        )

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
        history = target_angles.unsqueeze(1).repeat(1, 10, 1)
        pred_with, _ = model(history, target_position, target_orientation)
        mae_with = nn.functional.l1_loss(pred_with, target_angles).item()

        pred_without, _ = model(None, target_position, target_orientation)
        mae_without = nn.functional.l1_loss(pred_without, target_angles).item()

    logging.info(f"  有历史 MAE: {mae_with:.6f}")
    logging.info(f"  无历史 MAE: {mae_without:.6f}")


if __name__ == "__main__":
    finetune()
