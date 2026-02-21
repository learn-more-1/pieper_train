"""
测试微调效果：对比有无历史数据的性能
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/bonuli/pieper/casual')
sys.path.insert(0, '/home/bonuli/Pieper/casual/pieper1101')

from causal_ik_model_pieper2 import PieperCausalIK
from torch.utils.data import DataLoader
from dataset_generalized import create_windowed_dataloaders
from gpu_fk_wrapper import SimpleGPUFK
import numpy as np


class TestConfig:
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
    batch_size = 512
    num_joints = 7
    num_frames = 10
    hidden_dim = 256
    num_layers = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 模型路径
    pretrained_path = "/home/bonuli/Pieper/pieper1101/pieper_causal_ik_1101.pth"
    finetuned_path = "/home/bonuli/Pieper/pieper1101/pieper_causal_ik_no_history.pth"


def load_model(model_path, config):
    """加载模型"""
    model = PieperCausalIK(
        num_joints=config.num_joints,
        num_frames=config.num_frames,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers
    ).to(config.device)

    if torch.cuda.is_available():
        checkpoint = torch.load(model_path, map_location=config.device)
    else:
        checkpoint = torch.load(model_path, map_location='cpu')

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def test_model(model, data_loader, gpu_fk, config, use_history=True):
    """测试模型性能"""
    mse_criterion = nn.MSELoss()
    total_loss = 0.0
    total_count = 0
    preds = []
    trues = []

    with torch.no_grad():
        for batch_X, batch_y, batch_last_angle in data_loader:
            batch_X, batch_y, batch_last_angle = (
                batch_X.to(config.device, non_blocking=True),
                batch_y.to(config.device, non_blocking=True),
                batch_last_angle.to(config.device, non_blocking=True)
            )

            batch_size = batch_X.shape[0]

            # 判断y的格式
            if batch_y.shape[1] == 14:
                target_pose = batch_y[:, :7]
                target_angles = batch_y[:, 7:]
                target_position = target_pose[:, :3]
                target_orientation = target_pose[:, 3:7]
            else:
                target_angles = batch_y
                target_position = gpu_fk.forward(target_angles)
                target_orientation = None

            # 前向传播
            history_frames = batch_X if use_history else None
            pred_angles, _ = model(history_frames, target_position, target_orientation)

            # 计算损失
            loss = mse_criterion(pred_angles, target_angles)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            preds.append(pred_angles.cpu().numpy())
            trues.append(target_angles.cpu().numpy())

    # 计算指标
    avg_loss = total_loss / total_count
    preds = np.vstack(preds)
    trues = np.vstack(trues)

    # R² 分数
    r2 = 1 - (np.sum((trues - preds) ** 2) /
              (np.sum((trues - np.mean(trues)) ** 2) + 1e-8))

    # 平均绝对误差（度）
    mae = np.mean(np.abs(preds - trues))

    return avg_loss, r2, mae


def main():
    config = TestConfig()

    print("=" * 80)
    print("测试微调效果：对比有无历史数据的性能")
    print("=" * 80)

    # 加载数据
    print("\n加载数据集...")
    _, val_loader = create_windowed_dataloaders(config.data_path, config)
    gpu_fk = SimpleGPUFK()

    # 加载模型
    print("\n加载模型...")
    print(f"  预训练模型: {config.pretrained_path}")
    print(f"  微调模型: {config.finetuned_path}")

    model_pretrained = load_model(config.pretrained_path, config)
    model_finetuned = load_model(config.finetuned_path, config)

    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)

    # 测试1: 预训练模型 + 有历史
    loss, r2, mae = test_model(model_pretrained, val_loader, gpu_fk, config, use_history=True)
    print(f"\n[预训练模型] + 有历史:")
    print(f"  Loss:   {loss:.6f}")
    print(f"  R²:     {r2:.6f}")
    print(f"  MAE:    {mae:.4f} rad")

    # 测试2: 预训练模型 + 无历史
    loss, r2, mae = test_model(model_pretrained, val_loader, gpu_fk, config, use_history=False)
    print(f"\n[预训练模型] + 无历史:")
    print(f"  Loss:   {loss:.6f}")
    print(f"  R²:     {r2:.6f}")
    print(f"  MAE:    {mae:.4f} rad")

    # 测试3: 微调模型 + 有历史
    loss, r2, mae = test_model(model_finetuned, val_loader, gpu_fk, config, use_history=True)
    print(f"\n[微调模型] + 有历史:")
    print(f"  Loss:   {loss:.6f}")
    print(f"  R²:     {r2:.6f}")
    print(f"  MAE:    {mae:.4f} rad")

    # 测试4: 微调模型 + 无历史
    loss, r2, mae = test_model(model_finetuned, val_loader, gpu_fk, config, use_history=False)
    print(f"\n[微调模型] + 无历史:")
    print(f"  Loss:   {loss:.6f}")
    print(f"  R²:     {r2:.6f}")
    print(f"  MAE:    {mae:.4f} rad")

    # 对比分析
    print("\n" + "=" * 80)
    print("对比分析")
    print("=" * 80)

    # 计算 pre_no_hist 和 finetuned_no_hist 的对比
    loss_pre_no, _, _ = test_model(model_pretrained, val_loader, gpu_fk, config, use_history=False)
    loss_fine_no, _, _ = test_model(model_finetuned, val_loader, gpu_fk, config, use_history=False)

    improvement = (loss_pre_no - loss_fine_no) / loss_pre_no * 100
    print(f"\n无历史场景下的性能提升: {improvement:.2f}%")

    # 显示 default_history 值
    print(f"\n微调后的 default_history 值:")
    print(f"  第1帧: {model_finetuned.default_history.data[0, 0, :].cpu().numpy()}")
    print(f"  第5帧: {model_finetuned.default_history.data[0, 4, :].cpu().numpy()}")
    print(f"  第10帧: {model_finetuned.default_history.data[0, 9, :].cpu().numpy()}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
