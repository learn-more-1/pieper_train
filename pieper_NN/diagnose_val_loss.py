"""
诊断验证损失异常高的问题

观察：
- Train Loss: 0.000147, R²: 0.9997
- Val Loss: 0.0827, R²: 0.9997

R²都很高，说明角度预测准确。但Val Loss是Train Loss的500多倍！

可能原因：
1. 验证集的 GAP loss 特别大
2. 验证集和训练集分布不同
3. 损失计算有bug
"""

import torch
import numpy as np
from dataset_accad_cmu import create_accad_cmu_dataloaders
from causal_ik_model_pieper import PieperCausalIK
from gpu_fk_wrapper import SimpleGPUFK
import sys
sys.path.insert(0, '/home/wsy/Desktop/casual/pieper_NN')

class Config:
    data_path = "/home/wsy/Desktop/casual/ACCAD_CMU_merged_training_data.npz"
    batch_size = 512
    num_workers = 4
    pin_memory = True
    device = torch.device("cuda:0")

def diagnose_val_loss():
    """诊断验证损失"""

    print("=" * 70)
    print("诊断验证损失异常")
    print("=" * 70)

    # 加载模型
    model = PieperCausalIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=512,
        num_layers=2
    ).to(Config.device)

    try:
        checkpoint = torch.load('/home/wsy/Desktop/casual/pieper_NN/pieper_causal_ik.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 加载checkpoint，epoch: {checkpoint['epoch']}")
    except:
        print("⚠ 无法加载checkpoint，使用未训练模型")

    model.eval()

    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader = create_accad_cmu_dataloaders(Config.data_path, Config)

    gpu_fk = SimpleGPUFK()

    # 检查训练集
    print("\n" + "=" * 70)
    print("训练集损失分析（前10个batch）")
    print("=" * 70)

    with torch.no_grad():
        for i, (batch_X, batch_y, batch_last_angle, batch_human_pose) in enumerate(train_loader):
            if i >= 10:
                break

            batch_X = batch_X.to(Config.device)
            batch_y = batch_y.to(Config.device)
            batch_last_angle = batch_last_angle.to(Config.device)
            batch_human_pose = batch_human_pose.to(Config.device)

            batch_size = batch_X.shape[0]

            # 前向传播
            pred_angles, _ = model(
                batch_X,
                batch_human_pose[:, :3],
                batch_human_pose[:, 3:7]
            )

            # 计算各项损失
            ik_loss = torch.mean((pred_angles - batch_y) ** 2)

            pred_pos, _ = gpu_fk.forward(pred_angles), None
            target_pos, _ = gpu_fk.forward(batch_y), None
            fk_loss = torch.mean((pred_pos - target_pos) ** 2)

            gap_loss = torch.mean((pred_angles - batch_last_angle) ** 2)

            total_loss = ik_loss + 0.5 * fk_loss + gap_loss

            if i == 0:
                print(f"\nBatch {i}:")
                print(f"  IK Loss:    {ik_loss:.6f}")
                print(f"  FK Loss:    {fk_loss:.6f}")
                print(f"  GAP Loss:   {gap_loss:.6f}")
                print(f"  Total Loss: {total_loss:.6f}")

    # 检查验证集
    print("\n" + "=" * 70)
    print("验证集损失分析（前10个batch）")
    print("=" * 70)

    with torch.no_grad():
        for i, (batch_X, batch_y, batch_last_angle, batch_human_pose) in enumerate(val_loader):
            if i >= 10:
                break

            batch_X = batch_X.to(Config.device)
            batch_y = batch_y.to(Config.device)
            batch_last_angle = batch_last_angle.to(Config.device)
            batch_human_pose = batch_human_pose.to(Config.device)

            batch_size = batch_X.shape[0]

            # 前向传播
            pred_angles, _ = model(
                batch_X,
                batch_human_pose[:, :3],
                batch_human_pose[:, 3:7]
            )

            # 计算各项损失
            ik_loss = torch.mean((pred_angles - batch_y) ** 2)

            pred_pos, _ = gpu_fk.forward(pred_angles), None
            target_pos, _ = gpu_fk.forward(batch_y), None
            fk_loss = torch.mean((pred_pos - target_pos) ** 2)

            gap_loss = torch.mean((pred_angles - batch_last_angle) ** 2)

            total_loss = ik_loss + 0.5 * fk_loss + gap_loss

            if i == 0:
                print(f"\nBatch {i}:")
                print(f"  IK Loss:    {ik_loss:.6f}")
                print(f"  FK Loss:    {fk_loss:.6f}")
                print(f"  GAP Loss:   {gap_loss:.6f}")
                print(f"  Total Loss: {total_loss:.6f}")

    # 分析 GAP loss
    print("\n" + "=" * 70)
    print("GAP Loss 详细分析（验证集）")
    print("=" * 70)

    with torch.no_grad():
        gap_values_train = []
        gap_values_val = []

        # 训练集
        for i, (batch_X, batch_y, batch_last_angle, _) in enumerate(train_loader):
            if i >= 20:
                break
            gap = torch.mean((batch_y - batch_last_angle) ** 2)
            gap_values_train.append(gap.item())

        # 验证集
        for i, (batch_X, batch_y, batch_last_angle, _) in enumerate(val_loader):
            if i >= 20:
                break
            gap = torch.mean((batch_y - batch_last_angle) ** 2)
            gap_values_val.append(gap.item())

    print(f"\n真实数据的 GAP（目标角度 - 当前角度）:")
    print(f"  训练集: mean={np.mean(gap_values_train):.6f}, std={np.std(gap_values_train):.6f}")
    print(f"  验证集: mean={np.mean(gap_values_val):.6f}, std={np.std(gap_values_val):.6f}")

    print("\n" + "=" * 70)
    print("结论:")
    print("=" * 70)


if __name__ == '__main__':
    diagnose_val_loss()
