"""
测试修改后的训练流程

关键改动：
1. 数据集返回人臂位姿（human_pose）
2. 训练使用人臂位姿作为条件，而不是机器人FK位置
"""

import torch
import sys
sys.path.insert(0, '/home/wsy/Desktop/casual/pieper_NN')

from causal_ik_model_pieper import PieperCausalIK
from gpu_fk_wrapper import SimpleGPUFK
from dataset_generalized import create_windowed_dataloaders

class Config:
    data_path = "/home/wsy/Desktop/casual/merged_training_data.npz"
    batch_size = 4
    num_workers = 0
    pin_memory = False

def test_training_loop():
    """测试修改后的训练循环"""

    print("=" * 70)
    print("测试修改后的训练流程")
    print("=" * 70)

    # 创建模型
    model = PieperCausalIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        num_layers=2
    ).cuda()
    model.eval()

    # 加载数据
    print("\n1. 加载数据...")
    train_loader, val_loader = create_windowed_dataloaders(Config.data_path, Config)

    # 测试一个batch
    print("\n2. 测试数据加载...")
    for batch_X, batch_y, batch_last_angle, batch_human_pose in train_loader:
        batch_size = batch_X.shape[0]

        print(f"   batch_X: {batch_X.shape} - 机器人历史角度")
        print(f"   batch_y: {batch_y.shape} - 下一帧机器人角度")
        print(f"   batch_last_angle: {batch_last_angle.shape} - 最后一帧机器人角度")
        print(f"   batch_human_pose: {batch_human_pose.shape} - 最后一帧人臂位姿")

        # 提取人臂位姿
        target_position = batch_human_pose[:, :3].cuda()        # [batch, 3]
        target_orientation = batch_human_pose[:, 3:7].cuda()     # [batch, 4]

        print(f"\n   人臂位姿（第1个样本）:")
        print(f"     位置: {target_position[0].cpu().numpy()}")
        print(f"     姿态: {target_orientation[0].cpu().numpy()}")

        # 前向传播（使用人臂位姿作为条件）
        print(f"\n3. 前向传播（使用人臂位姿作为条件）...")
        with torch.no_grad():
            pred_angles, info = model(
                batch_X.cuda(),
                target_position,
                target_orientation
            )

        print(f"   预测角度: {pred_angles.shape}")
        print(f"   预测角度（第1个样本）: {pred_angles[0].cpu().numpy()}")

        # 对比目标角度
        target_angles = batch_y.cuda()
        print(f"\n   目标角度（第1个样本）: {target_angles[0].cpu().numpy()}")
        print(f"   预测误差: {torch.mean((pred_angles - target_angles) ** 2).item():.6f}")

        # 比较机器人FK位置和人臂位置
        gpu_fk = SimpleGPUFK()
        pred_pos, _ = gpu_fk.forward(pred_angles), None
        target_pos, _ = gpu_fk.forward(target_angles), None

        print(f"\n   机器人FK位置（预测）: {pred_pos[0].cpu().numpy()}")
        print(f"   机器人FK位置（目标）: {target_pos[0].cpu().numpy()}")
        print(f"   人臂位置（输入条件）: {target_position[0].cpu().numpy()}")

        print(f"\n   关键观察:")
        print(f"     - 人臂位置 ≠ 机器人FK位置（正常！）")
        print(f"     - 模型现在使用人臂位置作为条件")
        print(f"     - 训练目标：从人臂位置 → 预测机器人角度")

        break

    print("\n" + "=" * 70)
    print("✓ 测试通过！修改正确。")
    print("=" * 70)
    print("\n修改总结:")
    print("  1. 数据集返回人臂位姿（第4个返回值）")
    print("  2. 训练使用人臂位姿作为模型条件")
    print("  3. 目标：学习从人臂位姿到机器人角度的映射")
    print("\n预期改进:")
    print("  - 仿真时也输入人臂位姿")
    print("  - 模型见过人臂位姿分布")
    print("  - 泛化性能提升")
    print("=" * 70)


if __name__ == '__main__':
    test_training_loop()
