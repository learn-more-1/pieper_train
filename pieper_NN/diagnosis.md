"""
诊断仿真效果差的原因

训练指标：R²=0.9997, Loss=0.0001 ✓
仿真效果：不好 ✗

可能原因分析：
"""

import torch
import numpy as np
from causal_ik_model_pieper import PieperCausalIK
from gpu_fk_wrapper import SimpleGPUFK
from dataset_generalized import create_windowed_dataloaders
import sys

sys.path.insert(0, '/home/wsy/Desktop/casual/pieper_NN')

def analyze_model_behavior():
    """分析模型行为"""

    print("=" * 70)
    print("诊断模型：训练好 vs 仿真差")
    print("=" * 70)

    # 加载模型
    model = PieperCausalIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        num_layers=2
    )

    checkpoint_path = "/home/wsy/Desktop/casual/pieper_NN/pieper_causal_ik.pth"

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ 成功加载模型: {checkpoint_path}")
        print(f"  最优验证损失: {checkpoint['best_val_loss']:.6f}")
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        return

    model = model.cuda()
    model.eval()

    # 加载GPU FK
    gpu_fk = SimpleGPUFK()

    # 加载测试数据
    class Config:
        data_path = "/home/wsy/Desktop/casual/merged_training_data.npz"
        batch_size = 512
        num_workers = 4
        pin_memory = True

    print("\n加载测试数据...")
    _, val_loader = create_windowed_dataloaders(Config.data_path, Config)

    print("\n" + "=" * 70)
    print("问题1：模型学习的是什么？")
    print("=" * 70)

    with torch.no_grad():
        for batch_X, batch_y, batch_last_angle in val_loader:
            batch_X = batch_X.cuda()
            batch_y = batch_y.cuda()
            batch_last_angle = batch_last_angle.cuda()

            history_frames = batch_X[:10]  # 只分析前10个样本
            target_angles = batch_y[:10]

            # 模型预测（使用零位姿作为条件）
            pred_angles, info = model(
                history_frames,
                torch.zeros(10, 3).cuda(),  # 零位置
                torch.zeros(10, 4).cuda()   # 零姿态
            )

            # 计算FK位置
            pred_pos, _ = gpu_fk.forward(pred_angles), None
            target_pos, _ = gpu_fk.forward(target_angles), None

            print(f"\n前10个样本分析：")
            print(f"\n1. 历史最后一帧与目标的角度差异（真实数据中的运动幅度）:")
            last_angle = history_frames[:, -1, :]
            real_motion = torch.mean((target_angles - last_angle) ** 2, dim=1)
            print(f"  平均: {real_motion.mean():.6f}")
            print(f"  最小: {real_motion.min():.6f}")
            print(f"  最大: {real_motion.max():.6f}")

            print(f"\n2. 模型预测与目标的角度差异:")
            pred_error = torch.mean((pred_angles - target_angles) ** 2, dim=1)
            print(f"  平均: {pred_error.mean():.6f}")
            print(f"  最小: {pred_error.min():.6f}")
            print(f"  最大: {pred_error.max():.6f}")

            print(f"\n3. 模型预测与历史的连续性（GAP损失）:")
            gap = torch.mean((pred_angles - last_angle) ** 2, dim=1)
            print(f"  平均: {gap.mean():.6f}")
            print(f"  最小: {gap.min():.6f}")
            print(f"  最大: {gap.max():.6f}")

            print(f"\n4. FK位置误差（米）:")
            pos_error = torch.sqrt(torch.sum((pred_pos - target_pos) ** 2, dim=1))
            print(f"  平均: {pos_error.mean():.4f} m")
            print(f"  最小: {pos_error.min():.4f} m")
            print(f"  最大: {pos_error.max():.4f} m")

            # 详细分析第一个样本
            print(f"\n5. 第一个样本详细分析:")
            print(f"   历史最后一帧角度: {last_angle[0].cpu().numpy()}")
            print(f"   目标角度:         {target_angles[0].cpu().numpy()}")
            print(f"   预测角度:         {pred_angles[0].cpu().numpy()}")
            print(f"   差异(预测-目标):  {(pred_angles[0] - target_angles[0]).cpu().numpy()}")
            print(f"   FK位置（目标）:   {target_pos[0].cpu().numpy()}")
            print(f"   FK位置（预测）:   {pred_pos[0].cpu().numpy()}")
            print(f"   位置误差:         {pos_error[0]:.4f} m")

            break

    print("\n" + "=" * 70)
    print("关键诊断问题：")
    print("=" * 70)
    print("\n❓ 模型实际学习的任务是什么？")
    print("   训练时：历史10帧 → 预测下一帧角度（时序预测）")
    print("   仿真时：目标位姿 → 计算关节角度（IK求解）")
    print("   → 这两个任务不同！")

    print("\n❓ 测试时如何使用模型？")
    print("   如果使用零位姿作为条件，模型会输出什么？")
    print("   → 可能只是复制历史最后一帧（因为GAP损失很小）")

    print("\n❓ 仿真场景是什么？")
    print("   A. 给定目标手腕位置，求解关节角度（标准IK）")
    print("   B. 跟踪演示轨迹（运动模仿）")
    print("   C. 其他...")

    print("\n" + "=" * 70)
    print("建议的测试方法：")
    print("=" * 70)
    print("\n1. 测试不同的输入条件：")
    print("   - 使用随机位姿作为条件")
    print("   - 使用目标位姿作为条件")
    print("   - 观察预测角度的变化")

    print("\n2. 测试模型的IK能力：")
    print("   - 固定历史窗口")
    print("   - 改变目标位姿")
    print("   - 检查预测角度是否合理变化")

    print("\n3. 对比训练分布：")
    print("   - 仿真中的目标位姿是否在训练分布内？")
    print("   - 是否超出训练数据覆盖的范围？")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    analyze_model_behavior()
