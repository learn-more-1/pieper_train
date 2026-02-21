"""
简化模型推理示例

展示如何使用训练好的简化模型进行推理
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/bonuli/pieper/casual')
sys.path.insert(0, '/home/bonuli/Pieper/casual/pieper1101')

from causal_ik_model_pieper_simple import SimplifiedCausalIK


def inference_demo():
    print("=" * 70)
    print("简化IK模型推理示例")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = SimplifiedCausalIK(num_joints=7, hidden_dim=256).to(device)
    model.eval()

    # 加载训练好的权重
    checkpoint_path = "/home/bonuli/Pieper/pieper1101/simplified_causal_ik.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ 成功加载模型权重")
    except:
        print(f"⚠ 未找到权重文件，使用随机初始化")

    # 推理示例
    print("\n" + "-" * 70)
    print("推理场景：给定目标位姿，预测关节角度")
    print("-" * 70)

    # 示例1: 单个目标位姿
    print("\n[示例1] 单个目标位姿:")
    target_position = torch.tensor([[0.3, 0.2, 0.5]], device=device)  # [1, 3]
    target_orientation = torch.tensor([[0, 0, 0, 1]], device=device)  # [1, 4] 单位四元数

    with torch.no_grad():
        pred_angles, info = model(target_position, target_orientation)

    print(f"  目标位置: {target_position.cpu().numpy()[0]}")
    print(f"  预测角度: {pred_angles.cpu().numpy()[0]}")
    print(f"  关节权重: {info['position_weights'].cpu().numpy()[0]}")

    # 示例2: 批量推理
    print("\n[示例2] 批量推理:")
    batch_size = 8
    target_position = torch.randn(batch_size, 3, device=device)
    target_orientation = torch.randn(batch_size, 4, device=device)
    # 归一化为单位四元数
    target_orientation = target_orientation / target_orientation.norm(dim=1, keepdim=True)

    with torch.no_grad():
        pred_angles, info = model(target_position, target_orientation)

    print(f"  批次大小: {batch_size}")
    print(f"  输入形状: 位置={target_position.shape}, 姿态={target_orientation.shape}")
    print(f"  输出形状: {pred_angles.shape}")

    # 示例3: 实际使用
    print("\n[示例3] 实际使用代码:")
    print("""
    def solve_ik(model, target_position, target_orientation):
        '''
        IK求解器

        Args:
            model: 训练好的 SimplifiedCausalIK 模型
            target_position: np.array [3] 或 [batch, 3] 目标位置
            target_orientation: np.array [4] 或 [batch, 4] 目标姿态（四元数）

        Returns:
            joint_angles: np.array [7] 或 [batch, 7] 预测的关节角度
        '''
        model.eval()

        # 转换为tensor
        if target_position.ndim == 1:
            target_position = target_position[np.newaxis, :]
        if target_orientation.ndim == 1:
            target_orientation = target_orientation[np.newaxis, :]

        target_position = torch.from_numpy(target_position).float().to(device)
        target_orientation = torch.from_numpy(target_orientation).float().to(device)

        # 推理
        with torch.no_grad():
            pred_angles, _ = model(target_position, target_orientation)

        return pred_angles.cpu().numpy()

    # 使用示例
    target_pos = np.array([0.3, 0.2, 0.5])
    target_ori = np.array([0, 0, 0, 1])  # 单位四元数
    joint_angles = solve_ik(model, target_pos, target_ori)
    print(f"关节角度: {joint_angles}")
    """)

    # 展示学习的耦合关系
    print("\n" + "-" * 70)
    print("模型学习的关节耦合关系")
    print("-" * 70)

    coupling = model.coupling_embedding.coupling_prototype.data.cpu()  # [7, 256]
    print(f"\n耦合原型形状: {coupling.shape}")
    print(f"每个关节的耦合特征范数:")
    for i in range(7):
        norm = coupling[i].norm().item()
        print(f"  J{i}: {norm:.4f}")

    # 关联矩阵（通过相似度可视化）
    similarity = torch.mm(coupling, coupling.T)  # [7, 7]
    similarity = similarity / similarity.diag().unsqueeze(1) / similarity.diag().unsqueeze(0)

    print(f"\n关节间关联度（归一化相似度）:")
    joint_names = ['J0(shd_pitch)', 'J1(shd_roll)', 'J2(shd_yaw)', 'J3(elbow)', 'J4(forearm)', 'J5(wrist_0)', 'J6(wrist_1)']
    print("          " + "  ".join([f"{i:3d}" for i in range(7)]))
    for i in range(7):
        print(f"{joint_names[i][:12]:12s}", end=" ")
        for j in range(7):
            if i == j:
                print(f" 1.00", end=" ")
            else:
                print(f"{similarity[i,j].item():.2f}", end=" ")
        print()

    print("\n" + "=" * 70)
    print("总结:")
    print("  1. 推理时只需输入目标位姿 (位置 + 四元数)")
    print("  2. 模型内部的 coupling_prototype 记住了关节间的耦合关系")
    print("  3. 无需历史角度，模型参数量更少，推理更快")
    print("=" * 70)


if __name__ == "__main__":
    inference_demo()
