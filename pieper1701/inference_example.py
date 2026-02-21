"""
推理示例：处理无历史角度的情况

展示如何在推理时使用修改后的模型
"""

import torch
import sys
sys.path.insert(0, '/home/wsy/Desktop/casual/pieper1101')

from causal_ik_model_pieper2 import PieperCausalIK


def inference_with_history(model, end_position, end_orientation, history_frames=None):
    """
    IK推理

    Args:
        model: 训练好的模型
        end_position: [batch, 3] 目标末端位置
        end_orientation: [batch, 4] 目标末端姿态(四元数)，可以为None
        history_frames: [batch, 10, 7] 历史关节角度，可以为None

    Returns:
        pred_angles: [batch, 7] 预测的关节角度
    """
    model.eval()
    with torch.no_grad():
        pred_angles, info = model(history_frames, end_position, end_orientation)
    return pred_angles, info


def main():
    print("=" * 70)
    print("推理示例：处理无历史角度的情况")
    print("=" * 70)

    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PieperCausalIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        num_layers=2
    ).to(device)

    # 加载训练好的权重
    checkpoint_path = "/home/wsy/Desktop/casual/pieper_NN/pieper_causal_ik_1101.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ 成功加载模型权重")
    except Exception as e:
        print(f"⚠ 加载权重失败: {e}，使用随机初始化")

    model.eval()

    # 示例1: 有历史角度（与训练一致）
    print("\n[示例1] 有历史角度的推理:")
    batch_size = 4
    history_frames = torch.randn(batch_size, 10, 7).to(device)
    end_position = torch.randn(batch_size, 3).to(device)
    end_orientation = torch.randn(batch_size, 4).to(device)

    pred_angles, info = inference_with_history(
        model, end_position, end_orientation, history_frames
    )
    print(f"  输入历史: {history_frames.shape}")
    print(f"  预测角度: {pred_angles.shape}")

    # 示例2: 无历史角度（使用模型学习的默认历史）
    print("\n[示例2] 无历史角度的推理（使用默认历史）:")
    pred_angles_default, info = inference_with_history(
        model, end_position, end_orientation, history_frames=None
    )
    print(f"  使用默认历史嵌入")
    print(f"  预测角度: {pred_angles_default.shape}")

    # 示例3: 使用当前角度重复填充历史
    print("\n[示例3] 用当前角度重复填充历史:")
    # 假设我们知道当前机器人角度
    current_angles = torch.randn(batch_size, 7).to(device)
    # 重复10次作为历史
    history_filled = current_angles.unsqueeze(1).repeat(1, 10, 1)

    pred_angles_filled, info = inference_with_history(
        model, end_position, end_orientation, history_frames=history_filled
    )
    print(f"  当前角度重复填充历史")
    print(f"  预测角度: {pred_angles_filled.shape}")

    # 示例4: 实际使用场景
    print("\n[示例4] 实际使用场景:")
    print("""
    # 伪代码：实际机器人控制
    def robot_control_step(model, target_pose, current_angles=None):
        '''
        model: IK模型
        target_pose: 目标末端位姿 (position, orientation)
        current_angles: 当前关节角度 [7]，如果不知道则为None
        '''
        end_position = torch.from_numpy(target_pose['position']).unsqueeze(0).to(device)
        end_orientation = torch.from_numpy(target_pose['orientation']).unsqueeze(0).to(device)

        if current_angles is not None:
            # 情况A: 知道当前角度，用其填充历史
            current_angles = torch.from_numpy(current_angles).unsqueeze(0).to(device)
            history = current_angles.unsqueeze(1).repeat(1, 10, 1)
        else:
            # 情况B: 不知道当前角度，使用模型学习的默认历史
            history = None

        pred_angles, _ = model(history, end_position, end_orientation)
        return pred_angles.cpu().numpy()[0]
    """)

    print("\n" + "=" * 70)
    print("建议:")
    print("  1. 如果能获取当前关节角度 → 用当前角度重复填充历史")
    print("  2. 如果完全无法获取角度 → 使用 None（模型自动使用默认历史）")
    print("  3. 对于最优效果，可以用少量无历史数据微调模型")
    print("=" * 70)


if __name__ == "__main__":
    main()
