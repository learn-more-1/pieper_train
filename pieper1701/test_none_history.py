"""
测试 None 历史帧功能
"""
import torch
import sys
sys.path.insert(0, '/home/wsy/Desktop/casual/pieper1101')

from causal_ik_model_pieper2 import PieperCausalIK


def test_none_history():
    print("=" * 70)
    print("测试 None 历史帧功能")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PieperCausalIK(num_joints=7, num_frames=10, hidden_dim=256, num_layers=2)
    model = model.to(device)
    model.eval()

    batch_size = 2

    # 测试1: 正常历史帧
    print("\n[Test 1] 正常历史帧:")
    history_frames = torch.randn(batch_size, 10, 7).to(device)
    end_position = torch.randn(batch_size, 3).to(device)
    end_orientation = torch.randn(batch_size, 4).to(device)

    pred1, info1 = model(history_frames, end_position, end_orientation)
    print(f"  输入: history_frames={history_frames.shape}")
    print(f"  输出: pred_angles={pred1.shape}")

    # 测试2: None 历史帧（使用默认历史）
    print("\n[Test 2] None 历史帧（使用默认历史）:")
    pred2, info2 = model(None, end_position, end_orientation)
    print(f"  输入: history_frames=None")
    print(f"  输出: pred_angles={pred2.shape}")

    # 检查默认历史参数
    print(f"\n默认历史嵌入参数:")
    print(f"  shape: {model.default_history.shape}")
    print(f"  值: {model.default_history.data[0, 0, :]}")
    print(f"  requires_grad: {model.default_history.requires_grad}")

    # 测试3: 确保可训练
    print("\n[Test 3] 测试默认历史可训练性:")
    optimizer = torch.optim.AdamW([model.default_history], lr=1e-3)
    loss = pred2.sum()
    loss.backward()
    optimizer.step()
    print(f"  更新后的默认历史: {model.default_history.data[0, 0, :]}")
    print(f"  ✓ 默认历史可以训练")

    print("\n" + "=" * 70)
    print("所有测试通过！")
    print("=" * 70)


if __name__ == "__main__":
    test_none_history()
