"""
验证模型行为：历史 vs 目标位姿，谁主导输出？

测试场景：
1. 固定历史，改变目标位姿 -> 预测应该变化
2. 改变历史，固定目标位姿 -> 预测可能变化
3. 如果场景1变化很小，说明目标位姿影响弱，需要修复
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from causal_ik_model_pieper2 import PieperCausalIK


def test_model_sensitivity(model_path):
    """
    测试模型对输入的敏感度
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = PieperCausalIK(num_joints=7, num_frames=10, hidden_dim=256, num_layers=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("=" * 70)
    print("模型输入敏感度测试")
    print("=" * 70)
    
    # 测试1：固定历史，改变目标位姿
    print("\n测试1：固定历史（零姿态），改变目标位置")
    print("-" * 50)
    
    fixed_history = torch.zeros(1, 10, 7, device=device)
    
    positions = [
        torch.tensor([[0.4, 0.1, 0.3]], device=device),
        torch.tensor([[0.5, 0.2, 0.4]], device=device),
        torch.tensor([[0.6, 0.3, 0.5]], device=device),
    ]
    
    predictions_1 = []
    with torch.no_grad():
        for pos in positions:
            pred, _ = model(fixed_history, pos, None)
            predictions_1.append(pred[0].cpu().numpy())
            print(f"  目标位置 {pos[0].cpu().numpy().round(2)} -> 预测 {pred[0, :3].cpu().numpy().round(4)}...")
    
    # 计算变化量
    diff_1 = np.linalg.norm(predictions_1[1] - predictions_1[0])
    diff_2 = np.linalg.norm(predictions_1[2] - predictions_1[1])
    print(f"\n  目标变化引起角度变化量: {diff_1:.4f}, {diff_2:.4f}")
    
    # 测试2：固定目标位姿，改变历史
    print("\n测试2：固定目标位置，改变历史")
    print("-" * 50)
    
    fixed_pos = torch.tensor([[0.5, 0.2, 0.4]], device=device)
    
    histories = [
        torch.zeros(1, 10, 7, device=device),
        torch.ones(1, 10, 7, device=device) * 0.1,
        torch.ones(1, 10, 7, device=device) * 0.2,
    ]
    
    predictions_2 = []
    with torch.no_grad():
        for hist in histories:
            pred, _ = model(hist, fixed_pos, None)
            predictions_2.append(pred[0].cpu().numpy())
            print(f"  历史均值 {hist[0, :, 0].mean().item():.2f}... -> 预测 {pred[0, :3].cpu().numpy().round(4)}...")
    
    diff_h1 = np.linalg.norm(predictions_2[1] - predictions_2[0])
    diff_h2 = np.linalg.norm(predictions_2[2] - predictions_2[1])
    print(f"\n  历史变化引起角度变化量: {diff_h1:.4f}, {diff_h2:.4f}")
    
    # 结论
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    if diff_1 > 0.5:
        print("✓ 目标位姿对预测有强影响，模型工作正常")
        print("  如果机器人不动，检查目标位姿是否真的在变化")
    else:
        print("⚠ 目标位姿对预测影响很弱！")
        print("  模型可能过度依赖历史，需要修改")
    
    if diff_h1 > diff_1:
        print("⚠ 注意：历史变化的影响 > 目标位姿变化的影响")
        print("  这会导致目标位姿变化时输出变化不明显")


def test_real_scenario(model_path):
    """
    测试真实场景：人手跟踪
    
    模拟人手移动时的预测行为
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PieperCausalIK(num_joints=7, num_frames=10, hidden_dim=256, num_layers=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\n" + "=" * 70)
    print("真实场景模拟：人手跟踪")
    print("=" * 70)
    
    # 场景：人手从A点移动到B点
    print("\n场景：人手连续移动")
    print("-" * 50)
    
    # 起点和终点
    start_pos = np.array([0.4, 0.1, 0.3])
    end_pos = np.array([0.6, 0.3, 0.5])
    
    # 生成轨迹
    trajectory = []
    for t in np.linspace(0, 1, 10):
        pos = start_pos * (1 - t) + end_pos * t
        trajectory.append(pos)
    
    # 方式1：固定历史（纯位姿驱动）
    print("\n方式1：固定历史（纯位姿驱动）")
    fixed_history = torch.zeros(1, 10, 7, device=device)
    
    predictions_fixed = []
    with torch.no_grad():
        for pos in trajectory:
            pos_tensor = torch.tensor(pos, dtype=torch.float32).unsqueeze(0).to(device)
            pred, _ = model(fixed_history, pos_tensor, None)
            predictions_fixed.append(pred[0].cpu().numpy())
    
    for i in [0, 4, 9]:
        print(f"  目标{i+1} {trajectory[i].round(2)} -> 角度 {predictions_fixed[i][:3].round(3)}...")
    
    total_change_fixed = np.linalg.norm(predictions_fixed[-1] - predictions_fixed[0])
    print(f"\n  总变化量: {total_change_fixed:.4f}")
    
    # 方式2：自回归（历史更新）
    print("\n方式2：自回归（历史更新）")
    
    history = torch.zeros(10, 7, device=device)
    predictions_update = []
    
    with torch.no_grad():
        for pos in trajectory:
            pos_tensor = torch.tensor(pos, dtype=torch.float32).unsqueeze(0).to(device)
            pred, _ = model(history.unsqueeze(0), pos_tensor, None)
            predictions_update.append(pred[0].cpu().numpy())
            # 更新历史
            history = torch.cat([history[1:], pred], dim=0)
    
    for i in [0, 4, 9]:
        print(f"  目标{i+1} {trajectory[i].round(2)} -> 角度 {predictions_update[i][:3].round(3)}...")
    
    total_change_update = np.linalg.norm(predictions_update[-1] - predictions_update[0])
    print(f"\n  总变化量: {total_change_update:.4f}")
    
    # 对比
    print("\n" + "-" * 50)
    if abs(total_change_fixed - total_change_update) < 0.1:
        print("两种方式的预测变化相近，模型工作正常")
    elif total_change_fixed > total_change_update:
        print("纯位姿方式变化更大，自回归可能过于平滑")
    else:
        print("自回归变化更大")


def debug_why_not_moving(model_path):
    """
    调试：为什么机器人不动
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PieperCausalIK(num_joints=7, num_frames=10, hidden_dim=256, num_layers=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\n" + "=" * 70)
    print("调试：为什么机器人不动？")
    print("=" * 70)
    
    # 场景：目标位姿不变，历史不变
    print("\n场景1：目标位姿不变 + 历史不变 -> 预测应该不变（正确）")
    history = torch.zeros(1, 10, 7, device=device)
    pos = torch.tensor([[0.5, 0.2, 0.4]], device=device)
    
    with torch.no_grad():
        for i in range(3):
            pred, _ = model(history, pos, None)
            print(f"  重复{i+1}: {pred[0, :3].cpu().numpy().round(4)}...")
            # 更新历史（但预测不变，所以历史也不变）
            history = torch.cat([history[:, 1:, :], pred.unsqueeze(1)], dim=1)
    
    print("\n  这是正常的：目标不变，预测不变")
    
    # 场景：目标位姿变化，但历史固定
    print("\n场景2：目标位姿变化 + 历史固定 -> 预测应该变化")
    history = torch.zeros(1, 10, 7, device=device)
    
    positions = [
        torch.tensor([[0.4, 0.1, 0.3]], device=device),
        torch.tensor([[0.5, 0.2, 0.4]], device=device),
        torch.tensor([[0.6, 0.3, 0.5]], device=device),
    ]
    
    with torch.no_grad():
        for i, pos in enumerate(positions):
            pred, _ = model(history, pos, None)
            print(f"  目标{i+1} {pos[0].cpu().numpy().round(2)} -> {pred[0, :3].cpu().numpy().round(4)}...")
            # 历史固定不变
    
    print("\n  如果目标变化但预测几乎不变，说明模型过度依赖历史")


if __name__ == "__main__":
    model_path = "/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth"
    
    # 运行测试
    test_model_sensitivity(model_path)
    test_real_scenario(model_path)
    debug_why_not_moving(model_path)
    
    print("\n" + "=" * 70)
    print("诊断建议")
    print("=" * 70)
    print("""
    根据测试结果：
    
    情况A：目标变化引起预测变化（正常）
        -> 检查你的目标位姿是否真的在变化
        -> 检查位姿数据传输是否正常
    
    情况B：目标变化但预测几乎不变（模型问题）
        -> 使用 inference_pose_pure.py 的纯位姿方式
        -> 或考虑训练一个弱化历史依赖的模型
    
    常见错误：
    1. 每次传入相同的目标位姿（数据问题）
    2. 目标位姿变化太小（传感器精度问题）
    3. 历史更新方式错误（代码逻辑问题）
    """)
