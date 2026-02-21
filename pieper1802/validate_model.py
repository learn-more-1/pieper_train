"""
模型验证脚本

验证训练后的模型是否满足纯位姿输入要求
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from causal_ik_model_pieper2 import PieperCausalIK
from pose_focused_ik_model import PoseFocusedIK


def validate_model(model_path, model_type='original'):
    """
    验证模型性能
    
    Args:
        model_path: 模型权重路径
        model_type: 'original' 或 'pose_focused'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print(f"模型验证: {model_path}")
    print(f"模型类型: {model_type}")
    print("=" * 70)
    
    # 加载模型
    if model_type == 'original':
        model = PieperCausalIK(num_joints=7, num_frames=10, hidden_dim=256, num_layers=2)
    else:
        model = PoseFocusedIK(num_joints=7, num_frames=10, hidden_dim=256, num_layers=2)
    
    model = model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 模型加载成功")
    
    # 测试1：纯位姿输入（零历史）
    print("\n测试1：纯位姿输入性能（零历史）")
    print("-" * 50)
    
    test_positions = [
        np.array([0.4, 0.1, 0.3]),
        np.array([0.5, 0.2, 0.4]),
        np.array([0.6, 0.3, 0.5]),
    ]
    
    zero_history = torch.zeros(1, 10, 7, device=device)
    
    pose_only_predictions = []
    with torch.no_grad():
        for pos in test_positions:
            pos_tensor = torch.tensor(pos, dtype=torch.float32).unsqueeze(0).to(device)
            if model_type == 'original':
                pred, _ = model(zero_history, pos_tensor, None)
            else:
                pred, _ = model(zero_history, pos_tensor, None, history_alpha=0.3)
            pose_only_predictions.append(pred[0].cpu().numpy())
            print(f"  目标 {pos.round(2)} -> 角度 {pred[0, :3].cpu().numpy().round(4)}...")
    
    # 计算变化量
    changes = []
    for i in range(len(pose_only_predictions) - 1):
        diff = np.linalg.norm(pose_only_predictions[i+1] - pose_only_predictions[i])
        changes.append(diff)
    
    avg_change = np.mean(changes)
    print(f"\n  平均角度变化: {avg_change:.4f} rad")
    
    if avg_change > 0.1:
        print("  ✅ 纯位姿输入：目标变化驱动输出变化")
        pose_only_ok = True
    else:
        print("  ❌ 纯位姿输入：输出变化不明显（过度依赖历史）")
        pose_only_ok = False
    
    # 测试2：关节耦合关系
    print("\n测试2：关节耦合关系")
    print("-" * 50)
    
    # 在X方向移动，观察多个关节是否响应
    x_positions = np.linspace(0.4, 0.6, 5)
    shoulder_changes = []
    elbow_changes = []
    
    with torch.no_grad():
        prev_pred = None
        for x in x_positions:
            pos = torch.tensor([[x, 0.2, 0.4]], dtype=torch.float32).to(device)
            if model_type == 'original':
                pred, _ = model(zero_history, pos, None)
            else:
                pred, _ = model(zero_history, pos, None, history_alpha=0.3)
            
            if prev_pred is not None:
                shoulder_changes.append(abs(pred[0, 0].item() - prev_pred[0, 0].item()))
                elbow_changes.append(abs(pred[0, 3].item() - prev_pred[0, 3].item()))
            
            prev_pred = pred[0].cpu().numpy()
    
    avg_shoulder_change = np.mean(shoulder_changes)
    avg_elbow_change = np.mean(elbow_changes)
    
    print(f"  X移动时 Shoulder 平均变化: {avg_shoulder_change:.4f} rad")
    print(f"  X移动时 Elbow 平均变化: {avg_elbow_change:.4f} rad")
    
    if avg_elbow_change > 0.01:
        print("  ✅ 关节耦合：多个关节同时响应")
        coupling_ok = True
    else:
        print("  ⚠️ 关节耦合：只有主关节响应")
        coupling_ok = False
    
    # 测试3：一致性
    print("\n测试3：输出一致性")
    print("-" * 50)
    
    test_pos = torch.tensor([[0.5, 0.2, 0.4]], dtype=torch.float32).to(device)
    predictions = []
    
    with torch.no_grad():
        for _ in range(5):
            if model_type == 'original':
                pred, _ = model(zero_history, test_pos, None)
            else:
                pred, _ = model(zero_history, test_pos, None, history_alpha=0.3)
            predictions.append(pred[0].cpu().numpy())
    
    pred_std = np.std(predictions, axis=0).mean()
    print(f"  多次预测标准差: {pred_std:.6f}")
    
    if pred_std < 0.001:
        print("  ✅ 一致性：输出稳定")
        consistency_ok = True
    else:
        print("  ❌ 一致性：输出不稳定（有随机性）")
        consistency_ok = False
    
    # 总体评估
    print("\n" + "=" * 70)
    print("总体评估")
    print("=" * 70)
    
    score = 0
    if pose_only_ok:
        print("✅ 纯位姿输入性能: 通过")
        score += 40
    else:
        print("❌ 纯位姿输入性能: 未通过")
    
    if coupling_ok:
        print("✅ 关节耦合关系: 保留")
        score += 30
    else:
        print("⚠️ 关节耦合关系: 较弱")
        score += 15
    
    if consistency_ok:
        print("✅ 输出一致性: 稳定")
        score += 30
    else:
        print("❌ 输出一致性: 不稳定")
    
    print(f"\n综合得分: {score}/100")
    
    if score >= 80:
        print("🎉 模型优秀，可直接用于纯位姿推理")
    elif score >= 60:
        print("👌 模型良好，建议配合平滑后处理使用")
    else:
        print("⚠️ 模型需要继续训练或调整")
    
    return {
        'pose_only_ok': pose_only_ok,
        'coupling_ok': coupling_ok,
        'consistency_ok': consistency_ok,
        'score': score
    }


def compare_models(original_path, new_path):
    """
    对比原模型和新模型
    """
    print("\n" + "=" * 70)
    print("模型对比")
    print("=" * 70)
    
    print("\n>>> 原模型 <<<")
    orig_results = validate_model(original_path, model_type='original')
    
    print("\n>>> 新模型 <<<")
    new_results = validate_model(new_path, model_type='pose_focused')
    
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)
    
    print(f"\n原模型得分: {orig_results['score']}/100")
    print(f"新模型得分: {new_results['score']}/100")
    
    if new_results['score'] > orig_results['score']:
        print("✅ 新模型优于原模型")
    elif new_results['score'] == orig_results['score']:
        print("➡️ 新旧模型性能相当")
    else:
        print("⚠️ 新模型不如原模型，需要继续优化")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 2:
        model_path = sys.argv[1]
        model_type = sys.argv[2] if len(sys.argv) >= 3 else 'original'
        validate_model(model_path, model_type)
    else:
        print("用法:")
        print("  python validate_model.py <model_path> [model_type]")
        print("")
        print("示例:")
        print("  python validate_model.py pieper_causal_ik_1101.pth original")
        print("  python validate_model.py pose_focused_ik.pth pose_focused")
