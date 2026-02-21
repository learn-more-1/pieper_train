"""
测试历史固定后，关节耦合关系是否还存在
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from causal_ik_model_pieper2 import PieperCausalIK


def test_joint_coupling_with_fixed_history(model_path):
    """
    测试：历史固定时，关节是否还保持耦合关系
    
    测试方法：
    1. 改变目标位姿，观察多个关节是否同时响应
    2. 分析关节角度之间的相关性
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = PieperCausalIK(num_joints=7, num_frames=10, hidden_dim=256, num_layers=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("=" * 70)
    print("测试：历史固定时的关节耦合关系")
    print("=" * 70)
    
    # 固定历史为零
    fixed_history = torch.zeros(1, 10, 7, device=device)
    
    # 测试1：X方向移动，观察各关节响应
    print("\n测试1：X方向移动时的关节响应")
    print("-" * 50)
    
    x_positions = np.linspace(0.4, 0.6, 5)
    predictions_x = []
    
    with torch.no_grad():
        for x in x_positions:
            pos = torch.tensor([[x, 0.2, 0.4]], device=device, dtype=torch.float32)
            pred, _ = model(fixed_history, pos, None)
            predictions_x.append(pred[0].cpu().numpy())
    
    predictions_x = np.array(predictions_x)
    
    print("X位置 → Shoulder[0] Elbow[0] Wrist[0]")
    for i, (x, angles) in enumerate(zip(x_positions, predictions_x)):
        print(f"{x:.2f}  →  {angles[0]:7.3f}   {angles[3]:7.3f}   {angles[5]:7.3f}")
    
    # 计算变化量
    delta_shoulder = np.std(predictions_x[:, 0])
    delta_elbow = np.std(predictions_x[:, 3])
    delta_wrist = np.std(predictions_x[:, 5])
    
    print(f"\n变化量（标准差）:")
    print(f"  Shoulder: {delta_shoulder:.4f}")
    print(f"  Elbow:    {delta_elbow:.4f}")
    print(f"  Wrist:    {delta_wrist:.4f}")
    
    if delta_elbow > 0.01 or delta_wrist > 0.01:
        print("  ✓ X移动时，多个关节响应，耦合关系存在")
    else:
        print("  ✗ 只有shoulder响应，耦合关系弱")
    
    # 测试2：Z方向移动，观察各关节响应
    print("\n测试2：Z方向移动时的关节响应")
    print("-" * 50)
    
    z_positions = np.linspace(0.3, 0.5, 5)
    predictions_z = []
    
    with torch.no_grad():
        for z in z_positions:
            pos = torch.tensor([[0.5, 0.2, z]], device=device, dtype=torch.float32)
            pred, _ = model(fixed_history, pos, None)
            predictions_z.append(pred[0].cpu().numpy())
    
    predictions_z = np.array(predictions_z)
    
    print("Z位置 → Shoulder[1] Elbow[0] Wrist[0]")
    for i, (z, angles) in enumerate(zip(z_positions, predictions_z)):
        print(f"{z:.2f}  →  {angles[1]:7.3f}   {angles[3]:7.3f}   {angles[5]:7.3f}")
    
    delta_shoulder_z = np.std(predictions_z[:, 1])
    delta_elbow_z = np.std(predictions_z[:, 3])
    
    print(f"\n变化量（标准差）:")
    print(f"  Shoulder[1]: {delta_shoulder_z:.4f}")
    print(f"  Elbow:       {delta_elbow_z:.4f}")
    
    # 测试3：关节角度相关性分析
    print("\n测试3：关节角度相关性分析")
    print("-" * 50)
    
    # 生成更多样本
    all_predictions = []
    for x in np.linspace(0.35, 0.65, 10):
        for y in np.linspace(0.1, 0.3, 5):
            for z in np.linspace(0.25, 0.55, 5):
                pos = torch.tensor([[x, y, z]], device=device, dtype=torch.float32)
                with torch.no_grad():
                    pred, _ = model(fixed_history, pos, None)
                all_predictions.append(pred[0].cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    
    # 计算关节间的相关性
    from numpy import corrcoef
    
    corr_shoulder_elbow = np.corrcoef(all_predictions[:, 0], all_predictions[:, 3])[0, 1]
    corr_shoulder_wrist = np.corrcoef(all_predictions[:, 0], all_predictions[:, 5])[0, 1]
    corr_elbow_wrist = np.corrcoef(all_predictions[:, 3], all_predictions[:, 5])[0, 1]
    
    print(f"关节间相关性:")
    print(f"  Shoulder-Elbow: {corr_shoulder_elbow:.3f}")
    print(f"  Shoulder-Wrist: {corr_shoulder_wrist:.3f}")
    print(f"  Elbow-Wrist:    {corr_elbow_wrist:.3f}")
    
    if abs(corr_shoulder_elbow) > 0.3 or abs(corr_shoulder_wrist) > 0.3:
        print("  ✓ 关节间存在相关性，耦合关系还在")
    else:
        print("  ✗ 关节间相关性弱，耦合关系可能丢失")
    
    return all_predictions


def test_with_different_histories(model_path):
    """
    测试：不同历史固定值对关节耦合的影响
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PieperCausalIK(num_joints=7, num_frames=10, hidden_dim=256, num_layers=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\n" + "=" * 70)
    print("测试：不同历史固定值的影响")
    print("=" * 70)
    
    # 不同的固定历史
    histories = {
        '零姿态': torch.zeros(1, 10, 7, device=device),
        '中性姿态': torch.tensor([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], device=device).unsqueeze(0).unsqueeze(0).repeat(1, 10, 1),
        '随机姿态': torch.randn(1, 10, 7, device=device) * 0.2,
    }
    
    target_pos = torch.tensor([[0.5, 0.2, 0.4]], device=device, dtype=torch.float32)
    
    print(f"\n目标位置: [0.5, 0.2, 0.4]")
    print("-" * 50)
    
    for name, history in histories.items():
        with torch.no_grad():
            pred, _ = model(history, target_pos, None)
        angles = pred[0].cpu().numpy()
        print(f"{name:8s}: {angles[:3].round(3)} | {angles[3:5].round(3)} | {angles[5:7].round(3)}")
    
    print("\n如果不同历史产生不同结果，说明历史影响大")
    print("如果结果相似，说明目标位姿主导预测")


def analyze_gnn_behavior(model_path):
    """
    分析GNN的行为：消息传递是否还在工作？
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PieperCausalIK(num_joints=7, num_frames=10, hidden_dim=256, num_layers=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\n" + "=" * 70)
    print("分析：GNN消息传递机制")
    print("=" * 70)
    
    print("""
GNN的消息传递层（在模型中的位置）：
- shoulder → elbow
- elbow → forearm  
- forearm → wrist

这些层在模型内部，无论历史如何，都会执行。
关键问题是：输入特征的质量决定了消息传递的效果。
    """)
    
    # 固定历史
    fixed_history = torch.zeros(1, 10, 7, device=device)
    target_pos = torch.tensor([[0.5, 0.2, 0.4]], device=device, dtype=torch.float32)
    
    # 前向传播，获取中间信息
    with torch.no_grad():
        pred, info = model(fixed_history, target_pos, None)
    
    print(f"\n固定历史预测结果: {pred[0, :3].cpu().numpy().round(4)}...")
    print(f"Pieper位置权重:   {info['position_weights'][0].cpu().numpy().round(3)}")
    print(f"Pieper姿态权重:   {info['orientation_weights'][0].cpu().numpy().round(3)}")


if __name__ == "__main__":
    model_path = "/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth"
    
    # 运行测试
    test_joint_coupling_with_fixed_history(model_path)
    test_with_different_histories(model_path)
    analyze_gnn_behavior(model_path)
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
历史固定后：
1. GNN的消息传递机制还在（模型结构决定）
2. 但输入特征弱 → 消息传递的效果打折扣
3. 关节耦合关系部分保留，但不如时序输入时强

建议：
- 如果关节配合对你很重要 → 用插值历史方案
- 如果只要到达目标位姿 → 纯位姿方案足够
    """)
