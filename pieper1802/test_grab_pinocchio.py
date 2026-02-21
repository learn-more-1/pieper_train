"""
使用Pinocchio精确FK测试GRAB数据集

对比真实角度和预测角度的位置误差
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from explicit_coupling_ik import ExplicitCouplingIK
from pinocchio_fk import PinocchioFK


def load_model(model_path, device='cuda'):
    """加载模型"""
    model = ExplicitCouplingIK(
        num_joints=7, num_frames=10, hidden_dim=256, use_temporal=False
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def test_grab_with_pinocchio(model_path, grab_data_path, num_samples=10000):
    """
    使用Pinocchio精确FK测试
    
    分别计算：
    1. 真实角度 -> 真实肘部/腕部位置
    2. 预测角度 -> 预测肘部/腕部位置
    3. 对比位置误差
    """
    print("=" * 80)
    print("使用Pinocchio精确FK测试GRAB数据集")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    print(f"\n加载模型...")
    model = load_model(model_path, device)
    print("✓ 模型加载成功")
    
    # 加载Pinocchio FK
    print("\n初始化Pinocchio FK...")
    try:
        fk = PinocchioFK()
    except Exception as e:
        print(f"❌ Pinocchio初始化失败: {e}")
        print("请确保已安装: pip install pinocchio")
        return
    
    # 加载GRAB数据
    print(f"\n加载GRAB数据...")
    data = np.load(grab_data_path)
    y = data['y']
    
    # 随机采样
    np.random.seed(42)
    indices = np.random.choice(len(y), min(num_samples, len(y)), replace=False)
    y_test = y[indices]
    print(f"✓ 样本数: {len(y_test)}")
    
    # 准备数据
    y_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    human_pose = y_tensor[:, :7]
    human_pos = human_pose[:, :3]
    human_ori = human_pose[:, 3:7]
    target_angles = y_tensor[:, 7:]
    
    # 分批处理
    batch_size = 256
    n_samples = len(y_tensor)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"\n开始测试 ({n_batches} batches)...")
    print("=" * 80)
    
    all_target_elbow = []
    all_target_wrist = []
    all_pred_elbow = []
    all_pred_wrist = []
    all_ik_errors = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch_human_pos = human_pos[start_idx:end_idx]
        batch_human_ori = human_ori[start_idx:end_idx]
        batch_target_angles = target_angles[start_idx:end_idx]
        
        # 模型预测
        with torch.no_grad():
            pred_angles, _ = model(batch_human_pos, batch_human_ori)
        
        # 计算IK误差
        ik_error = (pred_angles - batch_target_angles).abs()
        all_ik_errors.append(ik_error.cpu().numpy())
        
        # Pinocchio FK - 真实角度
        target_elbow, target_wrist = fk.compute_positions(batch_target_angles)
        all_target_elbow.append(target_elbow.cpu().numpy())
        all_target_wrist.append(target_wrist.cpu().numpy())
        
        # Pinocchio FK - 预测角度
        pred_elbow, pred_wrist = fk.compute_positions(pred_angles)
        all_pred_elbow.append(pred_elbow.cpu().numpy())
        all_pred_wrist.append(pred_wrist.cpu().numpy())
        
        if (i + 1) % 5 == 0 or i == n_batches - 1:
            print(f"  进度: {end_idx}/{n_samples}")
    
    # 合并结果
    all_target_elbow = np.vstack(all_target_elbow)
    all_target_wrist = np.vstack(all_target_wrist)
    all_pred_elbow = np.vstack(all_pred_elbow)
    all_pred_wrist = np.vstack(all_pred_wrist)
    all_ik_errors = np.vstack(all_ik_errors)
    
    # 计算误差
    print("\n" + "=" * 80)
    print("测试结果（Pinocchio精确FK）")
    print("=" * 80)
    
    # 1. 关节角度误差
    ik_rmse = np.sqrt(np.mean(all_ik_errors ** 2))
    ik_mae = np.mean(all_ik_errors)
    
    print(f"\n【关节角度误差】")
    print(f"  RMSE: {ik_rmse:.6f} rad = {np.rad2deg(ik_rmse):.2f}°")
    print(f"  MAE:  {ik_mae:.6f} rad = {np.rad2deg(ik_mae):.2f}°")
    
    # 各关节误差
    joint_names = ['Shoulder0', 'Shoulder1', 'Shoulder2', 'Elbow', 'Forearm', 'Wrist0', 'Wrist1']
    print(f"\n  各关节RMSE:")
    for i, name in enumerate(joint_names):
        rmse = np.sqrt(np.mean(all_ik_errors[:, i] ** 2))
        print(f"    {name:12s}: {np.rad2deg(rmse):6.2f}°")
    
    # 2. 位置误差
    elbow_error = np.linalg.norm(all_pred_elbow - all_target_elbow, axis=1)
    wrist_error = np.linalg.norm(all_pred_wrist - all_target_wrist, axis=1)
    
    print(f"\n【位置误差（Pinocchio精确FK）】")
    print(f"  肘部位置:")
    print(f"    平均: {np.mean(elbow_error)*1000:.2f} mm")
    print(f"    中位: {np.median(elbow_error)*1000:.2f} mm")
    print(f"    95%:  {np.percentile(elbow_error*1000, 95):.2f} mm")
    print(f"    最大: {np.max(elbow_error)*1000:.2f} mm")
    
    print(f"\n  腕部位置:")
    print(f"    平均: {np.mean(wrist_error)*1000:.2f} mm")
    print(f"    中位: {np.median(wrist_error)*1000:.2f} mm")
    print(f"    95%:  {np.percentile(wrist_error*1000, 95):.2f} mm")
    print(f"    最大: {np.max(wrist_error)*1000:.2f} mm")
    
    # 各维度误差
    print(f"\n  腕部位置各维度误差:")
    for i, dim in enumerate(['X', 'Y', 'Z']):
        dim_error = np.abs(all_pred_wrist[:, i] - all_target_wrist[:, i])
        print(f"    {dim}: {np.mean(dim_error)*1000:.2f} mm")
    
    # R²
    ss_res = np.sum((all_target_wrist - all_pred_wrist) ** 2)
    ss_tot = np.sum((all_target_wrist - all_target_wrist.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    print(f"\n  R²: {r2:.4f}")
    
    # 结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    
    mean_wrist_error = np.mean(wrist_error) * 1000
    
    if mean_wrist_error < 10:
        rating = "✓✓ 泛化性优秀 (< 10mm)"
    elif mean_wrist_error < 20:
        rating = "✓ 泛化性良好 (< 20mm)"
    elif mean_wrist_error < 50:
        rating = "○ 泛化性一般 (< 50mm)"
    else:
        rating = "✗ 泛化性较差 (> 50mm)"
    
    print(f"\n腕部位置误差: {mean_wrist_error:.2f} mm")
    print(f"评级: {rating}")
    print("=" * 80)
    
    return {
        'wrist_error_mean': mean_wrist_error,
        'elbow_error_mean': np.mean(elbow_error) * 1000,
        'ik_rmse': np.rad2deg(ik_rmse),
        'r2': r2
    }


if __name__ == "__main__":
    model_path = "/home/bonuli/Pieper/pieper1802/explicit_coupling_ik_optimized.pth"
    grab_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(grab_path):
        print(f"❌ 数据不存在: {grab_path}")
        sys.exit(1)
    
    test_grab_with_pinocchio(model_path, grab_path)
