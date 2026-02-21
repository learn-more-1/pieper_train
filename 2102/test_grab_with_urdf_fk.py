"""
使用精确URDF-FK测试GRAB数据集
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper2102')

from explicit_coupling_ik import ExplicitCouplingIK
from fk_with_elbow_urdf import FKWithElbowURDF


def load_grab_data(data_path, num_samples=10000):
    """加载GRAB数据"""
    print(f"\n加载: {os.path.basename(data_path)}")
    data = np.load(data_path)
    y = data['y']
    
    if len(y) > num_samples:
        indices = np.random.choice(len(y), num_samples, replace=False)
        y = y[indices]
    
    print(f"  样本数: {len(y)}")
    return y


def test_model(model_path, grab_data_path, device='cuda'):
    """测试模型"""
    print("=" * 80)
    print("使用精确URDF-FK测试GRAB数据集")
    print("=" * 80)
    
    # 加载模型
    print(f"\n加载模型: {os.path.basename(model_path)}")
    model = ExplicitCouplingIK(
        num_joints=7, num_frames=10, hidden_dim=256, use_temporal=False
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ 模型加载成功")
    
    # 加载精确的FK
    print("\n加载精确URDF-FK...")
    fk = FKWithElbowURDF()
    print("✓ 使用精确的连杆长度（从URDF提取）")
    
    # 加载GRAB数据
    y_test = load_grab_data(grab_data_path, num_samples=10000)
    
    # 测试
    print("\n" + "=" * 80)
    print("开始测试...")
    print("=" * 80)
    
    y_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    human_pose = y_tensor[:, :7]
    human_pos = human_pose[:, :3]
    human_ori = human_pose[:, 3:7]
    target_angles = y_tensor[:, 7:]
    
    batch_size = 512
    n_samples = len(y_tensor)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_ik_errors = []
    all_elbow_errors = []
    all_end_errors = []
    all_r2_preds = []
    all_r2_targets = []
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_human_pos = human_pos[start_idx:end_idx]
            batch_human_ori = human_ori[start_idx:end_idx]
            batch_target_angles = target_angles[start_idx:end_idx]
            
            # 预测
            pred_angles, _ = model(batch_human_pos, batch_human_ori)
            
            # IK误差
            ik_error = (pred_angles - batch_target_angles).abs()
            all_ik_errors.append(ik_error.cpu().numpy())
            
            # 收集R2数据
            all_r2_preds.append(pred_angles.cpu())
            all_r2_targets.append(batch_target_angles.cpu())
            
            # 计算位置（使用精确FK）
            target_elbow, target_end = fk.compute_positions(batch_target_angles)
            pred_elbow, pred_end = fk.compute_positions(pred_angles)
            
            # 位置误差
            elbow_error = (pred_elbow - target_elbow).norm(dim=1)
            end_error = (pred_end - target_end).norm(dim=1)
            
            all_elbow_errors.append(elbow_error.cpu().numpy())
            all_end_errors.append(end_error.cpu().numpy())
            
            if (i + 1) % 5 == 0:
                print(f"  进度: {min(end_idx, n_samples)}/{n_samples}")
    
    # 合并结果
    all_ik_errors = np.concatenate(all_ik_errors, axis=0)
    all_elbow_errors = np.concatenate(all_elbow_errors)
    all_end_errors = np.concatenate(all_end_errors)
    all_r2_preds = torch.cat(all_r2_preds, dim=0)
    all_r2_targets = torch.cat(all_r2_targets, dim=0)
    
    # 计算指标
    print("\n" + "=" * 80)
    print("测试结果（使用精确URDF-FK）")
    print("=" * 80)
    
    # IK指标
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
    
    # R²
    ss_res = ((all_r2_targets - all_r2_preds) ** 2).sum()
    ss_tot = ((all_r2_targets - all_r2_targets.mean(dim=0)) ** 2).sum()
    r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
    print(f"\n  R²: {r2:.4f}")
    
    # 位置误差（精确FK）
    print(f"\n【位置误差（精确URDF-FK）】")
    print(f"  肘部位置:")
    print(f"    平均: {np.mean(all_elbow_errors)*1000:.2f} mm")
    print(f"    中位: {np.median(all_elbow_errors)*1000:.2f} mm")
    print(f"    最大: {np.max(all_elbow_errors)*1000:.2f} mm")
    
    print(f"\n  末端位置:")
    print(f"    平均: {np.mean(all_end_errors)*1000:.2f} mm")
    print(f"    中位: {np.median(all_end_errors)*1000:.2f} mm")
    print(f"    最大: {np.max(all_end_errors)*1000:.2f} mm")
    
    # 误差分布
    print(f"\n  末端位置误差分布:")
    for p in [50, 75, 90, 95]:
        val = np.percentile(all_end_errors * 1000, p)
        print(f"    {p}%: {val:.2f} mm")
    
    # 结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    
    mean_end_error = np.mean(all_end_errors) * 1000
    
    if mean_end_error < 10:
        rating = "✓✓ 泛化性优秀"
    elif mean_end_error < 20:
        rating = "✓ 泛化性良好"
    elif mean_end_error < 50:
        rating = "○ 泛化性一般"
    else:
        rating = "✗ 泛化性较差"
    
    print(f"\n末端位置误差: {mean_end_error:.2f} mm")
    print(f"评级: {rating}")
    print("=" * 80)


if __name__ == "__main__":
    model_path = "/home/bonuli/Pieper/pieper1802/elbow_constraint_ik.pth"
    grab_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(grab_path):
        print(f"❌ 数据不存在: {grab_path}")
        sys.exit(1)
    
    test_model(model_path, grab_path)
