"""
模型泛化性测试

在GRAB数据集上测试训练好的模型，评估泛化性能
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from explicit_coupling_ik import ExplicitCouplingIK
from gpu_fk_wrapper import SimpleGPUFK
from dataset_generalized import RobotArmDataset


def load_grab_data(data_path):
    """加载GRAB数据集"""
    print(f"加载GRAB数据集: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None, None
    
    data = np.load(data_path)
    
    print(f"数据键: {list(data.keys())}")
    
    # 获取数据
    if 'X' in data and 'y' in data:
        X = data['X']
        y = data['y']
    else:
        print(f"❌ 数据格式不正确")
        return None, None
    
    print(f"X形状: {X.shape}")
    print(f"y形状: {y.shape}")
    
    # 只取验证集部分（后20%）
    n_samples = len(X)
    split_idx = int(n_samples * 0.8)
    
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    
    print(f"验证集大小: {len(X_val)}")
    
    return X_val, y_val


def evaluate_model(model, X_data, y_data, device='cuda'):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        X_data: 输入数据 (历史帧)
        y_data: 目标数据 (位姿+角度)
    """
    model.eval()
    gpu_fk = SimpleGPUFK()
    mse_criterion = nn.MSELoss()
    
    # 准备数据
    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).to(device)
    
    # 提取人类位姿（输入）和机器人角度（目标）
    if y_tensor.shape[1] == 14:
        human_pose = y_tensor[:, :7]
        human_position = human_pose[:, :3]
        human_orientation = human_pose[:, 3:7]
        target_angles = y_tensor[:, 7:]
    else:
        print(f"❌ y数据维度不正确: {y_tensor.shape}")
        return None
    
    # 批量推理
    batch_size = 512
    n_samples = len(X_tensor)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_pred_angles = []
    all_target_angles = []
    
    print(f"开始推理 ({n_batches} batches)...")
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_human_pos = human_position[start_idx:end_idx]
            batch_human_ori = human_orientation[start_idx:end_idx]
            batch_target_angles = target_angles[start_idx:end_idx]
            
            # 模型推理
            pred_angles, _ = model(batch_human_pos, batch_human_ori)
            
            all_pred_angles.append(pred_angles.cpu())
            all_target_angles.append(batch_target_angles.cpu())
            
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{n_batches}")
    
    # 合并结果
    all_pred_angles = torch.cat(all_pred_angles, dim=0)
    all_target_angles = torch.cat(all_target_angles, dim=0)
    
    # 计算IK误差
    ik_errors = (all_pred_angles - all_target_angles).abs().numpy()
    ik_mse = mse_criterion(all_pred_angles, all_target_angles).item()
    ik_rmse = np.sqrt(ik_mse)
    
    # 计算FK误差
    print("计算FK误差...")
    all_pred_angles_gpu = all_pred_angles.cuda()
    all_target_angles_gpu = all_target_angles.cuda()
    
    pred_positions = gpu_fk.forward(all_pred_angles_gpu)
    target_positions = gpu_fk.forward(all_target_angles_gpu)
    
    fk_errors = (pred_positions - target_positions).abs().cpu().numpy()
    fk_mse = mse_criterion(pred_positions, target_positions).item()
    fk_rmse = np.sqrt(fk_mse)
    
    return {
        'ik_rmse': ik_rmse,
        'ik_mae': ik_errors.mean(),
        'ik_errors_per_joint': ik_errors.mean(axis=0),
        'fk_rmse': fk_rmse,
        'fk_mae': fk_errors.mean(),
        'fk_errors_per_dim': fk_errors.mean(axis=0),
        'n_samples': n_samples
    }


def compare_with_training_set(model_path, grab_data_path, device='cuda'):
    """
    对比训练集(ACCAD+CMU)和测试集(GRAB)的性能
    """
    print("=" * 70)
    print("模型泛化性测试")
    print("=" * 70)
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    model = ExplicitCouplingIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        use_temporal=False
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ 模型加载成功")
    
    # 加载GRAB数据
    X_grab, y_grab = load_grab_data(grab_data_path)
    
    if X_grab is None:
        print("❌ 无法加载GRAB数据")
        return
    
    # 评估GRAB数据集
    print("\n" + "=" * 70)
    print("在GRAB数据集上评估")
    print("=" * 70)
    
    results = evaluate_model(model, X_grab, y_grab, device)
    
    if results is None:
        return
    
    # 打印结果
    print(f"\n样本数: {results['n_samples']}")
    print(f"\n关节角度误差 (IK):")
    print(f"  RMSE: {results['ik_rmse']:.6f} rad ({results['ik_rmse'] * 180 / np.pi:.2f} deg)")
    print(f"  MAE:  {results['ik_mae']:.6f} rad ({results['ik_mae'] * 180 / np.pi:.2f} deg)")
    print(f"\n  各关节平均误差:")
    joint_names = ['Shoulder0', 'Shoulder1', 'Shoulder2', 'Elbow', 'Forearm', 'Wrist0', 'Wrist1']
    for i, (name, error) in enumerate(zip(joint_names, results['ik_errors_per_joint'])):
        print(f"    {name:12s}: {error:.6f} rad ({error * 180 / np.pi:.2f} deg)")
    
    print(f"\n位置误差 (FK):")
    print(f"  RMSE: {results['fk_rmse']:.6f} m ({results['fk_rmse'] * 100:.2f} cm)")
    print(f"  MAE:  {results['fk_mae']:.6f} m ({results['fk_mae'] * 100:.2f} cm)")
    print(f"\n  各维度平均误差:")
    dim_names = ['X', 'Y', 'Z']
    for i, (name, error) in enumerate(zip(dim_names, results['fk_errors_per_dim'])):
        print(f"    {name}: {error:.6f} m ({error * 100:.2f} cm)")
    
    # 泛化性评估
    print("\n" + "=" * 70)
    print("泛化性评估")
    print("=" * 70)
    
    # 参考训练集性能（从日志中获取的最后几轮）
    train_ik_rmse = 0.090  # ~0.008 loss
    train_fk_rmse = 0.044  # ~0.002 loss
    
    print(f"\n训练集(ACCAD+CMU) vs 测试集(GRAB):")
    print(f"\nIK RMSE:")
    print(f"  训练集: {train_ik_rmse:.4f} rad ({train_ik_rmse * 180 / np.pi:.2f} deg)")
    print(f"  GRAB:   {results['ik_rmse']:.4f} rad ({results['ik_rmse'] * 180 / np.pi:.2f} deg)")
    print(f"  差距:   {abs(results['ik_rmse'] - train_ik_rmse):.4f} rad")
    
    print(f"\nFK RMSE:")
    print(f"  训练集: {train_fk_rmse:.4f} m ({train_fk_rmse * 100:.2f} cm)")
    print(f"  GRAB:   {results['fk_rmse']:.4f} m ({results['fk_rmse'] * 100:.2f} cm)")
    print(f"  差距:   {abs(results['fk_rmse'] - train_fk_rmse):.4f} m")
    
    # 泛化性判断
    ik_gap = abs(results['ik_rmse'] - train_ik_rmse) / train_ik_rmse * 100
    fk_gap = abs(results['fk_rmse'] - train_fk_rmse) / train_fk_rmse * 100
    
    print(f"\n泛化性判断:")
    if ik_gap < 20 and fk_gap < 20:
        print("  ✅ 泛化性良好 (误差增加 < 20%)")
    elif ik_gap < 50 and fk_gap < 50:
        print("  ⚠️ 泛化性一般 (误差增加 20-50%)")
    else:
        print("  ❌ 泛化性较差 (误差增加 > 50%)")
    
    return results


def visualize_predictions(model_path, grab_data_path, n_samples=5, device='cuda'):
    """
    可视化几个样本的预测结果
    """
    print("\n" + "=" * 70)
    print("预测可视化")
    print("=" * 70)
    
    # 加载模型和数据
    model = ExplicitCouplingIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        use_temporal=False
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    X_grab, y_grab = load_grab_data(grab_data_path)
    if X_grab is None:
        return
    
    gpu_fk = SimpleGPUFK()
    
    # 随机选几个样本
    indices = np.random.choice(len(X_grab), n_samples, replace=False)
    
    print(f"\n随机选择 {n_samples} 个样本:")
    print("-" * 70)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            X_sample = torch.tensor(X_grab[idx:idx+1], dtype=torch.float32).to(device)
            y_sample = torch.tensor(y_grab[idx:idx+1], dtype=torch.float32).to(device)
            
            # 提取输入和目标
            human_pose = y_sample[:, :7]
            human_pos = human_pose[:, :3]
            human_ori = human_pose[:, 3:7]
            target_angles = y_sample[:, 7:]
            
            # 预测
            pred_angles, info = model(human_pos, human_ori)
            
            # FK验证
            target_pos = gpu_fk.forward(target_angles)
            pred_pos = gpu_fk.forward(pred_angles)
            
            # 计算误差
            angle_error = (pred_angles - target_angles).abs().cpu().numpy()[0]
            pos_error = (pred_pos - target_pos).abs().cpu().numpy()[0]
            
            print(f"\n样本 {i+1}:")
            print(f"  人类位姿: [{human_pos[0].cpu().numpy().round(3)}]")
            print(f"  目标角度: [{target_angles[0].cpu().numpy().round(3)}]")
            print(f"  预测角度: [{pred_angles[0].cpu().numpy().round(3)}]")
            print(f"  角度误差: [{angle_error.round(4)}] rad")
            print(f"  位置误差: [{pos_error.round(4)}] m")
            print(f"  耦合强度: elbow={info['coupling_info']['elbow_shoulder_coupling'].item():.3f}, "
                  f"forearm={info['coupling_info']['forearm_elbow_coupling'].item():.3f}")


if __name__ == "__main__":
    model_path = "/home/bonuli/Pieper/pieper1802/explicit_coupling_ik_optimized.pth"
    grab_data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"
    
    # 检查文件
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请确认训练已完成并保存模型")
        sys.exit(1)
    
    # 运行测试
    results = compare_with_training_set(model_path, grab_data_path)
    
    # 可视化几个样本
    if results:
        visualize_predictions(model_path, grab_data_path, n_samples=5)
