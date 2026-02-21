"""
简化版GRAB数据集测试

不依赖pytorch_kinematics，只评估IK性能
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from explicit_coupling_ik import ExplicitCouplingIK


def load_grab_data(data_path):
    """加载GRAB数据集"""
    print(f"加载GRAB数据集: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None, None
    
    data = np.load(data_path)
    
    print(f"数据键: {list(data.keys())}")
    
    X = data['X']
    y = data['y']
    
    print(f"X形状: {X.shape}")
    print(f"y形状: {y.shape}")
    print(f"总样本数: {len(y)}")
    
    # 只取验证集部分（后20%）
    n_samples = len(y)
    split_idx = int(n_samples * 0.8)
    
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    
    print(f"验证集大小: {len(y_val)}")
    
    return X_val, y_val


def evaluate_ik_only(model, X_data, y_data, device='cuda', batch_size=1024):
    """
    只评估IK性能（关节角度预测）
    """
    model.eval()
    mse_criterion = nn.MSELoss()
    
    # 准备数据
    y_tensor = torch.tensor(y_data, dtype=torch.float32).to(device)
    
    # 提取人类位姿（输入）和机器人角度（目标）
    human_pose = y_tensor[:, :7]
    human_position = human_pose[:, :3]
    human_orientation = human_pose[:, 3:7]
    target_angles = y_tensor[:, 7:]
    
    n_samples = len(y_tensor)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_pred_angles = []
    
    print(f"开始推理 ({n_batches} batches)...")
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_human_pos = human_position[start_idx:end_idx]
            batch_human_ori = human_orientation[start_idx:end_idx]
            
            # 模型推理
            pred_angles, _ = model(batch_human_pos, batch_human_ori)
            
            all_pred_angles.append(pred_angles.cpu())
            
            if (i + 1) % 10 == 0 or i == n_batches - 1:
                print(f"  进度: {i+1}/{n_batches}")
    
    # 合并结果
    all_pred_angles = torch.cat(all_pred_angles, dim=0)
    target_angles_cpu = target_angles.cpu()
    
    # 计算误差
    errors = (all_pred_angles - target_angles_cpu).abs().numpy()
    mse = mse_criterion(all_pred_angles, target_angles_cpu).item()
    rmse = np.sqrt(mse)
    mae = errors.mean()
    
    # 各关节误差
    per_joint_mae = errors.mean(axis=0)
    per_joint_rmse = np.sqrt(((all_pred_angles - target_angles_cpu).numpy() ** 2).mean(axis=0))
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mse': mse,
        'per_joint_mae': per_joint_mae,
        'per_joint_rmse': per_joint_rmse,
        'n_samples': n_samples
    }


def main():
    model_path = "/home/bonuli/Pieper/pieper1802/explicit_coupling_ik_optimized.pth"
    grab_data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"
    
    print("=" * 70)
    print("模型泛化性测试 - GRAB数据集")
    print("=" * 70)
    
    # 检查文件
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(grab_data_path):
        print(f"❌ 数据文件不存在: {grab_data_path}")
        return
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = ExplicitCouplingIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        use_temporal=False
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # 处理torch.compile的权重前缀
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("检测到torch.compile权重，移除前缀...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ 模型加载成功")
    print(f"训练轮数: {checkpoint.get('epoch', 'unknown')}")
    if 'best_val_loss' in checkpoint:
        print(f"最佳验证损失: {checkpoint['best_val_loss']:.6f}")
    
    # 加载数据
    X_grab, y_grab = load_grab_data(grab_data_path)
    if X_grab is None:
        return
    
    # 评估
    print("\n" + "=" * 70)
    print("在GRAB数据集上评估IK性能")
    print("=" * 70)
    
    results = evaluate_ik_only(model, X_grab, y_grab, device)
    
    # 打印结果
    print(f"\n样本数: {results['n_samples']}")
    print(f"\n关节角度误差:")
    print(f"  MSE:  {results['mse']:.6f} rad²")
    print(f"  RMSE: {results['rmse']:.6f} rad ({results['rmse'] * 180 / np.pi:.2f} deg)")
    print(f"  MAE:  {results['mae']:.6f} rad ({results['mae'] * 180 / np.pi:.2f} deg)")
    
    print(f"\n各关节误差 (RMSE / MAE):")
    joint_names = ['Shoulder0', 'Shoulder1', 'Shoulder2', 'Elbow', 'Forearm', 'Wrist0', 'Wrist1']
    for i, name in enumerate(joint_names):
        rmse_deg = results['per_joint_rmse'][i] * 180 / np.pi
        mae_deg = results['per_joint_mae'][i] * 180 / np.pi
        print(f"  {name:12s}: RMSE={rmse_deg:5.2f}°  MAE={mae_deg:5.2f}°")
    
    # 泛化性评估
    print("\n" + "=" * 70)
    print("泛化性评估")
    print("=" * 70)
    
    # 参考训练集性能
    train_rmse = 0.090  # ~0.008 loss
    train_mae = 0.070
    
    print(f"\n训练集(ACCAD+CMU) vs 测试集(GRAB):")
    print(f"\nIK RMSE:")
    print(f"  训练集: {train_rmse:.4f} rad ({train_rmse * 180 / np.pi:.2f} deg)")
    print(f"  GRAB:   {results['rmse']:.4f} rad ({results['rmse'] * 180 / np.pi:.2f} deg)")
    
    gap = abs(results['rmse'] - train_rmse) / train_rmse * 100
    print(f"  相对差距: {gap:.1f}%")
    
    print(f"\nIK MAE:")
    print(f"  训练集: {train_mae:.4f} rad ({train_mae * 180 / np.pi:.2f} deg)")
    print(f"  GRAB:   {results['mae']:.4f} rad ({results['mae'] * 180 / np.pi:.2f} deg)")
    
    # 判断
    print(f"\n泛化性判断:")
    if gap < 20:
        print("  ✅ 泛化性优秀 (误差增加 < 20%)")
    elif gap < 50:
        print("  ⚠️ 泛化性良好 (误差增加 20-50%)")
    elif gap < 100:
        print("  ⚠️ 泛化性一般 (误差增加 50-100%)")
    else:
        print("  ❌ 泛化性较差 (误差增加 > 100%)")
    
    # 可视化几个样本
    print("\n" + "=" * 70)
    print("随机样本预测对比")
    print("=" * 70)
    
    y_tensor = torch.tensor(y_grab, dtype=torch.float32).to(device)
    indices = np.random.choice(len(y_grab), 5, replace=False)
    
    with torch.no_grad():
        for idx in indices:
            y_sample = y_tensor[idx:idx+1]
            human_pose = y_sample[:, :7]
            human_pos = human_pose[:, :3]
            human_ori = human_pose[:, 3:7]
            target_angles = y_sample[:, 7:]
            
            pred_angles, info = model(human_pos, human_ori)
            
            error = (pred_angles - target_angles).abs().cpu().numpy()[0]
            
            print(f"\n样本 {idx}:")
            print(f"  人类位置: [{human_pos[0].cpu().numpy().round(3)}]")
            print(f"  目标角度: [{target_angles[0].cpu().numpy().round(3)}]")
            print(f"  预测角度: [{pred_angles[0].cpu().numpy().round(3)}]")
            print(f"  绝对误差: [{error.round(4)}] rad (max={error.max():.4f})")


if __name__ == "__main__":
    main()
