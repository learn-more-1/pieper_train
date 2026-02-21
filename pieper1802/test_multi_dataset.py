"""
多数据集泛化性测试

在多个数据集上测试模型，评估泛化性能
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


def load_dataset(data_path, num_samples=10000):
    """加载数据集"""
    print(f"\n加载: {os.path.basename(data_path)}")
    
    if not os.path.exists(data_path):
        print(f"  ❌ 文件不存在")
        return None
    
    data = np.load(data_path)
    y = data['y']
    
    # 随机采样
    if len(y) > num_samples:
        indices = np.random.choice(len(y), num_samples, replace=False)
        y = y[indices]
    
    print(f"  ✓ 样本数: {len(y)}")
    return y


def evaluate_on_dataset(model, y_data, gpu_fk, device='cuda', batch_size=512):
    """在数据集上评估"""
    model.eval()
    mse_criterion = nn.MSELoss()
    
    y_tensor = torch.tensor(y_data, dtype=torch.float32).to(device)
    
    # 提取人类位姿和机器人角度
    human_pose = y_tensor[:, :7]
    human_pos = human_pose[:, :3]
    human_ori = human_pose[:, 3:7]
    target_angles = y_tensor[:, 7:]
    
    n_samples = len(y_tensor)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_pred_angles = []
    all_pred_pos = []
    all_target_pos = []
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_human_pos = human_pos[start_idx:end_idx]
            batch_human_ori = human_ori[start_idx:end_idx]
            batch_target_angles = target_angles[start_idx:end_idx]
            
            # 预测
            pred_angles, _ = model(batch_human_pos, batch_human_ori)
            
            # FK
            pred_pos = gpu_fk.forward(pred_angles)
            target_pos = gpu_fk.forward(batch_target_angles)
            
            all_pred_angles.append(pred_angles.cpu())
            all_pred_pos.append(pred_pos.cpu())
            all_target_pos.append(target_pos.cpu())
    
    # 合并
    all_pred_angles = torch.cat(all_pred_angles, dim=0)
    all_pred_pos = torch.cat(all_pred_pos, dim=0)
    all_target_pos = torch.cat(all_target_pos, dim=0)
    target_angles_cpu = target_angles.cpu()
    
    # 计算指标
    # IK误差
    ik_mse = mse_criterion(all_pred_angles, target_angles_cpu).item()
    ik_rmse = np.sqrt(ik_mse)
    ik_mae = (all_pred_angles - target_angles_cpu).abs().mean().item()
    
    # FK误差
    pos_error = (all_pred_pos - all_target_pos).norm(dim=1)
    pos_mean = pos_error.mean().item()
    pos_median = pos_error.median().item()
    pos_max = pos_error.max().item()
    
    # R²
    ss_res = ((target_angles_cpu - all_pred_angles) ** 2).sum()
    ss_tot = ((target_angles_cpu - target_angles_cpu.mean(dim=0)) ** 2).sum()
    r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
    
    return {
        'ik_rmse': ik_rmse,
        'ik_mae': ik_mae,
        'pos_mean': pos_mean,
        'pos_median': pos_median,
        'pos_max': pos_max,
        'r2': r2,
        'n_samples': n_samples
    }


def test_model(model_path, test_datasets):
    """
    在多个数据集上测试模型
    
    Args:
        model_path: 模型路径
        test_datasets: {名称: 路径} 字典
    """
    print("=" * 80)
    print("多数据集泛化性测试")
    print("=" * 80)
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n加载模型: {model_path}")
    
    model = ExplicitCouplingIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        use_temporal=False
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    print("✓ 模型加载成功")
    
    # 加载FK
    try:
        gpu_fk = SimpleGPUFK()
        print("✓ GPU FK加载成功")
    except:
        print("❌ 无法加载GPU FK")
        return
    
    # 测试每个数据集
    print("\n" + "=" * 80)
    results = {}
    
    for name, path in test_datasets.items():
        y_data = load_dataset(path, num_samples=10000)
        if y_data is None:
            continue
        
        print(f"  评估中...")
        result = evaluate_on_dataset(model, y_data, gpu_fk, device)
        results[name] = result
    
    # 打印结果表格
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    print(f"\n{'数据集':<20} | {'位置误差(mm)':>12} | {'位置中位(mm)':>12} | {'角度RMSE(°)':>12} | {'R²':>8}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<20} | {result['pos_mean']*1000:>12.2f} | "
              f"{result['pos_median']*1000:>12.2f} | "
              f"{np.rad2deg(result['ik_rmse']):>12.2f} | {result['r2']:>8.4f}")
    
    # 计算平均性能
    if results:
        avg_pos = np.mean([r['pos_mean'] for r in results.values()])
        avg_r2 = np.mean([r['r2'] for r in results.values()])
        
        print(f"\n{'平均':<20} | {avg_pos*1000:>12.2f} | {'':<12} | {'':<12} | {avg_r2:>8.4f}")
    
    # 泛化性评级
    print("\n" + "=" * 80)
    print("泛化性评级")
    print("=" * 80)
    
    for name, result in results.items():
        pos_mm = result['pos_mean'] * 1000
        
        if pos_mm < 10:
            rating = "✓✓ 优秀"
        elif pos_mm < 20:
            rating = "✓ 良好"
        elif pos_mm < 50:
            rating = "○ 一般"
        else:
            rating = "✗ 较差"
        
        print(f"  {name:<20}: {pos_mm:>6.1f}mm {rating}")
    
    # 总体评价
    if results:
        max_pos = max(r['pos_mean'] for r in results.values()) * 1000
        min_pos = min(r['pos_mean'] for r in results.values()) * 1000
        
        print(f"\n跨数据集差异:")
        print(f"  最佳: {min_pos:.1f}mm")
        print(f"  最差: {max_pos:.1f}mm")
        print(f"  差异: {max_pos - min_pos:.1f}mm")
        
        if max_pos - min_pos < 10:
            print("  ✓ 泛化性优秀（跨数据集差异小）")
        elif max_pos - min_pos < 20:
            print("  ○ 泛化性良好")
        else:
            print("  ✗ 泛化性一般（跨数据集差异大）")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # 配置
    model_path = "/home/bonuli/Pieper/pieper1802/generalized_ik.pth"
    
    # 测试数据集
    test_datasets = {
        "ACCAD+CMU(验证)": "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz",
        "GRAB": "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz",
    }
    
    # 如果模型不存在，尝试其他路径
    if not os.path.exists(model_path):
        alt_paths = [
            "/home/bonuli/Pieper/pieper1802/explicit_coupling_ik_optimized.pth",
            "/home/bonuli/Pieper/pieper1802/explicit_coupling_ik_finetuned.pth",
        ]
        for path in alt_paths:
            if os.path.exists(path):
                model_path = path
                print(f"使用替代模型: {path}")
                break
    
    # 运行测试
    if os.path.exists(model_path):
        test_model(model_path, test_datasets)
    else:
        print(f"❌ 找不到模型文件")
        print(f"请确认模型已训练并保存")
