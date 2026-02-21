"""
对比不同模型在GRAB上的泛化性
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from explicit_coupling_ik import ExplicitCouplingIK


def load_model(model_path, device='cuda'):
    """加载模型"""
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
    model.eval()
    
    return model, checkpoint


def quick_test(model, y_data, device='cuda', n_samples=5000):
    """快速测试"""
    y_tensor = torch.tensor(y_data[:n_samples], dtype=torch.float32).to(device)
    
    human_pose = y_tensor[:, :7]
    human_pos = human_pose[:, :3]
    human_ori = human_pose[:, 3:7]
    target_angles = y_tensor[:, 7:]
    
    with torch.no_grad():
        pred_angles, _ = model(human_pos, human_ori)
        
        # IK误差
        ik_error = (pred_angles - target_angles).abs()
        ik_rmse = torch.sqrt((ik_error ** 2).mean()).item()
        ik_mae = ik_error.mean().item()
        
        # R²
        ss_res = ((target_angles - pred_angles) ** 2).sum()
        ss_tot = ((target_angles - target_angles.mean(dim=0)) ** 2).sum()
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
    
    return {
        'ik_rmse': ik_rmse,
        'ik_mae': ik_mae,
        'ik_rmse_deg': np.rad2deg(ik_rmse),
        'ik_mae_deg': np.rad2deg(ik_mae),
        'r2': r2
    }


def main():
    print("=" * 80)
    print("GRAB泛化性对比测试")
    print("=" * 80)
    
    # 加载GRAB数据
    grab_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"
    print(f"\n加载GRAB数据...")
    data = np.load(grab_path)
    y_grab = data['y']
    print(f"总样本: {len(y_grab)}")
    
    # 随机采样
    np.random.seed(42)
    indices = np.random.choice(len(y_grab), 10000, replace=False)
    y_test = y_grab[indices]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试的模型
    models_to_test = {
        "原模型": "/home/bonuli/Pieper/pieper1802/explicit_coupling_ik_optimized.pth",
        "鲁棒性模型": "/home/bonuli/Pieper/pieper1802/robust_ik.pth",
    }
    
    results = {}
    
    for name, path in models_to_test.items():
        if not os.path.exists(path):
            print(f"\n⚠ {name} 不存在: {path}")
            continue
        
        print(f"\n测试 {name}...")
        try:
            model, ckpt = load_model(path, device)
            result = quick_test(model, y_test, device)
            results[name] = result
            
            if 'epoch' in ckpt:
                print(f"  训练轮数: {ckpt['epoch']}")
        except Exception as e:
            print(f"  错误: {e}")
    
    # 打印对比
    if results:
        print("\n" + "=" * 80)
        print("GRAB泛化性对比")
        print("=" * 80)
        
        print(f"\n{'模型':<20} | {'角度RMSE(°)':>12} | {'角度MAE(°)':>12} | {'R²':>10}")
        print("-" * 80)
        
        for name, result in results.items():
            print(f"{name:<20} | {result['ik_rmse_deg']:>12.2f} | "
                  f"{result['ik_mae_deg']:>12.2f} | {result['r2']:>10.4f}")
        
        # 评价
        print("\n" + "=" * 80)
        print("评价")
        print("=" * 80)
        
        for name, result in results.items():
            rmse = result['ik_rmse_deg']
            r2 = result['r2']
            
            if rmse < 10 and r2 > 0.9:
                rating = "✓✓ 泛化性优秀"
            elif rmse < 15 and r2 > 0.8:
                rating = "✓ 泛化性良好"
            elif rmse < 20 and r2 > 0.7:
                rating = "○ 泛化性一般"
            else:
                rating = "✗ 需要改进"
            
            print(f"  {name:<20}: {rating}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
