"""
在GRAB数据上Fine-tune模型

策略：冻结耦合图，只训练编码器和解码器
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from explicit_coupling_ik import ExplicitCouplingIK


def load_grab_data(data_path, val_split=0.2):
    """加载GRAB数据集"""
    print(f"加载GRAB数据: {data_path}")
    data = np.load(data_path)
    
    y = data['y']
    n_samples = len(y)
    
    # 划分训练/验证
    split_idx = int(n_samples * (1 - val_split))
    
    y_train = y[:split_idx]
    y_val = y[split_idx:]
    
    print(f"训练集: {len(y_train)}, 验证集: {len(y_val)}")
    
    return y_train, y_val


def train_epoch(model, y_data, optimizer, device, batch_size=1024):
    """训练一个epoch"""
    model.train()
    
    y_tensor = torch.tensor(y_data, dtype=torch.float32).to(device)
    
    # 提取输入和目标
    human_pose = y_tensor[:, :7]
    human_position = human_pose[:, :3]
    human_orientation = human_pose[:, 3:7]
    target_angles = y_tensor[:, 7:]
    
    n_samples = len(y_tensor)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    total_loss = 0.0
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch_human_pos = human_position[start_idx:end_idx]
        batch_human_ori = human_orientation[start_idx:end_idx]
        batch_target_angles = target_angles[start_idx:end_idx]
        
        optimizer.zero_grad()
        
        # 前向
        pred_angles, _ = model(batch_human_pos, batch_human_ori)
        
        # 只计算IK loss（不需要FK，因为耦合图冻结了）
        loss = nn.MSELoss()(pred_angles, batch_target_angles)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * (end_idx - start_idx)
    
    return total_loss / n_samples


def validate(model, y_data, device, batch_size=1024):
    """验证"""
    model.eval()
    
    y_tensor = torch.tensor(y_data, dtype=torch.float32).to(device)
    
    human_pose = y_tensor[:, :7]
    human_position = human_pose[:, :3]
    human_orientation = human_pose[:, 3:7]
    target_angles = y_tensor[:, 7:]
    
    n_samples = len(y_tensor)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_errors = []
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_human_pos = human_position[start_idx:end_idx]
            batch_human_ori = human_orientation[start_idx:end_idx]
            batch_target_angles = target_angles[start_idx:end_idx]
            
            pred_angles, _ = model(batch_human_pos, batch_human_ori)
            
            error = (pred_angles - batch_target_angles).abs()
            all_errors.append(error.cpu().numpy())
    
    all_errors = np.concatenate(all_errors, axis=0)
    
    return {
        'rmse': np.sqrt((all_errors ** 2).mean()),
        'mae': all_errors.mean(),
        'max_error': all_errors.max()
    }


def main():
    # 配置
    model_path = "/home/bonuli/Pieper/pieper1802/explicit_coupling_ik_optimized.pth"
    grab_data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"
    output_path = "/home/bonuli/Pieper/pieper1802/explicit_coupling_ik_finetuned.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("GRAB数据集 Fine-tune")
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
    state_dict = checkpoint['model_state_dict']
    
    # 处理torch.compile前缀
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    print("✓ 模型加载成功")
    
    # 冻结耦合图
    print("\n冻结层:")
    for name, param in model.named_parameters():
        if 'coupling_graph' in name:
            param.requires_grad = False
            print(f"  - {name}")
    
    # 只训练编码器和解码器
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    print(f"\n可训练参数: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")
    
    # 优化器
    optimizer = optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    
    # 加载数据
    y_train, y_val = load_grab_data(grab_data_path)
    
    # 训练前验证
    print("\n训练前验证:")
    pre_results = validate(model, y_val, device)
    print(f"  RMSE: {pre_results['rmse']:.4f} rad ({pre_results['rmse'] * 180 / np.pi:.2f} deg)")
    print(f"  MAE:  {pre_results['mae']:.4f} rad ({pre_results['mae'] * 180 / np.pi:.2f} deg)")
    
    # Fine-tune
    print("\n" + "=" * 70)
    print("开始Fine-tune")
    print("=" * 70)
    
    best_mae = pre_results['mae']
    epochs = 20
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, y_train, optimizer, device)
        val_results = validate(model, y_val, device)
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val RMSE: {val_results['rmse']:.4f} | "
              f"Val MAE: {val_results['mae']:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳
        if val_results['mae'] < best_mae:
            best_mae = val_results['mae']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_results': val_results,
                'best_mae': best_mae
            }, output_path)
            print(f"  >>> 保存最佳模型 (MAE: {best_mae:.6f})")
    
    # 最终验证
    print("\n" + "=" * 70)
    print("Fine-tune完成")
    print("=" * 70)
    
    print(f"\n训练前: RMSE={pre_results['rmse']:.4f}, MAE={pre_results['mae']:.4f}")
    print(f"训练后: RMSE={val_results['rmse']:.4f}, MAE={val_results['mae']:.4f}")
    
    improvement = (pre_results['mae'] - val_results['mae']) / pre_results['mae'] * 100
    print(f"\n提升: {improvement:.1f}%")
    
    print(f"\n模型保存至: {output_path}")


if __name__ == "__main__":
    main()
