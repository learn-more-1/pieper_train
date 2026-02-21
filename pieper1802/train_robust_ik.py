"""
鲁棒性IK模型训练 - 提高泛化性

策略：
1. 只用ACCAD+CMU训练（保持泛化性）
2. 强数据增强（提高鲁棒性）
3. Dropout正则化（防止过拟合）
4. 输出train loss和R2
5. GRAB仅用于测试，不参与训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import sys
import time
import os

sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from explicit_coupling_ik import ExplicitCouplingIK
from dataset_generalized import create_windowed_dataloaders
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("robust_ik_train.log"), logging.StreamHandler()]
)


class RobustAugmentation:
    """
    强数据增强 - 提高泛化性
    """
    
    def __init__(self, 
                 position_noise=0.03,      # 3cm位置噪声
                 angle_noise=0.08,          # 0.08rad角度噪声
                 dropout_prob=0.1):         # 特征dropout
        self.position_noise = position_noise
        self.angle_noise = angle_noise
        self.dropout_prob = dropout_prob
        
    def __call__(self, human_pose, target_angles):
        """
        Args:
            human_pose: [batch, 7]
            target_angles: [batch, 7]
        """
        # 1. 位置噪声
        if self.position_noise > 0:
            noise = torch.randn_like(human_pose[:, :3]) * self.position_noise
            human_pose[:, :3] = human_pose[:, :3] + noise
        
        # 2. 姿态噪声（四元数小扰动）
        if np.random.rand() < 0.5:
            ori_noise = torch.randn_like(human_pose[:, 3:7]) * 0.05
            human_pose[:, 3:7] = human_pose[:, 3:7] + ori_noise
            # 归一化
            human_pose[:, 3:7] = human_pose[:, 3:7] / (human_pose[:, 3:7].norm(dim=1, keepdim=True) + 1e-8)
        
        # 3. 角度噪声
        if self.angle_noise > 0:
            angle_noise = torch.randn_like(target_angles) * self.angle_noise
            target_angles = target_angles + angle_noise
        
        # 4. 随机mask（模拟传感器丢失）
        if self.dropout_prob > 0 and np.random.rand() < 0.3:
            mask = torch.rand_like(human_pose[:, :3]) > self.dropout_prob
            human_pose[:, :3] = human_pose[:, :3] * mask.float()
        
        return human_pose, target_angles


class Config:
    # 数据配置 - 只用ACCAD+CMU
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
    num_joints = 7
    num_frames = 10
    
    # 训练参数
    batch_size = 1024
    epochs = 300
    initial_lr = 2e-3
    min_lr = 1e-6
    weight_decay = 5e-4  # 增大权重衰减
    
    # 早停
    early_stopping_patience = 30
    
    # 数据增强
    use_augmentation = True
    position_noise = 0.03
    angle_noise = 0.08
    dropout_prob = 0.1
    
    # 学习率调度
    warmup_epochs = 10
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 模型配置
    hidden_dim = 256
    
    # 损失权重
    ik_weight = 1.0
    fk_weight = 0.5
    coupling_reg = 0.01
    
    # 数据加载
    num_workers = 4
    pin_memory = True


def calculate_r2(pred, target):
    """计算R²分数"""
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean(dim=0)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2.item()


def train():
    config = Config()
    
    model_name = 'robust_ik'
    
    logging.info("=" * 90)
    logging.info("鲁棒性IK模型训练")
    logging.info("=" * 90)
    logging.info("策略:")
    logging.info("  1. 只用ACCAD+CMU训练（保持泛化性）")
    logging.info("  2. 强数据增强（提高鲁棒性）")
    logging.info("  3. 大权重衰减（防止过拟合）")
    logging.info("  4. GRAB仅用于测试验证")
    
    # 加载数据
    logging.info("\n加载ACCAD+CMU数据集...")
    
    class DataConfig:
        data_path = config.data_path
        num_joints = config.num_joints
        num_frames = config.num_frames
        batch_size = config.batch_size
        num_workers = config.num_workers
        pin_memory = config.pin_memory
    
    train_loader, val_loader = create_windowed_dataloaders(config.data_path, DataConfig())
    
    # 创建模型
    logging.info(f"\n创建模型: {model_name}")
    model = ExplicitCouplingIK(
        num_joints=config.joint_dim if hasattr(config, 'joint_dim') else 7,
        num_frames=config.num_frames,
        hidden_dim=config.hidden_dim,
        use_temporal=False
    ).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"模型参数量: {total_params:.2f}M")
    
    # 加载GPU FK
    logging.info("\n加载GPU FK模型...")
    gpu_fk = SimpleGPUFK()
    
    # 数据增强
    augmentation = RobustAugmentation(
        position_noise=config.position_noise,
        angle_noise=config.angle_noise,
        dropout_prob=config.dropout_prob
    ) if config.use_augmentation else None
    
    # 损失函数
    mse_criterion = nn.MSELoss()
    
    # 优化器 - 大权重衰减防止过拟合
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.initial_lr,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return epoch / config.warmup_epochs
        else:
            progress = (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 早停
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    
    # 断点路径
    checkpoint_path = f"/home/bonuli/Pieper/pieper1802/{model_name}.pth"
    
    logging.info(f"\n开始训练（{config.epochs} epochs）...")
    logging.info("=" * 90)
    logging.info(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>11} | {'IK RMSE':>9} | {'FK RMSE':>9} | {'Train R²':>9} | {'Val R²':>9} | {'LR':>10}")
    logging.info("-" * 90)
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        
        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        train_ik_loss = 0.0
        train_fk_loss = 0.0
        
        all_train_preds = []
        all_train_targets = []
        
        for batch_X, batch_y, _ in train_loader:
            batch_y = batch_y.to(config.device, non_blocking=True)
            batch_size = batch_y.shape[0]
            
            # 提取输入和目标
            if batch_y.shape[1] == 14:
                human_pose = batch_y[:, :7]
                target_angles = batch_y[:, 7:]
            else:
                continue
            
            # 数据增强
            if augmentation:
                human_pose_aug = human_pose.clone()
                target_angles_aug = target_angles.clone()
                human_pose_aug, target_angles_aug = augmentation(human_pose_aug, target_angles_aug)
            else:
                human_pose_aug = human_pose
                target_angles_aug = target_angles
            
            human_pos = human_pose_aug[:, :3]
            human_ori = human_pose_aug[:, 3:7]
            
            optimizer.zero_grad()
            
            # 前向
            pred_angles, info = model(human_pos, human_ori)
            
            # 损失
            ik_loss = mse_criterion(pred_angles, target_angles_aug)
            
            with torch.no_grad():
                target_pos = gpu_fk.forward(target_angles)
            pred_pos = gpu_fk.forward(pred_angles)
            fk_loss = mse_criterion(pred_pos, target_pos)
            
            coupling_loss = torch.tensor(0.0, device=config.device)
            for key, value in info['coupling_info'].items():
                coupling_loss += ((value - 0.5) ** 2)
            
            total_loss = (
                config.ik_weight * ik_loss +
                config.fk_weight * fk_loss +
                config.coupling_reg * coupling_loss
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item() * batch_size
            train_ik_loss += ik_loss.item() * batch_size
            train_fk_loss += fk_loss.item() * batch_size
            
            # 收集R²计算数据（用原始目标，不是增强后的）
            all_train_preds.append(pred_angles.detach().cpu())
            all_train_targets.append(target_angles.cpu())
        
        train_loss /= len(train_loader.dataset)
        train_ik_loss /= len(train_loader.dataset)
        train_fk_loss /= len(train_loader.dataset)
        
        # 训练R²
        all_train_preds = torch.cat(all_train_preds, dim=0)
        all_train_targets = torch.cat(all_train_targets, dim=0)
        train_r2 = calculate_r2(all_train_preds, all_train_targets)
        
        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0.0
        val_ik_loss = 0.0
        val_fk_loss = 0.0
        
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y, _ in val_loader:
                batch_y = batch_y.to(config.device, non_blocking=True)
                batch_size = batch_y.shape[0]
                
                if batch_y.shape[1] == 14:
                    human_pose = batch_y[:, :7]
                    target_angles = batch_y[:, 7:]
                else:
                    continue
                
                human_pos = human_pose[:, :3]
                human_ori = human_pose[:, 3:7]
                
                pred_angles, info = model(human_pos, human_ori)
                
                ik_loss = mse_criterion(pred_angles, target_angles)
                
                target_pos = gpu_fk.forward(target_angles)
                pred_pos = gpu_fk.forward(pred_angles)
                fk_loss = mse_criterion(pred_pos, target_pos)
                
                total_loss = ik_loss + 0.5 * fk_loss
                
                val_loss += total_loss.item() * batch_size
                val_ik_loss += ik_loss.item() * batch_size
                val_fk_loss += fk_loss.item() * batch_size
                
                all_val_preds.append(pred_angles.cpu())
                all_val_targets.append(target_angles.cpu())
        
        val_loss /= len(val_loader.dataset)
        val_ik_loss /= len(val_loader.dataset)
        val_fk_loss /= len(val_loader.dataset)
        
        # 验证R²
        all_val_preds = torch.cat(all_val_preds, dim=0)
        all_val_targets = torch.cat(all_val_targets, dim=0)
        val_r2 = calculate_r2(all_val_preds, all_val_targets)
        
        # RMSE
        train_ik_rmse = np.sqrt(train_ik_loss)
        train_fk_rmse = np.sqrt(train_fk_loss)
        val_ik_rmse = np.sqrt(val_ik_loss)
        val_fk_rmse = np.sqrt(val_fk_loss)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 日志
        logging.info(
            f"{epoch:>6} | {train_loss:>11.6f} | {val_loss:>11.6f} | "
            f"{val_ik_rmse:>9.4f} | {val_fk_rmse:>9.4f} | "
            f"{train_r2:>9.4f} | {val_r2:>9.4f} | {current_lr:>10.6f}"
        )
        
        # 保存最优
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'val_r2': val_r2,
                'train_r2': train_r2,
            }, checkpoint_path)
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= config.early_stopping_patience:
            logging.info(f"\n>>> 早停触发 (patience={config.early_stopping_patience})")
            break
    
    logging.info("=" * 90)
    logging.info("训练完成!")
    logging.info(f"最优验证损失: {best_val_loss:.6f} (epoch {best_epoch})")
    logging.info(f"模型保存至: {checkpoint_path}")
    logging.info("\n下一步: 在GRAB上测试泛化性")
    logging.info("  python test_grab_with_fk.py")


if __name__ == "__main__":
    train()
