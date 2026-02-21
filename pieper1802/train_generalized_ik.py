"""
泛化性优化版IK模型训练

核心改进：
1. 混合多数据集训练（ACCAD+CMU+GRAB）
2. 强数据增强（噪声、缩放、旋转）
3. 领域随机化（Domain Randomization）
4. 输出train loss和R2指标
5. 早停机制防止过拟合
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
import sys
import time
import os

sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from explicit_coupling_ik import ExplicitCouplingIK
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("generalized_ik_train.log"), logging.StreamHandler()]
)


class StrongAugmentation:
    """
    强数据增强 - 提高泛化性
    """
    
    def __init__(self, 
                 position_noise=0.02,      # 位置噪声 2cm
                 angle_noise=0.05,          # 角度噪声 0.05rad
                 scale_range=(0.9, 1.1),    # 缩放范围
                 rotation_perturb=0.1):     # 旋转扰动
        self.position_noise = position_noise
        self.angle_noise = angle_noise
        self.scale_range = scale_range
        self.rotation_perturb = rotation_perturb
        
    def __call__(self, human_pose, target_angles):
        """
        Args:
            human_pose: [batch, 7] (position + orientation)
            target_angles: [batch, 7]
        """
        # 1. 位置加噪声
        if self.position_noise > 0:
            noise_pos = torch.randn_like(human_pose[:, :3]) * self.position_noise
            human_pose[:, :3] += noise_pos
        
        # 2. 姿态加噪声（四元数扰动）
        if self.rotation_perturb > 0:
            # 简化的四元数扰动
            noise_ori = torch.randn_like(human_pose[:, 3:7]) * self.rotation_perturb
            human_pose[:, 3:7] += noise_ori
            # 重新归一化
            human_pose[:, 3:7] = human_pose[:, 3:7] / (human_pose[:, 3:7].norm(dim=1, keepdim=True) + 1e-8)
        
        # 3. 角度加噪声
        if self.angle_noise > 0:
            noise_angle = torch.randn_like(target_angles) * self.angle_noise
            target_angles = target_angles + noise_angle
        
        # 4. 随机缩放
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            scale = torch.rand(1) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            human_pose[:, :3] *= scale
        
        return human_pose, target_angles


class MultiDatasetLoader:
    """
    加载多个数据集并混合
    """
    
    def __init__(self, data_paths, mix_ratio=None):
        """
        Args:
            data_paths: 数据文件路径列表
            mix_ratio: 混合比例，None表示按数据集大小比例
        """
        self.data_paths = data_paths
        self.mix_ratio = mix_ratio
        self.datasets = []
        
        for path in data_paths:
            if os.path.exists(path):
                data = np.load(path)
                y = data['y']  # (N, 14)
                self.datasets.append({
                    'path': path,
                    'y': y,
                    'size': len(y)
                })
                print(f"✓ 加载: {os.path.basename(path)} - {len(y)} 样本")
            else:
                print(f"✗ 不存在: {path}")
        
        # 计算混合比例
        if mix_ratio is None:
            total = sum(d['size'] for d in self.datasets)
            self.mix_ratio = [d['size'] / total for d in self.datasets]
        else:
            self.mix_ratio = mix_ratio
    
    def create_mixed_loader(self, batch_size=512, train_split=0.8):
        """创建混合的DataLoader"""
        
        # 分别划分训练/验证
        train_data = []
        val_data = []
        
        for dataset, ratio in zip(self.datasets, self.mix_ratio):
            y = dataset['y']
            n = len(y)
            n_train = int(n * train_split)
            
            # 随机打乱
            indices = np.random.permutation(n)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:]
            
            train_data.append(y[train_idx])
            val_data.append(y[val_idx])
        
        # 合并
        y_train = np.vstack(train_data)
        y_val = np.vstack(val_data)
        
        # 再次打乱
        train_perm = np.random.permutation(len(y_train))
        val_perm = np.random.permutation(len(y_val))
        
        y_train = y_train[train_perm]
        y_val = y_val[val_perm]
        
        print(f"\n混合数据集:")
        print(f"  训练集: {len(y_train)} 样本")
        print(f"  验证集: {len(y_val)} 样本")
        
        # 创建Dataset
        class MixedDataset(Dataset):
            def __init__(self, y_data):
                self.y = torch.from_numpy(y_data).float()
            
            def __len__(self):
                return len(self.y)
            
            def __getitem__(self, idx):
                y = self.y[idx]
                human_pose = y[:7]
                target_angles = y[7:]
                return human_pose, target_angles
        
        train_dataset = MixedDataset(y_train)
        val_dataset = MixedDataset(y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader


class Config:
    # 数据配置
    data_paths = [
        "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz",
        "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz",
    ]
    
    # 训练参数
    batch_size = 1024
    epochs = 200
    initial_lr = 2e-3
    min_lr = 1e-6
    weight_decay = 1e-4
    
    # 早停
    early_stopping_patience = 20
    
    # 数据增强
    use_augmentation = True
    position_noise = 0.02
    angle_noise = 0.05
    scale_range = (0.95, 1.05)
    rotation_perturb = 0.05
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 模型配置
    hidden_dim = 256
    
    # 损失权重
    ik_weight = 1.0
    fk_weight = 0.5
    coupling_reg = 0.01
    
    # 学习率调度
    warmup_epochs = 10


def calculate_r2(pred, target):
    """计算R²分数"""
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean(dim=0)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2.item()


def train():
    config = Config()
    
    model_name = 'generalized_ik'
    
    logging.info("=" * 80)
    logging.info("训练泛化性优化版IK模型")
    logging.info("=" * 80)
    logging.info("核心改进:")
    logging.info("  1. 混合多数据集（ACCAD+CMU+GRAB）")
    logging.info("  2. 强数据增强")
    logging.info("  3. 早停机制")
    logging.info("  4. 输出train loss和R2")
    
    # 加载混合数据集
    logging.info("\n加载混合数据集...")
    data_loader = MultiDatasetLoader(config.data_paths)
    train_loader, val_loader = data_loader.create_mixed_loader(config.batch_size)
    
    # 创建模型
    logging.info(f"\n创建模型: {model_name}")
    model = ExplicitCouplingIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=config.hidden_dim,
        use_temporal=False
    ).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"模型参数量: {total_params:.2f}M")
    
    # 加载GPU FK
    logging.info("\n加载GPU FK模型...")
    gpu_fk = SimpleGPUFK()
    
    # 数据增强
    augmentation = StrongAugmentation(
        position_noise=config.position_noise,
        angle_noise=config.angle_noise,
        scale_range=config.scale_range,
        rotation_perturb=config.rotation_perturb
    ) if config.use_augmentation else None
    
    # 损失函数
    mse_criterion = nn.MSELoss()
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.initial_lr,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度 - warmup + cosine
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
    logging.info("=" * 80)
    logging.info(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'IK RMSE':>10} | {'FK RMSE':>10} | {'Train R²':>10} | {'Val R²':>10} | {'LR':>10}")
    logging.info("-" * 80)
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        
        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        train_ik_loss = 0.0
        train_fk_loss = 0.0
        
        all_train_preds = []
        all_train_targets = []
        
        for batch_human_pose, batch_target_angles in train_loader:
            batch_human_pose = batch_human_pose.to(config.device, non_blocking=True)
            batch_target_angles = batch_target_angles.to(config.device, non_blocking=True)
            
            batch_size = batch_human_pose.shape[0]
            
            # 数据增强
            if augmentation:
                batch_human_pose, batch_target_angles = augmentation(
                    batch_human_pose.clone(),
                    batch_target_angles.clone()
                )
            
            # 提取输入
            human_pos = batch_human_pose[:, :3]
            human_ori = batch_human_pose[:, 3:7]
            
            optimizer.zero_grad()
            
            # 前向
            pred_angles, info = model(human_pos, human_ori)
            
            # 计算损失
            ik_loss = mse_criterion(pred_angles, batch_target_angles)
            
            # FK验证
            with torch.no_grad():
                target_pos = gpu_fk.forward(batch_target_angles)
            pred_pos = gpu_fk.forward(pred_angles)
            fk_loss = mse_criterion(pred_pos, target_pos)
            
            # 耦合正则化
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
            
            # 收集用于计算R²
            all_train_preds.append(pred_angles.detach().cpu())
            all_train_targets.append(batch_target_angles.cpu())
        
        train_loss /= len(train_loader.dataset)
        train_ik_loss /= len(train_loader.dataset)
        train_fk_loss /= len(train_loader.dataset)
        
        # 计算训练R²
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
            for batch_human_pose, batch_target_angles in val_loader:
                batch_human_pose = batch_human_pose.to(config.device, non_blocking=True)
                batch_target_angles = batch_target_angles.to(config.device, non_blocking=True)
                
                batch_size = batch_human_pose.shape[0]
                
                human_pos = batch_human_pose[:, :3]
                human_ori = batch_human_pose[:, 3:7]
                
                pred_angles, info = model(human_pos, human_ori)
                
                ik_loss = mse_criterion(pred_angles, batch_target_angles)
                
                target_pos = gpu_fk.forward(batch_target_angles)
                pred_pos = gpu_fk.forward(pred_angles)
                fk_loss = mse_criterion(pred_pos, target_pos)
                
                total_loss = ik_loss + 0.5 * fk_loss
                
                val_loss += total_loss.item() * batch_size
                val_ik_loss += ik_loss.item() * batch_size
                val_fk_loss += fk_loss.item() * batch_size
                
                all_val_preds.append(pred_angles.cpu())
                all_val_targets.append(batch_target_angles.cpu())
        
        val_loss /= len(val_loader.dataset)
        val_ik_loss /= len(val_loader.dataset)
        val_fk_loss /= len(val_loader.dataset)
        
        # 计算验证R²
        all_val_preds = torch.cat(all_val_preds, dim=0)
        all_val_targets = torch.cat(all_val_targets, dim=0)
        val_r2 = calculate_r2(all_val_preds, all_val_targets)
        
        # 计算RMSE
        train_ik_rmse = np.sqrt(train_ik_loss)
        train_fk_rmse = np.sqrt(train_fk_loss)
        val_ik_rmse = np.sqrt(val_ik_loss)
        val_fk_rmse = np.sqrt(val_fk_loss)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 日志
        logging.info(
            f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
            f"{val_ik_rmse:>10.4f} | {val_fk_rmse:>10.4f} | "
            f"{train_r2:>10.4f} | {val_r2:>10.4f} | {current_lr:>10.6f}"
        )
        
        # 保存最优模型
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
                'config': config.__dict__
            }, checkpoint_path)
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= config.early_stopping_patience:
            logging.info(f"\n>>> 早停触发 (patience={config.early_stopping_patience})")
            break
    
    logging.info("=" * 80)
    logging.info("训练完成!")
    logging.info(f"最优验证损失: {best_val_loss:.6f} (epoch {best_epoch})")
    logging.info(f"模型保存至: {checkpoint_path}")


if __name__ == "__main__":
    train()
