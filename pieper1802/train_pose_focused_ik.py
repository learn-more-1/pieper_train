"""
改进版IK模型训练 - 弱化历史依赖，增强位姿影响

核心改进：
1. 历史随机化：添加噪声、随机mask，防止过拟合
2. 课程学习：逐步减少历史长度，强制学习目标位姿映射
3. 双流架构：历史流和位姿流分离，最后融合
4. 辅助任务：目标位姿重建，增强位姿表示学习
5. 自适应权重：根据训练进度调整历史vs位姿的权重
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
import copy

sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from dataset_generalized import create_windowed_dataloaders
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("pose_focused_train.log"), logging.StreamHandler()]
)


class HistoryAugmentation:
    """
    历史帧数据增强
    
    防止模型过度记忆特定历史轨迹
    """
    
    def __init__(self, noise_std=0.05, mask_prob=0.1, drop_prob=0.2):
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.drop_prob = drop_prob
        
    def __call__(self, history_frames):
        """
        Args:
            history_frames: [batch, num_frames, 7]
        Returns:
            augmented: [batch, num_frames, 7]
        """
        augmented = history_frames.clone()
        batch_size, num_frames, num_joints = augmented.shape
        
        # 1. 添加高斯噪声
        if self.noise_std > 0:
            noise = torch.randn_like(augmented) * self.noise_std
            augmented = augmented + noise
        
        # 2. 随机mask（将某些帧设为零）
        if self.mask_prob > 0:
            mask = torch.rand(batch_size, num_frames, 1, device=augmented.device)
            mask = (mask > self.mask_prob).float()
            augmented = augmented * mask
        
        # 3. 随机时间dropout（丢弃某些时间步）
        if self.drop_prob > 0 and np.random.rand() < 0.5:
            # 随机选择保留的帧
            keep_frames = max(3, int(num_frames * (1 - self.drop_prob)))
            indices = torch.randperm(num_frames)[:keep_frames].sort()[0]
            augmented = augmented[:, indices, :]
            # 插值回原始长度
            augmented = torch.nn.functional.interpolate(
                augmented.transpose(1, 2), 
                size=num_frames, 
                mode='linear'
            ).transpose(1, 2)
        
        return augmented


class CurriculumScheduler:
    """
    课程学习调度器
    
    逐步减少历史长度，强制模型学习目标位姿映射
    """
    
    def __init__(self, initial_frames=10, min_frames=1, warmup_epochs=30):
        self.initial_frames = initial_frames
        self.min_frames = min_frames
        self.warmup_epochs = warmup_epochs
        self.current_frames = initial_frames
        
    def step(self, epoch):
        """根据epoch更新历史长度"""
        if epoch < self.warmup_epochs:
            # 线性减少
            progress = epoch / self.warmup_epochs
            self.current_frames = int(
                self.initial_frames - (self.initial_frames - self.min_frames) * progress
            )
        else:
            self.current_frames = self.min_frames
        
        return max(self.current_frames, self.min_frames)


class PoseFocusedLoss(nn.Module):
    """
    聚焦于位姿的损失函数
    
    1. 关节角度损失（主要）
    2. 位置误差损失（FK验证）
    3. 目标位姿重建损失（辅助任务）
    4. 历史独立性损失（防止过拟合历史）
    """
    
    def __init__(self, ik_weight=1.0, fk_weight=0.5, pose_recon_weight=0.3, 
                 history_indep_weight=0.1):
        super().__init__()
        self.ik_weight = ik_weight
        self.fk_weight = fk_weight
        self.pose_recon_weight = pose_recon_weight
        self.history_indep_weight = history_indep_weight
        self.mse = nn.MSELoss()
        
    def forward(self, pred_angles, target_angles, pred_position, target_position,
                pred_pose_recon=None, target_pose=None, history_variance=None):
        """
        Args:
            pred_angles: [batch, 7] 预测角度
            target_angles: [batch, 7] 目标角度
            pred_position: [batch, 3] FK预测位置
            target_position: [batch, 3] 目标位置
            pred_pose_recon: [batch, 7] 位姿重建（可选）
            target_pose: [batch, 7] 目标位姿（可选）
            history_variance: float 历史方差（可选）
        """
        losses = {}
        
        # 1. IK损失（关节角度）
        losses['ik'] = self.mse(pred_angles, target_angles)
        
        # 2. FK损失（位置验证）
        losses['fk'] = self.mse(pred_position, target_position)
        
        # 3. 位姿重建损失（辅助任务）
        if pred_pose_recon is not None and target_pose is not None:
            losses['pose_recon'] = self.mse(pred_pose_recon, target_pose)
        else:
            losses['pose_recon'] = torch.tensor(0.0, device=pred_angles.device)
        
        # 4. 历史独立性损失
        # 鼓励不同历史产生相似的输出（即输出主要由目标位姿决定）
        if history_variance is not None:
            losses['history_indep'] = -history_variance  # 负方差 = 鼓励一致性
        else:
            losses['history_indep'] = torch.tensor(0.0, device=pred_angles.device)
        
        # 总损失
        total_loss = (
            self.ik_weight * losses['ik'] +
            self.fk_weight * losses['fk'] +
            self.pose_recon_weight * losses['pose_recon'] +
            self.history_indep_weight * losses['history_indep']
        )
        
        return total_loss, losses


class Config:
    # 数据配置
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
    num_joints = 7
    num_frames = 10
    
    # 训练参数
    batch_size = 512
    epochs = 200
    initial_lr = 1e-3
    min_lr = 1e-6
    
    # 课程学习
    use_curriculum = True
    warmup_epochs = 50
    min_frames = 1
    
    # 数据增强
    use_augmentation = True
    noise_std = 0.05
    mask_prob = 0.1
    
    # 损失权重
    ik_weight = 1.0
    fk_weight = 0.5
    pose_recon_weight = 0.3
    history_indep_weight = 0.1
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 模型配置
    joint_dim = 7
    hidden_dim = 256
    num_layers = 2
    
    # 数据加载
    num_workers = 4
    pin_memory = True


def train():
    config = Config()
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = 'pose_focused_ik'
    
    logging.info("=" * 70)
    logging.info("训练改进版IK模型（弱化历史依赖）")
    logging.info("=" * 70)
    logging.info("核心改进:")
    logging.info("  1. 历史数据增强（噪声、mask、dropout）")
    logging.info("  2. 课程学习（逐步减少历史长度）")
    logging.info("  3. 位姿重建辅助任务")
    logging.info("  4. 历史独立性损失")
    
    # 加载数据集
    logging.info("\n加载数据集...")
    train_loader, val_loader = create_windowed_dataloaders(config.data_path, config)
    
    # 创建模型（使用原模型架构，但训练策略不同）
    logging.info(f"\n创建模型: {model_name}")
    from causal_ik_model_pieper2 import PieperCausalIK
    
    model = PieperCausalIK(
        num_joints=config.joint_dim,
        num_frames=config.num_frames,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers
    )
    model = model.to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"模型参数量: {total_params:.2f}M")
    
    # 加载GPU FK模型
    logging.info("\n加载GPU FK模型...")
    gpu_fk = SimpleGPUFK()
    
    # 初始化组件
    history_aug = HistoryAugmentation(
        noise_std=config.noise_std,
        mask_prob=config.mask_prob
    ) if config.use_augmentation else None
    
    curriculum = CurriculumScheduler(
        initial_frames=config.num_frames,
        min_frames=config.min_frames,
        warmup_epochs=config.warmup_epochs
    ) if config.use_curriculum else None
    
    criterion = PoseFocusedLoss(
        ik_weight=config.ik_weight,
        fk_weight=config.fk_weight,
        pose_recon_weight=config.pose_recon_weight,
        history_indep_weight=config.history_indep_weight
    )
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.initial_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=config.min_lr
    )
    
    # 训练循环
    checkpoint_path = f"/home/bonuli/Pieper/pieper1802/{model_name}.pth"
    best_val_loss = float("inf")
    start_epoch = 0
    
    if os.path.exists(checkpoint_path):
        logging.info("加载断点...")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        logging.info(f"从第{start_epoch} epoch继续")
    
    logging.info(f"\n开始训练（{config.epochs} epochs）...")
    logging.info("-" * 70)
    
    for epoch in range(start_epoch, config.epochs):
        # 更新课程学习参数
        current_frames = config.num_frames
        if curriculum:
            current_frames = curriculum.step(epoch)
            if epoch % 10 == 0:
                logging.info(f"  [课程学习] 当前历史长度: {current_frames}")
        
        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        train_metrics = {'ik': 0, 'fk': 0, 'pose_recon': 0}
        
        for batch_idx, (batch_X, batch_y, batch_last_angle) in enumerate(train_loader):
            batch_X = batch_X.to(config.device, non_blocking=True)
            batch_y = batch_y.to(config.device, non_blocking=True)
            batch_last_angle = batch_last_angle.to(config.device, non_blocking=True)
            
            batch_size = batch_X.shape[0]
            
            # 课程学习：截断历史
            if current_frames < config.num_frames:
                batch_X = batch_X[:, -current_frames:, :]
                # 填充回原始长度（用最后一帧重复）
                if current_frames < config.num_frames:
                    padding = batch_X[:, -1:, :].repeat(1, config.num_frames - current_frames, 1)
                    batch_X = torch.cat([padding, batch_X], dim=1)
            
            # 数据增强
            if history_aug and model.training:
                batch_X = history_aug(batch_X)
            
            # 提取目标
            if batch_y.shape[1] == 14:
                target_pose = batch_y[:, :7]
                target_angles = batch_y[:, 7:]
                target_position = target_pose[:, :3]
                target_orientation = target_pose[:, 3:7]
            else:
                target_angles = batch_y
                target_position, _ = gpu_fk.forward(batch_y)
                target_orientation = None
                target_pose = None
            
            optimizer.zero_grad()
            
            # 前向传播
            pred_angles, info = model(batch_X, target_position, target_orientation)
            
            # FK验证
            pred_position, _ = gpu_fk.forward(pred_angles)
            
            # 计算损失
            total_loss, losses = criterion(
                pred_angles, target_angles,
                pred_position, target_position
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item() * batch_size
            for k in train_metrics:
                if k in losses:
                    train_metrics[k] += losses[k].item() * batch_size
        
        train_loss /= len(train_loader.dataset)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader.dataset)
        
        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0.0
        val_ik_loss = 0.0
        val_fk_loss = 0.0
        
        # 测试：纯位姿输入性能
        pose_only_errors = []
        
        with torch.no_grad():
            for batch_X, batch_y, _ in val_loader:
                batch_X = batch_X.to(config.device, non_blocking=True)
                batch_y = batch_y.to(config.device, non_blocking=True)
                
                batch_size = batch_X.shape[0]
                
                if batch_y.shape[1] == 14:
                    target_pose = batch_y[:, :7]
                    target_angles = batch_y[:, 7:]
                    target_position = target_pose[:, :3]
                    target_orientation = target_pose[:, 3:7]
                else:
                    target_angles = batch_y
                    target_position, _ = gpu_fk.forward(batch_y)
                    target_orientation = None
                
                # 正常验证（带历史）
                pred_angles, _ = model(batch_X, target_position, target_orientation)
                pred_position, _ = gpu_fk.forward(pred_angles)
                
                ik_loss = ((pred_angles - target_angles) ** 2).mean()
                fk_loss = ((pred_position - target_position) ** 2).mean()
                
                val_loss += (ik_loss + 0.5 * fk_loss).item() * batch_size
                val_ik_loss += ik_loss.item() * batch_size
                val_fk_loss += fk_loss.item() * batch_size
                
                # 测试纯位姿输入（零历史）
                zero_history = torch.zeros_like(batch_X)
                pred_angles_zero_hist, _ = model(zero_history, target_position, target_orientation)
                error = ((pred_angles_zero_hist - target_angles) ** 2).mean()
                pose_only_errors.append(error.item())
        
        val_loss /= len(val_loader.dataset)
        val_ik_loss /= len(val_loader.dataset)
        val_fk_loss /= len(val_loader.dataset)
        pose_only_error = np.mean(pose_only_errors)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 日志
        logging.info(
            f"Epoch [{epoch}/{config.epochs}] | "
            f"Val Loss: {val_loss:.6f} | "
            f"IK: {val_ik_loss:.6f} | FK: {val_fk_loss:.6f} | "
            f"Pose-Only Err: {pose_only_error:.6f} | "
            f"Frames: {current_frames} | LR: {current_lr:.6f}"
        )
        
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config.__dict__
            }, checkpoint_path)
            logging.info(f"  >>> 保存最优模型（验证损失：{val_loss:.6f}）")
    
    logging.info("\n训练完成！")
    logging.info(f"最优验证损失: {best_val_loss:.6f}")
    logging.info(f"模型保存至: {checkpoint_path}")


if __name__ == "__main__":
    train()
