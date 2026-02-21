"""
对比学习风格 IK 专用数据集

关键修改：返回完整的 X_full (14维) 而不是 X (7维)
使得可以同时获取 pose 历史和 joint 历史
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging


class ContrastiveIKDataset(Dataset):
    """
    对比学习风格 IK 数据集
    
    与 WindowedIKDataset 的区别：
    - 返回 X_full [window_size, 14] 而不是 X [window_size, 7]
    - 包含人位姿历史 (7维) 和机器人关节历史 (7维)
    """
    
    def __init__(self, data_path, use_augmentation=False, augmentation_level='light'):
        """
        Args:
            data_path: 数据集路径 (包含X, y的npz文件)
            use_augmentation: 是否使用数据增强
            augmentation_level: 增强级别 ('light', 'moderate', 'none')
        """
        super().__init__()
        self.data_path = data_path
        self.use_augmentation = use_augmentation
        self.augmentation_level = augmentation_level
        
        # 加载数据
        logging.info(f"加载对比学习数据集: {data_path}")
        data = np.load(data_path, allow_pickle=True)
        
        # X_full: (N, window_size, 14) - 前7维人位姿 + 后7维机器人角度
        self.X_full = data['X'].astype(np.float32)
        self.y = data['y'].astype(np.float32)
        
        self.window_size = self.X_full.shape[1]
        self.num_samples = self.X_full.shape[0]
        
        # 检查y的格式
        self.y_has_pose = (self.y.shape[1] == 14)
        
        logging.info(f"  样本数: {self.num_samples}")
        logging.info(f"  窗口大小: {self.window_size}")
        logging.info(f"  X_full 维度（人位姿+机器人角度）: {self.X_full.shape[2]}")
        logging.info(f"  y 有 pose: {self.y_has_pose}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        返回:
            X: (window_size, 14) - 人位姿历史(7) + 机器人角度历史(7)
            y: (14,) or (7,) - 目标位姿+角度 或 仅角度
            last_angle: (7,) - 最后一帧的机器人关节角度
        """
        X = self.X_full[idx]  # (window_size, 14) - 完整历史
        y = self.y[idx]       # (14,) or (7,)
        
        # 最后一帧的机器人角度
        last_angle = self.X_full[idx, -1, 7:]  # (7,)
        
        # 数据增强
        if self.use_augmentation and self.augmentation_level != 'none':
            X, y = self._apply_augmentation(X, y)
        
        return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(last_angle)
    
    def _apply_augmentation(self, X, y):
        """应用数据增强"""
        if self.augmentation_level == 'light':
            # 添加高斯噪声
            noise_X = np.random.normal(0, 0.01, X.shape).astype(np.float32)
            X = X + 0.1 * noise_X
            
            if len(y) == 14:
                noise_y = np.random.normal(0, 0.01, y.shape).astype(np.float32)
                y = y + 0.05 * noise_y  # 目标噪声更小
        
        elif self.augmentation_level == 'moderate':
            # 时序抖动
            if np.random.rand() < 0.3:
                t = np.random.randint(1, self.window_size - 1)
                noise_t = np.random.randn(*X[t].shape).astype(np.float32) * 0.05
                X[t] = X[t] + noise_t
        
        return X, y


def create_contrastive_dataloaders(data_path, config):
    """
    创建对比学习风格 IK 的数据加载器
    
    Args:
        data_path: 数据集路径 (.npz 文件)
        config: 配置对象，需要包含:
            - batch_size
            - use_augmentation (可选)
            - augmentation_level (可选)
            - num_workers (可选)
            - pin_memory (可选)
    
    Returns:
        train_loader, val_loader
    """
    # 创建训练集
    train_dataset = ContrastiveIKDataset(
        data_path,
        use_augmentation=getattr(config, 'use_augmentation', False),
        augmentation_level=getattr(config, 'augmentation_level', 'light')
    )
    
    # 创建验证集
    val_dataset = ContrastiveIKDataset(
        data_path,
        use_augmentation=False,
        augmentation_level='none'
    )
    
    # 分割训练集和验证集
    from torch.utils.data import random_split
    
    total = len(train_dataset)
    train_size = int(0.9 * total)
    val_size = total - train_size
    
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=getattr(config, 'num_workers', 4),
        pin_memory=getattr(config, 'pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, 'num_workers', 4),
        pin_memory=getattr(config, 'pin_memory', True)
    )
    
    logging.info(f"训练集样本数: {train_size}")
    logging.info(f"验证集样本数: {val_size}")
    
    return train_loader, val_loader
