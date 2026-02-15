"""
ACCAD/CMU合并数据集（新格式）

数据格式：
- X: (N, 10, 14) - 10帧历史，每帧7维位姿+7维角度
- y: (N, 14) - 目标帧的7维位姿+7维角度
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging


class ACCAD_CMU_Dataset(Dataset):
    """
    ACCAD/CMU合并数据集

    返回格式:
        X: (window_size, 7) - 历史帧的机器人关节角度
        y: (7,) - 目标帧的机器人关节角度
        last_angle: (7,) - 历史最后一帧的机器人关节角度（连续性约束）
        target_pose: (7,) - 目标帧的人臂位姿（位置3维+姿态4维）
    """

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

        # 加载数据
        logging.info(f"加载ACCAD/CMU数据集: {data_path}")
        data = np.load(data_path, allow_pickle=True)

        self.X_full = data['X']  # (N, window_size, 14) - 位姿+角度
        self.y_full = data['y']  # (N, 14) - 目标帧位姿+角度

        # 提取机器人角度作为输入
        self.X = self.X_full[:, :, 7:]  # (N, window_size, 7) - 只用角度
        self.y = self.y_full[:, 7:]      # (N, 7) - 目标帧角度

        self.window_size = self.X.shape[1]
        self.num_samples = self.X.shape[0]

        logging.info(f"  样本数: {self.num_samples}")
        logging.info(f"  窗口大小: {self.window_size}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        返回:
            X: (window_size, 7) - 历史帧的机器人关节角度
            y: (7,) - 目标帧的机器人关节角度
            last_angle: (7,) - 历史最后一帧的机器人关节角度
            target_pose: (7,) - 目标帧的人臂位姿（位置3维+姿态4维）
        """
        X = self.X[idx].astype(np.float32)  # (window_size, 7)
        y = self.y[idx].astype(np.float32)  # (7,)
        last_angle = self.X[idx, -1, :].astype(np.float32)  # (7,) - 最后一帧角度

        # 提取目标帧的人臂位姿（从y的前7维）
        target_pose = self.y_full[idx, :7].astype(np.float32)  # (7,) - 人臂位姿

        return (
            torch.from_numpy(X),
            torch.from_numpy(y),
            torch.from_numpy(last_angle),
            torch.from_numpy(target_pose)
        )


def create_accad_cmu_dataloaders(data_path, config):
    """
    创建ACCAD/CMU数据加载器

    Args:
        data_path: 数据集路径
        config: 配置对象（需要包含batch_size, num_workers, pin_memory）

    Returns:
        train_loader, val_loader
    """
    # 创建数据集
    dataset = ACCAD_CMU_Dataset(data_path)

    # 分割训练集和验证集
    from torch.utils.data import random_split

    total = len(dataset)
    train_size = int(0.9 * total)
    val_size = total - train_size

    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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


# 测试
if __name__ == '__main__':
    data_path = "/home/wsy/Desktop/casual/ACCAD_CMU_merged_training_data.npz"

    dataset = ACCAD_CMU_Dataset(data_path)

    # 测试一个样本
    X, y, last_angle, target_pose = dataset[0]

    print("=" * 70)
    print("测试ACCAD/CMU数据集")
    print("=" * 70)
    print(f"\n返回值:")
    print(f"  X shape: {X.shape} - 历史帧机器人角度")
    print(f"  y shape: {y.shape} - 目标帧机器人角度")
    print(f"  last_angle shape: {last_angle.shape} - 最后一帧机器人角度")
    print(f"  target_pose shape: {target_pose.shape} - 目标帧人臂位姿")

    print(f"\n第一个样本:")
    print(f"  目标人臂位姿: {target_pose.numpy()}")
    print(f"    - 位置: {target_pose[:3].numpy()}")
    print(f"    - 姿态: {target_pose[3:].numpy()}")
    print(f"  目标机器人角度: {y.numpy()}")
    print(f"  历史最后一帧角度: {last_angle.numpy()}")

    print(f"\n✓ 数据集加载成功！")
    print("=" * 70)
