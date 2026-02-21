"""
隐式神经场IK模型（支持历史位姿版本）

核心特性：
1. 使用历史位姿序列（10帧）代替当前关节角条件
2. 避免自回归误差累积
3. 捕捉运动趋势和动态信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model, max_freq=10):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq

    def forward(self, x):
        """
        Args:
            x: [batch, dim] 输入
        Returns:
            encoded: [batch, dim * (2 * max_freq + 1)] 编码后的特征
        """
        batch_size = x.shape[0]
        device = x.device

        # 生成不同频率
        freqs = 2.0 ** torch.arange(0, self.max_freq, device=device)
        # [batch, dim, max_freq]
        x_expanded = x.unsqueeze(-1) * freqs.view(1, 1, -1)
        # 正弦和余弦
        sin_features = torch.sin(x_expanded)  # [batch, dim, max_freq]
        cos_features = torch.cos(x_expanded)  # [batch, dim, max_freq]

        # 拼接: [batch, dim, 2*max_freq]
        features = torch.cat([sin_features, cos_features], dim=-1)
        # 展平: [batch, dim * 2 * max_freq]
        features = features.reshape(batch_size, -1)

        # 拼接原始输入
        encoded = torch.cat([x, features], dim=-1)

        return encoded


class TemporalPoseEncoder(nn.Module):
    """时序位姿编码器 - 处理历史位姿序列"""

    def __init__(self, pose_dim=7, hidden_dim=256, num_frames=10):
        super().__init__()
        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames

        # 1D卷积提取时序特征
        self.conv_layers = nn.Sequential(
            nn.Conv1d(pose_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, history_poses):
        """
        Args:
            history_poses: [batch, num_frames, pose_dim] 历史位姿序列

        Returns:
            temporal_feature: [batch, hidden_dim] 时序特征
        """
        batch_size = history_poses.shape[0]

        # 转置: [batch, pose_dim, num_frames]
        x = history_poses.transpose(1, 2)

        # 卷积特征提取
        conv_out = self.conv_layers(x)  # [batch, hidden_dim, 1]

        # 压缩: [batch, hidden_dim]
        conv_out = conv_out.squeeze(-1)

        # 投影
        temporal_feature = self.output_proj(conv_out)

        return temporal_feature


class ImplicitIKWithHistory(nn.Module):
    """
    带历史位姿的隐式IK模型

    输入:
        - target_pose: [batch, pose_dim] 目标位姿
        - history_poses: [batch, num_frames, pose_dim] 历史位姿序列

    输出:
        - joint_angles: [batch, joint_dim] 预测的关节角度
    """

    def __init__(self, pose_dim=7, joint_dim=7, hidden_dim=1000,
                 num_freqs=10, num_frames=10, temporal_hidden=256):
        super().__init__()

        self.pose_dim = pose_dim
        self.joint_dim = joint_dim
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames

        # 目标位姿编码器
        self.pose_encoder = PositionalEncoding(pose_dim, num_freqs)
        pose_encoded_dim = pose_dim * (2 * num_freqs + 1)

        # 时序位姿编码器
        self.temporal_encoder = TemporalPoseEncoder(pose_dim, temporal_hidden, num_frames)

        # 输入维度 = 目标位姿编码 + 时序特征
        input_dim = pose_encoded_dim + temporal_hidden

        # 主网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in [hidden_dim, hidden_dim // 2, hidden_dim // 4]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, joint_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, target_pose, history_poses):
        """
        Args:
            target_pose: [batch, pose_dim] 目标位姿
            history_poses: [batch, num_frames, pose_dim] 历史位姿序列

        Returns:
            joint_angles: [batch, joint_dim] 预测的关节角度
        """
        # 编码目标位姿
        pose_encoded = self.pose_encoder(target_pose)

        # 编码历史位姿
        temporal_feature = self.temporal_encoder(history_poses)

        # 拼接特征
        features = torch.cat([pose_encoded, temporal_feature], dim=-1)

        # 预测关节角
        joint_angles = self.network(features)

        return joint_angles


class ImplicitIKWithHistoryEnsemble(nn.Module):
    """
    集成版本 - 多个模型取平均
    """

    def __init__(self, pose_dim=7, joint_dim=7, hidden_dim=1000,
                 num_freqs=10, num_frames=10, temporal_hidden=256, num_models=5):
        super().__init__()

        self.num_models = num_models
        self.models = nn.ModuleList([
            ImplicitIKWithHistory(pose_dim, joint_dim, hidden_dim, num_freqs, num_frames, temporal_hidden)
            for _ in range(num_models)
        ])

    def forward(self, target_pose, history_poses):
        """
        Args:
            target_pose: [batch, pose_dim] 目标位姿
            history_poses: [batch, num_frames, pose_dim] 历史位姿序列

        Returns:
            joint_angles: [batch, joint_dim] 平均预测
        """
        predictions = []
        for model in self.models:
            pred = model(target_pose, history_poses)
            predictions.append(pred)

        # 平均
        joint_angles = torch.stack(predictions, dim=0).mean(dim=0)

        return joint_angles


class NormalizationLayer(nn.Module):
    """归一化层"""

    def __init__(self, pose_mean, pose_std, joint_mean, joint_std):
        super().__init__()
        self.register_buffer('pose_mean', torch.tensor(pose_mean))
        self.register_buffer('pose_std', torch.tensor(pose_std))
        self.register_buffer('joint_mean', torch.tensor(joint_mean))
        self.register_buffer('joint_std', torch.tensor(joint_std))

    def normalize_pose(self, pose):
        return (pose - self.pose_mean) / (self.pose_std + 1e-8)

    def normalize_history_poses(self, history_poses):
        """归一化历史位姿序列"""
        return (history_poses - self.pose_mean) / (self.pose_std + 1e-8)

    def denormalize_joint(self, joint):
        return joint * self.joint_std + self.joint_mean


def compute_dataset_statistics(dataloader, pose_dim, joint_dim):
    """计算数据集统计信息"""
    print("计算数据集统计信息...")

    pose_sum = 0
    pose_sq_sum = 0
    joint_sum = 0
    joint_sq_sum = 0
    count = 0

    for batch_X, batch_y, _ in dataloader:
        batch_size = batch_X.shape[0]

        # 提取位姿和关节角
        if batch_y.shape[1] == 14:
            poses = batch_y[:, :pose_dim]
            joints = batch_y[:, pose_dim:pose_dim + joint_dim]
        else:
            # y只包含关节角，从X提取历史位姿
            poses = batch_X[:, -1, :pose_dim]
            joints = batch_y

        pose_sum += poses.sum(dim=0)
        pose_sq_sum += (poses ** 2).sum(dim=0)
        joint_sum += joints.sum(dim=0)
        joint_sq_sum += (joints ** 2).sum(dim=0)
        count += batch_size

    pose_mean = pose_sum / count
    pose_std = torch.sqrt(pose_sq_sum / count - pose_mean ** 2)
    joint_mean = joint_sum / count
    joint_std = torch.sqrt(joint_sq_sum / count - joint_mean ** 2)

    print(f"位姿均值: {pose_mean}")
    print(f"位姿标准差: {pose_std}")
    print(f"关节均值: {joint_mean}")
    print(f"关节标准差: {joint_std}")

    return pose_mean, pose_std, joint_mean, joint_std
