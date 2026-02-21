"""
对比学习风格 IK 模型 (2103版本)

核心特性:
1. 训练时用 history_joints 监督风格编码器
2. 推理时只用 history_poses 推断风格
3. 通过对比学习对齐两种风格表示
4. 零自回归，保持泛化性

架构:
- PoseStyleEncoder: 从末端轨迹推断风格 (学生，推理用)
- JointStyleEncoder: 从关节历史提取风格 (教师，训练用)
- 对比损失: 让学生的输出接近教师
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model, max_freq=15):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq
    
    def forward(self, x):
        """
        Args:
            x: [batch, dim]
        Returns:
            encoded: [batch, dim * (2 * max_freq + 1)]
        """
        batch_size = x.shape[0]
        device = x.device
        
        freqs = 2.0 ** torch.arange(0, self.max_freq, device=device)
        x_expanded = x.unsqueeze(-1) * freqs.view(1, 1, -1)
        sin_features = torch.sin(x_expanded)
        cos_features = torch.cos(x_expanded)
        features = torch.cat([sin_features, cos_features], dim=-1)
        features = features.reshape(batch_size, -1)
        encoded = torch.cat([x, features], dim=-1)
        
        return encoded


class TemporalPoseEncoder(nn.Module):
    """时序位姿编码器 - 处理末端轨迹"""
    
    def __init__(self, pose_dim=7, hidden_dim=256, num_frames=10):
        super().__init__()
        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(pose_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, history_poses):
        """
        Args:
            history_poses: [batch, num_frames, pose_dim]
        Returns:
            feature: [batch, hidden_dim]
        """
        x = history_poses.transpose(1, 2)  # [B, pose_dim, num_frames]
        conv_out = self.conv_layers(x).squeeze(-1)  # [B, hidden_dim]
        return self.output_proj(conv_out)


class ContrastiveStyleIK(nn.Module):
    """
    对比学习风格 IK 模型
    
    训练: 用 history_joints 监督 history_poses 提取的风格
    推理: 只用 history_poses 推断风格，无自回归
    """
    
    def __init__(self,
                 pose_dim=7,
                 joint_dim=7,
                 hidden_dim=1200,
                 temporal_hidden=256,
                 num_freqs=15,
                 num_frames=10,
                 style_dim=128,
                 use_projection=True,
                 temperature=0.07):
        super().__init__()
        
        self.pose_dim = pose_dim
        self.joint_dim = joint_dim
        self.num_frames = num_frames
        self.style_dim = style_dim
        self.temperature = temperature
        self.use_projection = use_projection
        
        # ========== 1. 基础编码器 ==========
        self.pose_encoder = PositionalEncoding(pose_dim, num_freqs)
        pose_encoded_dim = pose_dim * (2 * num_freqs + 1)
        
        self.temporal_encoder = TemporalPoseEncoder(pose_dim, temporal_hidden, num_frames)
        
        # ========== 2. 风格编码器（核心）==========
        
        # 2a. Pose风格编码器 - 学生网络（推理用）
        self.pose_style_encoder = nn.Sequential(
            nn.Linear(temporal_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, style_dim)
        )
        
        # 2b. Joint风格编码器 - 教师网络（训练用）
        # 注意：这个网络在推理时不用，只用于训练监督
        self.joint_style_encoder = nn.Sequential(
            nn.Flatten(),  # [B, 10, 7] -> [B, 70]
            nn.Linear(joint_dim * num_frames, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, style_dim)
        )
        
        # 2c. 投影头（对比学习用）
        if use_projection:
            self.style_projector = nn.Sequential(
                nn.Linear(style_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            )
        else:
            self.style_projector = nn.Identity()
        
        # ========== 3. 生成网络 ==========
        # 输入：目标位姿编码 + 时序特征 + 风格
        input_dim = pose_encoded_dim + temporal_hidden + style_dim
        
        self.generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, joint_dim)
        )
        
        # 风格残差 - 让风格影响更直接
        self.style_residual = nn.Sequential(
            nn.Linear(style_dim, 64),
            nn.ReLU(),
            nn.Linear(64, joint_dim),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def extract_pose_style(self, history_poses):
        """
        从末端轨迹提取风格 - 推理时使用
        
        Args:
            history_poses: [B, num_frames, pose_dim]
        Returns:
            style: [B, style_dim]
        """
        temporal_feat = self.temporal_encoder(history_poses)
        style = self.pose_style_encoder(temporal_feat)
        return style
    
    def extract_joint_style(self, history_joints):
        """
        从关节历史提取风格 - 训练时使用
        
        Args:
            history_joints: [B, num_frames, joint_dim]
        Returns:
            style: [B, style_dim]
        """
        style = self.joint_style_encoder(history_joints)
        return style
    
    def forward(self, target_pose, history_poses, history_joints=None,
                mode='inference', return_aux=False):
        """
        前向传播
        
        Args:
            target_pose: [B, pose_dim] 目标位姿
            history_poses: [B, num_frames, pose_dim] 末端历史
            history_joints: [B, num_frames, joint_dim] 关节历史（仅训练）
            mode: 'inference' 或 'training'
            return_aux: 是否返回辅助信息
        
        Returns:
            pred: [B, joint_dim] 预测关节角
            aux: dict (optional) 包含风格向量等
        """
        B = target_pose.shape[0]
        
        # 基础特征
        pose_feat = self.pose_encoder(target_pose)  # [B, 217]
        temporal_feat = self.temporal_encoder(history_poses)  # [B, 256]
        
        # 风格提取
        if mode == 'training' and history_joints is not None:
            # 训练模式：两种风格
            pred_style = self.extract_pose_style(history_poses)
            true_style = self.extract_joint_style(history_joints)
            
            # 使用教师风格进行生成（训练更稳定）
            style = true_style
        else:
            # 推理模式：只用 poses
            pred_style = self.extract_pose_style(history_poses)
            true_style = None
            style = pred_style
        
        # 生成
        combined = torch.cat([pose_feat, temporal_feat, style], dim=-1)
        base_pred = self.generator(combined)
        
        # 风格残差
        style_adj = self.style_residual(style) * 0.1
        pred = base_pred + style_adj
        
        if return_aux or mode == 'training':
            aux = {
                'pred_style': pred_style,
                'true_style': true_style,
                'style': style,
                'base_pred': base_pred,
                'temporal_feat': temporal_feat
            }
            return pred, aux
        
        return pred
    
    def compute_contrastive_loss(self, pred_style, true_style):
        """
        计算对比损失 - 让 pred_style 接近 true_style
        
        使用 InfoNCE Loss（对比学习标准）
        """
        if true_style is None:
            return torch.tensor(0.0, device=pred_style.device)
        
        # 投影
        z_i = self.style_projector(pred_style)   # 学生 [B, 64]
        z_j = self.style_projector(true_style)   # 教师 [B, 64]
        
        # 归一化
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # InfoNCE: batch内其他样本作为负样本
        batch_size = z_i.shape[0]
        
        # 相似度矩阵 [B, B]
        similarity = torch.mm(z_i, z_j.t()) / self.temperature
        
        # 正样本在对角线上
        labels = torch.arange(batch_size, device=z_i.device)
        
        # 双向损失
        loss_i2j = F.cross_entropy(similarity, labels)
        loss_j2i = F.cross_entropy(similarity.t(), labels)
        
        return (loss_i2j + loss_j2i) / 2
    
    def compute_mse_alignment_loss(self, pred_style, true_style):
        """
        简单的 MSE 对齐（备用方案）
        """
        if true_style is None:
            return torch.tensor(0.0, device=pred_style.device)
        
        # 投影后计算 MSE
        z_i = self.style_projector(pred_style)
        z_j = self.style_projector(true_style)
        
        return F.mse_loss(z_i, z_j.detach())


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
    
    def normalize_joints(self, joints):
        """归一化关节角"""
        return (joints - self.joint_mean) / (self.joint_std + 1e-8)
    
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
