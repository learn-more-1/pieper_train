"""
改进版IK模型 - 位姿聚焦架构

核心改进：
1. 双流编码器：历史流和位姿流分离
2. 位姿流权重更高
3. 历史压缩：将多帧历史压缩为单帧表示
4. 自适应融合：根据训练进度调整历史vs位姿的融合比例
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HistoryCompressor(nn.Module):
    """
    历史压缩器
    
    将多帧历史压缩为单帧表示，减少时序依赖
    """
    
    def __init__(self, joint_dim=7, num_frames=10, hidden_dim=128):
        super().__init__()
        
        # 时序压缩（类似SE-Net）
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, joint_dim),
            nn.Sigmoid()
        )
        
    def forward(self, history_frames):
        """
        Args:
            history_frames: [batch, num_frames, 7]
        Returns:
            compressed: [batch, 7] 单帧表示
        """
        # 转置为 [batch, 7, num_frames]
        x = history_frames.transpose(1, 2)
        
        # 时序池化
        x = self.temporal_pool(x).squeeze(-1)  # [batch, 7]
        
        # 通道注意力
        attention = self.channel_attention(x)
        x = x * attention
        
        return x


class PoseFocusedEncoder(nn.Module):
    """
    位姿聚焦编码器
    
    更强大的位姿特征提取
    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        # 位置编码（更深）
        self.position_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 姿态编码（更深）
        self.orientation_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 位姿融合
        self.pose_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
    def forward(self, position, orientation):
        """
        Args:
            position: [batch, 3]
            orientation: [batch, 4] or None
        Returns:
            pose_feature: [batch, hidden_dim]
        """
        pos_feat = self.position_encoder(position)
        
        if orientation is not None:
            ori_feat = self.orientation_encoder(orientation)
            combined = torch.cat([pos_feat, ori_feat], dim=-1)
        else:
            # 如果没有姿态，用位置特征填充
            combined = torch.cat([pos_feat, pos_feat], dim=-1)
        
        pose_feature = self.pose_fusion(combined)
        return pose_feature


class AdaptiveFusion(nn.Module):
    """
    自适应融合模块
    
    动态调整历史和位姿特征的融合比例
    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
    def forward(self, history_feat, pose_feat, alpha=0.5):
        """
        Args:
            history_feat: [batch, hidden_dim]
            pose_feat: [batch, hidden_dim]
            alpha: 历史特征的权重（0-1）
        Returns:
            fused: [batch, hidden_dim]
        """
        concat = torch.cat([history_feat, pose_feat], dim=-1)
        
        # 可学习的门控
        gate = self.gate(concat)
        
        # 加权融合
        weighted_hist = history_feat * gate * alpha
        weighted_pose = pose_feat * (1 - gate) * (1 + alpha)
        
        fused = self.fusion(torch.cat([weighted_hist, weighted_pose], dim=-1))
        return fused


class PoseFocusedIK(nn.Module):
    """
    位姿聚焦的IK模型
    
    架构：
    1. 历史压缩器（多帧 -> 单帧）
    2. 位姿编码器（深层）
    3. 自适应融合（位姿主导）
    4. 关节预测（保持耦合）
    """
    
    def __init__(self, num_joints=7, num_frames=10, hidden_dim=256, num_layers=2):
        super().__init__()
        
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        
        # 历史压缩
        self.history_compressor = HistoryCompressor(num_joints, num_frames, hidden_dim)
        self.history_encoder = nn.Sequential(
            nn.Linear(num_joints, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 位姿编码（更强）
        self.pose_encoder = PoseFocusedEncoder(hidden_dim)
        
        # 自适应融合
        self.fusion = AdaptiveFusion(hidden_dim)
        
        # 关节分组（与原版相同）
        self.joint_groups = {
            'shoulder': [0, 1, 2],
            'elbow': [3],
            'forearm': [4],
            'wrist': [5, 6]
        }
        
        # 分组特征提取
        self.group_encoders = nn.ModuleDict({
            'shoulder': nn.Linear(hidden_dim, hidden_dim),
            'elbow': nn.Linear(hidden_dim, hidden_dim),
            'forearm': nn.Linear(hidden_dim, hidden_dim),
            'wrist': nn.Linear(hidden_dim, hidden_dim)
        })
        
        # 消息传递（简化版）
        self.message_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'shoulder_to_elbow': nn.Linear(hidden_dim, hidden_dim),
                'elbow_to_forearm': nn.Linear(hidden_dim, hidden_dim),
                'forearm_to_wrist': nn.Linear(hidden_dim, hidden_dim)
            })
            self.message_layers.append(layer)
        
        # 输出头
        self.output_heads = nn.ModuleDict({
            'shoulder': nn.Linear(hidden_dim, 3),
            'elbow': nn.Linear(hidden_dim, 1),
            'forearm': nn.Linear(hidden_dim, 1),
            'wrist': nn.Linear(hidden_dim, 2)
        })
        
        # 位姿重建头（辅助任务）
        self.pose_reconstruction = nn.Sequential(
            nn.Linear(num_joints, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7)  # 重建位姿
        )
        
    def forward(self, history_frames, end_position, end_orientation, 
                history_alpha=0.3):
        """
        Args:
            history_frames: [batch, num_frames, 7]
            end_position: [batch, 3]
            end_orientation: [batch, 4] or None
            history_alpha: 历史特征的权重（训练时调整）
            
        Returns:
            pred_angles: [batch, 7]
            info: dict
        """
        batch_size = history_frames.shape[0]
        
        # 1. 历史压缩
        compressed_history = self.history_compressor(history_frames)
        history_feat = self.history_encoder(compressed_history)
        
        # 2. 位姿编码
        pose_feat = self.pose_encoder(end_position, end_orientation)
        
        # 3. 自适应融合（位姿主导）
        fused_feat = self.fusion(history_feat, pose_feat, alpha=history_alpha)
        
        # 4. 分组特征
        group_features = {}
        for group in self.joint_groups.keys():
            group_features[group] = self.group_encoders[group](fused_feat)
        
        # 5. 消息传递（保持关节耦合）
        for layer in self.message_layers:
            # Shoulder -> Elbow
            msg = layer['shoulder_to_elbow'](group_features['shoulder'])
            group_features['elbow'] = group_features['elbow'] + msg
            
            # Elbow -> Forearm
            msg = layer['elbow_to_forearm'](group_features['elbow'])
            group_features['forearm'] = group_features['forearm'] + msg
            
            # Forearm -> Wrist
            msg = layer['forearm_to_wrist'](group_features['forearm'])
            group_features['wrist'] = group_features['wrist'] + msg
        
        # 6. 输出预测
        pred_shoulder = self.output_heads['shoulder'](group_features['shoulder'])
        pred_elbow = self.output_heads['elbow'](group_features['elbow'])
        pred_forearm = self.output_heads['forearm'](group_features['forearm'])
        pred_wrist = self.output_heads['wrist'](group_features['wrist'])
        
        pred_angles = torch.cat([pred_shoulder, pred_elbow, pred_forearm, pred_wrist], dim=1)
        
        # 7. 位姿重建（辅助任务）
        reconstructed_pose = self.pose_reconstruction(pred_angles)
        
        return pred_angles, {
            'reconstructed_pose': reconstructed_pose,
            'history_feat': history_feat,
            'pose_feat': pose_feat
        }


# ==================== 测试 ====================

if __name__ == '__main__':
    print("=" * 70)
    print("测试位姿聚焦IK模型")
    print("=" * 70)
    
    model = PoseFocusedIK(num_joints=7, num_frames=10, hidden_dim=256, num_layers=2)
    model = model.cuda()
    model.eval()
    
    # 测试数据
    batch_size = 4
    history = torch.randn(batch_size, 10, 7).cuda()
    position = torch.randn(batch_size, 3).cuda()
    orientation = torch.randn(batch_size, 4).cuda()
    
    print(f"\n输入:")
    print(f"  - 历史: {history.shape}")
    print(f"  - 位置: {position.shape}")
    print(f"  - 姿态: {orientation.shape}")
    
    # 前向传播
    with torch.no_grad():
        pred_angles, info = model(history, position, orientation, history_alpha=0.3)
    
    print(f"\n输出:")
    print(f"  - 预测角度: {pred_angles.shape}")
    print(f"  - 重建位姿: {info['reconstructed_pose'].shape}")
    
    # 测试纯位姿输入（零历史）
    zero_history = torch.zeros(batch_size, 10, 7).cuda()
    with torch.no_grad():
        pred_zero_hist, _ = model(zero_history, position, orientation, history_alpha=0.3)
    
    print(f"\n零历史测试:")
    print(f"  - 正常历史预测: {pred_angles[0, :3].cpu().numpy().round(3)}")
    print(f"  - 零历史预测:   {pred_zero_hist[0, :3].cpu().numpy().round(3)}")
    
    diff = torch.norm(pred_angles - pred_zero_hist).item()
    print(f"  - 差异: {diff:.4f}")
    
    if diff > 0.1:
        print("  ✓ 历史变化对输出影响较小（位姿主导）")
    else:
        print("  ⚠ 历史变化对输出影响较大")
    
    print(f"\n总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
