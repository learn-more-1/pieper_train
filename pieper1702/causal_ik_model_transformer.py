"""
基于 Transformer 的简化IK模型

核心创新：
1. 使用 Self-Attention 学习关节间的关系
2. 使用 Cross-Attention 让关节特征感知末端位姿
3. 多层 Transformer 堆叠提升表达能力
4. 无需历史帧，直接从末端位姿预测关节角度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch, query_len, hidden_dim]
            key: [batch, key_len, hidden_dim]
            value: [batch, value_len, hidden_dim]
        """
        batch_size = query.shape[0]

        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        output = self.out_proj(context)
        return output


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, hidden_dim, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    """Transformer 编码层"""

    def __init__(self, hidden_dim, num_heads=8, ff_dim=1024, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForward(hidden_dim, ff_dim, dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual
        x = x + self.dropout(self.self_attn(x, x, x, mask))
        x = self.norm1(x)

        # FFN + residual
        x = x + self.ffn(x)
        x = self.norm2(x)

        return x


class JointTokenEmbedding(nn.Module):
    """关节 token 嵌入"""

    def __init__(self, num_joints=7, hidden_dim=256):
        super().__init__()

        # 可学习的关节 token
        self.joint_tokens = nn.Parameter(torch.randn(num_joints, hidden_dim) * 0.02)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(num_joints, hidden_dim) * 0.02)

    def forward(self, batch_size):
        """
        Returns:
            tokens: [batch, num_joints, hidden_dim]
        """
        tokens = self.joint_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
        tokens = tokens + self.pos_encoding.unsqueeze(0).repeat(batch_size, 1, 1)
        return tokens


class EndPoseConditioning(nn.Module):
    """末端位姿条件编码"""

    def __init__(self, hidden_dim=256):
        super().__init__()

        # 位置编码
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 姿态编码
        self.ori_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, end_position, end_orientation):
        """
        Returns:
            condition: [batch, hidden_dim]
        """
        pos_feat = self.pos_encoder(end_position)
        ori_feat = self.ori_encoder(end_orientation)

        concat = torch.cat([pos_feat, ori_feat], dim=1)
        condition = self.fusion(concat)

        return condition


class CrossAttentionConditioning(nn.Module):
    """Cross-Attention 条件注入"""

    def __init__(self, hidden_dim=256, num_heads=8):
        super().__init__()

        # 用末端位姿作为 query，关节特征作为 key/value
        self.cross_attn = MultiHeadAttention(hidden_dim, num_heads)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, joint_tokens, end_pose_condition):
        """
        Args:
            joint_tokens: [batch, num_joints, hidden_dim]
            end_pose_condition: [batch, hidden_dim]

        Returns:
            conditioned_tokens: [batch, num_joints, hidden_dim]
        """
        # 将末端位姿扩展为 query
        batch_size = joint_tokens.shape[0]
        query = end_pose_condition.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Cross-attention
        conditioned = self.cross_attn(
            query.repeat(1, joint_tokens.shape[1], 1),  # [batch, num_joints, hidden_dim]
            joint_tokens,
            joint_tokens
        )

        # 残差连接
        conditioned = joint_tokens + conditioned
        conditioned = self.norm(conditioned)

        return conditioned


class TransformerIK(nn.Module):
    """
    基于 Transformer 的 IK 模型（无需历史帧）

    核心思想：
    1. 将每个关节视为一个 token
    2. 使用 Self-Attention 学习关节间关系
    3. 使用 Cross-Attention 注入末端位姿信息
    4. 多层 Transformer 编码
    5. 预测每个关节的角度
    """

    def __init__(self, num_joints=7, hidden_dim=256, num_layers=6, num_heads=8, ff_dim=512):
        super().__init__()

        self.num_joints = num_joints
        self.hidden_dim = hidden_dim

        # 关节 token 嵌入
        self.joint_embedding = JointTokenEmbedding(num_joints, hidden_dim)

        # 末端位姿条件编码
        self.endpose_conditioning = EndPoseConditioning(hidden_dim)

        # Cross-attention 条件注入
        self.cross_condition = CrossAttentionConditioning(hidden_dim, num_heads)

        # Transformer 编码层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        # 输出头（为每个关节预测角度）
        # 注意：输入是 [batch, num_joints, hidden_dim]，需要分别处理每个关节
        self.shoulder_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)  # 每个关节输出1维
        )
        self.elbow_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.forearm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.wrist_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, end_position, end_orientation):
        """
        Args:
            end_position: [batch, 3] 末端位置
            end_orientation: [batch, 4] 末端姿态（四元数）

        Returns:
            pred_angles: [batch, 7] 预测的关节角度
        """
        batch_size = end_position.shape[0]
        device = end_position.device

        if end_orientation is None:
            end_orientation = torch.zeros(batch_size, 4, device=device)

        # 1. 关节 token 嵌入
        joint_tokens = self.joint_embedding(batch_size)  # [batch, 7, hidden_dim]

        # 2. 末端位姿条件编码
        end_pose_cond = self.endpose_conditioning(end_position, end_orientation)  # [batch, hidden_dim]

        # 3. Cross-attention 注入末端位姿信息
        joint_tokens = self.cross_condition(joint_tokens, end_pose_cond)

        # 4. Transformer 编码
        for layer in self.transformer_layers:
            joint_tokens = layer(joint_tokens)

        # 5. 预测每个关节的角度
        # joint_tokens: [batch, 7, hidden_dim]
        # 分别为每个关节预测一个角度值
        joint_preds = []
        for i in range(7):
            if i < 3:
                pred = self.shoulder_head(joint_tokens[:, i, :])  # [batch, 1]
            elif i == 3:
                pred = self.elbow_head(joint_tokens[:, i, :])      # [batch, 1]
            elif i == 4:
                pred = self.forearm_head(joint_tokens[:, i, :])   # [batch, 1]
            else:  # i == 5, 6
                pred = self.wrist_head(joint_tokens[:, i, :])     # [batch, 1]
            joint_preds.append(pred)

        # 拼接所有关节预测
        pred_angles = torch.cat(joint_preds, dim=1)  # [batch, 7]

        return pred_angles, {
            'joint_tokens': joint_tokens,
            'end_pose_cond': end_pose_cond
        }


# 测试代码
if __name__ == '__main__':
    print("=" * 70)
    print("测试 Transformer IK 模型")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TransformerIK(
        num_joints=7,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        ff_dim=512
    ).to(device)
    model.eval()

    batch_size = 4
    end_position = torch.randn(batch_size, 3).to(device)
    end_orientation = torch.randn(batch_size, 4).to(device)

    print(f"\n输入:")
    print(f"  - 末端位置: {end_position.shape}")
    print(f"  - 末端姿态: {end_orientation.shape}")

    with torch.no_grad():
        pred_angles, info = model(end_position, end_orientation)

    print(f"\n输出:")
    print(f"  - 预测角度: {pred_angles.shape}")
    print(f"  - 关节 tokens: {info['joint_tokens'].shape}")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n参数量: {total_params:.2f}M")

    print("\n" + "=" * 70)
