"""
大容量改进IK模型 - 无需历史帧

容量改进：
1. hidden_dim: 256 → 512
2. num_heads: 4 → 8
3. decoder_layers: 3 → 5
4. FFN expansion: 4x
5. 多级 Cross-Attention
6. 更深的特征融合
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
    """前馈网络 - 更大容量"""

    def __init__(self, hidden_dim, ff_dim=2048, dropout=0.1):
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


class LargeDynamicJointCoupling(nn.Module):
    """
    大容量动态关节耦合模块

    改进：
    1. 增大 hidden_dim 到 512
    2. 使用更多注意力头（8）
    3. 多层 Cross-Attention
    4. 更大的 FFN
    """

    def __init__(self, num_joints=7, hidden_dim=512, num_heads=8):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim

        # 静态关节原型
        self.joint_prototype = nn.Parameter(
            torch.randn(num_joints, hidden_dim) * 0.02,
            requires_grad=True
        )

        # 关节组偏置
        self.group_bias = nn.ParameterDict({
            'shoulder': nn.Parameter(torch.zeros(3, hidden_dim)),
            'elbow': nn.Parameter(torch.zeros(1, hidden_dim)),
            'forearm': nn.Parameter(torch.zeros(1, hidden_dim)),
            'wrist': nn.Parameter(torch.zeros(2, hidden_dim)),
        })

        # 多层 Cross-attention
        self.cross_attn_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads, dropout=0.1)
            for _ in range(3)  # 3层 Cross-Attention
        ])

        # 自注意力（让关节之间互相感知）
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout=0.1)

        # 前馈网络（更大）
        self.ffn = FeedForward(hidden_dim, ff_dim=hidden_dim * 4, dropout=0.1)

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        self.norm5 = nn.LayerNorm(hidden_dim)

    def forward(self, end_pose_feat, batch_size):
        """
        Args:
            end_pose_feat: [batch, hidden_dim] 末端位姿特征
            batch_size: batch size

        Returns:
            joint_features: [batch, 7, hidden_dim]
        """
        device = end_pose_feat.device

        # 基础关节特征
        joint_feat = self.joint_prototype  # [7, hidden_dim]

        # 添加关节组偏置
        joint_feat = joint_feat.clone()
        joint_feat[:3] += self.group_bias['shoulder']
        joint_feat[3:4] += self.group_bias['elbow']
        joint_feat[4:5] += self.group_bias['forearm']
        joint_feat[5:7] += self.group_bias['wrist']

        # 扩展到 batch
        joint_feat = joint_feat.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, 7, hidden_dim]

        # 多层 Cross-attention + 残差连接
        end_pose_expanded = end_pose_feat.unsqueeze(1)  # [batch, 1, hidden_dim]

        for i, cross_attn in enumerate(self.cross_attn_layers):
            # 使用末端位姿作为 query，关节特征作为 key/value
            joint_feat = joint_feat + cross_attn(
                end_pose_expanded.repeat(1, 7, 1),
                joint_feat,
                joint_feat
            )
            joint_feat = self.norm1(joint_feat)

        # Self-attention：让关节之间互相感知
        joint_feat = joint_feat + self.self_attn(joint_feat, joint_feat, joint_feat)
        joint_feat = self.norm2(joint_feat)

        # FFN
        joint_feat = joint_feat + self.ffn(joint_feat)
        joint_feat = self.norm3(joint_feat)

        return joint_feat


class LargeEndPoseEncoder(nn.Module):
    """大容量末端位姿编码器"""

    def __init__(self, hidden_dim=512):
        super().__init__()

        # 更深的位置编码器
        self.position_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 更深的姿态编码器
        self.orientation_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, end_position, end_orientation):
        pos_feat = self.position_encoder(end_position)
        ori_feat = self.orientation_encoder(end_orientation)

        concat = torch.cat([pos_feat, ori_feat], dim=1)
        end_feat = self.fusion(concat)

        return pos_feat, ori_feat, end_feat


class LargeMultiScaleFeatureFusion(nn.Module):
    """
    大容量多尺度特征融合模块

    改进：
    1. 更深的融合网络
    2. 更大的 FFN
    3. 多级残差连接
    """

    def __init__(self, hidden_dim=512):
        super().__init__()

        # 为每个关节组创建更深的融合层
        self.fusion_layers = nn.ModuleDict({
            'shoulder': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ),
            'elbow': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ),
            'forearm': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ),
            'wrist': nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
        })

        # 自适应门控
        self.gate_layers = nn.ModuleDict({
            'shoulder': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            ),
            'elbow': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            ),
            'forearm': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            ),
            'wrist': nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.Sigmoid()
            )
        })

    def forward(self, joint_groups, end_pos_feat, end_ori_feat, end_feat):
        fused = {}

        for group in ['shoulder', 'elbow', 'forearm']:
            if group == 'shoulder':
                end_feat_group = end_pos_feat
            elif group == 'wrist':
                end_feat_group = end_ori_feat
            else:
                end_feat_group = end_feat

            concat = torch.cat([joint_groups[group], end_feat_group], dim=1)

            gate = self.gate_layers[group](concat)
            transformed = self.fusion_layers[group](concat)
            fused[group] = joint_groups[group] * (1 - gate) + transformed * gate

        # Wrist 特殊处理
        concat_wrist = torch.cat([joint_groups['wrist'], end_pos_feat, end_ori_feat], dim=1)
        gate = self.gate_layers['wrist'](concat_wrist)
        transformed = self.fusion_layers['wrist'](concat_wrist)
        fused['wrist'] = joint_groups['wrist'] * (1 - gate) + transformed * gate

        return fused


class LargeJointGroupDecoder(nn.Module):
    """
    大容量关节组解码器

    改进：
    1. 更多消息传递层（5层）
    2. 更大的 FFN
    3. 多级残差连接
    """

    def __init__(self, hidden_dim=512, num_layers=5):
        super().__init__()
        self.num_layers = num_layers

        # 多层消息传递
        self.message_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'shoulder_to_elbow': nn.Linear(hidden_dim, hidden_dim),
                'to_forearm': nn.Linear(hidden_dim * 2, hidden_dim),
                'forearm_to_wrist': nn.Linear(hidden_dim * 2, hidden_dim),
                'forearm_gate': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid()
                ),
                'wrist_gate': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid()
                )
            })
            self.message_layers.append(layer)

        # 输出头（更深层）
        self.output_heads = nn.ModuleDict({
            'shoulder': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3)
            ),
            'elbow': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)
            ),
            'forearm': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)
            ),
            'wrist': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2)
            )
        })

    def forward(self, fused_features):
        nodes = {k: v.clone() for k, v in fused_features.items()}

        # 多层消息传递
        for layer in self.message_layers:
            # Shoulder → Elbow
            elbow_msg = layer['shoulder_to_elbow'](nodes['shoulder'])
            nodes['elbow'] = nodes['elbow'] + elbow_msg

            # Shoulder + Elbow → Forearm
            se_concat = torch.cat([nodes['shoulder'], nodes['elbow']], dim=1)
            forearm_msg = layer['to_forearm'](se_concat)
            gate = layer['forearm_gate'](forearm_msg)
            nodes['forearm'] = nodes['forearm'] * (1 - gate) + forearm_msg * gate

            # Elbow + Forearm → Wrist
            ef_concat = torch.cat([nodes['elbow'], nodes['forearm']], dim=1)
            wrist_msg = layer['forearm_to_wrist'](ef_concat)
            gate = layer['wrist_gate'](wrist_msg)
            nodes['wrist'] = nodes['wrist'] * (1 - gate) + wrist_msg * gate

        # 输出预测
        pred_shoulder = self.output_heads['shoulder'](nodes['shoulder'])
        pred_elbow = self.output_heads['elbow'](nodes['elbow'])
        pred_forearm = self.output_heads['forearm'](nodes['forearm'])
        pred_wrist = self.output_heads['wrist'](nodes['wrist'])

        pred_angles = torch.cat([pred_shoulder, pred_elbow, pred_forearm, pred_wrist], dim=1)

        return pred_angles


class LargeImprovedSimplifiedCausalIK(nn.Module):
    """
    大容量改进IK模型 - 无需历史帧

    容量改进：
    1. hidden_dim: 256 → 512 (2x)
    2. num_heads: 4 → 8 (2x)
    3. decoder_layers: 3 → 5 (1.67x)
    4. FFN: hidden_dim*2 → hidden_dim*4 (2x)
    5. 多级 Cross-Attention (3层)
    6. 更深的特征融合

    预期参数量：约 15M+
    """

    def __init__(self, num_joints=7, hidden_dim=512, num_heads=8, num_decoder_layers=5):
        super().__init__()

        self.num_joints = num_joints
        self.hidden_dim = hidden_dim

        # 1. 大容量动态关节耦合模块
        self.dynamic_coupling = LargeDynamicJointCoupling(num_joints, hidden_dim, num_heads)

        # 2. 大容量末端位姿编码器
        self.endpose_encoder = LargeEndPoseEncoder(hidden_dim)

        # 3. 大容量多尺度特征融合
        self.feature_fusion = LargeMultiScaleFeatureFusion(hidden_dim)

        # 4. 大容量解码器
        self.decoder = LargeJointGroupDecoder(hidden_dim, num_decoder_layers)

        # 关节分组
        self.joint_groups = {
            'shoulder': [0, 1, 2],
            'elbow': [3],
            'forearm': [4],
            'wrist': [5, 6]
        }

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

        # 1. 编码末端位姿
        end_pos_feat, end_ori_feat, end_feat = self.endpose_encoder(end_position, end_orientation)

        # 2. 动态关节耦合（多层 Cross-Attention 让关节特征感知末端位姿）
        joint_features = self.dynamic_coupling(end_feat, batch_size)  # [batch, 7, hidden_dim]

        # 3. 按组聚合关节特征
        joint_groups = {
            'shoulder': joint_features[:, :3, :].mean(dim=1),
            'elbow': joint_features[:, 3:4, :].squeeze(1),
            'forearm': joint_features[:, 4:5, :].squeeze(1),
            'wrist': joint_features[:, 5:7, :].mean(dim=1)
        }

        # 4. 多尺度特征融合
        fused_features = self.feature_fusion(joint_groups, end_pos_feat, end_ori_feat, end_feat)

        # 5. 解码预测
        pred_angles = self.decoder(fused_features)

        return pred_angles, {
            'joint_features': joint_features
        }


# 测试代码
if __name__ == '__main__':
    print("=" * 70)
    print("测试大容量改进IK模型")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LargeImprovedSimplifiedCausalIK(
        num_joints=7,
        hidden_dim=512,
        num_heads=8,
        num_decoder_layers=5
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
    print(f"  - 关节特征: {info['joint_features'].shape}")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n参数量: {total_params:.2f}M")

    print("\n" + "=" * 70)
