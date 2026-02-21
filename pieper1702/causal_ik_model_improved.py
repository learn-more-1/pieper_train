"""
改进的简化IK模型 - 无需历史帧

改进点：
1. 动态关节耦合：末端位姿通过 cross-attention 调制关节特征
2. 多层特征交互：关节特征和末端位姿特征多次交互
3. 残差连接：加深网络同时保持梯度流动
4. 自适应特征融合：根据末端位置动态调整融合权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

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

        # [batch, num_heads, query_len, key_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # [batch, num_heads, query_len, head_dim]
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        output = self.out_proj(context)
        return output


class DynamicJointCoupling(nn.Module):
    """
    动态关节耦合模块

    用末端位姿特征通过 cross-attention 动态调制关节特征
    """

    def __init__(self, num_joints=7, hidden_dim=256, num_heads=4):
        super().__init__()
        self.num_joints = num_joints

        # 静态关节原型（作为基础）
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

        # Cross-attention：用末端位姿调制关节特征
        self.cross_attention = MultiHeadAttention(hidden_dim, num_heads)

        # 自注意力：让关节之间互相感知
        self.self_attention = MultiHeadAttention(hidden_dim, num_heads)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

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

        # 扩展到batch
        joint_feat = joint_feat.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, 7, hidden_dim]

        # 末端位姿作为 query，关节特征作为 key/value
        end_pose_expanded = end_pose_feat.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Cross-attention：让关节特征感知末端位姿
        joint_feat = joint_feat + self.cross_attention(
            query=end_pose_expanded.repeat(1, 7, 1),  # [batch, 7, hidden_dim]
            key=joint_feat,
            value=joint_feat
        )
        joint_feat = self.norm1(joint_feat)

        # Self-attention：让关节之间互相感知
        joint_feat = joint_feat + self.self_attention(
            query=joint_feat,
            key=joint_feat,
            value=joint_feat
        )
        joint_feat = self.norm2(joint_feat)

        # FFN
        joint_feat = joint_feat + self.ffn(joint_feat)
        joint_feat = self.norm3(joint_feat)

        return joint_feat  # [batch, 7, hidden_dim]


class AdaptiveEndPoseEncoder(nn.Module):
    """
    自适应末端位姿编码器

    为位置和姿态分别编码，并根据不同关节组的需求调整特征
    """

    def __init__(self, hidden_dim=256):
        super().__init__()

        # 位置编码（多层）
        self.position_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 姿态编码（多层）
        self.orientation_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
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
            end_pos_feat: [batch, hidden_dim]
            end_ori_feat: [batch, hidden_dim]
            end_feat: [batch, hidden_dim] 融合特征
        """
        pos_feat = self.position_encoder(end_position)
        ori_feat = self.orientation_encoder(end_orientation)

        # 融合
        concat = torch.cat([pos_feat, ori_feat], dim=1)
        end_feat = self.fusion(concat)

        return pos_feat, ori_feat, end_feat


class MultiScaleFeatureFusion(nn.Module):
    """
    多尺度特征融合模块

    将关节特征和末端位姿特征在多个尺度上进行融合
    """

    def __init__(self, hidden_dim=256):
        super().__init__()

        # 为每个关节组创建融合层
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
        """
        Args:
            joint_groups: dict, 每个关节组的特征
            end_pos_feat: [batch, hidden_dim]
            end_ori_feat: [batch, hidden_dim]
            end_feat: [batch, hidden_dim]

        Returns:
            fused_features: dict
        """
        fused = {}

        for group in ['shoulder', 'elbow', 'forearm']:
            # 选择末端特征
            if group == 'shoulder':
                end_feat_group = end_pos_feat
            elif group == 'wrist':
                end_feat_group = end_ori_feat
            else:
                end_feat_group = end_feat

            # 拼接
            concat = torch.cat([joint_groups[group], end_feat_group], dim=1)

            # 门控
            gate = self.gate_layers[group](concat)
            transformed = self.fusion_layers[group](concat)

            # 融合
            fused[group] = joint_groups[group] * (1 - gate) + transformed * gate

        # Wrist 特殊处理
        concat_wrist = torch.cat([joint_groups['wrist'], end_pos_feat, end_ori_feat], dim=1)
        gate = self.gate_layers['wrist'](concat_wrist)
        transformed = self.fusion_layers['wrist'](concat_wrist)
        fused['wrist'] = joint_groups['wrist'] * (1 - gate) + transformed * gate

        return fused


class ImprovedJointGroupDecoder(nn.Module):
    """
    改进的关节组解码器

    使用更深的消息传递网络和残差连接
    """

    def __init__(self, hidden_dim=256, num_layers=3):
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

        # 输出头
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
        """
        Args:
            fused_features: dict with keys: shoulder, elbow, forearm, wrist

        Returns:
            pred_angles: [batch, 7]
        """
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


class ImprovedSimplifiedCausalIK(nn.Module):
    """
    改进的简化IK模型

    主要改进：
    1. 动态关节耦合：用 cross-attention 让关节特征感知末端位姿
    2. 自适应末端位姿编码：多层编码 + 融合
    3. 多尺度特征融合：更深、更智能的特征融合
    4. 改进的解码器：多层消息传递 + 残差输出头
    """

    def __init__(self, num_joints=7, hidden_dim=256, num_heads=4, num_decoder_layers=3):
        super().__init__()

        self.num_joints = num_joints
        self.hidden_dim = hidden_dim

        # 1. 动态关节耦合模块
        self.dynamic_coupling = DynamicJointCoupling(num_joints, hidden_dim, num_heads)

        # 2. 自适应末端位姿编码器
        self.endpose_encoder = AdaptiveEndPoseEncoder(hidden_dim)

        # 3. 多尺度特征融合
        self.feature_fusion = MultiScaleFeatureFusion(hidden_dim)

        # 4. 改进的解码器
        self.decoder = ImprovedJointGroupDecoder(hidden_dim, num_decoder_layers)

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

        # 2. 动态关节耦合（让关节特征感知末端位姿）
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
            'joint_features': joint_features,
            'fused_features': fused_features
        }


# 测试代码
if __name__ == '__main__':
    print("=" * 70)
    print("测试改进的简化IK模型")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ImprovedSimplifiedCausalIK(
        num_joints=7,
        hidden_dim=256,
        num_heads=4,
        num_decoder_layers=3
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

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n参数量: {total_params:.2f}M")

    print("\n" + "=" * 70)
