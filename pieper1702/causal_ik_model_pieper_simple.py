"""
基于关节耦合关系的简化IK模型

核心思想：
- 不使用历史帧序列
- 用可学习的关节耦合嵌入直接编码关节间的相关性
- 用注意力机制学习关节对末端位姿的影响权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointCouplingEmbedding(nn.Module):
    """
    关节耦合嵌入 - 学习关节间的相关性关系

    不使用历史序列，直接学习一个静态的关节关系表示
    """

    def __init__(self, num_joints=7, hidden_dim=256):
        super().__init__()
        self.num_joints = num_joints

        # 可学习的关节耦合原型
        # 初始化为接近0的小值，让模型自由学习
        self.coupling_prototype = nn.Parameter(
            torch.randn(num_joints, hidden_dim) * 0.01,
            requires_grad=True
        )

        # 关节组特定的耦合偏置
        # shoulder(3), elbow(1), forearm(1), wrist(2)
        self.group_bias = nn.ParameterDict({
            'shoulder': nn.Parameter(torch.zeros(3, hidden_dim)),
            'elbow': nn.Parameter(torch.zeros(1, hidden_dim)),
            'forearm': nn.Parameter(torch.zeros(1, hidden_dim)),
            'wrist': nn.Parameter(torch.zeros(2, hidden_dim)),
        })

    def forward(self, batch_size):
        """
        Args:
            batch_size: batch size

        Returns:
            coupling_features: [batch, 7, hidden_dim] 每个关节的耦合特征
        """
        # 基础耦合特征 [7, hidden_dim]
        base_coupling = self.coupling_prototype

        # 添加关节组偏置
        coupling = base_coupling.clone()
        coupling[:3] += self.group_bias['shoulder']   # shoulder
        coupling[3:4] += self.group_bias['elbow']      # elbow
        coupling[4:5] += self.group_bias['forearm']    # forearm
        coupling[5:7] += self.group_bias['wrist']      # wrist

        # 扩展到batch维度
        coupling_features = coupling.unsqueeze(0).repeat(batch_size, 1, 1)

        return coupling_features  # [batch, 7, hidden_dim]


class JointPositionAttention(nn.Module):
    """
    基于关节耦合特征计算关节对末端位置/姿态的影响权重
    """

    def __init__(self, hidden_dim=256):
        super().__init__()

        # 位置影响注意力
        self.position_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 每个关节一个权重
        )

        # 姿态影响注意力
        self.orientation_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, coupling_features):
        """
        Args:
            coupling_features: [batch, 7, hidden_dim] 关节耦合特征

        Returns:
            pos_weights: [batch, 7] 位置影响权重（softmax归一化）
            ori_weights: [batch, 7] 姿态影响权重（softmax归一化）
        """
        # 计算每个关节的权重
        pos_logits = self.position_attention(coupling_features).squeeze(-1)  # [batch, 7]
        ori_logits = self.orientation_attention(coupling_features).squeeze(-1)  # [batch, 7]

        # Softmax归一化
        pos_weights = F.softmax(pos_logits, dim=-1)
        ori_weights = F.softmax(ori_logits, dim=-1)

        return pos_weights, ori_weights


class FiLMGenerator(nn.Module):
    """从关节特征生成FiLM参数"""

    def __init__(self, hidden_dim=256):
        super().__init__()

        self.position_gamma_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        self.position_beta_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

        self.orientation_gamma_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        self.orientation_beta_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, coupling_features, pos_weights, ori_weights):
        """
        Args:
            coupling_features: [batch, 7, hidden_dim]
            pos_weights: [batch, 7] 位置权重
            ori_weights: [batch, 7] 姿态权重

        Returns:
            pos_gamma, pos_beta: [batch, 3]
            ori_gamma, ori_beta: [batch, 4]
        """
        # 加权聚合关节特征
        weighted_pos = (coupling_features * pos_weights.unsqueeze(-1)).sum(dim=1)  # [batch, hidden_dim]
        weighted_ori = (coupling_features * ori_weights.unsqueeze(-1)).sum(dim=1)  # [batch, hidden_dim]

        pos_gamma = self.position_gamma_net(weighted_pos)
        pos_beta = self.position_beta_net(weighted_pos)
        ori_gamma = self.orientation_gamma_net(weighted_ori)
        ori_beta = self.orientation_beta_net(weighted_ori)

        return pos_gamma, pos_beta, ori_gamma, ori_beta


class EndPoseEncoder(nn.Module):
    """编码末端位姿"""

    def __init__(self, hidden_dim=256):
        super().__init__()

        self.position_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        self.orientation_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

    def forward(self, end_position, end_orientation):
        pos_feat = self.position_encoder(end_position)
        ori_feat = self.orientation_encoder(end_orientation)
        return pos_feat, ori_feat


class JointGroupDecoder(nn.Module):
    """
    关节组解码器 - 从融合特征预测关节角度

    遵循因果链: shoulder → elbow → forearm → wrist
    """

    def __init__(self, hidden_dim=256):
        super().__init__()

        # 消息传递层
        self.shoulder_to_elbow = nn.Linear(hidden_dim, hidden_dim)
        self.to_forearm = nn.Linear(hidden_dim * 2, hidden_dim)
        self.forearm_to_wrist = nn.Linear(hidden_dim * 2, hidden_dim)

        # 门控
        self.forearm_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.wrist_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # 输出头
        self.shoulder_head = nn.Linear(hidden_dim, 3)
        self.elbow_head = nn.Linear(hidden_dim, 1)
        self.forearm_head = nn.Linear(hidden_dim, 1)
        self.wrist_head = nn.Linear(hidden_dim, 2)

    def forward(self, fused_features):
        """
        Args:
            fused_features: dict with keys: shoulder, elbow, forearm, wrist
                          each value is [batch, hidden_dim]

        Returns:
            pred_angles: [batch, 7]
        """
        nodes = fused_features.copy()

        # Shoulder → Elbow
        elbow_msg = self.shoulder_to_elbow(nodes['shoulder'])
        nodes['elbow'] = nodes['elbow'] + elbow_msg

        # Shoulder + Elbow → Forearm
        se_concat = torch.cat([nodes['shoulder'], nodes['elbow']], dim=1)
        forearm_msg = self.to_forearm(se_concat)
        gate = self.forearm_gate(forearm_msg)
        nodes['forearm'] = nodes['forearm'] * (1 - gate) + forearm_msg * gate

        # Elbow + Forearm → Wrist
        ef_concat = torch.cat([nodes['elbow'], nodes['forearm']], dim=1)
        wrist_msg = self.forearm_to_wrist(ef_concat)
        gate = self.wrist_gate(wrist_msg)
        nodes['wrist'] = nodes['wrist'] * (1 - gate) + wrist_msg * gate

        # 输出预测
        pred_shoulder = self.shoulder_head(nodes['shoulder'])
        pred_elbow = self.elbow_head(nodes['elbow'])
        pred_forearm = self.forearm_head(nodes['forearm'])
        pred_wrist = self.wrist_head(nodes['wrist'])

        pred_angles = torch.cat([pred_shoulder, pred_elbow, pred_forearm, pred_wrist], dim=1)

        return pred_angles


class SimplifiedCausalIK(nn.Module):
    """
    简化的因果IK模型 - 不使用历史帧

    核心思想：
    1. 用关节耦合嵌入直接学习关节间的关系
    2. 用注意力机制计算关节对末端的影响权重
    3. 用FiLM调制末端位姿
    4. 用因果链解码器预测关节角度
    """

    def __init__(self, num_joints=7, hidden_dim=256):
        super().__init__()

        self.num_joints = num_joints
        self.hidden_dim = hidden_dim

        # 1. 关节耦合嵌入
        self.coupling_embedding = JointCouplingEmbedding(num_joints, hidden_dim)

        # 2. 关节位置注意力
        self.joint_attention = JointPositionAttention(hidden_dim)

        # 3. FiLM生成器
        self.film_generator = FiLMGenerator(hidden_dim)

        # 4. 末端位姿编码器
        self.endpose_encoder = EndPoseEncoder(hidden_dim)

        # 5. 特征融合（为每个关节组分配调制后的末端位姿特征）
        self.fusion_layers = nn.ModuleDict({
            'shoulder': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ),
            'elbow': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ),
            'forearm': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ),
            'wrist': nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
        })

        # 6. 关节组解码器
        self.decoder = JointGroupDecoder(hidden_dim)

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

        # 如果姿态为None，使用零tensor
        if end_orientation is None:
            end_orientation = torch.zeros(batch_size, 4, device=device)

        # 1. 获取关节耦合特征
        coupling_features = self.coupling_embedding(batch_size)  # [batch, 7, hidden_dim]

        # 2. 计算关节影响权重
        pos_weights, ori_weights = self.joint_attention(coupling_features)  # [batch, 7]

        # 3. 生成FiLM参数
        pos_gamma, pos_beta, ori_gamma, ori_beta = self.film_generator(
            coupling_features, pos_weights, ori_weights
        )

        # 4. FiLM调制末端位姿
        modulated_pos = pos_gamma * end_position + pos_beta  # [batch, 3]
        modulated_ori = ori_gamma * end_orientation + ori_beta  # [batch, 4]

        # 5. 编码调制后的末端位姿
        end_pos_feat, end_ori_feat = self.endpose_encoder(modulated_pos, modulated_ori)

        # 6. 融合特征（关节耦合特征 + 调制后的末端位姿特征）
        # 将关节耦合特征按组分配
        coupling_groups = {
            'shoulder': coupling_features[:, :3, :].mean(dim=1),  # [batch, hidden_dim]
            'elbow': coupling_features[:, 3:4, :].squeeze(1),     # [batch, hidden_dim]
            'forearm': coupling_features[:, 4:5, :].squeeze(1),   # [batch, hidden_dim]
            'wrist': coupling_features[:, 5:7, :].mean(dim=1),    # [batch, hidden_dim]
        }

        # 融合
        fused_features = {}
        for group in ['shoulder', 'elbow', 'forearm']:
            # 根据关节组特性选择末端特征
            if group == 'shoulder':
                end_feat = end_pos_feat
            else:
                end_feat = (end_pos_feat + end_ori_feat) / 2

            concat = torch.cat([coupling_groups[group], end_feat], dim=1)
            fused_features[group] = self.fusion_layers[group](concat)

        # Wrist特殊处理（融合位置+姿态特征）
        concat_wrist = torch.cat([
            coupling_groups['wrist'],
            end_pos_feat,
            end_ori_feat
        ], dim=1)
        fused_features['wrist'] = self.fusion_layers['wrist'](concat_wrist)

        # 7. 解码预测关节角度
        pred_angles = self.decoder(fused_features)

        return pred_angles, {
            'position_weights': pos_weights,
            'orientation_weights': ori_weights,
            'modulated_pos': modulated_pos,
            'modulated_ori': modulated_ori
        }


# 测试代码
if __name__ == '__main__':
    print("=" * 70)
    print("测试简化的因果IK模型（无历史帧）")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimplifiedCausalIK(num_joints=7, hidden_dim=256).to(device)
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
    print(f"  - 位置权重: {info['position_weights'].shape}")

    print(f"\n关节对位置的影响权重（第一个样本）:")
    for i, w in enumerate(info['position_weights'][0]):
        print(f"  J{i}: {w:.4f}")

    print(f"\n关节耦合嵌入参数:")
    print(f"  coupling_prototype: {model.coupling_embedding.coupling_prototype.shape}")
    print(f"  可学习参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    print("\n" + "=" * 70)
