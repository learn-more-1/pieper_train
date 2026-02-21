"""
基于 Pieper 准则的因果IK模型

核心思想：
1. Pieper 准则：不同关节对末端位姿的影响程度不同
2. Shoulder 的 rpy 对末端位置影响大
3. Shoulder 和 wrist 的 rp 对 wrist yaw 影响大
4. 使用 Attention 动态学习影响权重
5. 分别对末端位置和姿态做 FiLM 调制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PieperAttention(nn.Module):
    """
    基于 Pieper 准则的注意力机制（支持时序历史输入）

    计算每个关节对末端位置和姿态的影响权重
    """

    def __init__(self, joint_dim=7, num_frames=10, hidden_dim=256):
        super().__init__()
        self.joint_dim = joint_dim
        self.num_frames = num_frames

        # 时序编码器（编码全部历史帧）
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(joint_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 池化到固定长度
        )

        # 关节特征融合（将时序特征映射到关节表示）
        self.joint_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 位置影响注意力（Joint → Position）
        # Pieper: Shoulder rpy 对位置影响大
        self.position_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7),  # 7个关节的权重
            nn.Softmax(dim=-1)
        )

        # 姿态影响注意力（Joint → Orientation）
        # Pieper: Shoulder & wrist rp 对 wrist yaw 影响大
        self.orientation_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7),  # 7个关节的权重
            nn.Softmax(dim=-1)
        )

    def forward(self, history_frames):
        """
        Args:
            history_frames: [batch, num_frames, 7] 历史关节角度

        Returns:
            position_weights: [batch, 7] 每个关节对位置的影响权重
            orientation_weights: [batch, 7] 每个关节对姿态的影响权重
        """
        # 转换为 [batch, joint_dim, num_frames] 用于 Conv1d
        x = history_frames.transpose(1, 2)  # [batch, 7, num_frames]

        # 时序编码
        temporal_features = self.temporal_encoder(x).squeeze(-1)  # [batch, hidden_dim]

        # 特征融合
        joint_features = self.joint_fusion(temporal_features)  # [batch, hidden_dim]

        # 计算影响权重
        pos_weights = self.position_attention(joint_features)   # [batch, 7]
        ori_weights = self.orientation_attention(joint_features) # [batch, 7]

        return pos_weights, ori_weights


class PieperFiLMGenerator(nn.Module):
    """
    基于 Pieper 权重的 FiLM 生成器

    使用注意力权重调制关节特征，然后生成 FiLM 参数
    """

    def __init__(self, joint_dim=7, hidden_dim=256):
        super().__init__()

        # 生成位置 FiLM 参数（直接从加权关节角度）
        self.position_gamma_net = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)  # 位置是3维
        )

        self.position_beta_net = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)
        )

        # 生成姿态 FiLM 参数
        self.orientation_gamma_net = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4)  # 姿态用4维（四元数）
        )

        self.orientation_beta_net = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, joint_angles, position_weights, orientation_weights):
        """
        Args:
            joint_angles: [batch, 7] 关节角度
            position_weights: [batch, 7] 位置影响权重
            orientation_weights: [batch, 7] 姿态影响权重

        Returns:
            position_gamma, position_beta: [batch, 3] 位置的 FiLM 参数
            orientation_gamma, orientation_beta: [batch, 4] 姿态的 FiLM 参数
        """
        # 1. 应用 Pieper 权重到关节角度
        weighted_joints_pos = joint_angles * position_weights  # [batch, 7]
        weighted_joints_ori = joint_angles * orientation_weights  # [batch, 7]

        # 2. 生成 FiLM 参数
        pos_gamma = self.position_gamma_net(weighted_joints_pos)  # [batch, 3]
        pos_beta = self.position_beta_net(weighted_joints_pos)

        ori_gamma = self.orientation_gamma_net(weighted_joints_ori)  # [batch, 4]
        ori_beta = self.orientation_beta_net(weighted_joints_ori)

        return pos_gamma, pos_beta, ori_gamma, ori_beta


class EndPoseEncoder(nn.Module):
    """编码末端位姿（位置 + 四元数）"""

    def __init__(self, hidden_dim=256):
        super().__init__()

        # 位置编码器
        self.position_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # 四元数编码器
        self.orientation_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

    def forward(self, end_position, end_orientation):
        """
        Args:
            end_position: [batch, 3] 末端位置
            end_orientation: [batch, 4] 末端姿态（四元数）

        Returns:
            position_feat: [batch, hidden_dim]
            orientation_feat: [batch, hidden_dim]
        """
        pos_feat = self.position_encoder(end_position)
        ori_feat = self.orientation_encoder(end_orientation)

        return pos_feat, ori_feat


class MultiFeatureFusion(nn.Module):
    """
    多特征融合模块

    融合 FiLM 调制后的末端位姿特征和 GNN 输出的节点特征
    """

    def __init__(self, hidden_dim=256):
        super().__init__()

        # 为每个关节组创建融合层
        self.fusion_layers = nn.ModuleDict({
            'shoulder': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            'elbow': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            'forearm': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            'wrist': nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),  # wrist 包含末端位姿特征
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        })

        # 门控机制（控制融合比例）
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

    def forward(self, gnn_features, film_features, endeff_feat=None):
        """
        Args:
            gnn_features: dict, GNN 输出的节点特征
            film_features: dict, FiLM 调制后的特征
            endeff_feat: [batch, hidden_dim] or None, 末端位姿特征（仅用于 wrist）

        Returns:
            fused_features: dict, 融合后的特征
        """
        fused = {}

        for joint in ['shoulder', 'elbow', 'forearm']:
            # 拼接 GNN 和 FiLM 特征
            concat = torch.cat([gnn_features[joint], film_features[joint]], dim=1)

            # 门控融合
            gate = self.gate_layers[joint](concat)
            transformed = self.fusion_layers[joint](concat)
            fused[joint] = gnn_features[joint] * (1 - gate) + transformed * gate

        # Wrist 特殊处理（额外融合末端位姿特征）
        if endeff_feat is not None:
            concat_wrist = torch.cat([gnn_features['wrist'], film_features['wrist'], endeff_feat], dim=1)
        else:
            concat_wrist = torch.cat([gnn_features['wrist'], film_features['wrist']], dim=1)

        gate = self.gate_layers['wrist'](concat_wrist)
        transformed = self.fusion_layers['wrist'](concat_wrist)
        fused['wrist'] = gnn_features['wrist'] * (1 - gate) + transformed * gate

        return fused


class JointwiseTemporalEncoder(nn.Module):
    """为每个关节单独编码历史特征"""

    def __init__(self, num_frames=10, hidden_dim=256):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

        # 为每个关节组创建独立的时序编码器
        self.encoders = nn.ModuleDict({
            'shoulder': nn.Sequential(
                nn.Conv1d(3, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            ),
            'elbow': nn.Sequential(
                nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            ),
            'forearm': nn.Sequential(
                nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            ),
            'wrist': nn.Sequential(
                nn.Conv1d(2, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
        })

    def forward(self, history_frames):
        """
        Args:
            history_frames: [batch, num_frames, 7] 7个关节的历史角度

        Returns:
            dict: 每个关节组的时序特征
        """
        batch_size = history_frames.shape[0]

        # 提取各关节组的历史
        history_shoulder = history_frames[:, :, :3].transpose(1, 2)  # [batch, 3, num_frames]
        history_elbow = history_frames[:, :, 3:4].transpose(1, 2)     # [batch, 1, num_frames]
        history_forearm = history_frames[:, :, 4:5].transpose(1, 2)   # [batch, 1, num_frames]
        history_wrist = history_frames[:, :, 5:7].transpose(1, 2)     # [batch, 2, num_frames]

        # 分别编码
        features = {
            'shoulder': self.encoders['shoulder'](history_shoulder).squeeze(-1),
            'elbow': self.encoders['elbow'](history_elbow).squeeze(-1),
            'forearm': self.encoders['forearm'](history_forearm).squeeze(-1),
            'wrist': self.encoders['wrist'](history_wrist).squeeze(-1)
        }

        return features


class PieperCausalIK(nn.Module):
    """
    基于 Pieper 准则的因果IK模型（支持历史输入）

    核心创新：
    1. 使用时序编码器分别编码每个关节的历史
    2. 使用 Attention 学习关节对末端的影响权重（符合 Pieper 准则）
    3. 分别对位置和姿态做 FiLM 调制
    4. 结合 GNN 学习关节间的因果关系
    """

    def __init__(self, num_joints=7, num_frames=10, hidden_dim=256, num_layers=2):
        super().__init__()

        self.joint_dim = num_joints
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

        # 1. 关节级别的时序编码器
        self.temporal_encoder = JointwiseTemporalEncoder(num_frames, hidden_dim)

        # 可学习的默认历史嵌入（用于推理时无历史数据的情况）
        # 在微调时会学习到合适的默认值
        self.default_history = nn.Parameter(
            torch.zeros(1, num_frames, num_joints),
            requires_grad=True
        )

        # 2. Pieper 注意力模块（使用全部历史帧）
        self.pieper_attention = PieperAttention(num_joints, num_frames, hidden_dim)

        # 3. Pieper FiLM 生成器
        self.pieper_film = PieperFiLMGenerator(num_joints, hidden_dim)

        # 4. 末端位姿编码器
        self.endpose_encoder = EndPoseEncoder(hidden_dim)

        # 5. 关节分组（因果链）
        self.joint_groups = {
            'shoulder': [0, 1, 2],
            'elbow': [3],
            'forearm': [4],
            'wrist': [5, 6]
        }

        # 6. 关节嵌入（将当前关节角度映射到特征空间，用于计算attention）
        self.joint_embeddings = nn.ModuleDict({
            'shoulder': nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ),
            'elbow': nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ),
            'forearm': nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ),
            'wrist': nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
        })

        # 6. 消息传递层（因果链）
        self.message_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'shoulder_to_elbow': nn.Linear(hidden_dim, hidden_dim),
                'to_forearm': nn.Linear(hidden_dim * 2, hidden_dim),
                'forearm_to_wrist': nn.Linear(hidden_dim * 2, hidden_dim),
                'forearm_gate': nn.Sigmoid(),  # 门控
                'wrist_gate': nn.Sigmoid()
            })
            self.message_layers.append(layer)

        # 7. 多特征融合模块
        self.feature_fusion = MultiFeatureFusion(hidden_dim)

        # 8. 输出头
        self.output_heads = nn.ModuleDict({
            'shoulder': nn.Linear(hidden_dim, 3),
            'elbow': nn.Linear(hidden_dim, 1),
            'forearm': nn.Linear(hidden_dim, 1),
            'wrist': nn.Linear(hidden_dim, 2)
        })

    def forward(self, history_frames, end_position, end_orientation):
        """
        Args:
            history_frames: [batch, num_frames, 7] 历史关节角度，可以为None（推理时）
            end_position: [batch, 3] 末端位置
            end_orientation: [batch, 4] 末端姿态（四元数）- 可以为None

        Returns:
            pred_joint_angles: [batch, 7] 预测的关节角度变化
        """
        batch_size = end_position.shape[0]
        device = end_position.device

        # 如果历史帧为None，使用可学习的默认历史
        if history_frames is None:
            history_frames = self.default_history.repeat(batch_size, 1, 1)

        # 如果姿态为None，使用零tensor
        if end_orientation is None:
            end_orientation = torch.zeros(batch_size, 4, device=device)

        # 1. 关节级别的时序编码（提取历史特征）
        joint_features = self.temporal_encoder(history_frames)
        # joint_features = {
        #     'shoulder': [batch, hidden_dim],
        #     'elbow': [batch, hidden_dim],
        #     'forearm': [batch, hidden_dim],
        #     'wrist': [batch, hidden_dim]
        # }

        # 2. 计算 Pieper 权重（使用全部历史帧）
        pos_weights, ori_weights = self.pieper_attention(history_frames)

        # 3. 生成 FiLM 参数（从加权历史特征）
        # 对历史帧应用位置权重并聚合
        weighted_history = history_frames * pos_weights.unsqueeze(1)  # [batch, num_frames, 7]
        aggregated_joints = weighted_history.mean(dim=1)  # [batch, 7]

        pos_gamma, pos_beta, ori_gamma, ori_beta = self.pieper_film(
            aggregated_joints, pos_weights, ori_weights
        )

        # 5. FiLM 调制末端位姿（调制原始位置和四元数，然后编码）
        modulated_pos = pos_gamma * end_position + pos_beta  # [batch, 3]
        modulated_ori = ori_gamma * end_orientation + ori_beta  # [batch, 4]

        # 6. 编码调制后的末端位姿
        end_pos_feat, end_ori_feat = self.endpose_encoder(modulated_pos, modulated_ori)

        # 7. 融合末端位姿特征
        endeff_feat = end_pos_feat + end_ori_feat  # [batch, hidden_dim]

        # 8. 为每个关节组生成 FiLM 调制特征
        film_features = {}
        for group in self.joint_groups.keys():
            # 根据关节组与末端位姿的关联程度分配特征
            if group == 'shoulder':
                feat = end_pos_feat  # shoulder 与位置强相关
            elif group == 'wrist':
                feat = end_ori_feat  # wrist 与姿态强相关
            else:
                feat = (end_pos_feat + end_ori_feat) / 2  # 其他关节均衡

            film_features[group] = feat

        # 9. 初始化关节节点（使用时序编码的历史特征）
        nodes = joint_features.copy()  # 直接使用时序特征作为节点

        # 10. 消息传递（因果链）
        for layer in self.message_layers:
            # Shoulder → Elbow
            elbow_msg = layer['shoulder_to_elbow'](nodes['shoulder'])
            nodes['elbow'] = nodes['elbow'] + elbow_msg

            # Shoulder + Elbow → Forearm
            se_concat = torch.cat([nodes['shoulder'], nodes['elbow']], dim=1)
            forearm_msg = layer['to_forearm'](se_concat)
            # 门控融合
            gate = layer['forearm_gate'](forearm_msg)
            nodes['forearm'] = nodes['forearm'] * (1 - gate) + forearm_msg * gate

            # Elbow + Forearm → Wrist
            ef_concat = torch.cat([nodes['elbow'], nodes['forearm']], dim=1)
            wrist_msg = layer['forearm_to_wrist'](ef_concat)
            # 门控融合
            gate = layer['wrist_gate'](wrist_msg)
            nodes['wrist'] = nodes['wrist'] * (1 - gate) + wrist_msg * gate

        # 11. 多特征融合（GNN 特征 + FiLM 特征）
        fused_nodes = self.feature_fusion(nodes, film_features, endeff_feat)

        # 12. 输出预测（使用融合后的特征）
        pred_shoulder = self.output_heads['shoulder'](fused_nodes['shoulder'])
        pred_elbow = self.output_heads['elbow'](fused_nodes['elbow'])
        pred_forearm = self.output_heads['forearm'](fused_nodes['forearm'])
        pred_wrist = self.output_heads['wrist'](fused_nodes['wrist'])

        # 拼接
        pred_angles = torch.cat([pred_shoulder, pred_elbow, pred_forearm, pred_wrist], dim=1)

        return pred_angles, {
            'position_weights': pos_weights,
            'orientation_weights': ori_weights,
            'modulated_pos': modulated_pos,
            'modulated_ori': modulated_ori
        }


# 测试代码
if __name__ == '__main__':
    print("=" * 70)
    print("测试 Pieper 准则因果IK模型（支持历史输入）")
    print("=" * 70)

    # 创建模型
    model = PieperCausalIK(num_joints=7, num_frames=10, hidden_dim=256, num_layers=2)
    model = model.cuda()
    model.eval()

    # 测试数据
    batch_size = 4
    num_frames = 10
    history_frames = torch.randn(batch_size, num_frames, 7).cuda()
    end_position = torch.randn(batch_size, 3).cuda()
    end_orientation = torch.randn(batch_size, 4).cuda()

    print(f"\n输入:")
    print(f"  - 历史帧: {history_frames.shape}")
    print(f"  - 末端位置: {end_position.shape}")
    print(f"  - 末端姿态: {end_orientation.shape}")

    # 前向传播
    with torch.no_grad():
        pred_angles, info = model(history_frames, end_position, end_orientation)

    print(f"\n输出:")
    print(f"  - 预测角度: {pred_angles.shape}")
    print(f"  - 位置影响权重: {info['position_weights'].shape}")
    print(f"  - 姿态影响权重: {info['orientation_weights'].shape}")

    # 分析权重
    print(f"\nPieper 权重分析（第一个样本）:")
    print(f"  关节对位置的影响:")
    for i, w in enumerate(info['position_weights'][0]):
        print(f"    J{i}: {w:.4f}")

    print(f"  关节对姿态的影响:")
    for i, w in enumerate(info['orientation_weights'][0]):
        print(f"    J{i}: {w:.4f}")

    print(f"\n总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("\n架构说明:")
    print("  1. 时序编码: 分别对每个关节组编码10帧历史")
    print("     - shoulder: [batch, 10, 3] → [batch, 256]")
    print("     - elbow: [batch, 10, 1] → [batch, 256]")
    print("     - forearm: [batch, 10, 1] → [batch, 256]")
    print("     - wrist: [batch, 10, 2] → [batch, 256]")
    print("  2. PieperAttention: 从全部历史帧计算关节影响权重")
    print("     - 输入: [batch, 10, 7] → 时序编码 → 权重 [batch, 7]")
    print("  3. FiLM: 使用权重调制末端位姿")
    print("  4. GNN: 沿因果链传播信息")
    print("  5. 多特征融合: GNN特征 + FiLM特征 → 融合特征")
    print("     - 门控融合机制，自适应融合比例")
    print("=" * 70)
