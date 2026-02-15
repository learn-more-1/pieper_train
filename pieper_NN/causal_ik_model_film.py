"""
基于图神经网络的因果IK模型

核心思想：
1. 将关节建模为图的节点
2. 边表示因果依赖关系（运动学链）
3. 通过消息传递学习关节间的耦合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MessagePassingBlock(nn.Module):
    """改进的消息传递块：包含归一化、激活、残差"""

    def __init__(self, in_dim, out_dim, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        # 如果维度不同且使用残差，需要投影
        if use_residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = None

    def forward(self, x):
        """
        Args:
            x: [batch, in_dim]
        Returns:
            out: [batch, out_dim]
        """
        identity = x

        out = self.linear1(x)
        out = self.norm1(out)
        out = F.gelu(out)  # GELU 比 ReLU 更平滑

        out = self.linear2(out)
        out = self.norm2(out)
        out = F.gelu(out)

        # 残差连接
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            out = out + identity

        return out



#接近 1：完全开放，让对应信息大量通过
#接近 0：完全关闭，阻断对应信息
#中间值：部分开放，按比例通过
class GatedMessageFusion(nn.Module):
    """门控消息融合：动态控制多条消息的权重"""

    def __init__(self, dim, num_inputs):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * num_inputs, dim),
            nn.Sigmoid()
        )

    def forward(self, *inputs):
        """
        Args:
            inputs: 多个 [batch, dim] 的张量
        Returns:
            fused: [batch, dim] 融合后的特征
        """
        concat = torch.cat(inputs, dim=-1)
        gate = self.gate(concat)

        # 加权平均
        stacked = torch.stack(inputs, dim=0)  # [num_inputs, batch, dim]
        weighted = stacked * gate.unsqueeze(0)
        fused = weighted.sum(dim=0)

        return fused


class FiLMGenerator(nn.Module):
    """FiLM 参数生成器：从条件 y 生成缩放(γ)和偏置(β)参数"""

    def __init__(self, condition_dim, feature_dim, hidden_dim=None):
        """
        Args:
            condition_dim: 条件 y 的维度（目标位姿）
            feature_dim: 要调制的特征维度
            hidden_dim: 隐藏层维度（可选，默认与 feature_dim 相同）
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = feature_dim

        # 生成 γ (scale) 参数
        self.gamma_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        # 生成 β (shift) 参数
        self.beta_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        # 初始化：γ 初始化为 1，β 初始化为 0（初始时无调制）
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.zeros_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, condition):
        """
        Args:
            condition: [batch, condition_dim] 条件特征（目标位姿）

        Returns:
            gamma: [batch, feature_dim] 缩放参数
            beta: [batch, feature_dim] 偏置参数
        """
        gamma = self.gamma_net(condition)
        beta = self.beta_net(condition)
        return gamma, beta


def apply_film(features, gamma, beta):
    """
    应用 FiLM 调制

    Args:
        features: [batch, feature_dim] 要调制的特征
        gamma: [batch, feature_dim] 缩放参数
        beta: [batch, feature_dim] 偏置参数

    Returns:
        modulated_features: [batch, feature_dim] 调制后的特征
    """
    # FiLM: γ ⊙ x + β
    return gamma * features + beta


class CausalIKGNN(nn.Module):
    """基于GNN的因果IK模型

    因果图结构：
        肩(J0-2) → 肘(J3) → 前臂(J4) → 手腕(J5-6)
    """

    def __init__(self, input_dim=21, output_dim=7, hidden_dim=256):
        super().__init__()

        # 关节分组（按照运动学链）
        self.joint_groups = {
            'shoulder': [0, 1, 2],      # 3个关节
            'elbow': [3],                # 1个关节
            'forearm': [4],              # 1个关节 wrist_roll
            'wrist': [5, 6]             # 2个关节
        }

        # 1. 输入编码（全局特征）
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 2. 关节节点初始化
        self.joint_embeddings = nn.ModuleDict({
            'shoulder': nn.Linear(hidden_dim, hidden_dim),
            'elbow': nn.Linear(hidden_dim, hidden_dim),
            'forearm': nn.Linear(hidden_dim, hidden_dim),
            'wrist': nn.Linear(hidden_dim, hidden_dim),
        })

        # 3. 消息传递层（学习因果耦合）
        # 肩 → 肘
        self.shoulder_to_elbow = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 肩 + 肘 → 前臂
        self.elbow_to_forearm = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 前臂 → 手腕
        self.forearm_to_wrist = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 4. 输出头（从每个关节节点预测对应角度）
        self.output_heads = nn.ModuleDict({
            'shoulder': nn.Linear(hidden_dim, 3),   # 3个肩关节
            'elbow': nn.Linear(hidden_dim, 1),     # 1个肘关节
            'forearm': nn.Linear(hidden_dim, 1),   # 1个前臂关节
            'wrist': nn.Linear(hidden_dim, 2),     # 2个腕关节
        })

    def forward(self, x):
        """
        Args:
            x: [batch, input_dim] 手腕位姿特征

        Returns:
            joint_angles: [batch, 7] 关节角度
        """
        batch_size = x.shape[0]

        # 1. 全局特征编码
        global_features = self.input_encoder(x)  # [batch, hidden_dim]

        # 2. 初始化关节节点
        nodes = {}
        for group in self.joint_groups:
            nodes[group] = self.joint_embeddings[group](global_features)

        # 3. 层次化消息传递（沿着因果链）

        # 步骤1: 肩 → 肘
        elbow_message = self.shoulder_to_elbow(nodes['shoulder'])
        nodes['elbow'] = nodes['elbow'] + elbow_message

        # 步骤2: 肩 + 肘 → 前臂
        elbow_forearm_concat = torch.cat([nodes['shoulder'], nodes['elbow']], dim=1)
        forearm_message = self.elbow_to_forearm(elbow_forearm_concat)
        nodes['forearm'] = nodes['forearm'] + forearm_message

        # 步骤3: 前臂 → 手腕
        wrist_message = self.forearm_to_wrist(nodes['forearm'])
        nodes['wrist'] = nodes['wrist'] + wrist_message

        # 4. 预测各关节角度
        pred_shoulder = self.output_heads['shoulder'](nodes['shoulder'])  # [batch, 3]
        pred_elbow = self.output_heads['elbow'](nodes['elbow'])          # [batch, 1]
        pred_forearm = self.output_heads['forearm'](nodes['forearm'])    # [batch, 1]
        pred_wrist = self.output_heads['wrist'](nodes['wrist'])          # [batch, 2]

        # 5. 拼接输出
        joint_angles = torch.cat([pred_shoulder, pred_elbow, pred_forearm, pred_wrist], dim=1)

        return joint_angles, nodes

    def get_causal_graph(self):
        """返回因果图的邻接矩阵（用于可视化）"""
        # 节点: 肩(3), 肘(1), 前臂(1), 手腕(2) = 7个关节
        adj_matrix = np.zeros((7, 7))

        # 因果边（父节点 → 子节点）
        # 肩(0-2) → 肘(3)
        adj_matrix[:3, 3] = 1

        # 肘(3) → 前臂(4)
        adj_matrix[3, 4] = 1

        # 前臂(4) → 手腕(5-6)
        adj_matrix[4, 5:] = 1

        return adj_matrix


class CausalIKGNNv2(nn.Module):
    """改进版因果IK模型 + FiLM条件调制

    改进点：
    1. GELU激活函数替代ReLU（更平滑的梯度）
    2. 每层添加LayerNorm（训练更稳定）
    3. 多层消息传递 + 残差连接（更强的特征表达）
    4. 门控机制（动态控制信息流动）
    5. FiLM条件调制（目标位姿作为条件）
    """

    def __init__(self, input_dim=21, output_dim=7, hidden_dim=256, num_layers=2,
                 condition_dim=7, use_film=True):
        """
        Args:
            input_dim: 历史帧特征维度
            output_dim: 输出关节角度维度
            hidden_dim: 隐藏层维度
            num_layers: 消息传递层数
            condition_dim: 条件维度（目标位姿维度，默认7）
            use_film: 是否使用FiLM条件调制
        """
        super().__init__()

        # 关节分组（按照运动学链）
        self.joint_groups = {
            'shoulder': [0, 1, 2],
            'elbow': [3],
            'forearm': [4],
            'wrist': [5, 6]
        }
        self.num_layers = num_layers
        self.use_film = use_film

        # 1. 输入编码（全局特征）- 使用改进的激活函数
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # FiLM: 条件编码器（编码目标位姿 y）
        if use_film:
            self.condition_encoder = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

            # 为每层消息传递生成 FiLM 参数（动态控制因果流）
            # 关键改进：调制消息本身，而非节点特征
            self.film_generators = nn.ModuleList()
            for _ in range(num_layers):
                layer_films = nn.ModuleDict({
                    'shoulder_to_elbow': FiLMGenerator(hidden_dim, hidden_dim),    # 调制肩→肘消息
                    'to_forearm': FiLMGenerator(hidden_dim, hidden_dim),          # 调制肩+肘→前臂消息
                    'forearm_to_wrist': FiLMGenerator(hidden_dim, hidden_dim),    # 调制肘+前臂→手腕消息
                })
                self.film_generators.append(layer_films)

        # 2. 关节节点初始化（带归一化）
        self.joint_embeddings = nn.ModuleDict({
            'shoulder': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ),
            'elbow': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ),
            'forearm': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ),
            'wrist': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ),
        })

        # 3. 多层消息传递（沿因果链）
        # 每一层都是独立的消息传递模块
        self.message_layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            layer_modules = nn.ModuleDict()

            # 肩 → 肘
            layer_modules['shoulder_to_elbow'] = MessagePassingBlock(hidden_dim, hidden_dim)

            # 肩 + 肘 → 前臂（使用门控融合）
            layer_modules['elbow_to_forearm_msg'] = MessagePassingBlock(hidden_dim * 2, hidden_dim)
            layer_modules['elbow_to_forearm_gate'] = GatedMessageFusion(hidden_dim, 2)

            # 肘 + 前臂 → 手腕（使用门控融合）
            layer_modules['forearm_to_wrist_msg'] = MessagePassingBlock(hidden_dim * 2, hidden_dim)
            layer_modules['forearm_to_wrist_gate'] = GatedMessageFusion(hidden_dim, 2)

            self.message_layers.append(layer_modules)

        # 4. 输出头（改进：添加归一化）
        self.output_heads = nn.ModuleDict({
            'shoulder': nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 3)
            ),
            'elbow': nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1)
            ),
            'forearm': nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1)
            ),
            'wrist': nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 2)
            ),
        })

        # 5. 初始化权重（Xavier初始化）
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, y=None):
        """
        Args:
            x: [batch, input_dim] 历史帧特征
            y: [batch, condition_dim] 目标位姿（条件，可选）

        Returns:
            joint_angles: [batch, 7] 关节角度
        """
        # 1. 全局特征编码
        global_features = self.input_encoder(x)  # [batch, hidden_dim]

        # 2. 编码条件（如果提供）
        if self.use_film and y is not None:
            condition_features = self.condition_encoder(y)  # [batch, hidden_dim]
        else:
            condition_features = None

        # 3. 初始化关节节点（不再应用 FiLM）
        nodes = {}
        for group in self.joint_groups:
            nodes[group] = self.joint_embeddings[group](global_features)

        # 4. 使用节点消息传递
        return self._forward_with_nodes(nodes, condition_features)

    def forward_with_nodes(self, nodes, y=None, return_intermediates=False):
        """
        使用预初始化的节点进行前向传播

        Args:
            nodes: dict, 预初始化的关节节点特征
            y: [batch, condition_dim] 目标位姿（条件，可选）
            return_intermediates: 是否返回中间节点特征

        Returns:
            joint_angles: [batch, 7] 关节角度
            nodes: dict (可选) 更新后的节点特征
        """
        # 1. 编码条件（如果提供）
        if self.use_film and y is not None:
            condition_features = self.condition_encoder(y)  # [batch, hidden_dim]
        else:
            condition_features = None

        # 2. 使用预初始化的节点进行消息传递
        joint_angles, updated_nodes = self._forward_with_nodes(nodes, condition_features)

        if return_intermediates:
            return joint_angles, updated_nodes
        return joint_angles

    def _forward_with_nodes(self, nodes, condition_features):
        """
        使用给定的节点特征进行消息传递

        Args:
            nodes: dict, 关节节点特征
            condition_features: [batch, hidden_dim] 条件特征（可选）

        Returns:
            joint_angles: [batch, 7] 关节角度
            nodes: dict 更新后的节点特征
        """
        # 多层消息传递（在消息传递中应用 FiLM）
        for layer_idx, layer in enumerate(self.message_layers):

            # === 肩 → 肘 ===
            elbow_msg = layer['shoulder_to_elbow'](nodes['shoulder'])
            # FiLM 调制消息（关键改进！）
            if self.use_film and condition_features is not None:
                gamma, beta = self.film_generators[layer_idx]['shoulder_to_elbow'](condition_features)
                elbow_msg = apply_film(elbow_msg, gamma, beta)
            nodes['elbow'] = nodes['elbow'] + elbow_msg

            # === 肩 + 肘 → 前臂 ===
            se_concat = torch.cat([nodes['shoulder'], nodes['elbow']], dim=1)
            forearm_msg = layer['elbow_to_forearm_msg'](se_concat)
            # FiLM 调制消息
            if self.use_film and condition_features is not None:
                gamma, beta = self.film_generators[layer_idx]['to_forearm'](condition_features)
                forearm_msg = apply_film(forearm_msg, gamma, beta)
            # 使用门控融合原始特征和消息
            nodes['forearm'] = layer['elbow_to_forearm_gate'](nodes['forearm'], forearm_msg)

            # === 肘 + 前臂 → 手腕 ===
            fw_concat = torch.cat([nodes['elbow'], nodes['forearm']], dim=1)
            wrist_msg = layer['forearm_to_wrist_msg'](fw_concat)
            # FiLM 调制消息
            if self.use_film and condition_features is not None:
                gamma, beta = self.film_generators[layer_idx]['forearm_to_wrist'](condition_features)
                wrist_msg = apply_film(wrist_msg, gamma, beta)
            # 使用门控融合原始特征和消息
            nodes['wrist'] = layer['forearm_to_wrist_gate'](nodes['wrist'], wrist_msg)

        # 5. 预测各关节角度
        pred_shoulder = self.output_heads['shoulder'](nodes['shoulder'])
        pred_elbow = self.output_heads['elbow'](nodes['elbow'])
        pred_forearm = self.output_heads['forearm'](nodes['forearm'])
        pred_wrist = self.output_heads['wrist'](nodes['wrist'])

        # 6. 拼接输出
        joint_angles = torch.cat([pred_shoulder, pred_elbow, pred_forearm, pred_wrist], dim=1)

        return joint_angles, nodes

    def get_causal_graph(self):
        """返回因果图的邻接矩阵（用于可视化）"""
        adj_matrix = np.zeros((7, 7))
        adj_matrix[:3, 3] = 1  # 肩 → 肘
        adj_matrix[3, 4] = 1   # 肘 → 前臂
        adj_matrix[4, 5:] = 1  # 前臂 → 手腕
        return adj_matrix



class TemporalEncoder(nn.Module):
    """提取历史帧的时序特征"""
    def __init__(self, frame_dim=7, num_frames=10, hidden_dim=256):
        super().__init__()
        self.frame_dim = frame_dim
        self.num_frames = num_frames

        # 方案1: 1D卷积（轻量、快速）
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(frame_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # [batch, hidden_dim, 1]
        )

    def forward(self, x):
        # x: [batch, num_frames * frame_dim]
        batch_size = x.shape[0]

        # 重塑为 [batch, frame_dim, num_frames]
        x = x.view(batch_size, self.frame_dim, self.num_frames)

        # 时序特征提取
        features = self.temporal_conv(x).squeeze(-1)  # [batch, hidden_dim]

        return features


class JointwiseTemporalEncoder(nn.Module):
    """为每个关节单独编码历史特征"""

    def __init__(self, num_frames=10, hidden_dim=256):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

        # 为每个关节组创建独立的时序编码器
        # shoulder: 3维, elbow: 1维, forearm: 1维, wrist: 2维
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
            'shoulder': self.encoders['shoulder'](history_shoulder).squeeze(-1),  # [batch, hidden_dim]
            'elbow': self.encoders['elbow'](history_elbow).squeeze(-1),
            'forearm': self.encoders['forearm'](history_forearm).squeeze(-1),
            'wrist': self.encoders['wrist'](history_wrist).squeeze(-1)
        }

        return features



class PhysicsAwareCausalIKWithHistory(nn.Module):
    def __init__(self, num_gnn_layers, frame_dim=7, num_frames=10, output_dim=7,
                 hidden_dim=256, condition_dim=7, use_film=True):
        """
        Args:
            num_gnn_layers: GNN消息传递层数
            frame_dim: 每帧维度
            num_frames: 历史帧数
            output_dim: 输出关节角度维度
            hidden_dim: 隐藏层维度
            condition_dim: 条件维度（目标位姿维度）
            use_film: 是否使用FiLM条件调制
        """
        super().__init__()

        # 关节分组
        self.joint_groups = {
            'shoulder': [0, 1, 2],
            'elbow': [3],
            'forearm': [4],
            'wrist': [5, 6]
        }
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

        # 关节级别的时序编码器：为每个关节单独编码历史
        self.jointwise_temporal_encoder = JointwiseTemporalEncoder(
            num_frames=num_frames,
            hidden_dim=hidden_dim
        )

        # GNN with FiLM
        self.gnn = CausalIKGNNv2(
            input_dim=hidden_dim,  # 使用关节级别的特征
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            condition_dim=condition_dim,
            use_film=use_film
        )

        # 关节限制（Unitree G1左臂）
        # 格式: (joint_idx, min_angle, max_angle)
        self.joint_limits = torch.tensor([
            [-2.87, 2.87],  # J0: shoulder_pitch
            [-0.59, 0.59],  # J1: shoulder_roll
            [-1.76, 1.76],  # J2: shoulder_yaw
            [-0.09, 2.49],  # J3: elbow
            [-1.67, 1.67],  # J4: forearm_roll
            [-1.57, 1.57],  # J5: wrist_yaw (实际受限)
            [-1.57, 1.57],  # J6: wrist_pitch (实际受限)
        ], dtype=torch.float32)

        # FK近似网络（用于计算FK一致性损失）
        self.fk_approximator = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 预测手腕位置
        )

    def forward(self, history_frames, target_pose=None, return_intermediates=False):
        """
        Args:
            history_frames: [batch, num_frames * 7] 历史关节角度
            target_pose: [batch, condition_dim] 目标位姿（条件，可选）
            return_intermediates: 是否返回中间节点特征（用于计算耦合损失）

        Returns:
            joint_angles: [batch, 7] 关节角度
            nodes: dict (可选) 中间节点特征
        """
        # 1. 重塑历史帧
        batch_size = history_frames.shape[0]
        history_reshaped = history_frames.view(batch_size, self.num_frames, 7)  # [batch, 10, 7]

        # 2. 关节级别的时序编码
        joint_features = self.jointwise_temporal_encoder(history_reshaped)
        # joint_features = {
        #     'shoulder': [batch, hidden_dim],
        #     'elbow': [batch, hidden_dim],
        #     'forearm': [batch, hidden_dim],
        #     'wrist': [batch, hidden_dim]
        # }

        # 3. 初始化关节节点（用自己的历史特征）
        nodes = {}
        for group in self.joint_groups:
            nodes[group] = joint_features[group]  # 直接使用时序编码的特征

        # 4. GNN推理（带条件调制）- 传入节点特征进行消息传递
        # 注意：需要修改 GNN forward 来接受预初始化的节点
        joint_angles, nodes = self.gnn.forward_with_nodes(
            nodes, target_pose, return_intermediates=True
        )

        if return_intermediates:
            return joint_angles, nodes
        return joint_angles

    def apply_joint_limits(self, joint_angles):
        """应用关节限制（使用sigmoid缩放）"""
        # 对每个关节单独缩放到物理范围内
        limited = torch.zeros_like(joint_angles)

        for i in range(7):
            min_val = self.joint_limits[i, 0]
            max_val = self.joint_limits[i, 1]
            range_val = max_val - min_val

            # 使用sigmoid将任意值缩放到[min, max]区间
            limited[:, i] = min_val + range_val * torch.sigmoid(joint_angles[:, i])

        return limited

    def compute_fk_loss(self, joint_angles, target_wrist_pos):
        """
        计算FK一致性损失（辅助训练）

        通过学习FK近似，让模型理解关节→位置的因果关系
        """
        predicted_wrist_pos = self.fk_approximator(joint_angles)
        fk_loss = F.mse_loss(predicted_wrist_pos, target_wrist_pos)
        return fk_loss

    def compute_joint_coupling_loss(self, nodes):
        """
        计算关节耦合一致性损失

        确保子节点的特征合理地依赖于父节点
        """
        # 肘应该依赖肩
        shoulder_elbow_coupling = F.cosine_similarity(
            nodes['shoulder'].detach(),
            nodes['elbow'],
            dim=-1
        ).mean()

        # 前臂应该依赖肘
        elbow_forearm_coupling = F.cosine_similarity(
            nodes['elbow'].detach(),
            nodes['forearm'],
            dim=-1
        ).mean()

        # 我们希望有一定的一致性，但不完全相关
        # 目标：0.3-0.7 之间（有因果关系但不完全线性）
        coupling_loss = (
            F.relu(shoulder_elbow_coupling - 0.7) +  # 不超过0.7
            F.relu(0.3 - shoulder_elbow_coupling) +  # 不低于0.3
            F.relu(elbow_forearm_coupling - 0.7) +
            F.relu(0.3 - elbow_forearm_coupling)
        )

        return coupling_loss


class PhysicsAwareCausalIK(nn.Module):
    """物理感知的因果IK模型

    结合：
    1. 图神经网络（学习因果结构）
    2. 正向运动学约束（物理规律）
    3. 关节限制（硬约束）
    """

    def __init__(self, input_dim=21, output_dim=7, hidden_dim=256,
                 use_v2=True, num_gnn_layers=2):
        """
        Args:
            use_v2: 是否使用改进版 GNN (默认True，推荐)
            num_gnn_layers: GNN 消息传递层数 (仅 v2)
        """
        super().__init__()

        # GNN backbone - 选择版本
        if use_v2:
            self.gnn = CausalIKGNNv2(input_dim, output_dim, hidden_dim, num_layers=num_gnn_layers)
        else:
            self.gnn = CausalIKGNN(input_dim, output_dim, hidden_dim)

        # FK近似网络（用于计算FK一致性损失）
        self.fk_approximator = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 预测手腕位置
        )

        # 关节限制（Unitree G1左臂）
        # 格式: (joint_idx, min_angle, max_angle)
        self.joint_limits = torch.tensor([
            [-2.87, 2.87],  # J0: shoulder_pitch
            [-0.59, 0.59],  # J1: shoulder_roll
            [-1.76, 1.76],  # J2: shoulder_yaw
            [-0.09, 2.49],  # J3: elbow
            [-1.67, 1.67],  # J4: forearm_roll
            [-1.57, 1.57],  # J5: wrist_yaw (实际受限)
            [-1.57, 1.57],  # J6: wrist_pitch (实际受限)
        ], dtype=torch.float32)

    def forward(self, wrist_pose, return_intermediates=False):
        """
        Args:
            wrist_pose: [batch, 21] 手腕位姿特征（包含位置、姿态、速度、加速度）

        Returns:
            joint_angles: [batch, 7] 关节角度
        """
        joint_angles, nodes = self.gnn(wrist_pose)

        # 暂时关闭关节限制（sigmoid导致梯度消失）
        # joint_angles = self.apply_joint_limits(joint_angles)

        if return_intermediates:
            return joint_angles, nodes
        return joint_angles

    def apply_joint_limits(self, joint_angles):
        """应用关节限制（使用sigmoid缩放）"""
        # 对每个关节单独缩放到物理范围内
        limited = torch.zeros_like(joint_angles)

        for i in range(7):
            min_val = self.joint_limits[i, 0]
            max_val = self.joint_limits[i, 1]
            range_val = max_val - min_val

            # 使用sigmoid将任意值缩放到[min, max]区间
            limited[:, i] = min_val + range_val * torch.sigmoid(joint_angles[:, i])

        return limited

    def compute_fk_loss(self, joint_angles, target_wrist_pos):
        """
        计算FK一致性损失（辅助训练）

        通过学习FK近似，让模型理解关节→位置的因果关系
        """
        predicted_wrist_pos = self.fk_approximator(joint_angles)
        fk_loss = F.mse_loss(predicted_wrist_pos, target_wrist_pos)
        return fk_loss

    def compute_joint_coupling_loss(self, nodes):
        """
        计算关节耦合一致性损失

        确保子节点的特征合理地依赖于父节点
        """
        # 肘应该依赖肩
        shoulder_elbow_coupling = F.cosine_similarity(
            nodes['shoulder'].detach(),
            nodes['elbow'],
            dim=-1
        ).mean()

        # 前臂应该依赖肘
        elbow_forearm_coupling = F.cosine_similarity(
            nodes['elbow'].detach(),
            nodes['forearm'],
            dim=-1
        ).mean()

        # 我们希望有一定的一致性，但不完全相关
        # 目标：0.3-0.7 之间（有因果关系但不完全线性）
        coupling_loss = (
            F.relu(shoulder_elbow_coupling - 0.7) +  # 不超过0.7
            F.relu(0.3 - shoulder_elbow_coupling) +  # 不低于0.3
            F.relu(elbow_forearm_coupling - 0.7) +
            F.relu(0.3 - elbow_forearm_coupling)
        )

        return coupling_loss


# ==================== 训练时使用的物理感知损失 ====================

class PhysicsAwareLoss(nn.Module):
    """物理感知的损失函数"""

    def __init__(self, ik_weight=1.0, fk_weight=0.1, coupling_weight=0.05):
        super().__init__()
        self.ik_weight = ik_weight
        self.fk_weight = fk_weight
        self.coupling_weight = coupling_weight

    def forward(self, model, batch):
        """
        计算多任务损失

        batch包含:
        - wrist_pose_features: [batch, 21]
        - target_joint_angles: [batch, 7]
        - target_wrist_pos: [batch, 3]（真实手腕位置，用于FK损失）
        """
        wrist_pose = batch['wrist_pose_features']
        target_angles = batch['target_joint_angles']
        target_pos = batch['target_wrist_pos']

        # 前向传播
        joint_angles, nodes = model(wrist_pose, return_intermediates=True)

        # 1. IK损失（主任务）
        ik_loss = F.mse_loss(joint_angles, target_angles)

        # 2. FK一致性损失（物理约束）
        fk_loss = model.compute_fk_loss(joint_angles, target_pos)

        # 3. 关节耦合损失（因果结构）
        coupling_loss = model.compute_joint_coupling_loss(nodes)

        # 4. 关节限制惩罚
        limit_penalty = self.compute_limit_penalty(joint_angles)

        # 总损失
        total_loss = (
            self.ik_weight * ik_loss +
            self.fk_weight * fk_loss +
            self.coupling_weight * coupling_loss +
            0.01 * limit_penalty
        )

        return total_loss, {
            'ik_loss': ik_loss.item(),
            'fk_loss': fk_loss.item(),
            'coupling_loss': coupling_loss.item(),
            'limit_penalty': limit_penalty.item()
        }

    def compute_limit_penalty(self, joint_angles):
        """计算关节限制惩罚"""
        # 这里简化处理，实际应该使用model中的joint_limits
        penalty = torch.zeros_like(joint_angles)

        # 假设所有关节限制在[-π, π]
        for i in range(7):
            # 超出[-π, π]的部分施加惩罚
            beyond_pos = F.relu(joint_angles[:, i] - np.pi)
            beyond_neg = F.relu(-np.pi - joint_angles[:, i])
            penalty[:, i] = beyond_pos + beyond_neg

        return penalty.mean()


class PositionBasedLoss(nn.Module):
    """直接优化末端位置误差的损失函数

    与 PhysicsAwareLoss 不同，这个损失函数直接以位置误差为主要优化目标，
    而不是关节角度误差。
    """

    def __init__(self, position_weight=1.0, joint_weight=0.01, coupling_weight=0.0):
        super().__init__()
        self.position_weight = position_weight
        self.joint_weight = joint_weight
        self.coupling_weight = coupling_weight

    def forward(self, pred_wrist_pos, true_wrist_pos, pred_angles, true_angles, nodes=None):
        """
        Args:
            pred_wrist_pos: [batch, 3] 预测的手腕位置（通过FK计算）
            true_wrist_pos: [batch, 3] 真实的手腕位置
            pred_angles: [batch, 7] 预测的关节角度
            true_angles: [batch, 7] 真实的关节角度
            nodes: dict, 关节节点特征（可选，用于耦合损失）

        Returns:
            total_loss, loss_dict
        """
        # 1. 位置误差损失（主任务）
        position_loss = F.mse_loss(pred_wrist_pos, true_wrist_pos)

        # 2. 关节角度损失（辅助任务，防止解太极端）
        joint_loss = F.mse_loss(pred_angles, true_angles)

        # 3. 关节耦合损失（可选）
        coupling_loss = 0.0
        if nodes is not None and self.coupling_weight > 0:
            shoulder_elbow_coupling = F.cosine_similarity(
                nodes['shoulder'].detach(),
                nodes['elbow'],
                dim=-1
            ).mean()

            elbow_forearm_coupling = F.cosine_similarity(
                nodes['elbow'].detach(),
                nodes['forearm'],
                dim=-1
            ).mean()

            coupling_loss = (
                F.relu(shoulder_elbow_coupling - 0.7) +
                F.relu(0.3 - shoulder_elbow_coupling) +
                F.relu(elbow_forearm_coupling - 0.7) +
                F.relu(0.3 - elbow_forearm_coupling)
            )

        # 总损失
        total_loss = (
            self.position_weight * position_loss +
            self.joint_weight * joint_loss +
            self.coupling_weight * coupling_loss
        )

        return total_loss, {
            'position_loss': position_loss.item(),
            'joint_loss': joint_loss.item(),
            'coupling_loss': coupling_loss if isinstance(coupling_loss, float) else coupling_loss.item()
        }


# ==================== 使用示例 ====================

if __name__ == '__main__':
    """测试因果IK模型 - 比较两个版本"""

    print("=" * 60)
    print("对比测试: CausalIKGNN vs CausalIKGNNv2")
    print("=" * 60)

    # 模拟输入
    batch_size = 4
    wrist_pose_features = torch.randn(batch_size, 21)

    # ==================== 原版模型 ====================
    print("\n【原版】 CausalIKGNN")
    model_v1 = PhysicsAwareCausalIK(
        input_dim=21,
        output_dim=7,
        hidden_dim=256,
        use_v2=False
    )

    joint_angles_v1, nodes_v1 = model_v1(wrist_pose_features, return_intermediates=True)
    params_v1 = sum(p.numel() for p in model_v1.parameters()) / 1e6

    print(f"  参数量: {params_v1:.2f}M")
    print(f"  输出形状: {joint_angles_v1.shape}")

    # ==================== 改进版模型 ====================
    print("\n【改进版】 CausalIKGNNv2")
    model_v2 = PhysicsAwareCausalIK(
        input_dim=21,
        output_dim=7,
        hidden_dim=256,
        use_v2=True,
        num_gnn_layers=2
    )

    joint_angles_v2, nodes_v2 = model_v2(wrist_pose_features, return_intermediates=True)
    params_v2 = sum(p.numel() for p in model_v2.parameters()) / 1e6

    print(f"  参数量: {params_v2:.2f}M")
    print(f"  输出形状: {joint_angles_v2.shape}")
    print(f"  参数增加: {params_v2 - params_v1:.2f}M ({(params_v2/params_v1-1)*100:.1f}%)")

    # ==================== 关节限制测试 ====================
    test_angles = torch.randn(1, 7) * 10  # 超出范围的测试值
    limited_angles = model_v2.apply_joint_limits(test_angles)
    print(f"\n关节限制测试:")
    print(f"  输入: {test_angles}")
    print(f"  限制后: {limited_angles}")
