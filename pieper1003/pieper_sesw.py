"""
基于 Pieper 准则的因果IK模型（注意力+FiLM两阶段）

核心架构：
1. 【第一阶段】Pieper Attention：使用多头自注意力提取历史帧特征
2. 【第二阶段】FiLM调制：用 y 的位姿作为条件调制历史特征
3. 移除GNN，直接从调制后的特征预测

数据流：
- history_frames（全部10帧） → PieperAttention → joint_features
- y的位姿 → endpose_encoder → target_feat
- target_feat → FiLM生成器 → gamma, beta
- gamma * joint_features + beta → modulated_features
- modulated_features → 输出头 → 预测关节角度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PieperAttention(nn.Module):
    """
    Pieper 注意力机制：提取历史帧中的关键信息

    对每个关节组使用多头注意力处理历史帧：
    - 学习哪些历史时刻对预测更重要
    - 捕捉时序依赖关系
    """

    def __init__(self, input_dim, hidden_dim=256, num_heads=4, num_frames=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_frames = num_frames

        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 输入投影：将关节角度投影到 hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, joint_history):
        """
        Args:
            joint_history: [batch, num_frames, input_dim] 某个关节组的历史帧

        Returns:
            attended_feat: [batch, hidden_dim] 注意力提取的特征
        """
        batch_size = joint_history.shape[0]

        # 投影到 hidden_dim
        x = self.input_projection(joint_history)  # [batch, num_frames, hidden_dim]

        # 多头自注意力
        attn_output, attn_weights = self.multihead_attn(
            query=x,
            key=x,
            value=x
        )  # attn_output: [batch, num_frames, hidden_dim]

        # 残差连接 + 层归一化
        x = self.layer_norm(x + attn_output)

        # 前馈网络 + 残差连接
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)  # [batch, num_frames, hidden_dim]

        # 平均池化得到单一特征向量
        attended_feat = x.mean(dim=1)  # [batch, hidden_dim]

        return attended_feat


class JointwiseAttentionEncoder(nn.Module):
    """为每个关节组使用独立的注意力编码器"""

    def __init__(self, num_frames=10, hidden_dim=256, num_heads=4):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

        # 为每个关节组创建独立的注意力编码器
        self.attention_encoders = nn.ModuleDict({
            'shoulder': PieperAttention(input_dim=3, hidden_dim=hidden_dim, num_heads=num_heads, num_frames=num_frames),
            'es': PieperAttention(input_dim=3, hidden_dim=hidden_dim, num_heads=num_heads, num_frames=num_frames),
            'wristyaw': PieperAttention(input_dim=1, hidden_dim=hidden_dim, num_heads=num_heads, num_frames=num_frames)
        })

    def forward(self, history_frames):
        """
        Args:
            history_frames: [batch, num_frames, 7] 7个关节的历史角度
                索引: [0:3]=shoulder, [3]=elbow, [4]=forearm, [5]=wrist_pitch, [6]=wrist_yaw

        Returns:
            dict: 每个关节组的注意力特征
        """
        # 提取各关节组的历史
        history_shoulder = history_frames[:, :, :3]          # [batch, num_frames, 3]
        history_es = history_frames[:, :, 3:6]               # [batch, num_frames, 3] (elbow, forearm, wrist_pitch)
        history_wristyaw = history_frames[:, :, 6:7]         # [batch, num_frames, 1] (wrist_yaw)

        # 分别用注意力提取特征
        features = {
            'shoulder': self.attention_encoders['shoulder'](history_shoulder),
            'es': self.attention_encoders['es'](history_es),
            'wristyaw': self.attention_encoders['wristyaw'](history_wristyaw)
        }

        return features


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


class EndPoseEncoder(nn.Module):
    """编码末端位姿（位置 + 四元数）→ 生成目标特征"""

    def __init__(self, hidden_dim=256):
        super().__init__()

        # 位置编码器
        self.position_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # 四元数编码器
        self.orientation_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

    def forward(self, end_position, end_orientation):
        """
        Args:
            end_position: [batch, 3] 末端位置
            end_orientation: [batch, 4] 末端姿态（四元数）

        Returns:
            target_feat: [batch, hidden_dim] 融合的末端位姿特征
        """
        pos_feat = self.position_encoder(end_position)   # [batch, hidden_dim]
        ori_feat = self.orientation_encoder(end_orientation)  # [batch, hidden_dim]

        # 融合位置和姿态特征
        target_feat = pos_feat + ori_feat  # [batch, hidden_dim]

        return target_feat


class TargetConditionedFiLM(nn.Module):
    """
    基于目标位姿的FiLM生成器（修正版）

    核心修正：
    - 从目标位姿生成FiLM参数（而不是从当前关节角度）
    - 用FiLM参数调制历史关节特征

    FiLM公式: modulated_features = gamma * features + beta
    """

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 从目标特征生成FiLM参数
        # 为每个关节组生成独立的gamma和beta
        self.film_generators = nn.ModuleDict({
            'shoulder': self._make_film_generator(hidden_dim),
            'es': self._make_film_generator(hidden_dim),
            'wristyaw': self._make_film_generator(hidden_dim)
        })

    def _make_film_generator(self, hidden_dim):
        """创建FiLM参数生成器"""
        return nn.ModuleDict({
            'gamma': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            'beta': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        })

    def forward(self, joint_features, target_feat):
        """
        Args:
            joint_features: dict, 每个关节的历史特征
                {
                    'shoulder': [batch, hidden_dim],
                    'es': [batch, hidden_dim],
                    'wristyaw': [batch, hidden_dim]
                }
            target_feat: [batch, hidden_dim] 目标位姿特征

        Returns:
            modulated_features: dict, 调制后的关节特征
            film_params: dict, FiLM参数（用于调试）
        """
        modulated_features = {}
        film_params = {}

        for joint_name in ['shoulder', 'es', 'wristyaw']:
            # 从目标特征生成FiLM参数
            gamma = self.film_generators[joint_name]['gamma'](target_feat)  # [batch, hidden_dim]
            beta = self.film_generators[joint_name]['beta'](target_feat)    # [batch, hidden_dim]

            # FiLM调制历史关节特征
            original_feat = joint_features[joint_name]
            modulated_features[joint_name] = gamma * original_feat + beta

            # 保存参数用于调试
            film_params[joint_name] = {'gamma': gamma, 'beta': beta}

        return modulated_features, film_params


class PieperCausalIK(nn.Module):
    """
    基于 Pieper 准则的因果IK模型（注意力+FiLM两阶段）

    核心架构：
    1. 第一阶段：Pieper Attention 提取历史帧特征
    2. 第二阶段：FiLM调制（目标位姿 → gamma/beta → 调制历史特征）
    3. 移除GNN，直接从调制后的特征预测
    """

    def __init__(self, num_joints=7, num_frames=10, hidden_dim=256, num_layers=2, num_heads=4):
        super().__init__()

        self.joint_dim = num_joints
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 1. Pieper注意力编码器（处理全部历史帧）
        self.attention_encoder = JointwiseAttentionEncoder(num_frames, hidden_dim, num_heads)

        # 2. 末端位姿编码器（生成条件）
        self.endpose_encoder = EndPoseEncoder(hidden_dim)

        # 3. FiLM生成器（从目标位姿生成调制参数）
        self.target_conditioned_film = TargetConditionedFiLM(hidden_dim)

        # 4. 输出头（直接从调制后的特征预测）
        self.output_heads = nn.ModuleDict({
            'shoulder': nn.Linear(hidden_dim, 3),
            'es': nn.Linear(hidden_dim, 3),
            'wristyaw': nn.Linear(hidden_dim, 1)
        })

    def forward(self, history_frames, end_position, end_orientation):
        """
        Args:
            history_frames: [batch, num_frames, 7] 全部历史帧（使用所有10帧）
            end_position: [batch, 3] 末端位置
            end_orientation: [batch, 4] 末端姿态（四元数）- 可以为None

        Returns:
            pred_joint_angles: [batch, 7] 预测的关节角度
            info: dict, 包含调试信息的字典
        """
        batch_size = history_frames.shape[0]
        device = history_frames.device

        # 如果姿态为None，使用零tensor
        if end_orientation is None:
            end_orientation = torch.zeros(batch_size, 4, device=device)

        # ============ 第一阶段：Pieper Attention ============
        # 从全部历史帧提取关节特征（注意力机制）
        joint_features = self.attention_encoder(history_frames)
        # joint_features = {
        #     'shoulder': [batch, hidden_dim],
        #     'es': [batch, hidden_dim],
        #     'wristyaw': [batch, hidden_dim]
        # }

        # ============ 第二阶段：FiLM调制 ============
        # 编码目标位姿（条件）
        target_feat = self.endpose_encoder(end_position, end_orientation)
        # target_feat: [batch, hidden_dim]

        # 3. FiLM调制：用目标位姿调制历史关节特征
        modulated_features, film_params = self.target_conditioned_film(joint_features, target_feat)
        # modulated_features: 每个关节的特征已被目标位姿调制

        # 4. 直接输出预测（无GNN）
        pred_shoulder = self.output_heads['shoulder'](modulated_features['shoulder'])
        pred_es = self.output_heads['es'](modulated_features['es'])
        pred_wristyaw = self.output_heads['wristyaw'](modulated_features['wristyaw'])

        # 拼接
        pred_angles = torch.cat([pred_shoulder, pred_es, pred_wristyaw], dim=1)

        # 返回调试信息
        return pred_angles, {
            'target_feat_norm': torch.norm(target_feat, dim=1).mean().item(),
            'film_gamma_mean': {k: v['gamma'].mean().item() for k, v in film_params.items()},
            'film_beta_mean': {k: v['beta'].mean().item() for k, v in film_params.items()}
        }


# 测试代码
if __name__ == '__main__':
    print("=" * 70)
    print("测试 Pieper 因果IK模型（注意力+FiLM两阶段）")
    print("=" * 70)

    # 创建模型
    model = PieperCausalIK(num_joints=7, num_frames=10, hidden_dim=256, num_layers=2, num_heads=4)
    model = model.cuda()
    model.eval()

    # 测试数据
    batch_size = 4
    num_frames = 10
    history_frames = torch.randn(batch_size, num_frames, 7).cuda()
    end_position = torch.randn(batch_size, 3).cuda()
    end_orientation = torch.randn(batch_size, 4).cuda()

    print(f"\n输入:")
    print(f"  - 历史帧（全部10帧）: {history_frames.shape}")
    print(f"  - 末端位置: {end_position.shape}")
    print(f"  - 末端姿态: {end_orientation.shape}")

    # 前向传播
    with torch.no_grad():
        pred_angles, info = model(history_frames, end_position, end_orientation)

    print(f"\n输出:")
    print(f"  - 预测角度: {pred_angles.shape}")
    print(f"\n调试信息:")
    print(f"  - 目标特征范数: {info['target_feat_norm']:.4f}")
    print(f"  - FiLM gamma均值:")
    for joint, gamma_mean in info['film_gamma_mean'].items():
        print(f"      {joint}: {gamma_mean:.4f}")
    print(f"  - FiLM beta均值:")
    for joint, beta_mean in info['film_beta_mean'].items():
        print(f"      {joint}: {beta_mean:.4f}")

    print(f"\n总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("\n两阶段架构:")
    print("  【第一阶段】Pieper Attention:")
    print("    - 对每个关节组使用多头自注意力处理历史10帧")
    print("    - shoulder: [batch, 10, 3] → MultiHeadAttn → [batch, 256]")
    print("    - es (elbow+forearm+wrist_pitch): [batch, 10, 3] → MultiHeadAttn → [batch, 256]")
    print("    - wristyaw: [batch, 10, 1] → MultiHeadAttn → [batch, 256]")
    print("  【第二阶段】FiLM调制:")
    print("    - y的位姿 → endpose_encoder → target_feat [batch, 256]")
    print("    - target_feat → FiLM生成器 → gamma, beta")
    print("    - modulated_feat = gamma * attention_feat + beta")
    print("  【输出】直接从调制后的特征预测（无GNN）")
    print("=" * 70)

    # 测试不同目标位姿的影响
    print("\n" + "=" * 70)
    print("验证FiLM调制：改变目标位姿，观察预测是否改变")
    print("=" * 70)

    history_frames_test = torch.randn(1, num_frames, 7).cuda()
    end_position_1 = torch.tensor([[0.3, 0.2, 0.1]]).cuda()
    end_position_2 = torch.tensor([[0.5, 0.4, 0.3]]).cuda()
    end_orientation_test = torch.randn(1, 4).cuda()

    with torch.no_grad():
        pred_1, _ = model(history_frames_test, end_position_1, end_orientation_test)
        pred_2, _ = model(history_frames_test, end_position_2, end_orientation_test)

    diff = torch.norm(pred_1 - pred_2).item()
    print(f"\n目标位置1: {end_position_1.cpu().numpy()[0]}")
    print(f"目标位置2: {end_position_2.cpu().numpy()[0]}")
    print(f"预测差异: {diff:.4f}")
    print(f"预测1前3关节: {pred_1[0, :3].cpu().numpy()}")
    print(f"预测2前3关节: {pred_2[0, :3].cpu().numpy()}")

    if diff > 0.01:
        print("✓ FiLM调制正常工作：目标位姿改变导致预测改变")
    else:
        print("✗ 警告：目标位姿改变未影响预测，FiLM可能未正确工作")
