"""
基于 Pieper 准则的因果IK模型（修正版）

核心修正：
1. FiLM的正确用法：条件（目标位姿）→ FiLM参数 → 调制特征（历史关节特征）
2. 目标位姿作为条件，生成gamma和beta来调制历史关节特征
3. 所有关节都能感受到目标位姿的影响，而不是只有wrist

数据流：
- history_frames → temporal_encoder → joint_features [每个关节的时序特征]
- target_position + target_orientation → endpose_encoder → target_feat
- target_feat → FiLM生成器 → gamma, beta
- gamma * joint_features + beta → modulated_features
- modulated_features → GNN消息传递 → 预测关节角度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
            'elbow': self._make_film_generator(hidden_dim),
            'forearm': self._make_film_generator(hidden_dim),
            'wrist': self._make_film_generator(hidden_dim)
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
                    'elbow': [batch, hidden_dim],
                    'forearm': [batch, hidden_dim],
                    'wrist': [batch, hidden_dim]
                }
            target_feat: [batch, hidden_dim] 目标位姿特征

        Returns:
            modulated_features: dict, 调制后的关节特征
            film_params: dict, FiLM参数（用于调试）
        """
        modulated_features = {}
        film_params = {}

        for joint_name in ['shoulder', 'elbow', 'forearm', 'wrist']:
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
    基于 Pieper 准则的因果IK模型（修正版）

    核心修正：
    1. FiLM调制：目标位姿 → gamma/beta → 调制历史关节特征
    2. 所有关节都能感受到目标位姿的影响
    3. GNN沿因果链（shoulder→elbow→forearm→wrist）传播信息
    """

    def __init__(self, num_joints=7, num_frames=10, hidden_dim=256, num_layers=2):
        super().__init__()

        self.joint_dim = num_joints
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

        # 1. 关节级别的时序编码器
        self.temporal_encoder = JointwiseTemporalEncoder(num_frames, hidden_dim)

        # 2. 末端位姿编码器（生成条件）
        self.endpose_encoder = EndPoseEncoder(hidden_dim)

        # 3. FiLM生成器（从目标位姿生成调制参数）
        self.target_conditioned_film = TargetConditionedFiLM(hidden_dim)

        # 4. 消息传递层（因果链：shoulder → elbow → forearm → wrist）
        self.message_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'shoulder_to_elbow': nn.Linear(hidden_dim, hidden_dim),
                'to_forearm': nn.Linear(hidden_dim * 2, hidden_dim),
                'forearm_to_wrist': nn.Linear(hidden_dim * 2, hidden_dim),
                'forearm_gate': nn.Sigmoid(),
                'wrist_gate': nn.Sigmoid()
            })
            self.message_layers.append(layer)

        # 5. 输出头
        self.output_heads = nn.ModuleDict({
            'shoulder': nn.Linear(hidden_dim, 3),
            'elbow': nn.Linear(hidden_dim, 1),
            'forearm': nn.Linear(hidden_dim, 1),
            'wrist': nn.Linear(hidden_dim, 2)
        })

    def forward(self, history_frames, end_position, end_orientation):
        """
        Args:
            history_frames: [batch, num_frames, 7] 历史关节角度
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

        # 1. 时序编码：从历史帧提取关节特征
        joint_features = self.temporal_encoder(history_frames)
        # joint_features = {
        #     'shoulder': [batch, hidden_dim],
        #     'elbow': [batch, hidden_dim],
        #     'forearm': [batch, hidden_dim],
        #     'wrist': [batch, hidden_dim]
        # }

        # 2. 编码目标位姿（条件）
        target_feat = self.endpose_encoder(end_position, end_orientation)
        # target_feat: [batch, hidden_dim]

        # 3. FiLM调制：用目标位姿调制历史关节特征
        modulated_features, film_params = self.target_conditioned_film(joint_features, target_feat)
        # modulated_features: 每个关节的特征已被目标位姿调制

        # 4. 初始化节点（使用调制后的特征）
        nodes = modulated_features.copy()

        # 5. 消息传递（因果链）
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

        # 6. 输出预测
        pred_shoulder = self.output_heads['shoulder'](nodes['shoulder'])
        pred_elbow = self.output_heads['elbow'](nodes['elbow'])
        pred_forearm = self.output_heads['forearm'](nodes['forearm'])
        pred_wrist = self.output_heads['wrist'](nodes['wrist'])

        # 拼接
        pred_angles = torch.cat([pred_shoulder, pred_elbow, pred_forearm, pred_wrist], dim=1)

        # 返回调试信息
        return pred_angles, {
            'target_feat_norm': torch.norm(target_feat, dim=1).mean().item(),
            'film_gamma_mean': {k: v['gamma'].mean().item() for k, v in film_params.items()},
            'film_beta_mean': {k: v['beta'].mean().item() for k, v in film_params.items()}
        }


# 测试代码
if __name__ == '__main__':
    print("=" * 70)
    print("测试修正版 Pieper 因果IK模型")
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
    print(f"\n调试信息:")
    print(f"  - 目标特征范数: {info['target_feat_norm']:.4f}")
    print(f"  - FiLM gamma均值:")
    for joint, gamma_mean in info['film_gamma_mean'].items():
        print(f"      {joint}: {gamma_mean:.4f}")
    print(f"  - FiLM beta均值:")
    for joint, beta_mean in info['film_beta_mean'].items():
        print(f"      {joint}: {beta_mean:.4f}")

    print(f"\n总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("\n修正后的架构:")
    print("  1. 时序编码: 对每个关节组编码10帧历史")
    print("     - shoulder: [batch, 10, 3] → [batch, 256]")
    print("     - elbow: [batch, 10, 1] → [batch, 256]")
    print("     - forearm: [batch, 10, 1] → [batch, 256]")
    print("     - wrist: [batch, 10, 2] → [batch, 256]")
    print("  2. 目标编码: 目标位姿 → [batch, 256]")
    print("  3. FiLM调制: 目标特征 → gamma/beta → 调制历史关节特征")
    print("     modulated_feat = gamma * history_feat + beta")
    print("  4. GNN: 沿因果链传播信息")
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
