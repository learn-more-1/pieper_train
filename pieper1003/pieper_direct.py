"""
基于 Pieper 准则的因果IK模型（直接相乘交互版）

核心架构：
1. 【第一阶段】Pieper Attention：使用多头自注意力提取历史帧特征
2. 【第二阶段】直接相乘交互：用目标位姿直接与注意力特征相乘
3. MLP 预测：从交互特征预测关节角度

与 FiLM 版本的差异：
- FiLM: modulated_feat = gamma * feat + beta (gamma/beta 由目标位姿生成)
- 直接相乘: interact_feat = feat * pose (元素级直接相乘)
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
                索引: [0:3]=shoulder, [3:6]=es(elbow,forearm,wrist_pitch), [6:7]=wrist_yaw

        Returns:
            dict: 每个关节组的注意力特征
        """
        # 提取各关节组的历史
        history_shoulder = history_frames[:, :, :3]          # [batch, num_frames, 3]
        history_es = history_frames[:, :, 3:6]               # [batch, num_frames, 3]
        history_wristyaw = history_frames[:, :, 6:7]         # [batch, num_frames, 1]

        # 分别用注意力提取特征
        features = {
            'shoulder': self.attention_encoders['shoulder'](history_shoulder),
            'es': self.attention_encoders['es'](history_es),
            'wristyaw': self.attention_encoders['wristyaw'](history_wristyaw)
        }

        return features


class DirectInteractionModule(nn.Module):
    """
    直接相乘交互模块

    将关节特征与目标位姿直接相乘，实现条件注入
    """

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 为每个关节组创建独立的投影层
        # 用于将交互后的特征映射回原始维度
        self.projections = nn.ModuleDict({
            'shoulder': nn.Linear(hidden_dim * 8, hidden_dim),  # (hidden + pos*3 + ori*4)
            'es': nn.Linear(hidden_dim * 8, hidden_dim),
            'wristyaw': nn.Linear(hidden_dim * 8, hidden_dim)
        })

    def forward(self, joint_features, end_position, end_orientation):
        """
        Args:
            joint_features: dict, 每个关节的历史特征
                {
                    'shoulder': [batch, hidden_dim],
                    'es': [batch, hidden_dim],
                    'wristyaw': [batch, hidden_dim]
                }
            end_position: [batch, 3] 末端位置
            end_orientation: [batch, 4] 末端姿态（四元数）

        Returns:
            interacted_features: dict, 交互后的关节特征
        """
        batch_size = end_position.shape[0]

        interacted_features = {}

        for joint_name in ['shoulder', 'es', 'wristyaw']:
            joint_feat = joint_features[joint_name]  # [batch, hidden_dim]

            # 位置交互：将关节特征与位置的每个维度相乘
            # joint_feat: [batch, hidden_dim], end_position: [batch, 3]
            pos_interact = joint_feat.unsqueeze(-1) * end_position.unsqueeze(1)  # [batch, hidden_dim, 3]
            pos_interact = pos_interact.view(batch_size, -1)  # [batch, hidden_dim * 3]

            # 姿态交互：将关节特征与姿态的每个维度相乘
            # joint_feat: [batch, hidden_dim], end_orientation: [batch, 4]
            ori_interact = joint_feat.unsqueeze(-1) * end_orientation.unsqueeze(1)  # [batch, hidden_dim, 4]
            ori_interact = ori_interact.view(batch_size, -1)  # [batch, hidden_dim * 4]

            # 拼接原始特征 + 位置交互 + 姿态交互
            combined = torch.cat([
                joint_feat,      # [batch, hidden_dim]
                pos_interact,    # [batch, hidden_dim * 3]
                ori_interact     # [batch, hidden_dim * 4]
            ], dim=1)  # [batch, hidden_dim * 8]

            # 通过投影层映射回原始维度
            interacted_features[joint_name] = self.projections[joint_name](combined)  # [batch, hidden_dim]

        return interacted_features


class PieperCausalIKDirect(nn.Module):
    """
    基于 Pieper 准则的因果IK模型（直接相乘版）

    核心架构：
    1. 第一阶段：Pieper Attention 提取历史帧特征
    2. 第二阶段：直接相乘交互（关节特征 * 目标位姿）
    3. MLP 预测：从交互特征预测关节角度
    """

    def __init__(self, num_joints=7, num_frames=10, hidden_dim=256, num_layers=2, num_heads=4):
        super().__init__()

        self.joint_dim = num_joints
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 1. Pieper注意力编码器（处理全部历史帧）
        self.attention_encoder = JointwiseAttentionEncoder(num_frames, hidden_dim, num_heads)

        # 2. 直接交互模块
        self.interaction_module = DirectInteractionModule(hidden_dim)

        # 3. 输出头（从交互后的特征预测）
        self.output_heads = nn.ModuleDict({
            'shoulder': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3)
            ),
            'es': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3)
            ),
            'wristyaw': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )
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

        # ============ 第二阶段：直接相乘交互 ============
        # 将关节特征与目标位姿直接相乘
        interacted_features = self.interaction_module(joint_features, end_position, end_orientation)

        # ============ 第三阶段：MLP 预测 ============
        pred_shoulder = self.output_heads['shoulder'](interacted_features['shoulder'])
        pred_es = self.output_heads['es'](interacted_features['es'])
        pred_wristyaw = self.output_heads['wristyaw'](interacted_features['wristyaw'])

        # 拼接
        pred_angles = torch.cat([pred_shoulder, pred_es, pred_wristyaw], dim=1)

        # 返回调试信息
        return pred_angles, {
            'joint_feat_norm': {
                k: torch.norm(v, dim=1).mean().item()
                for k, v in joint_features.items()
            },
            'interacted_feat_norm': {
                k: torch.norm(v, dim=1).mean().item()
                for k, v in interacted_features.items()
            }
        }


# 测试代码
if __name__ == '__main__':
    print("=" * 70)
    print("测试 Pieper 因果IK模型（直接相乘版）")
    print("=" * 70)

    # 创建模型
    model = PieperCausalIKDirect(num_joints=7, num_frames=10, hidden_dim=256, num_layers=2, num_heads=4)
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
    print(f"  - 关节特征范数:")
    for joint, norm in info['joint_feat_norm'].items():
        print(f"      {joint}: {norm:.4f}")
    print(f"  - 交互特征范数:")
    for joint, norm in info['interacted_feat_norm'].items():
        print(f"      {joint}: {norm:.4f}")

    print(f"\n总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("\n两阶段架构:")
    print("  【第一阶段】Pieper Attention:")
    print("    - 对每个关节组使用多头自注意力处理历史10帧")
    print("    - shoulder: [batch, 10, 3] → MultiHeadAttn → [batch, 256]")
    print("    - es (elbow+forearm+wrist_pitch): [batch, 10, 3] → MultiHeadAttn → [batch, 256]")
    print("    - wristyaw: [batch, 10, 1] → MultiHeadAttn → [batch, 256]")
    print("  【第二阶段】直接相乘交互:")
    print("    - joint_feat [batch, 256]")
    print("    - pos_interact = joint_feat * end_position.unsqueeze(1) → [batch, 256, 3]")
    print("    - ori_interact = joint_feat * end_orientation.unsqueeze(1) → [batch, 256, 4]")
    print("    - combined = [joint_feat, pos_interact, ori_interact] → [batch, 256*8]")
    print("    - projection → [batch, 256]")
    print("  【输出】MLP 预测")
    print("=" * 70)

    # 测试不同目标位姿的影响
    print("\n" + "=" * 70)
    print("验证交互：改变目标位姿，观察预测是否改变")
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
    print(f"预测1: {pred_1[0].cpu().numpy()}")
    print(f"预测2: {pred_2[0].cpu().numpy()}")

    if diff > 0.01:
        print("✓ 直接相乘交互正常工作：目标位姿改变导致预测改变")
    else:
        print("✗ 警告：目标位姿改变未影响预测，交互可能未正确工作")
