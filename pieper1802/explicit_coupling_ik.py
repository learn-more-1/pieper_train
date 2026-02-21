"""
显式耦合IK模型

核心思想：
关节耦合是结构性的、客观的，应该被显式建模，而不是隐式学习。

架构设计：
1. 位姿编码器：将目标位姿编码为"运动意图"
2. 显式耦合图：图神经网络建模关节间的物理约束
3. 耦合约束求解器：在耦合约束下求解各关节角度
4. 时序参考（可选）：仅用于平滑，不主导预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KinematicCouplingGraph(nn.Module):
    """
    运动学耦合图
    
    显式建模7轴机械臂的关节耦合关系：
    - Shoulder (J0, J1, J2): 基座到上臂
    - Elbow (J3): 肘关节
    - Forearm (J4): 前臂旋转
    - Wrist (J5, J6): 手腕
    
    耦合关系（边）：
    - J0-J1-J2: shoulder内部协调
    - J2-J3: shoulder到elbow
    - J3-J4: elbow到forearm
    - J4-J5-J6: forearm到wrist
    - J0-J3, J1-J3: 位置耦合（Pieper准则）
    """
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        
        # 定义关节耦合的邻接关系
        # 0=shoulder0, 1=shoulder1, 2=shoulder2, 3=elbow
        # 4=forearm, 5=wrist0, 6=wrist1
        
        # 物理连接（运动链）
        self.chain_edges = [
            (0, 1), (1, 2), (2, 3),  # shoulder -> elbow
            (3, 4), (4, 5), (5, 6)   # elbow -> forearm -> wrist
        ]
        
        # 功能耦合（Pieper准则）
        self.coupling_edges = [
            (0, 3),  # shoulder0 与 elbow 位置耦合
            (1, 3),  # shoulder1 与 elbow 位置耦合
            (2, 5),  # shoulder2 与 wrist 姿态耦合
            (3, 5),  # elbow 与 wrist 姿态耦合
        ]
        
        # 边类型编码
        self.edge_types = len(self.chain_edges) + len(self.coupling_edges)
        self.edge_embedding = nn.Embedding(self.edge_types, hidden_dim)
        
        # 消息传递：每种边类型一个MLP
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),  # 源+目标+边类型
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(self.edge_types)
        ])
        
        # 节点更新
        self.node_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ) for _ in range(7)  # 每个关节一个
        ])
        
    def forward(self, joint_features, return_messages=False):
        """
        Args:
            joint_features: [batch, 7, hidden_dim] 每个关节的初始特征
            return_messages: 是否返回中间消息（用于可视化）
            
        Returns:
            updated_features: [batch, 7, hidden_dim]
            messages (optional): dict 耦合消息
        """
        batch_size = joint_features.shape[0]
        all_edges = self.chain_edges + self.coupling_edges
        
        messages = {i: [] for i in range(7)}
        
        # 消息传递
        for edge_idx, (src, dst) in enumerate(all_edges):
            # 获取源和目标特征
            src_feat = joint_features[:, src, :]  # [batch, hidden_dim]
            dst_feat = joint_features[:, dst, :]  # [batch, hidden_dim]
            edge_feat = self.edge_embedding(torch.tensor(edge_idx, device=src_feat.device))
            edge_feat = edge_feat.unsqueeze(0).expand(batch_size, -1)
            
            # 计算消息
            concat = torch.cat([src_feat, dst_feat, edge_feat], dim=-1)
            message = self.message_mlps[edge_idx](concat)
            
            # 双向消息
            messages[dst].append(message)
            # 反向边（不同方向可能有不同语义）
            if edge_idx < len(self.chain_edges):  # 只给运动链加反向
                reverse_edge_idx = edge_idx + len(self.chain_edges) if edge_idx < len(self.coupling_edges) else edge_idx
                edge_feat_rev = self.edge_embedding(torch.tensor(reverse_edge_idx % self.edge_types, device=src_feat.device))
                edge_feat_rev = edge_feat_rev.unsqueeze(0).expand(batch_size, -1)
                concat_rev = torch.cat([dst_feat, src_feat, edge_feat_rev], dim=-1)
                message_rev = self.message_mlps[edge_idx](concat_rev)
                messages[src].append(message_rev)
        
        # 聚合消息并更新节点
        updated_features = []
        for i in range(7):
            if len(messages[i]) > 0:
                # 聚合所有入边消息
                aggregated = torch.stack(messages[i], dim=1).mean(dim=1)  # [batch, hidden_dim]
                # 与自身特征融合
                combined = torch.cat([joint_features[:, i, :], aggregated], dim=-1)
                updated = self.node_update[i](combined)
            else:
                updated = joint_features[:, i, :]
            updated_features.append(updated)
        
        updated_features = torch.stack(updated_features, dim=1)  # [batch, 7, hidden_dim]
        
        if return_messages:
            return updated_features, messages
        return updated_features


class MotionIntentionEncoder(nn.Module):
    """
    运动意图编码器
    
    将目标位姿编码为"运动意图"向量，
    这个向量会被分发到各个关节，驱动它们协调运动。
    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        # 位置意图
        self.position_intention = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 姿态意图
        self.orientation_intention = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 意图融合
        self.intention_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
    def forward(self, position, orientation=None):
        """
        Args:
            position: [batch, 3]
            orientation: [batch, 4] or None
            
        Returns:
            intention: [batch, hidden_dim] 运动意图
        """
        pos_intent = self.position_intention(position)
        
        if orientation is not None:
            ori_intent = self.orientation_intention(orientation)
            combined = torch.cat([pos_intent, ori_intent], dim=-1)
        else:
            combined = torch.cat([pos_intent, pos_intent], dim=-1)
        
        intention = self.intention_fusion(combined)
        return intention


class JointAngleDecoder(nn.Module):
    """
    关节角度解码器
    
    在耦合约束下，从运动意图解码各关节角度。
    每个关节的解码都考虑了与其他关节的耦合关系。
    """
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        
        # 关节分组
        self.groups = {
            'shoulder': [0, 1, 2],
            'elbow': [3],
            'forearm': [4],
            'wrist': [5, 6]
        }
        
        # 组内协调解码
        self.group_decoders = nn.ModuleDict({
            'shoulder': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3)
            ),
            'elbow': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1)
            ),
            'forearm': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1)
            ),
            'wrist': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2)
            )
        })
        
        # 组间协调门控
        self.gates = nn.ModuleDict({
            'elbow_from_shoulder': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            ),
            'forearm_from_elbow': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            ),
            'wrist_from_forearm': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        })
        
    def forward(self, coupled_features):
        """
        Args:
            coupled_features: [batch, 7, hidden_dim] 经过耦合图处理的特征
            
        Returns:
            angles: [batch, 7] 预测的关节角度
            coupling_info: dict 耦合信息
        """
        batch_size = coupled_features.shape[0]
        
        # 提取各组特征
        shoulder_feat = coupled_features[:, 0:3, :].mean(dim=1)  # [batch, hidden_dim]
        elbow_feat = coupled_features[:, 3, :]  # [batch, hidden_dim]
        forearm_feat = coupled_features[:, 4, :]  # [batch, hidden_dim]
        wrist_feat = coupled_features[:, 5:7, :].mean(dim=1)  # [batch, hidden_dim]
        
        coupling_info = {}
        
        # 1. Shoulder解码（基座）
        shoulder_angles = self.group_decoders['shoulder'](shoulder_feat)
        
        # 2. Elbow解码（受shoulder影响）
        gate_elbow = self.gates['elbow_from_shoulder'](
            torch.cat([elbow_feat, shoulder_feat], dim=-1)
        )
        elbow_input = elbow_feat * gate_elbow + shoulder_feat * (1 - gate_elbow)
        elbow_angle = self.group_decoders['elbow'](elbow_input)
        coupling_info['elbow_shoulder_coupling'] = gate_elbow.mean()
        
        # 3. Forearm解码（受elbow影响）
        gate_forearm = self.gates['forearm_from_elbow'](
            torch.cat([forearm_feat, elbow_feat], dim=-1)
        )
        forearm_input = forearm_feat * gate_forearm + elbow_feat * (1 - gate_forearm)
        forearm_angle = self.group_decoders['forearm'](forearm_input)
        coupling_info['forearm_elbow_coupling'] = gate_forearm.mean()
        
        # 4. Wrist解码（受forearm影响）
        gate_wrist = self.gates['wrist_from_forearm'](
            torch.cat([wrist_feat, forearm_feat], dim=-1)
        )
        wrist_input = wrist_feat * gate_wrist + forearm_feat * (1 - gate_wrist)
        wrist_angles = self.group_decoders['wrist'](wrist_input)
        coupling_info['wrist_forearm_coupling'] = gate_wrist.mean()
        
        # 拼接
        angles = torch.cat([
            shoulder_angles,
            elbow_angle,
            forearm_angle,
            wrist_angles
        ], dim=-1)
        
        return angles, coupling_info


class ExplicitCouplingIK(nn.Module):
    """
    显式耦合IK模型
    
    完全解耦时序和耦合：
    1. 运动意图 ← 目标位姿（主导）
    2. 关节特征 ← 运动意图分发
    3. 耦合特征 ← 显式图网络处理
    4. 关节角度 ← 耦合约束解码
    
    可选：历史时序仅用于平滑参考
    """
    
    def __init__(self, num_joints=7, num_frames=10, hidden_dim=256, 
                 use_temporal=False):
        super().__init__()
        
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.use_temporal = use_temporal
        
        # 运动意图编码（位姿 → 意图）
        self.intention_encoder = MotionIntentionEncoder(hidden_dim)
        
        # 意图到关节分发
        self.intention_to_joints = nn.Linear(hidden_dim, num_joints * (hidden_dim // 2))
        
        # 可选：时序参考编码
        if use_temporal:
            self.temporal_encoder = nn.Sequential(
                nn.Linear(num_frames * num_joints, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU()
            )
        
        # 显式耦合图
        self.coupling_graph = KinematicCouplingGraph(hidden_dim // 2)
        
        # 关节角度解码
        self.angle_decoder = JointAngleDecoder(hidden_dim // 2)
        
    def forward(self, target_position, target_orientation=None, 
                history_frames=None, return_coupling=False):
        """
        Args:
            target_position: [batch, 3] 目标位置（主导）
            target_orientation: [batch, 4] 目标姿态（可选）
            history_frames: [batch, num_frames, 7] 历史参考（可选）
            return_coupling: 是否返回耦合信息
            
        Returns:
            pred_angles: [batch, 7]
            info: dict 包含耦合信息
        """
        batch_size = target_position.shape[0]
        device = target_position.device
        
        # 确保是batch维度
        if target_position.dim() == 1:
            target_position = target_position.unsqueeze(0)
            if target_orientation is not None and target_orientation.dim() == 1:
                target_orientation = target_orientation.unsqueeze(0)
        
        # 重新获取batch_size（可能在unsqueeze后变化）
        batch_size = target_position.shape[0]
        
        # 1. 编码运动意图 ← 目标位姿（主导）
        intention = self.intention_encoder(target_position, target_orientation)
        
        # 2. 意图分发到各关节
        joint_features_flat = self.intention_to_joints(intention)
        joint_features = joint_features_flat.view(batch_size, self.num_joints, -1)
        
        # 3. 可选：融合时序参考
        if self.use_temporal and history_frames is not None:
            temporal_feat = self.temporal_encoder(
                history_frames.view(batch_size, -1)
            )
            # 时序仅作为参考，不主导
            joint_features = joint_features + temporal_feat.unsqueeze(1) * 0.1
        
        # 4. 显式耦合图处理
        coupled_features, messages = self.coupling_graph(
            joint_features, return_messages=True
        )
        
        # 5. 耦合约束解码
        pred_angles, coupling_info = self.angle_decoder(coupled_features)
        
        info = {
            'intention': intention,
            'coupling_info': coupling_info,
            'messages': messages if return_coupling else None
        }
        
        return pred_angles, info


# ==================== 可视化耦合关系 ====================

def visualize_coupling(model, target_position, target_orientation=None):
    """
    可视化关节耦合关系
    
    显示哪些关节在配合运动，以及耦合强度。
    """
    model.eval()
    
    with torch.no_grad():
        pred_angles, info = model(
            target_position.unsqueeze(0),
            target_orientation.unsqueeze(0) if target_orientation is not None else None,
            return_coupling=True
        )
    
    print("=" * 60)
    print("关节耦合可视化")
    print("=" * 60)
    
    print(f"\n目标位姿: {target_position.cpu().numpy().round(3)}")
    print(f"预测角度: {pred_angles[0].cpu().numpy().round(3)}")
    
    print(f"\n耦合强度:")
    for key, value in info['coupling_info'].items():
        v = value.item() if isinstance(value, torch.Tensor) else value
        bar = "█" * int(v * 20)
        print(f"  {key:25s}: {v:.3f} {bar}")
    
    # 关节间消息强度
    if info['messages']:
        print(f"\n关节间消息传递:")
        for joint_idx, msgs in info['messages'].items():
            if len(msgs) > 0:
                avg_strength = torch.stack(msgs).norm(dim=-1).mean().item()
                bar = "█" * int(avg_strength * 10)
                print(f"  Joint {joint_idx}: {avg_strength:.3f} {bar}")


# ==================== 测试 ====================

if __name__ == '__main__':
    print("=" * 70)
    print("显式耦合IK模型测试")
    print("=" * 70)
    
    # 创建模型
    model = ExplicitCouplingIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        use_temporal=False  # 纯位姿输入，不用时序
    ).cuda()
    
    # 测试数据
    batch_size = 4
    target_pos = torch.randn(batch_size, 3).cuda()
    target_ori = torch.randn(batch_size, 4).cuda()
    
    print(f"\n输入:")
    print(f"  - 目标位置: {target_pos.shape}")
    print(f"  - 目标姿态: {target_ori.shape}")
    
    # 前向传播
    with torch.no_grad():
        pred_angles, info = model(target_pos, target_ori)
    
    print(f"\n输出:")
    print(f"  - 预测角度: {pred_angles.shape}")
    print(f"  - 运动意图: {info['intention'].shape}")
    
    # 可视化
    print("\n")
    visualize_coupling(model, target_pos[0], target_ori[0])
    
    # 测试纯位姿输入（无历史）
    print("\n" + "=" * 70)
    print("纯位姿输入测试（无历史）")
    print("=" * 70)
    
    test_positions = [
        torch.tensor([0.4, 0.1, 0.3], device='cuda'),
        torch.tensor([0.5, 0.2, 0.4], device='cuda'),
        torch.tensor([0.6, 0.3, 0.5], device='cuda'),
    ]
    
    for i, pos in enumerate(test_positions):
        with torch.no_grad():
            angles, _ = model(pos)
        print(f"  目标 {pos.cpu().numpy().round(2)} -> 角度 {angles[0, :3].cpu().numpy().round(4)}...")
    
    print("\n✓ 纯位姿输入直接产生预测，无需历史！")
    
    print(f"\n总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    print("\n架构优势:")
    print("  1. 显式耦合图：关节关系清晰可见")
    print("  2. 位姿驱动：目标位姿直接生成运动意图")
    print("  3. 可解释性：能看到哪些关节在配合")
    print("  4. 纯位姿输入：无需历史时序")
