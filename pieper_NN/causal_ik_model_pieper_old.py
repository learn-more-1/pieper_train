"""
旧版本 Pieper 因果IK模型 - 用于加载 pieper_causal_ik_092.pth

这个版本的结构与旧checkpoint完全匹配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """时间编码器 - 处理历史10帧"""
    def __init__(self, num_joints=7, hidden_dim=512, num_layers=2):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim

        # 每个关节独立的历史编码
        self.encoders = nn.ModuleDict({
            'shoulder': nn.Sequential(
                nn.Linear(10, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ),
            'elbow': nn.Sequential(
                nn.Linear(10, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ),
            'forearm': nn.Sequential(
                nn.Linear(10, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ),
            'wrist': nn.Sequential(
                nn.Linear(10, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        })

    def forward(self, x):
        """
        x: [batch, 10, 7] 历史角度
        返回: 每个关节的特征 dict
        """
        batch_size = x.shape[0]
        features = {}

        # Shoulder joints (J0, J1, J2)
        shoulder_feat = self.encoders['shoulder'](x[:, :, :3].transpose(1, 2))  # [batch, 3, hidden]
        shoulder_feat = shoulder_feat.mean(dim=1)  # [batch, hidden]
        features['shoulder'] = shoulder_feat

        # Elbow (J3)
        elbow_feat = self.encoders['elbow'](x[:, :, 3:4].transpose(1, 2))  # [batch, 1, hidden]
        elbow_feat = elbow_feat.mean(dim=1)
        features['elbow'] = elbow_feat

        # Forearm (J4)
        forearm_feat = self.encoders['forearm'](x[:, :, 4:5].transpose(1, 2))
        forearm_feat = forearm_feat.mean(dim=1)
        features['forearm'] = forearm_feat

        # Wrist joints (J5, J6)
        wrist_feat = self.encoders['wrist'](x[:, :, 5:7].transpose(1, 2))  # [batch, 2, hidden]
        wrist_feat = wrist_feat.mean(dim=1)
        features['wrist'] = wrist_feat

        return features


class JointEmbeddings(nn.Module):
    """关节嵌入层"""
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            'shoulder': nn.Sequential(nn.Linear(512, 512), nn.ReLU()),
            'elbow': nn.Sequential(nn.Linear(512, 512), nn.ReLU()),
            'forearm': nn.Sequential(nn.Linear(512, 512), nn.ReLU()),
            'wrist': nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        })

    def forward(self, joint_features):
        """
        joint_features: dict of {'shoulder': [batch, hidden], ...}
        返回: 嵌入后的特征 dict
        """
        embedded = {}
        for name, feat in joint_features.items():
            embedded[name] = self.embeddings[name](feat)
        return embedded


class PieperAttention(nn.Module):
    """Pieper准则注意力模块"""
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 关节编码
        self.joint_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 位置和姿态的注意力
        self.position_attention = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.orientation_attention = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, joint_embeddings, target_position, target_orientation):
        """
        joint_embeddings: dict of embedded features
        target_position: [batch, 3]
        target_orientation: [batch, 4]
        返回: (position_weights, orientation_weights)
        """
        batch_size = target_position.shape[0]

        # 编码目标位姿
        pos_feat = self.position_attention(target_position)  # [batch, hidden]
        ori_feat = self.orientation_attention(target_orientation)  # [batch, hidden]

        # 计算每个关节的注意力权重
        position_weights = {}
        orientation_weights = {}

        for name, feat in joint_embeddings.items():
            encoded = self.joint_encoder(feat)  # [batch, hidden]

            # 与目标位姿的相似度
            pos_sim = torch.sum(encoded * pos_feat, dim=1, keepdim=True)  # [batch, 1]
            ori_sim = torch.sum(encoded * ori_feat, dim=1, keepdim=True)  # [batch, 1]

            position_weights[name] = pos_sim
            orientation_weights[name] = ori_sim

        # 归一化为7个关节的权重
        # shoulder -> J0, J1, J2
        # elbow -> J3
        # forearm -> J4
        # wrist -> J5, J6

        full_pos_weights = torch.cat([
            position_weights['shoulder'].repeat(1, 3),
            position_weights['elbow'],
            position_weights['forearm'],
            position_weights['wrist'].repeat(1, 2)
        ], dim=1)  # [batch, 7]

        full_ori_weights = torch.cat([
            orientation_weights['shoulder'].repeat(1, 3),
            orientation_weights['elbow'],
            orientation_weights['forearm'],
            orientation_weights['wrist'].repeat(1, 2)
        ], dim=1)  # [batch, 7]

        return full_pos_weights, full_ori_weights


class PieperFiLM(nn.Module):
    """Pieper准则的FiLM调制"""
    def __init__(self, hidden_dim=512):
        super().__init__()

        # 位置调制
        self.position_gamma_net = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.position_beta_net = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 姿态调制
        self.orientation_gamma_net = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.orientation_beta_net = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, joint_features, position_weights, orientation_weights):
        """
        应用FiLM调制
        返回: modulated_features
        """
        modulated = {}

        # 为每个关节生成调制参数
        for name, feat in joint_features.items():
            # 位置调制
            pos_gamma = self.position_gamma_net(feat)  # [batch, hidden]
            pos_beta = self.position_beta_net(feat)

            # 姿态调制
            ori_gamma = self.orientation_gamma_net(feat)
            ori_beta = self.orientation_beta_net(feat)

            # 应用调制
            modulated[name] = feat * (pos_gamma + ori_gamma) + (pos_beta + ori_beta)

        return modulated


class MessagePassing(nn.Module):
    """消息传递层 - 关节间通信"""
    def __init__(self, hidden_dim=512, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, joint_features):
        """
        关节间消息传递
        """
        # shoulder
        s = joint_features['shoulder']
        e = joint_features['elbow']
        f = joint_features['forearm']
        w = joint_features['wrist']

        # 消息传递
        for layer in self.layers:
            s_new = F.relu(layer(s + e))
            e_new = F.relu(layer(e + s + f))
            f_new = F.relu(layer(f + e + w))
            w_new = F.relu(layer(w + f))

            s, e, f, w = s_new, e_new, f_new, w_new

        return {'shoulder': s, 'elbow': e, 'forearm': f, 'wrist': w}


class OutputHeads(nn.Module):
    """输出头 - 预测关节角度"""
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.heads = nn.ModuleDict({
            'shoulder': nn.Linear(hidden_dim, 3),  # J0, J1, J2
            'elbow': nn.Linear(hidden_dim, 1),      # J3
            'forearm': nn.Linear(hidden_dim, 1),    # J4
            'wrist': nn.Linear(hidden_dim, 2)       # J5, J6
        })

    def forward(self, joint_features):
        """预测7个关节的角度"""
        angles = torch.cat([
            self.heads['shoulder'](joint_features['shoulder']),
            self.heads['elbow'](joint_features['elbow']),
            self.heads['forearm'](joint_features['forearm']),
            self.heads['wrist'](joint_features['wrist'])
        ], dim=1)  # [batch, 7]

        return angles


class EndPoseEncoder(nn.Module):
    """目标位姿编码器"""
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.position_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU()
        )
        self.orientation_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU()
        )

    def forward(self, position, orientation):
        """
        position: [batch, 3]
        orientation: [batch, 4]
        返回: target_feat [batch, hidden]
        """
        pos_feat = self.position_encoder(position)
        ori_feat = self.orientation_encoder(orientation)
        target_feat = pos_feat + ori_feat
        return target_feat


class PieperCausalIK(nn.Module):
    """完整的Pieper因果IK模型（旧版本）"""

    def __init__(self, num_joints=7, num_frames=10, hidden_dim=512, num_layers=2):
        super().__init__()
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

        # 核心模块
        self.temporal_encoder = TemporalEncoder(num_joints, hidden_dim, num_layers)
        self.joint_embeddings = JointEmbeddings(hidden_dim)
        self.pieper_attention = PieperAttention(hidden_dim)
        self.pieper_film = PieperFiLM(hidden_dim)
        self.message_layers = MessagePassing(hidden_dim, num_layers)
        self.output_heads = OutputHeads(hidden_dim)
        self.endpose_encoder = EndPoseEncoder(hidden_dim)

    def forward(self, history_angles, target_position, target_orientation):
        """
        history_angles: [batch, 10, 7] 历史角度
        target_position: [batch, 3] 目标位置
        target_orientation: [batch, 4] 目标姿态
        """
        # 1. 时间编码
        joint_features = self.temporal_encoder(history_angles)

        # 2. 关节嵌入
        joint_embeddings = self.joint_embeddings(joint_features)

        # 3. Pieper注意力
        position_weights, orientation_weights = self.pieper_attention(
            joint_embeddings, target_position, target_orientation
        )

        # 4. FiLM调制
        modulated_features = self.pieper_film(
            joint_features, position_weights, orientation_weights
        )

        # 5. 消息传递
        refined_features = self.message_layers(modulated_features)

        # 6. 输出预测
        pred_angles = self.output_heads(refined_features)

        # 返回预测和注意力权重（用于分析）
        info = {
            'position_weights': position_weights,
            'orientation_weights': orientation_weights
        }

        return pred_angles, info
