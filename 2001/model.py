"""
基于隐式神经场的IK模型（无历史帧版本）

核心思想:
1. 使用隐式神经场学习 位姿->关节角 的映射
2. 随机傅里叶特征提升泛化能力
3. 直接回归（无需优化求解），支持实时推理
4. 无需历史帧信息，仅用目标位姿预测关节角
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


class RandomFourierFeatures(nn.Module):
    """
    随机傅里叶特征（神经正切核逼近）

    将输入映射到高频空间，极大提升模型的泛化能力
    参考: "Fourier Features Let Networks Learn High Frequency Functions"
    """

    def __init__(self, input_dim, output_dim, sigma=1.0):
        super().__init__()
        # 随机采样权重（固定，不训练）
        weight = torch.randn(input_dim, output_dim) * sigma
        bias = torch.rand(output_dim) * 2 * np.pi

        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

    def forward(self, x):
        """
        Args:
            x: [batch, input_dim]
        Returns:
            features: [batch, output_dim]
        """
        # 使用sin和cos编码
        x = x @ self.weight + self.bias
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1) / np.sqrt(2.0)


class PositionalEncoding(nn.Module):
    """
    位置编码（多尺度正弦/余弦编码）

    替代随机傅里叶特征，提供确定性的高频特征
    """

    def __init__(self, input_dim, num_freqs=10, max_freq_log2=8):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs

        # 生成频率
        freqs = 2.0 ** torch.linspace(0, max_freq_log2, num_freqs)
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        """
        Args:
            x: [batch, input_dim]
        Returns:
            features: [batch, input_dim * 2 * num_freqs]
        """
        # 对每个维度和每个频率进行编码
        batch_size = x.shape[0]
        features = []

        for dim in range(self.input_dim):
            x_dim = x[:, dim:dim+1]  # [batch, 1]
            for freq in self.freqs:
                features.append(torch.sin(freq * x_dim))
                features.append(torch.cos(freq * x_dim))

        return torch.cat(features, dim=-1)


class ImplicitIKDecoder(nn.Module):
    """
    隐式IK解码器

    输入: 目标位姿 [batch, 7] (位置3 + 四元数4)
    输出: 关节角度 [batch, 7]
    """

    def __init__(self, pose_dim=7, joint_dim=7, hidden_dim=512,
                 use_fourier=True, num_freqs=10):
        super().__init__()

        self.pose_dim = pose_dim
        self.joint_dim = joint_dim
        self.use_fourier = use_fourier

        if use_fourier:
            # 使用随机傅里叶特征
            fourier_dim = hidden_dim // 2
            self.pose_encoder = RandomFourierFeatures(pose_dim, fourier_dim // 2, sigma=2.0)
            feature_dim = fourier_dim  # sin+cos
        else:
            # 使用位置编码
            self.pose_encoder = PositionalEncoding(pose_dim, num_freqs=num_freqs)
            feature_dim = pose_dim * 2 * num_freqs

        # 主网络（带残差连接）
        self.input_layer = nn.Linear(feature_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # 隐藏层（带跳跃连接）
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            for _ in range(6)  # 6层隐藏层
        ])

        # 跳跃连接的投影层
        self.skip_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) if i % 2 == 0 else nn.Identity()
            for i in range(6)
        ])

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, joint_dim)
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, target_pose):
        """
        Args:
            target_pose: [batch, 7] 目标位姿 (位置3 + 四元数4)

        Returns:
            joint_angles: [batch, 7] 预测的关节角度
        """
        # 编码位姿
        x = self.pose_encoder(target_pose)  # [batch, feature_dim]
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = F.gelu(x)

        # 通过隐藏层（带跳跃连接）
        for i, (layer, skip_proj) in enumerate(zip(self.hidden_layers, self.skip_proj)):
            identity = x
            x = layer(x)
            if isinstance(skip_proj, nn.Linear):
                x = x + skip_proj(identity)  # 残差连接

        # 输出
        joint_angles = self.output_layer(x)

        return joint_angles


class ImplicitIK(nn.Module):
    """
    隐式IK模型（完整版）

    特性:
    1. 无需历史帧
    2. 使用隐式神经场
    3. 支持条件编码（可选的当前关节角作为条件）
    """

    def __init__(self, pose_dim=7, joint_dim=7, hidden_dim=512,
                 use_fourier=True, num_freqs=10, use_condition=False):
        super().__init__()

        self.use_condition = use_condition

        # 位姿编码器
        self.pose_encoder = PositionalEncoding(pose_dim, num_freqs=num_freqs)
        pose_feat_dim = pose_dim * 2 * num_freqs

        if use_condition:
            # 条件编码器（当前关节角）
            self.condition_encoder = nn.Sequential(
                nn.Linear(joint_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, hidden_dim // 4)
            )
            feature_dim = pose_feat_dim + hidden_dim // 4
        else:
            feature_dim = pose_feat_dim

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            # 隐藏层
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),

            # 输出
            nn.Linear(hidden_dim // 2, joint_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, target_pose, current_angles=None):
        """
        Args:
            target_pose: [batch, 7] 目标位姿 (位置3 + 四元数4)
            current_angles: [batch, 7] 当前关节角（可选，仅当use_condition=True时使用）

        Returns:
            joint_angles: [batch, 7] 预测的关节角度
        """
        # 编码目标位姿
        pose_feat = self.pose_encoder(target_pose)

        if self.use_condition and current_angles is not None:
            # 融合当前关节角信息
            cond_feat = self.condition_encoder(current_angles)
            x = torch.cat([pose_feat, cond_feat], dim=-1)
        else:
            x = pose_feat

        # 解码
        joint_angles = self.decoder(x)

        return joint_angles


class ImplicitIKEnsemble(nn.Module):
    """
    隐式IK集成模型

    使用多个子模型集成，提升鲁棒性
    """

    def __init__(self, pose_dim=7, joint_dim=7, hidden_dim=512,
                 num_models=5, num_freqs=10):
        super().__init__()

        self.num_models = num_models

        # 创建多个子模型
        self.models = nn.ModuleList([
            ImplicitIK(
                pose_dim=pose_dim,
                joint_dim=joint_dim,
                hidden_dim=hidden_dim,
                use_fourier=False,
                num_freqs=num_freqs,
                use_condition=False
            )
            for _ in range(num_models)
        ])

    def forward(self, target_pose, current_angles=None):
        """
        Args:
            target_pose: [batch, 7] 目标位姿
            current_angles: [batch, 7] 当前关节角（不使用）

        Returns:
            joint_angles: [batch, 7] 集成预测的关节角度（平均）
        """
        predictions = []

        for model in self.models:
            pred = model(target_pose)
            predictions.append(pred)

        # 集成（平均）
        joint_angles = torch.stack(predictions, dim=0).mean(dim=0)

        return joint_angles


# 归一化层
class NormalizationLayer(nn.Module):
    """
    数据归一化层

    对输入进行归一化，对输出进行反归一化
    """

    def __init__(self, pose_mean, pose_std, joint_mean, joint_std):
        super().__init__()

        self.register_buffer('pose_mean', torch.tensor(pose_mean))
        self.register_buffer('pose_std', torch.tensor(pose_std))
        self.register_buffer('joint_mean', torch.tensor(joint_mean))
        self.register_buffer('joint_std', torch.tensor(joint_std))

    def normalize_pose(self, pose):
        """归一化位姿"""
        return (pose - self.pose_mean) / (self.pose_std + 1e-8)

    def denormalize_joint(self, joint):
        """反归一化关节角"""
        return joint * self.joint_std + self.joint_mean


def compute_dataset_statistics(dataloader, pose_dim=7, joint_dim=7):
    """
    计算数据集的统计信息（均值和标准差）

    Args:
        dataloader: 数据加载器
        pose_dim: 位姿维度
        joint_dim: 关节维度

    Returns:
        pose_mean, pose_std, joint_mean, joint_std
    """
    logging.info("计算数据集统计信息...")

    pose_sum = 0
    pose_sq_sum = 0
    joint_sum = 0
    joint_sq_sum = 0
    count = 0

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                # WindowedIKDataset格式: (X, y, last_angle)
                X, y, last_angle = batch
                # 从y中提取位姿和关节数据
                if y.shape[1] == 14:
                    poses = y[:, :7]
                    joints = y[:, 7:]
                else:
                    # 只有关节数据
                    joints = y
                    poses = None
            else:
                # 其他格式
                poses = batch[0] if len(batch) > 0 else None
                joints = batch[1] if len(batch) > 1 else None
        else:
            # 字典格式
            poses = batch.get('pose')
            joints = batch.get('joint')

        if poses is not None:
            pose_sum += poses.sum(dim=0)
            pose_sq_sum += (poses ** 2).sum(dim=0)
            count += poses.shape[0]

        if joints is not None:
            joint_sum += joints.sum(dim=0)
            joint_sq_sum += (joints ** 2).sum(dim=0)

    # 计算均值和标准差
    pose_mean = pose_sum / count
    pose_std = torch.sqrt(pose_sq_sum / count - pose_mean ** 2)

    joint_mean = joint_sum / (count if joints is not None else count)
    joint_std = torch.sqrt(joint_sq_sum / count - joint_mean ** 2)

    logging.info(f"位姿均值: {pose_mean}")
    logging.info(f"位姿标准差: {pose_std}")
    logging.info(f"关节均值: {joint_mean}")
    logging.info(f"关节标准差: {joint_std}")

    return pose_mean.numpy(), pose_std.numpy(), joint_mean.numpy(), joint_std.numpy()


# 测试代码
if __name__ == '__main__':
    print("=" * 70)
    print("测试隐式IK模型")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 测试1: 基础ImplicitIK
    print("\n测试1: ImplicitIK (无条件)")
    model = ImplicitIK(pose_dim=7, joint_dim=7, hidden_dim=256, use_condition=False).to(device)
    model.eval()

    batch_size = 4
    target_pose = torch.randn(batch_size, 7).to(device)

    with torch.no_grad():
        pred_joints = model(target_pose)

    print(f"输入位姿: {target_pose.shape}")
    print(f"输出关节角: {pred_joints.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试2: 条件ImplicitIK
    print("\n测试2: ImplicitIK (带条件)")
    model_cond = ImplicitIK(pose_dim=7, joint_dim=7, hidden_dim=256, use_condition=True).to(device)
    model_cond.eval()

    current_angles = torch.randn(batch_size, 7).to(device)

    with torch.no_grad():
        pred_joints_cond = model_cond(target_pose, current_angles)

    print(f"输入位姿: {target_pose.shape}")
    print(f"当前关节角: {current_angles.shape}")
    print(f"输出关节角: {pred_joints_cond.shape}")
    print(f"参数量: {sum(p.numel() for p in model_cond.parameters()) / 1e6:.2f}M")

    # 测试3: 集成模型
    print("\n测试3: ImplicitIKEnsemble")
    ensemble = ImplicitIKEnsemble(pose_dim=7, joint_dim=7, hidden_dim=128, num_models=3).to(device)
    ensemble.eval()

    with torch.no_grad():
        pred_joints_ens = ensemble(target_pose)

    print(f"输入位姿: {target_pose.shape}")
    print(f"输出关节角: {pred_joints_ens.shape}")
    print(f"参数量: {sum(p.numel() for p in ensemble.parameters()) / 1e6:.2f}M")

    print("\n" + "=" * 70)
