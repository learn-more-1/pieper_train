"""
训练 GNN + FiLM IK模型 (消融实验版本 - 无Pieper)

核心思想:
1. 使用 GNN 学习关节间的因果关系
2. 使用 FiLM 调制末端位姿信息
3. 移除 Pieper 注意力机制（消融实验对比）

数据流:
- 输入: 历史关节角度 [batch, 10, 7]
- 预处理: 提取最后一帧作为当前关节角度
- FK计算: 从目标关节角度计算末端位姿(位置+四元数)
- 模型: (当前关节角度, 目标末端位姿) -> 预测关节角度
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import sys
import time
import os

sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1101')

from dataset_generalized import create_windowed_dataloaders
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("gnn_film_train.log"), logging.StreamHandler()]
)


class Config:
    # 数据配置
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
    num_joints = 7
    num_frames = 10

    # 训练参数
    batch_size = 512
    epochs = 300
    initial_lr = 1e-3
    min_lr = 1e-6
    patience = 10
    factor = 0.5

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 模型配置
    joint_dim = 7
    hidden_dim = 256
    num_layers = 2

    # 损失权重
    ik_weight = 1.0       # 关节角度损失
    fk_weight = 0.5       # 位置误差损失
    continuity_weight = 1.0  # 连续性损失

    # 数据加载优化
    num_workers = 4
    pin_memory = True


class SimpleFiLMGenerator(nn.Module):
    """
    简化的 FiLM 生成器（无Pieper权重）

    直接从关节角度生成 FiLM 参数，不使用注意力权重
    """

    def __init__(self, joint_dim=7, hidden_dim=256):
        super().__init__()

        # 生成位置 FiLM 参数
        self.position_gamma_net = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)
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
            nn.Linear(hidden_dim, 4)
        )

        self.orientation_beta_net = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, joint_angles):
        """
        Args:
            joint_angles: [batch, 7] 关节角度

        Returns:
            position_gamma, position_beta: [batch, 3] 位置的 FiLM 参数
            orientation_gamma, orientation_beta: [batch, 4] 姿态的 FiLM 参数
        """
        pos_gamma = self.position_gamma_net(joint_angles)
        pos_beta = self.position_beta_net(joint_angles)
        ori_gamma = self.orientation_gamma_net(joint_angles)
        ori_beta = self.orientation_beta_net(joint_angles)

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


class GNNFiLMIK(nn.Module):
    """
    基于 GNN + FiLM 的IK模型（消融实验版本 - 无Pieper）

    核心特性：
    1. 使用时序编码器分别编码每个关节的历史
    2. 使用简化版 FiLM（无注意力权重）
    3. 使用 GNN 学习关节间的因果关系
    """

    def __init__(self, num_joints=7, num_frames=10, hidden_dim=256, num_layers=2):
        super().__init__()

        self.joint_dim = num_joints
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

        # 1. 关节级别的时序编码器
        self.temporal_encoder = JointwiseTemporalEncoder(num_frames, hidden_dim)

        # 2. 简化版 FiLM 生成器（无Pieper注意力）
        self.simple_film = SimpleFiLMGenerator(num_joints, hidden_dim)

        # 3. 末端位姿编码器
        self.endpose_encoder = EndPoseEncoder(hidden_dim)

        # 4. 关节分组（因果链）
        self.joint_groups = {
            'shoulder': [0, 1, 2],
            'elbow': [3],
            'forearm': [4],
            'wrist': [5, 6]
        }

        # 5. 消息传递层（因果链）
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

        # 6. 输出头
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
            info: dict 包含调制信息
        """
        batch_size = history_frames.shape[0]
        device = history_frames.device

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

        # 2. 从历史最后一帧提取当前关节角度
        current_joint_angles = history_frames[:, -1, :]  # [batch, 7]

        # 3. 生成 FiLM 参数（无Pieper权重）
        pos_gamma, pos_beta, ori_gamma, ori_beta = self.simple_film(current_joint_angles)

        # 4. FiLM 调制末端位姿
        modulated_pos = pos_gamma * end_position + pos_beta  # [batch, 3]
        modulated_ori = ori_gamma * end_orientation + ori_beta  # [batch, 4]

        # 5. 编码调制后的末端位姿
        end_pos_feat, end_ori_feat = self.endpose_encoder(modulated_pos, modulated_ori)

        # 6. 融合末端位姿特征
        endeff_feat = end_pos_feat + end_ori_feat  # [batch, hidden_dim]

        # 7. 初始化关节节点（使用时序编码的历史特征）
        nodes = joint_features.copy()

        # Wrist 节点额外融合末端位姿特征
        nodes['wrist'] = nodes['wrist'] + endeff_feat

        # 8. 消息传递（因果链）
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

        # 9. 输出预测
        pred_shoulder = self.output_heads['shoulder'](nodes['shoulder'])
        pred_elbow = self.output_heads['elbow'](nodes['elbow'])
        pred_forearm = self.output_heads['forearm'](nodes['forearm'])
        pred_wrist = self.output_heads['wrist'](nodes['wrist'])

        # 拼接
        pred_angles = torch.cat([pred_shoulder, pred_elbow, pred_forearm, pred_wrist], dim=1)

        return pred_angles, {
            'modulated_pos': modulated_pos,
            'modulated_ori': modulated_ori
        }


def load_robot_model():
    """加载GPU FK模型"""
    try:
        gpu_fk = SimpleGPUFK()
        left_arm_joints = [16, 17, 18, 19, 20, 21, 22]
        logging.info(f"✓ 成功加载GPU FK模型")
        logging.info(f"  左臂关节 ID: {left_arm_joints}")
        return gpu_fk, left_arm_joints
    except Exception as e:
        logging.error(f"✗ 加载GPU FK模型失败: {e}")
        return None, None


def forward_kinematics_with_pose(gpu_fk, joint_angles):
    """
    GPU加速的批量FK计算（仅位置）

    Args:
        gpu_fk: GPU FK模型
        joint_angles: [batch, 7] 关节角度

    Returns:
        positions: [batch, 3] 手腕位置
        orientations: [batch, 4] None
    """
    positions = gpu_fk.forward(joint_angles)
    return positions, None


def train():
    config = Config()

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = 'gnn_film_ik'

    logging.info("=" * 70)
    logging.info("训练 GNN + FiLM IK模型（消融实验版本 - 无Pieper）")
    logging.info("=" * 70)
    logging.info("核心特性:")
    logging.info("  1. 简化版FiLM（无Pieper注意力权重）")
    logging.info("  2. GNN学习关节间因果耦合")
    logging.info("  3. 消融实验对比：验证Pieper模块的作用")

    # 加载数据集
    logging.info("\n加载数据集...")
    train_loader, val_loader = create_windowed_dataloaders(config.data_path, config)

    # 创建模型
    logging.info(f"\n创建模型: {model_name}")
    model = GNNFiLMIK(
        num_joints=config.joint_dim,
        num_frames=config.num_frames,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers
    )
    model = model.to(config.device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"模型参数量: {total_params:.2f}M")

    # 加载GPU FK模型
    logging.info("\n加载GPU FK模型（用于计算末端位姿）...")
    gpu_fk, joint_ids = load_robot_model()

    if gpu_fk is None:
        logging.error("无法加载GPU FK模型，退出")
        return

    # 损失函数
    mse_criterion = nn.MSELoss()

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.initial_lr, weight_decay=1e-4)

    # 学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=70,
        T_mult=2,
        eta_min=config.min_lr
    )

    # 断点路径
    checkpoint_path = f"/home/bonuli/Pieper/pieper1101/{model_name}_1101.pth"

    # 加载断点
    best_val_loss = float("inf")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        logging.info("加载断点...")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        logging.info(f"从第{start_epoch} epoch继续")

    # 训练循环
    logging.info(f"\n开始训练（{config.epochs} epochs）...")

    for epoch in range(start_epoch, config.epochs):
        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        train_ik_loss = 0.0
        train_fk_loss = 0.0
        train_continuity_loss = 0.0
        train_preds = []
        train_trues = []
        start_time = time.time()

        for batch_X, batch_y, batch_last_angle in train_loader:
            batch_X, batch_y, batch_last_angle = (
                batch_X.to(config.device, non_blocking=True),
                batch_y.to(config.device, non_blocking=True),
                batch_last_angle.to(config.device, non_blocking=True)
            )

            batch_size = batch_X.shape[0]

            # 1. 使用历史窗口（10帧）
            history_frames = batch_X  # [batch, 10, 7]

            # 2. 判断y的格式并提取目标位姿和角度
            if batch_y.shape[1] == 14:
                target_pose = batch_y[:, :7]
                target_angles = batch_y[:, 7:]
                target_position = target_pose[:, :3]
                target_orientation = target_pose[:, 3:7]
            else:
                target_angles = batch_y
                target_position, target_orientation = forward_kinematics_with_pose(gpu_fk, target_angles)
                target_orientation = None

            optimizer.zero_grad()

            # 3. 前向传播
            pred_joint_angles, info = model(
                history_frames,
                target_position,
                target_orientation
            )

            # 4. 计算损失
            # IK损失: 预测角度 vs 目标角度
            ik_loss = mse_criterion(pred_joint_angles, target_angles)

            # FK损失: 预测角度的末端位置 vs 目标末端位置
            pred_position, _ = forward_kinematics_with_pose(gpu_fk, pred_joint_angles)
            fk_position, _ = forward_kinematics_with_pose(gpu_fk, target_angles)
            fk_loss = mse_criterion(pred_position, fk_position)

            # 连续性损失: 预测角度与当前角度的差异
            continuity_loss = torch.mean((pred_joint_angles - batch_last_angle) ** 2)

            # 总损失
            total_loss = (
                config.ik_weight * ik_loss +
                config.fk_weight * fk_loss +
                config.continuity_weight * continuity_loss
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * batch_size
            train_ik_loss += ik_loss.item() * batch_size
            train_fk_loss += fk_loss.item() * batch_size
            train_continuity_loss += continuity_loss.item() * batch_size

            if len(train_preds) < 10000:
                train_preds.append(pred_joint_angles.cpu().detach().numpy())
                train_trues.append(target_angles.cpu().detach().numpy())

        # 计算训练指标
        train_preds = np.vstack(train_preds)
        train_trues = np.vstack(train_trues)
        train_r2 = 1 - (np.sum((train_trues - train_preds) ** 2) /
                      (np.sum((train_trues - np.mean(train_trues)) ** 2) + 1e-8))
        train_loss = train_loss / len(train_loader.dataset)
        train_time = time.time() - start_time

        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0.0
        val_ik_loss = 0.0
        val_fk_loss = 0.0
        val_preds = []
        val_trues = []

        with torch.no_grad():
            for batch_X, batch_y, batch_last_angle in val_loader:
                batch_X, batch_y, batch_last_angle = (
                    batch_X.to(config.device, non_blocking=True),
                    batch_y.to(config.device, non_blocking=True),
                    batch_last_angle.to(config.device, non_blocking=True)
                )

                batch_size = batch_X.shape[0]

                history_frames = batch_X

                if batch_y.shape[1] == 14:
                    target_pose = batch_y[:, :7]
                    target_angles = batch_y[:, 7:]
                    target_position = target_pose[:, :3]
                    target_orientation = target_pose[:, 3:7]
                else:
                    target_angles = batch_y
                    target_position, target_orientation = forward_kinematics_with_pose(gpu_fk, target_angles)
                    target_orientation = None

                pred_joint_angles, info = model(
                    history_frames,
                    target_position,
                    target_orientation
                )

                ik_loss = mse_criterion(pred_joint_angles, target_angles)

                pred_position, _ = forward_kinematics_with_pose(gpu_fk, pred_joint_angles)
                fk_loss = mse_criterion(pred_position, target_position)

                continuity_loss = torch.mean((pred_joint_angles - batch_last_angle) ** 2)

                total_loss = (
                    config.ik_weight * ik_loss +
                    config.fk_weight * fk_loss +
                    config.continuity_weight * continuity_loss
                )

                val_loss += total_loss.item() * batch_size
                val_ik_loss += ik_loss.item() * batch_size
                val_fk_loss += fk_loss.item() * batch_size

                val_preds.append(pred_joint_angles.cpu().numpy())
                val_trues.append(target_angles.cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_trues = np.vstack(val_trues)
        val_r2 = 1 - (np.sum((val_trues - val_preds) ** 2) /
                      (np.sum((val_trues - np.mean(val_trues)) ** 2) + 1e-8))
        val_loss = val_loss / len(val_loader.dataset)

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 日志
        logging.info(
            f"Epoch [{epoch}/{config.epochs}] | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Train R²: {train_r2:.6f} | Val R²: {val_r2:.6f} | "
            f"IK: {train_ik_loss/len(train_loader.dataset):.4f} | "
            f"FK: {train_fk_loss/len(train_loader.dataset):.6f} | "
            f"GAP: {train_continuity_loss/len(train_loader.dataset):.4f} | "
            f"LR: {current_lr:.6f} | Time: {train_time:.1f}s"
        )

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            logging.info(f"  >>> 保存最优模型（验证损失：{val_loss:.6f}）")

    logging.info("\n训练完成！")
    logging.info(f"最优验证损失: {best_val_loss:.6f}")


# if __name__ == "__main__":
#     train()
