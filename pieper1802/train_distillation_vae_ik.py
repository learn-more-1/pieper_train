"""
训练基于VAE蒸馏的IK模型

核心思想:
1. 使用VAE将目标位姿(7D)和历史关节角(7D)编码到潜在空间
2. 从潜在空间解码重建关节角度
3. 无需直接输入角度信息，只使用目标位姿

数据流:
- 输入1: 目标位姿 (7维: 位置3 + 四元数4)
- 输入2: 历史关节角 (7维)
- 编码器: (7+7) -> 256 -> 128 -> 64 -> (z_mean, z_log_var)
- 潜在变量: z = z_mean + exp(z_log_var/2) * epsilon
- 解码器: (z + 7) -> 128 -> 256 -> 128 -> 7 (重建关节角)
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

# 添加路径
sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/hhome/bonuli/Pieper/pieper1803')

from dataset_generalized import create_windowed_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("distillation_vae_train.log"), logging.StreamHandler()]
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

    # VAE模型配置
    latent_dim = 64
    encoder_hidden_dims = [256, 128]
    decoder_hidden_dims = [128, 256, 128]

    # 损失权重
    reconstruction_weight = 1.0    # 重建损失 (关节角度)
    kl_weight = 0.001              # KL散度权重
    fk_weight = 0.5                # 位置误差损失
    continuity_weight = 1.0        # 连续性损失

    # 数据加载优化
    num_workers = 4
    pin_memory = True

    # FK计算方式选择
    urdf_path = "/home/bonuli/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"


class VAEIKModel(nn.Module):
    """
    基于VAE的IK模型

    输入:
        - target_pose: [batch, 7] 目标位姿 (位置3 + 四元数4)
        - history_angles: [batch, 7] 历史关节角

    输出:
        - reconstructed_angles: [batch, 7] 重建的关节角度
        - z_mean: [batch, latent_dim] 潜在变量均值
        - z_log_var: [batch, latent_dim] 潜在变量对数方差
    """

    def __init__(self, pose_dim=7, joint_dim=7, latent_dim=64,
                 encoder_hidden_dims=[256, 128],
                 decoder_hidden_dims=[128, 256, 128]):
        super(VAEIKModel, self).__init__()

        self.pose_dim = pose_dim
        self.joint_dim = joint_dim
        self.latent_dim = latent_dim

        # 编码器输入维度 = 位姿(7) + 历史关节角(7) = 14
        encoder_input_dim = pose_dim + joint_dim

        # 构建编码器
        encoder_layers = []
        prev_dim = encoder_input_dim
        for hidden_dim in encoder_hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # 输出均值和方差
        self.fc_mean = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

        # 解码器输入维度 = 潜在变量(64) + 目标位姿(7) = 71
        decoder_input_dim = latent_dim + pose_dim

        # 构建解码器
        decoder_layers = []
        prev_dim = decoder_input_dim
        for hidden_dim in decoder_hidden_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        self.decoder = nn.Sequential(*decoder_layers)

        # 输出层：重建关节角度
        self.fc_output = nn.Linear(prev_dim, joint_dim)

    def encode(self, x):
        """编码器：输入 -> 潜在空间参数"""
        h = self.encoder(x)
        z_mean = self.fc_mean(h)
        z_log_var = self.fc_log_var(h)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        """重参数化技巧：采样潜在变量"""
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z, target_pose):
        """解码器：潜在变量 + 目标位姿 -> 重建关节角度"""
        # 拼接潜在变量和目标位姿
        decoder_input = torch.cat([z, target_pose], dim=1)
        h = self.decoder(decoder_input)
        reconstructed_angles = self.fc_output(h)
        return reconstructed_angles

    def forward(self, target_pose, history_angles):
        """
        前向传播

        Args:
            target_pose: [batch, 7] 目标位姿
            history_angles: [batch, 7] 历史关节角

        Returns:
            reconstructed_angles: [batch, 7] 重建的关节角度
            z_mean: [batch, latent_dim]
            z_log_var: [batch, latent_dim]
        """
        # 拼接目标位姿和历史关节角
        encoder_input = torch.cat([target_pose, history_angles], dim=1)

        # 编码
        z_mean, z_log_var = self.encode(encoder_input)

        # 采样潜在变量
        z = self.reparameterize(z_mean, z_log_var)

        # 解码（使用目标位姿作为条件）
        reconstructed_angles = self.decode(z, target_pose)

        return reconstructed_angles, z_mean, z_log_var


# Pinocchio FK模型（全局变量）
_pinocchio_model = None
_pinocchio_data = None
_joint_ids = []
_wrist_frame_id = None


def load_pinocchio_model(urdf_path):
    """
    加载Pinocchio模型（全局单例）

    Args:
        urdf_path: URDF文件路径
    """
    global _pinocchio_model, _pinocchio_data, _joint_ids, _wrist_frame_id

    try:
        import pinocchio as pin

        # 加载URDF
        _pinocchio_model = pin.buildModelFromUrdf(urdf_path)
        _pinocchio_data = _pinocchio_model.createData()

        # 左臂关节ID
        joint_names = [
            'left_shoulder_pitch_joint',
            'left_shoulder_roll_joint',
            'left_shoulder_yaw_joint',
            'left_elbow_joint',
            'left_wrist_roll_joint',
            'left_wrist_pitch_joint',
            'left_wrist_yaw_joint'
        ]

        # 获取关节ID
        _joint_ids = []
        for name in joint_names:
            joint_id = _pinocchio_model.getJointId(name)
            if joint_id < _pinocchio_model.njoints:
                _joint_ids.append(joint_id)

        # 腕部link名称
        wrist_link = "left_wrist_yaw_link"
        _wrist_frame_id = _pinocchio_model.getFrameId(wrist_link)

        logging.info(f"✓ Pinocchio模型加载成功")
        logging.info(f"  URDF: {urdf_path}")
        logging.info(f"  关节数: {len(_joint_ids)}")
        logging.info(f"  腕部frame: {wrist_link} (ID: {_wrist_frame_id})")

        return True

    except ImportError:
        logging.error("❌ 未安装Pinocchio，请运行: pip install pinocchio")
        return False
    except Exception as e:
        logging.error(f"❌ 加载Pinocchio模型失败: {e}")
        return False


def pinocchio_fk_batch(joint_angles):
    """
    使用Pinocchio批量计算FK

    Args:
        joint_angles: [batch, 7] tensor 关节角度

    Returns:
        positions: [batch, 3] tensor 腕部位置
    """
    global _pinocchio_model, _pinocchio_data, _joint_ids, _wrist_frame_id

    if _pinocchio_model is None:
        raise ValueError("Pinocchio模型未加载，请先调用load_pinocchio_model()")

    batch_size = joint_angles.shape[0]
    device = joint_angles.device

    # 转换为numpy
    if isinstance(joint_angles, torch.Tensor):
        angles_np = joint_angles.detach().cpu().numpy()
    else:
        angles_np = joint_angles

    wrist_positions = []

    for i in range(batch_size):
        # 设置关节角度（零位配置）
        q = np.zeros(_pinocchio_model.nq)

        # 填入左臂角度
        for j, joint_id in enumerate(_joint_ids):
            if j < len(angles_np[i]):
                joint = _pinocchio_model.joints[joint_id]
                idx_q = joint.idx_q
                if idx_q < _pinocchio_model.nq:
                    q[idx_q] = angles_np[i][j]

        # 正运动学
        import pinocchio as pin
        pin.forwardKinematics(_pinocchio_model, _pinocchio_data, q)
        pin.updateFramePlacements(_pinocchio_model, _pinocchio_data)

        # 获取腕部位置
        wrist_pos = _pinocchio_data.oMf[_wrist_frame_id].translation
        wrist_positions.append(wrist_pos.copy())

    # 转换为tensor
    wrist_tensor = torch.tensor(np.array(wrist_positions),
                                dtype=torch.float32, device=device)

    return wrist_tensor


def load_robot_model():
    """加载Pinocchio模型"""
    success = load_pinocchio_model("/home/bonuli/xr_teleoperate/assets/g1/g1_body29_hand14.urdf")
    return success


def forward_kinematics_with_pose(joint_angles):
    """
    使用Pinocchio计算FK

    Args:
        joint_angles: [batch, 7] 关节角度

    Returns:
        positions: [batch, 3] 手腕位置
    """
    return pinocchio_fk_batch(joint_angles)


def kl_divergence(z_mean, z_log_var):
    """计算KL散度（潜在变量与标准正态分布的散度）"""
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
    return torch.mean(kl_loss)


def train():
    config = Config()

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = 'distillation_vae_ik'

    logging.info("=" * 70)
    logging.info("训练基于VAE蒸馏的IK模型")
    logging.info("=" * 70)
    logging.info("核心特性:")
    logging.info("  1. VAE编码目标位姿和历史关节角到潜在空间")
    logging.info("  2. 从潜在空间解码重建关节角度")
    logging.info("  3. 无需直接输入目标角度信息")
    logging.info(f"  4. 潜在空间维度: {config.latent_dim}")

    # 加载数据集
    logging.info("\n加载数据集...")
    train_loader, val_loader = create_windowed_dataloaders(config.data_path, config)

    # 创建VAE模型
    logging.info(f"\n创建模型: {model_name}")
    model = VAEIKModel(
        pose_dim=7,
        joint_dim=config.num_joints,
        latent_dim=config.latent_dim,
        encoder_hidden_dims=config.encoder_hidden_dims,
        decoder_hidden_dims=config.decoder_hidden_dims
    )
    model = model.to(config.device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"模型参数量: {total_params:.2f}M")

    # 打印模型结构
    logging.info("\n模型结构:")
    logging.info(f"  编码器: 7+7 -> {config.encoder_hidden_dims} -> {config.latent_dim}*2")
    logging.info(f"  解码器: {config.latent_dim}+7 -> {config.decoder_hidden_dims} -> 7")

    # 加载Pinocchio模型
    logging.info("\n加载Pinocchio FK模型...")
    success = load_robot_model()

    if not success:
        logging.error("无法加载Pinocchio模型，退出")
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
    checkpoint_path = f"/home/bonuli/Piper/pieper1803/{model_name}_1803.pth"

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
        train_reconstruction_loss = 0.0
        train_kl_loss = 0.0
        train_fk_loss = 0.0
        train_continuity_loss = 0.0
        train_preds = []
        train_trues = []
        start_time = time.time()

        for batch_X, batch_y, batch_last_angle in train_loader:
            # batch_X: [batch, 10, 7] 机器人历史角度
            # batch_y: [batch, 7] or [batch, 14] - 机器人角度，或(人位姿+机器人角度)
            # batch_last_angle: [batch, 7] 历史窗口最后一帧的角度

            batch_X, batch_y, batch_last_angle = (
                batch_X.to(config.device, non_blocking=True),
                batch_y.to(config.device, non_blocking=True),
                batch_last_angle.to(config.device, non_blocking=True)
            )

            batch_size = batch_X.shape[0]

            # 提取历史关节角（使用最后一帧）
            history_angles = batch_last_angle  # [batch, 7]

            # 判断y的格式并提取目标位姿和角度
            if batch_y.shape[1] == 14:
                # y包含位姿信息: 前7维人位姿，后7维机器人角度
                target_pose = batch_y[:, :7]      # [batch, 7] 人目标位姿
                target_angles = batch_y[:, 7:]    # [batch, 7] 机器人目标角度
            else:
                # y只包含机器人角度，需要用FK计算位姿
                target_angles = batch_y  # [batch, 7]
                # 从机器人目标角度计算末端位姿（使用FK）
                target_position = forward_kinematics_with_pose(target_angles)

                # 由于Pinocchio FK只返回位置，我们需要构建目标位姿
                # 对于姿态，我们可以使用零或历史信息（这里简化处理）
                # 在实际应用中，可能需要完整的FK计算位置+姿态
                target_orientation = torch.zeros(batch_size, 4, device=config.device)
                target_orientation[:, 3] = 1.0  # 单位四元数
                target_pose = torch.cat([target_position, target_orientation], dim=1)  # [batch, 7]

            optimizer.zero_grad()

            # 前向传播
            pred_joint_angles, z_mean, z_log_var = model(target_pose, history_angles)

            # 计算损失
            # 1. 重建损失：预测角度 vs 目标角度
            reconstruction_loss = mse_criterion(pred_joint_angles, target_angles)

            # 2. KL散度损失
            kl_loss = kl_divergence(z_mean, z_log_var)

            # 3. FK损失：预测角度的末端位置 vs 真实角度的末端位置
            pred_position = forward_kinematics_with_pose(pred_joint_angles)
            target_position_from_angles = forward_kinematics_with_pose(target_angles)
            fk_loss = mse_criterion(pred_position, target_position_from_angles)

            # 4. 连续性损失：预测角度与当前角度的差异
            continuity_loss = torch.mean((pred_joint_angles - history_angles) ** 2)

            # 总损失
            total_loss = (
                config.reconstruction_weight * reconstruction_loss +
                config.kl_weight * kl_loss +
                config.fk_weight * fk_loss +
                config.continuity_weight * continuity_loss
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * batch_size
            train_reconstruction_loss += reconstruction_loss.item() * batch_size
            train_kl_loss += kl_loss.item() * batch_size
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
        val_reconstruction_loss = 0.0
        val_kl_loss = 0.0
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

                # 提取历史关节角
                history_angles = batch_last_angle

                # 判断y的格式并提取目标位姿和角度
                if batch_y.shape[1] == 14:
                    target_pose = batch_y[:, :7]
                    target_angles = batch_y[:, 7:]
                else:
                    target_angles = batch_y
                    target_position = forward_kinematics_with_pose(target_angles)
                    target_orientation = torch.zeros(batch_size, 4, device=config.device)
                    target_orientation[:, 3] = 1.0
                    target_pose = torch.cat([target_position, target_orientation], dim=1)

                # 前向传播
                pred_joint_angles, z_mean, z_log_var = model(target_pose, history_angles)

                # 计算损失
                reconstruction_loss = mse_criterion(pred_joint_angles, target_angles)
                kl_loss = kl_divergence(z_mean, z_log_var)

                # FK损失：预测角度的末端位置 vs 真实角度的末端位置
                pred_position = forward_kinematics_with_pose(pred_joint_angles)
                target_position_from_angles = forward_kinematics_with_pose(target_angles)
                fk_loss = mse_criterion(pred_position, target_position_from_angles)

                continuity_loss = torch.mean((pred_joint_angles - history_angles) ** 2)

                total_loss = (
                    config.reconstruction_weight * reconstruction_loss +
                    config.kl_weight * kl_loss +
                    config.fk_weight * fk_loss +
                    config.continuity_weight * continuity_loss
                )

                val_loss += total_loss.item() * batch_size
                val_reconstruction_loss += reconstruction_loss.item() * batch_size
                val_kl_loss += kl_loss.item() * batch_size
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
            f"Recon: {train_reconstruction_loss/len(train_loader.dataset):.4f} | "
            f"KL: {train_kl_loss/len(train_loader.dataset):.6f} | "
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


if __name__ == "__main__":
    train()
