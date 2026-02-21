"""
改进的知识蒸馏：特征蒸馏方案

核心思想：
1. 教师模型使用历史输入，提取中间特征（时序编码特征、注意力权重）
2. 学生模型尝试从末端位姿重建这些特征
3. 蒸馏目标：让学生学习教师内部的历史耦合表示

蒸馏层次：
1. Pieper权重蒸馏：position_weights, orientation_weights
2. 时序特征蒸馏：temporal_features（压缩历史信息）
3. 节点特征蒸馏：GNN节点特征
4. 输出蒸馏（可选）：最终预测角度
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import time
import sys

sys.path.insert(0, '/home/bonuli/Pieper/pieper1702')

from causal_ik_model_pieper2 import PieperCausalIK
from causal_ik_model_improved import ImprovedSimplifiedCausalIK
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train_feature_distillation.log")]
)


class FeatureDistillationConfig:
    # 数据路径
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
    teacher_path = "/home/bonuli/Pieper/pieper1101/pieper_causal_ik_1101.pth"

    # 训练参数
    batch_size = 2048
    epochs = 100
    lr = 2e-4
    weight_decay = 1e-4
    warmup_epochs = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 蒸馏参数
    temperature = 2.0

    # 损失权重
    alpha_output = 0.3        # 输出蒸馏权重
    alpha_weights = 0.3        # 权重蒸馏
    alpha_temporal = 0.2      # 时序特征蒸馏（通过投影）
    fk_weight = 0.2

    # 早停
    patience = 15
    min_delta = 1e-5

    # 保存路径
    checkpoint_path = "/home/bonuli/Pieper/pieper1702/improved_simplified_feature_distilled_best.pth"


class TempuralFeatureProjector(nn.Module):
    """将学生特征投影到教师时序特征空间"""

    def __init__(self, student_hidden_dim=256, teacher_temporal_dim=256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(student_hidden_dim, student_hidden_dim),
            nn.LayerNorm(student_hidden_dim),
            nn.GELU(),
            nn.Linear(student_hidden_dim, teacher_temporal_dim)
        )

    def forward(self, student_features):
        """
        Args:
            student_features: [batch, hidden_dim] 或 [batch, 7, hidden_dim]

        Returns:
            projected: [batch, teacher_temporal_dim]
        """
        if student_features.dim() == 3:
            # 聚合
            student_features = student_features.mean(dim=1)

        return self.projector(student_features)


def distillation_loss_with_features(student_output, teacher_output, target,
                                     student_features_dict, teacher_features_dict,
                                     config, projectors):
    """
    计算多层次蒸馏损失

    Args:
        student_output: [batch, 7] 学生预测
        teacher_output: [batch, 7] 教师预测
        target: [batch, 7] 真实标签
        student_features_dict: dict, 学生中间特征
        teacher_features_dict: dict, 教师中间特征
        config: 配置
        projectors: 投影器

    Returns:
        total_loss: 总损失
        loss_dict: 各项损失的详细信息
    """
    losses = {}

    # 1. Hard loss: 学生 vs 真实标签
    hard_loss = F.mse_loss(student_output, target)
    losses['hard'] = hard_loss

    # 2. Soft loss: 学生 vs 教师输出
    soft_loss = F.mse_loss(
        student_output / config.temperature,
        teacher_output / config.temperature
    ) * (config.temperature ** 2)
    losses['soft_output'] = soft_loss

    # 3. Pieper权重蒸馏
    if 'position_weights' in teacher_features_dict and 'joint_features' in student_features_dict:
        # 教师的权重: [batch, 7]
        teacher_pos_weights = teacher_features_dict['position_weights']
        teacher_ori_weights = teacher_features_dict['orientation_weights']

        # 从学生的joint特征重建权重（通过注意力）
        student_joint_feat = student_features_dict['joint_features']  # [batch, 7, hidden_dim]
        batch_size, num_joints, hidden_dim = student_joint_feat.shape

        # 简单的权重重建：用hidden_dim的均值作为权重
        student_pos_logits = student_joint_feat.mean(dim=-1)  # [batch, 7]
        student_ori_logits = student_joint_feat.mean(dim=-1)

        # Softmax归一化
        student_pos_weights = F.softmax(student_pos_logits / config.temperature, dim=-1)
        student_ori_weights = F.softmax(student_ori_logits / config.temperature, dim=-1)

        # 权重蒸馏损失
        pos_weight_loss = F.kl_div(
            F.log_softmax(student_pos_logits / config.temperature, dim=-1),
            F.softmax(teacher_pos_weights / config.temperature, dim=-1),
            reduction='batchmean'
        ) * (config.temperature ** 2)

        ori_weight_loss = F.kl_div(
            F.log_softmax(student_ori_logits / config.temperature, dim=-1),
            F.softmax(teacher_ori_weights / config.temperature, dim=-1),
            reduction='batchmean'
        ) * (config.temperature ** 2)

        weight_loss = (pos_weight_loss + ori_weight_loss) / 2
        losses['weights'] = weight_loss
    else:
        losses['weights'] = torch.tensor(0.0, device=config.device)

    # 4. 时序特征蒸馏（可选，需要投影器）
    if 'temporal_features' in teacher_features_dict and config.alpha_temporal > 0:
        # 教师时序特征: [batch, hidden_dim]
        teacher_temporal = teacher_features_dict['temporal_features']

        # 学生特征投影（使用末端位姿特征作为代理）
        if 'end_pose_feat' in student_features_dict:
            student_end_feat = student_features_dict['end_pose_feat']
            # 投影到教师特征空间
            if projectors is not None:
                projected = projectors['temporal'](student_end_feat)

                # 对齐损失
                temporal_loss = F.mse_loss(projected, teacher_temporal.detach())
                losses['temporal'] = temporal_loss
            else:
                losses['temporal'] = torch.tensor(0.0, device=config.device)
        else:
            losses['temporal'] = torch.tensor(0.0, device=config.device)
    else:
        losses['temporal'] = torch.tensor(0.0, device=config.device)

    # 组合损失
    total_loss = (
        (1 - config.alpha_output - config.alpha_weights - config.alpha_temporal) * hard_loss +
        config.alpha_output * soft_loss +
        config.alpha_weights * losses['weights'] +
        config.alpha_temporal * losses['temporal']
    )

    return total_loss, losses


class WarmupCosineScheduler:
    """预热 + 余弦退火学习率调度"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


def load_data(data_path, device=None, preload=False):
    """加载数据"""
    logging.info(f"加载数据: {data_path}")
    start = time.time()

    data = np.load(data_path)
    y = data['y'].astype(np.float32)

    split_idx = int(len(y) * 0.9)
    train_data = y[:split_idx]
    val_data = y[split_idx:]

    if preload and device is not None and device.type == 'cuda':
        logging.info(f"  预加载到GPU...")
        train_data = torch.from_numpy(train_data).to(device)
        val_data = torch.from_numpy(val_data).to(device)
        logging.info(f"  ✓ 数据已预加载到GPU")

    elapsed = time.time() - start
    logging.info(f"  训练集: {len(train_data)}")
    logging.info(f"  验证集: {len(val_data)}")
    logging.info(f"  加载时间: {elapsed:.2f}s")

    return train_data, val_data


def train_epoch_feature_distill(student, teacher, train_data, config, gpu_fk,
                                   optimizer, scheduler, projectors, epoch):
    """训练一个epoch（特征蒸馏）"""
    student.train()
    train_loss = 0.0
    train_hard_loss = 0.0
    train_soft_loss = 0.0
    train_weight_loss = 0.0
    train_fk_loss = 0.0

    n_batches = (len(train_data) + config.batch_size - 1) // config.batch_size

    for i in range(n_batches):
        start_idx = i * config.batch_size
        end_idx = min((i + 1) * config.batch_size, len(train_data))

        if isinstance(train_data, torch.Tensor):
            batch_y = train_data[start_idx:end_idx]
        else:
            batch_y = torch.from_numpy(train_data[start_idx:end_idx]).to(config.device)

        target_pose = batch_y[:, :7]
        target_angles = batch_y[:, 7:]
        target_position = target_pose[:, :3]
        target_orientation = target_pose[:, 3:7]

        optimizer.zero_grad()

        # 教师模型推理（用历史帧）- 提取中间特征
        with torch.no_grad():
            history_frames = target_angles.unsqueeze(1).repeat(1, 10, 1)
            teacher_output, teacher_info = teacher(history_frames, target_position, target_orientation)

            # 提取教师中间特征（需要修改教师模型返回更多信息）
            # 当前PieperCausalIK的info包含：position_weights, orientation_weights, modulated_pos, modulated_ori

        # 学生模型推理（不需要历史）
        student_output, student_info = student(target_position, target_orientation)

        # 构建特征字典
        teacher_features = {
            'position_weights': teacher_info['position_weights'],
            'orientation_weights': teacher_info['orientation_weights']
        }

        student_features = {
            'joint_features': student_info['joint_features'],  # [batch, 7, hidden_dim]
            'end_pose_feat': None  # 可以从student_info提取
        }

        # 计算蒸馏损失
        total_loss, loss_dict = distillation_loss_with_features(
            student_output, teacher_output, target_angles,
            student_features, teacher_features, config, projectors
        )

        # FK 损失
        pred_position = gpu_fk.forward(student_output)
        target_position_fk = gpu_fk.forward(target_angles)
        fk_loss = F.mse_loss(pred_position, target_position_fk)

        total_loss = total_loss + config.fk_weight * fk_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = len(batch_y)
        train_loss += total_loss.item() * batch_size
        train_hard_loss += loss_dict['hard'].item() * batch_size
        train_soft_loss += loss_dict['soft_output'].item() * batch_size
        train_weight_loss += loss_dict['weights'].item() * batch_size
        train_fk_loss += fk_loss.item() * batch_size

        if (i + 1) % 100 == 0:
            progress = (i + 1) / n_batches * 100
            logging.info(
                f"  Progress: {progress:.1f}% | "
                f"Loss: {total_loss.item():.6f} | "
                f"Hard: {loss_dict['hard']:.4f} | Soft: {loss_dict['soft_output']:.4f} | "
                f"Weight: {loss_dict['weights']:.4f}"
            )

    train_loss = train_loss / len(train_data)
    train_hard_loss = train_hard_loss / len(train_data)
    train_soft_loss = train_soft_loss / len(train_data)
    train_weight_loss = train_weight_loss / len(train_data)
    train_fk_loss = train_fk_loss / len(train_data)

    return train_loss, train_hard_loss, train_soft_loss, train_weight_loss, train_fk_loss


def validate(student, val_data, config):
    """验证"""
    student.eval()
    val_mae = 0.0
    val_mse = 0.0

    n_batches = (len(val_data) + config.batch_size - 1) // config.batch_size

    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * config.batch_size
            end_idx = min((i + 1) * config.batch_size, len(val_data))

            if isinstance(val_data, torch.Tensor):
                batch_y = val_data[start_idx:end_idx]
            else:
                batch_y = torch.from_numpy(val_data[start_idx:end_idx]).to(config.device)

            target_pose = batch_y[:, :7]
            target_angles = batch_y[:, 7:]
            target_position = target_pose[:, :3]
            target_orientation = target_pose[:, 3:7]

            pred_angles, _ = student(target_position, target_orientation)

            mae = F.l1_loss(pred_angles, target_angles)
            mse = F.mse_loss(pred_angles, target_angles)

            val_mae += mae.item() * len(batch_y)
            val_mse += mse.item() * len(batch_y)

    val_mae = val_mae / len(val_data)
    val_mse = val_mse / len(val_data)

    return val_mae, val_mse


def train_feature_distillation():
    config = FeatureDistillationConfig()

    logging.info("=" * 70)
    logging.info("特征蒸馏训练：从教师的历史耦合特征学习")
    logging.info("=" * 70)
    logging.info(f"  蒸馏层次：输出 + Pieper权重 + 时序特征")
    logging.info(f"  alpha_output: {config.alpha_output}")
    logging.info(f"  alpha_weights: {config.alpha_weights}")

    # 加载数据
    train_data, val_data = load_data(
        config.data_path,
        device=config.device,
        preload=True
    )

    # 加载教师模型
    logging.info(f"\n加载教师模型...")
    teacher = PieperCausalIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        num_layers=2
    ).to(config.device)

    checkpoint = torch.load(config.teacher_path, map_location=config.device)
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    logging.info(f"  ✓ 教师: {teacher_params:.2f}M 参数")

    # 创建学生模型
    logging.info(f"\n创建学生模型（改进简化版）...")
    student = ImprovedSimplifiedCausalIK(
        num_joints=7,
        hidden_dim=256,
        num_heads=4,
        num_decoder_layers=3
    ).to(config.device)

    student_params = sum(p.numel() for p in student.parameters()) / 1e6
    logging.info(f"  ✓ 学生: {student_params:.2f}M 参数")

    # 创建特征投影器
    projectors = nn.ModuleDict({
        'temporal': nn.Linear(256, 256).to(config.device)
    })

    # 加载 FK
    try:
        gpu_fk = SimpleGPUFK()
        logging.info(f"\n✓ GPU FK 加载成功")
    except:
        logging.error(f"\n✗ GPU FK 加载失败")
        return

    # 优化器
    optimizer = optim.AdamW(
        list(student.parameters()) + list(projectors.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.epochs,
        min_lr=1e-6
    )

    # 早停
    best_val_mae = float("inf")
    patience_counter = 0

    n_train_batches = (len(train_data) + config.batch_size - 1) // config.batch_size
    logging.info(f"\n开始特征蒸馏训练...")
    logging.info(f"  batches/epoch: {n_train_batches}")

    for epoch in range(config.epochs):
        epoch_start = time.time()
        current_lr = scheduler.step(epoch)

        logging.info(f"\nEpoch [{epoch}/{config.epochs}] | LR: {current_lr:.6f}")

        # 训练
        train_loss, train_hard_loss, train_soft_loss, train_weight_loss, train_fk_loss = train_epoch_feature_distill(
            student, teacher, train_data, config, gpu_fk, optimizer, scheduler, projectors, epoch
        )

        # 验证
        val_mae, val_mse = validate(student, val_data, config)

        epoch_time = time.time() - epoch_start

        logging.info(
            f"\n{'='*60}\n"
            f"Epoch [{epoch}/{config.epochs}] | Time: {epoch_time:.1f}s\n"
            f"  Train Loss: {train_loss:.6f}\n"
            f"  Hard Loss: {train_hard_loss:.6f} (vs 真实)\n"
            f"  Soft Loss: {train_soft_loss:.6f} (vs 教师输出)\n"
            f"  Weight Loss: {train_weight_loss:.6f} (vs 教师权重)\n"
            f"  FK Loss: {train_fk_loss:.6f}\n"
            f"  Val MAE: {val_mae:.6f} | Val MSE: {val_mse:.6f}\n"
            f"{'='*60}"
        )

        # 保存
        if val_mae < best_val_mae - config.min_delta:
            best_val_mae = val_mae
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'projectors_state_dict': projectors.state_dict(),
                'best_val_mae': best_val_mae,
            }, config.checkpoint_path)

            logging.info(f"  >>> 保存最优模型 (val_mae: {val_mae:.6f})")
        else:
            patience_counter += 1
            logging.info(f"  无改善 ({patience_counter}/{config.patience})")

        if patience_counter >= config.patience:
            logging.info(f"\n早停触发！")
            break

    logging.info(f"\n特征蒸馏训练完成！")
    logging.info(f"  最优验证 MAE: {best_val_mae:.6f}")
    logging.info(f"  模型保存到: {config.checkpoint_path}")


if __name__ == "__main__":
    train_feature_distillation()
