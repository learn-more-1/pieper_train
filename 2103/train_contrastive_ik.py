"""
对比学习风格 IK 训练脚本 (2103版本)

核心特点:
1. 训练时用 history_joints 监督风格编码器
2. 验证时只用 history_poses 测试泛化性
3. 对比损失 + IK 损失联合优化
4. 监控风格相似度（学生 vs 教师）

用法:
    cd /home/bonuli/Pieper/2103
    python train_contrastive_ik.py [model_name]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import sys
import time
import os

sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/2003')  # 复用数据集
sys.path.insert(0, '/home/bonuli/Pieper/2103')  # 必须最后插入，确保在最前面

from model import ContrastiveStyleIK, compute_dataset_statistics, NormalizationLayer
from dataset_contrastive import create_contrastive_dataloaders

sys.path.insert(0, '/home/bonuli/Pieper/pieper1101')
try:
    from gpu_fk_wrapper import SimpleGPUFK
except ImportError:
    SimpleGPUFK = None

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("contrastive_ik_train.log"),
        logging.StreamHandler()
    ]
)


class Config:
    """训练配置"""
    # 数据配置
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
    num_joints = 7
    num_frames = 10
    
    # 训练参数
    batch_size = 1024
    epochs = 150
    initial_lr = 1e-3
    min_lr = 1e-6
    
    # 模型参数
    pose_dim = 7
    joint_dim = 7
    hidden_dim = 1200
    temporal_hidden = 256
    num_freqs = 15
    style_dim = 128  # 风格维度
    
    # 损失权重
    ik_weight = 1.0
    contrastive_weight = 0.05  # 降低，防止过度依赖 joints 监督
    
    # 训练时模拟推理的概率（关键防过拟合参数）
    # 以此概率只用 poses 训练，模拟验证场景
    inference_simulation_prob = 0.3  # 30% 概率训练时也不用 joints
    fk_weight = 1.0
    consistency_weight = 0.1
    
    # 对比学习参数
    temperature = 0.07
    use_infonce = True  # 使用 InfoNCE，否则用 MSE
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 关节损失权重
    joint_loss_weights = torch.tensor([
        1.6, 2.0, 1.6, 2.5, 1.0, 0.7, 0.6
    ])
    
    num_workers = 8
    pin_memory = True
    
    urdf_path = "/home/bonuli/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
    use_pinocchio_fk = True


def load_pinocchio_model(urdf_path):
    """加载 Pinocchio FK 模型"""
    try:
        from pinocchio_fk import PinocchioFK
        fk = PinocchioFK(urdf_path)
        logging.info("✓ 成功加载 Pinocchio FK 模型")
        return fk
    except Exception as e:
        logging.warning(f"Pinocchio 加载失败: {e}")
        return None


def weighted_mse_loss(pred, target, weights):
    """加权 MSE 损失"""
    squared_error = (pred - target) ** 2
    weighted_error = squared_error * weights.unsqueeze(0)
    return torch.mean(weighted_error)


def compute_style_similarity(pred_style, true_style):
    """计算风格余弦相似度"""
    if pred_style is None or true_style is None:
        return 0.0
    
    pred_norm = F.normalize(pred_style, dim=1)
    true_norm = F.normalize(true_style, dim=1)
    similarity = (pred_norm * true_norm).sum(dim=1).mean()
    return similarity.item()


def train():
    """主训练函数"""
    config = Config()
    
    # 早停配置
    patience = 10  # 早停耐心值
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = 'contrastive_ik'
    
    logging.info("=" * 80)
    logging.info("对比学习风格 IK 模型训练 (2103版本)")
    logging.info("=" * 80)
    logging.info("核心特性:")
    logging.info("  1. 训练时用 history_joints 监督风格编码器")
    logging.info("  2. 验证时只用 history_poses 测试泛化性")
    logging.info("  3. 对比损失让模型学会从末端轨迹推断运动风格")
    logging.info(f"  4. 风格维度: {config.style_dim}")
    logging.info(f"  5. 对比损失权重: {config.contrastive_weight}")
    logging.info(f"  6. 温度系数: {config.temperature}")
    logging.info(f"  7. 训练时推理模拟概率: {config.inference_simulation_prob}（防过拟合）")
    logging.info(f"  8. 早停耐心值: {patience} 轮")
    
    # 打印关节权重
    joint_names = ['ShoulderPitch', 'ShoulderRoll', 'ShoulderYaw',
                   'Elbow', 'ForearmRoll', 'WristYaw', 'WristPitch']
    logging.info("\n关节损失权重:")
    for i, (name, weight) in enumerate(zip(joint_names, config.joint_loss_weights)):
        logging.info(f"  J{i} ({name:15s}): {weight:.2f}")
    
    joint_weights = config.joint_loss_weights.to(config.device)
    
    # 加载数据集
    logging.info("\n加载数据集...")
    train_loader, val_loader = create_contrastive_dataloaders(config.data_path, config)
    
    # 计算统计信息
    logging.info("\n计算数据集统计信息...")
    pose_mean, pose_std, joint_mean, joint_std = compute_dataset_statistics(
        train_loader, config.pose_dim, config.joint_dim
    )
    
    norm_layer = NormalizationLayer(pose_mean, pose_std, joint_mean, joint_std)
    norm_layer = norm_layer.to(config.device)
    
    # 创建模型
    logging.info(f"\n创建模型: {model_name}")
    model = ContrastiveStyleIK(
        pose_dim=config.pose_dim,
        joint_dim=config.joint_dim,
        hidden_dim=config.hidden_dim,
        temporal_hidden=config.temporal_hidden,
        num_freqs=config.num_freqs,
        num_frames=config.num_frames,
        style_dim=config.style_dim,
        temperature=config.temperature
    )
    
    model = model.to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logging.info(f"模型总参数量: {total_params:.2f}M")
    logging.info(f"可训练参数量: {trainable_params:.2f}M")
    
    # 加载 FK 模型
    logging.info("\n加载 FK 模型...")
    gpu_fk = None
    if config.use_pinocchio_fk:
        gpu_fk = load_pinocchio_model(config.urdf_path)
    
    if gpu_fk is None and SimpleGPUFK is not None:
        try:
            gpu_fk = SimpleGPUFK()
            logging.info("使用 SimpleGPUFK")
        except:
            pass
    
    if gpu_fk is None:
        logging.warning("FK 模型加载失败，将不使用 FK 损失")
        config.fk_weight = 0.0
        config.consistency_weight = 0.0
    else:
        logging.info("✓ FK 模型加载成功")
    
    mse_criterion = nn.MSELoss()
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.initial_lr, weight_decay=1e-4)
    
    # 学习率调度
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,
        T_mult=2,
        eta_min=config.min_lr
    )
    
    # 检查点路径
    checkpoint_path = f"/home/bonuli/Pieper/2103/{model_name}_2103.pth"
    
    # 加载断点
    best_val_loss = float("inf")
    best_val_r2 = -float("inf")
    no_improve_epochs = 0
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        logging.info("\n加载断点...")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_val_r2 = checkpoint.get("best_val_r2", -float("inf"))
        logging.info(f"从第 {start_epoch} epoch 继续")
        logging.info(f"恢复最优 Val R²: {best_val_r2:.4f}")
    
    # 训练循环
    logging.info(f"\n开始训练（{config.epochs} epochs）...")
    logging.info("-" * 80)
    
    for epoch in range(start_epoch, config.epochs):
        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        train_ik_loss = 0.0
        train_contrastive_loss = 0.0
        train_fk_loss = 0.0
        train_style_sim = 0.0
        train_preds = []
        train_trues = []
        start_time = time.time()
        
        for batch_X, batch_y, _ in train_loader:
            batch_X = batch_X.to(config.device, non_blocking=True)
            batch_y = batch_y.to(config.device, non_blocking=True)
            
            batch_size = batch_X.shape[0]
            
            # 分解输入
            # batch_X 可能 shape 为 [B, 10, 14] 或 [B, 10, 7]
            # 需要根据实际情况拆分
            
            if batch_X.shape[-1] == 14:
                # 包含 pose + joint
                history_poses = batch_X[:, :, :config.pose_dim]    # [B, 10, 7]
                history_joints = batch_X[:, :, config.pose_dim:]   # [B, 10, 7]
            elif batch_X.shape[-1] == 7:
                # 只有 pose，需要报错或从其他方式获取 joint
                logging.error(f"batch_X 只有 7 维，需要 14 维 (pose+joint)。实际形状: {batch_X.shape}")
                logging.error("请检查 dataset_generalized.py 是否返回完整数据")
                raise RuntimeError(f"batch_X 维度不足: {batch_X.shape}, 需要 [B, 10, 14]")
            else:
                raise RuntimeError(f"batch_X 未知维度: {batch_X.shape}")
            
            # 目标
            if batch_y.shape[1] == 14:
                target_pose = batch_y[:, :config.pose_dim]
                target_angles = batch_y[:, config.pose_dim:]
            else:
                target_angles = batch_y
                target_pose = batch_X[:, -1, :config.pose_dim]
            
            optimizer.zero_grad()
            
            # 归一化
            target_pose_norm = norm_layer.normalize_pose(target_pose)
            history_poses_norm = norm_layer.normalize_history_poses(history_poses)
            history_joints_norm = norm_layer.normalize_joints(history_joints)
            
            # ===== 关键：随机决定是否模拟推理场景 =====
            # 以此概率只用 poses（不传入 joints），强制模型学会从 poses 推断
            simulate_inference = np.random.rand() < config.inference_simulation_prob
            
            if simulate_inference:
                # 模拟验证场景：只用 poses（和验证时一样）
                pred_joint, aux = model(
                    target_pose_norm,
                    history_poses_norm,
                    history_joints=None,  # 不传 joints！
                    mode='inference',
                    return_aux=True
                )
                contrastive_loss = torch.tensor(0.0, device=config.device)  # 无对比损失
            else:
                # 正常训练：用 joints 监督风格
                pred_joint, aux = model(
                    target_pose_norm,
                    history_poses_norm,
                    history_joints_norm,
                    mode='training',
                    return_aux=True
                )
                # ===== 对比损失（核心）=====
                contrastive_loss = model.compute_contrastive_loss(
                    aux['pred_style'],
                    aux['true_style']
                )
            
            # 反归一化
            pred_joint_denorm = norm_layer.denormalize_joint(pred_joint)
            
            # ===== 1. IK 损失 =====
            ik_loss = weighted_mse_loss(pred_joint_denorm, target_angles, joint_weights)
            
            # ===== 3. FK 损失 =====
            fk_loss = 0.0
            if config.fk_weight > 0 and gpu_fk is not None:
                pred_pos = gpu_fk.forward(pred_joint_denorm)
                target_pos = gpu_fk.forward(target_angles)
                if pred_pos is not None and target_pos is not None:
                    fk_loss = mse_criterion(pred_pos, target_pos)
            
            # ===== 4. 一致性损失 =====
            consistency_loss = 0.0
            if config.consistency_weight > 0:
                consistency_loss = torch.mean((pred_joint_denorm - torch.clamp(
                    pred_joint_denorm, -np.pi, np.pi
                )) ** 2)
            
            # ===== 总损失 =====
            total_loss = (
                config.ik_weight * ik_loss +
                config.contrastive_weight * contrastive_loss +
                config.fk_weight * fk_loss +
                config.consistency_weight * consistency_loss
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            train_loss += total_loss.item() * batch_size
            train_ik_loss += ik_loss.item() * batch_size
            train_contrastive_loss += contrastive_loss.item() * batch_size
            train_fk_loss += fk_loss.item() * batch_size if isinstance(fk_loss, torch.Tensor) else 0
            
            # 计算风格相似度
            style_sim = compute_style_similarity(aux['pred_style'], aux['true_style'])
            train_style_sim += style_sim * batch_size
            
            if len(train_preds) < 10000:
                train_preds.append(pred_joint_denorm.cpu().detach().numpy())
                train_trues.append(target_angles.cpu().detach().numpy())
        
        # 训练统计
        train_preds = np.vstack(train_preds)
        train_trues = np.vstack(train_trues)
        train_r2 = 1 - (np.sum((train_trues - train_preds) ** 2) /
                       (np.sum((train_trues - np.mean(train_trues)) ** 2) + 1e-8))
        
        num_train = len(train_loader.dataset)
        train_loss /= num_train
        train_ik_loss /= num_train
        train_contrastive_loss /= num_train
        train_fk_loss /= num_train
        train_style_sim /= num_train
        train_time = time.time() - start_time
        
        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0.0
        val_ik_loss = 0.0
        val_fk_loss = 0.0
        val_style_sim = 0.0  # 验证时也计算风格相似度（测试泛化性）
        val_preds = []
        val_trues = []
        
        with torch.no_grad():
            for batch_X, batch_y, _ in val_loader:
                batch_X = batch_X.to(config.device, non_blocking=True)
                batch_y = batch_y.to(config.device, non_blocking=True)
                
                batch_size = batch_X.shape[0]
                
                # 分解输入
                history_poses = batch_X[:, :, :config.pose_dim]
                history_joints = batch_X[:, :, config.pose_dim:]  # 用于计算风格相似度
                
                if batch_y.shape[1] == 14:
                    target_pose = batch_y[:, :config.pose_dim]
                    target_angles = batch_y[:, config.pose_dim:]
                else:
                    target_angles = batch_y
                    target_pose = batch_X[:, -1, :config.pose_dim]
                
                # 归一化
                target_pose_norm = norm_layer.normalize_pose(target_pose)
                history_poses_norm = norm_layer.normalize_history_poses(history_poses)
                history_joints_norm = norm_layer.normalize_joints(history_joints)
                
                # ===== 关键：验证时用 inference 模式（只用 poses）=====
                pred_joint, aux = model(
                    target_pose_norm,
                    history_poses_norm,
                    history_joints=None,  # ★ 不传 joints！
                    mode='inference',
                    return_aux=True
                )
                
                pred_joint_denorm = norm_layer.denormalize_joint(pred_joint)
                
                # IK 损失
                ik_loss = weighted_mse_loss(pred_joint_denorm, target_angles, joint_weights)
                
                # FK 损失
                fk_loss = 0.0
                if config.fk_weight > 0 and gpu_fk is not None:
                    pred_pos = gpu_fk.forward(pred_joint_denorm)
                    target_pos = gpu_fk.forward(target_angles)
                    if pred_pos is not None and target_pos is not None:
                        fk_loss = mse_criterion(pred_pos, target_pos)
                
                total_loss = config.ik_weight * ik_loss + config.fk_weight * fk_loss
                
                val_loss += total_loss.item() * batch_size
                val_ik_loss += ik_loss.item() * batch_size
                val_fk_loss += fk_loss.item() * batch_size if isinstance(fk_loss, torch.Tensor) else 0
                
                # 计算风格相似度（测试泛化性：只用 poses 推断的风格 vs 真实风格）
                # 注意：这里用教师网络提取 true_style 只是为了评估
                with torch.enable_grad() if False else torch.no_grad():
                    true_style = model.extract_joint_style(history_joints_norm)
                style_sim = compute_style_similarity(aux['pred_style'], true_style)
                val_style_sim += style_sim * batch_size
                
                val_preds.append(pred_joint_denorm.cpu().numpy())
                val_trues.append(target_angles.cpu().numpy())
        
        # 验证统计
        val_preds = np.vstack(val_preds)
        val_trues = np.vstack(val_trues)
        val_r2 = 1 - (np.sum((val_trues - val_preds) ** 2) /
                     (np.sum((val_trues - np.mean(val_trues)) ** 2) + 1e-8))
        
        num_val = len(val_loader.dataset)
        val_loss /= num_val
        val_ik_loss /= num_val
        val_fk_loss /= num_val
        val_style_sim /= num_val
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印日志
        logging.info(
            f"Epoch [{epoch:3d}/{config.epochs}] | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f} | "
            f"IK: {train_ik_loss:.4f}/{val_ik_loss:.4f} | "
            f"Cont: {train_contrastive_loss:.4f} | "
            f"StyleSim: {train_style_sim:.3f}/{val_style_sim:.3f} | "
            f"LR: {current_lr:.6f} | Time: {train_time:.1f}s"
        )
        
        # 保存最优模型（基于 Val R² 而不是 Val Loss）
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_r2': best_val_r2,
                'pose_mean': pose_mean,
                'pose_std': pose_std,
                'joint_mean': joint_mean,
                'joint_std': joint_std,
                'config': {
                    'pose_dim': config.pose_dim,
                    'joint_dim': config.joint_dim,
                    'hidden_dim': config.hidden_dim,
                    'temporal_hidden': config.temporal_hidden,
                    'num_freqs': config.num_freqs,
                    'num_frames': config.num_frames,
                    'style_dim': config.style_dim,
                }
            }, checkpoint_path)
            logging.info(f"  >>> 保存最优模型（Val R²：{val_r2:.4f}，Val Loss：{val_loss:.6f}）")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logging.info(f"\n早停：验证 R² {patience} 轮未改善，最佳 Val R²: {best_val_r2:.4f}")
                break
    
    logging.info("\n训练完成！")
    logging.info(f"最优验证 R²: {best_val_r2:.4f}")
    logging.info(f"最优验证损失: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
