"""
肘部位置约束训练

同时优化肘部和末端位置，提高中间关节精度
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import sys
import time

sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from explicit_coupling_ik import ExplicitCouplingIK
from dataset_generalized import create_windowed_dataloaders
from fk_with_elbow import FKWithElbow, MultiPositionLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("elbow_constraint_train.log"), logging.StreamHandler()]
)


class Config:
    # 数据配置
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
    num_joints = 7
    num_frames = 10
    
    # 训练参数
    batch_size = 1024
    epochs = 77
    initial_lr = 2e-3
    min_lr = 1e-6
    weight_decay = 5e-4
    
    # 早停
    early_stopping_patience = 30
    
    # 多位置损失权重
    elbow_weight = 0.5  # 肘部位置权重
    end_weight = 0.5    # 末端位置权重
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 模型配置
    hidden_dim = 256
    
    # 损失权重
    ik_weight = 1.0      # IK角度损失
    pos_weight = 0.5     # 位置损失（包含肘部+末端）
    coupling_reg = 0.01  # 耦合正则化
    
    # 数据加载
    num_workers = 4
    pin_memory = True


def calculate_r2(pred, target):
    """计算R²分数"""
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean(dim=0)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2.item()


def train():
    config = Config()
    
    model_name = 'elbow_constraint_ik'
    
    logging.info("=" * 90)
    logging.info("肘部位置约束训练")
    logging.info("=" * 90)
    logging.info("改进:")
    logging.info(f"  1. 同时约束肘部位置 (weight={config.elbow_weight})")
    logging.info(f"  2. 同时约束末端位置 (weight={config.end_weight})")
    logging.info("  3. 提高中间关节精度")
    
    # 加载数据
    logging.info("\n加载ACCAD+CMU数据集...")
    
    class DataConfig:
        data_path = config.data_path
        num_joints = config.num_joints
        num_frames = config.num_frames
        batch_size = config.batch_size
        num_workers = config.num_workers
        pin_memory = config.pin_memory
    
    train_loader, val_loader = create_windowed_dataloaders(config.data_path, DataConfig())
    
    # 创建模型
    logging.info(f"\n创建模型: {model_name}")
    model = ExplicitCouplingIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=config.hidden_dim,
        use_temporal=False
    ).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"模型参数量: {total_params:.2f}M")
    
    # 加载FK（支持肘部计算）
    logging.info("\n加载FK模型（支持肘部位置计算）...")
    fk = FKWithElbow()
    
    # 多位置损失
    pos_criterion = MultiPositionLoss(
        elbow_weight=config.elbow_weight,
        end_weight=config.end_weight
    )
    mse_criterion = nn.MSELoss()
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.initial_lr,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度
    def lr_lambda(epoch):
        if epoch < 10:
            return epoch / 10
        else:
            progress = (epoch - 10) / (config.epochs - 10)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 早停
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    
    checkpoint_path = f"/home/bonuli/Pieper/pieper1802/{model_name}.pth"
    
    logging.info(f"\n开始训练（{config.epochs} epochs）...")
    logging.info("=" * 90)
    logging.info(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>11} | {'Elbow(mm)':>10} | {'End(mm)':>10} | {'IK RMSE(°)':>11} | {'Val R²':>8} | {'LR':>9}")
    logging.info("-" * 90)
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        
        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        train_ik_loss = 0.0
        train_pos_loss = 0.0
        
        all_train_preds = []
        all_train_targets = []
        
        for batch_X, batch_y, _ in train_loader:
            batch_y = batch_y.to(config.device, non_blocking=True)
            batch_size = batch_y.shape[0]
            
            if batch_y.shape[1] != 14:
                continue
            
            human_pose = batch_y[:, :7]
            target_angles = batch_y[:, 7:]
            
            human_pos = human_pose[:, :3]
            human_ori = human_pose[:, 3:7]
            
            optimizer.zero_grad()
            
            # 预测角度
            pred_angles, info = model(human_pos, human_ori)
            
            # IK损失
            ik_loss = mse_criterion(pred_angles, target_angles)
            
            # 计算位置（肘部+末端）
            with torch.no_grad():
                target_elbow, target_end = fk.compute_positions(target_angles)
            pred_elbow, pred_end = fk.compute_positions(pred_angles)
            
            pos_loss, pos_loss_dict = pos_criterion(
                pred_elbow, pred_end,
                target_elbow, target_end
            )
            
            # 耦合正则化
            coupling_loss = torch.tensor(0.0, device=config.device)
            for key, value in info['coupling_info'].items():
                coupling_loss += ((value - 0.5) ** 2)
            
            # 总损失
            total_loss = (
                config.ik_weight * ik_loss +
                config.pos_weight * pos_loss +
                config.coupling_reg * coupling_loss
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item() * batch_size
            train_ik_loss += ik_loss.item() * batch_size
            train_pos_loss += pos_loss.item() * batch_size
            
            all_train_preds.append(pred_angles.detach().cpu())
            all_train_targets.append(target_angles.cpu())
        
        train_loss /= len(train_loader.dataset)
        train_ik_loss /= len(train_loader.dataset)
        train_pos_loss /= len(train_loader.dataset)
        
        # 训练R²
        all_train_preds = torch.cat(all_train_preds, dim=0)
        all_train_targets = torch.cat(all_train_targets, dim=0)
        train_r2 = calculate_r2(all_train_preds, all_train_targets)
        
        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0.0
        val_ik_loss = 0.0
        val_pos_loss = 0.0
        val_elbow_error = 0.0
        val_end_error = 0.0
        
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y, _ in val_loader:
                batch_y = batch_y.to(config.device, non_blocking=True)
                batch_size = batch_y.shape[0]
                
                if batch_y.shape[1] != 14:
                    continue
                
                human_pose = batch_y[:, :7]
                target_angles = batch_y[:, 7:]
                
                human_pos = human_pose[:, :3]
                human_ori = human_pose[:, 3:7]
                
                pred_angles, info = model(human_pos, human_ori)
                
                ik_loss = mse_criterion(pred_angles, target_angles)
                
                # 位置误差
                target_elbow, target_end = fk.compute_positions(target_angles)
                pred_elbow, pred_end = fk.compute_positions(pred_angles)
                
                pos_loss, _ = pos_criterion(
                    pred_elbow, pred_end,
                    target_elbow, target_end
                )
                
                # 统计位置误差（毫米）
                elbow_error = (pred_elbow - target_elbow).norm(dim=1).mean()
                end_error = (pred_end - target_end).norm(dim=1).mean()
                
                total_loss = ik_loss + 0.5 * pos_loss
                
                val_loss += total_loss.item() * batch_size
                val_ik_loss += ik_loss.item() * batch_size
                val_pos_loss += pos_loss.item() * batch_size
                val_elbow_error += elbow_error.item() * batch_size
                val_end_error += end_error.item() * batch_size
                
                all_val_preds.append(pred_angles.cpu())
                all_val_targets.append(target_angles.cpu())
        
        val_loss /= len(val_loader.dataset)
        val_ik_loss /= len(val_loader.dataset)
        val_pos_loss /= len(val_loader.dataset)
        val_elbow_error /= len(val_loader.dataset)
        val_end_error /= len(val_loader.dataset)
        
        # 验证R²
        all_val_preds = torch.cat(all_val_preds, dim=0)
        all_val_targets = torch.cat(all_val_targets, dim=0)
        val_r2 = calculate_r2(all_val_preds, all_val_targets)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 日志
        ik_rmse_deg = np.rad2deg(np.sqrt(val_ik_loss))
        
        logging.info(
            f"{epoch:>6} | {train_loss:>11.6f} | {val_loss:>11.6f} | "
            f"{val_elbow_error*1000:>10.2f} | {val_end_error*1000:>10.2f} | "
            f"{ik_rmse_deg:>11.2f} | {val_r2:>8.4f} | {current_lr:>9.6f}"
        )
        
        # 保存最优
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_r2': val_r2,
                'config': config.__dict__
            }, checkpoint_path)
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= config.early_stopping_patience:
            logging.info(f"\n>>> 早停触发 (patience={config.early_stopping_patience})")
            break
    
    logging.info("=" * 90)
    logging.info("训练完成!")
    logging.info(f"最优验证损失: {best_val_loss:.6f} (epoch {best_epoch})")
    logging.info(f"模型保存至: {checkpoint_path}")


if __name__ == "__main__":
    train()
