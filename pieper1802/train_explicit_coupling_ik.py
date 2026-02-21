"""
训练显式耦合IK模型

特点：
1. 纯位姿输入（无需历史）
2. 显式建模关节耦合关系
3. 可解释的耦合强度
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
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from explicit_coupling_ik import ExplicitCouplingIK
from dataset_generalized import create_windowed_dataloaders
from gpu_fk_wrapper import SimpleGPUFK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("explicit_coupling_train.log"), logging.StreamHandler()]
)


class Config:
    # 数据配置
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
    num_joints = 7
    num_frames = 10  # 可选，用于平滑参考
    
    # 训练参数
    batch_size = 512
    epochs = 200
    initial_lr = 1e-3
    min_lr = 1e-6
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 模型配置
    hidden_dim = 256
    use_temporal = False  # 纯位姿模式
    
    # 损失权重
    ik_weight = 1.0      # 关节角度损失
    fk_weight = 0.5      # FK位置损失
    coupling_reg = 0.01  # 耦合正则化（鼓励合理的耦合强度）
    
    # 数据加载
    num_workers = 4
    pin_memory = True


def train():
    config = Config()
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = 'explicit_coupling_ik'
    
    logging.info("=" * 70)
    logging.info("训练显式耦合IK模型")
    logging.info("=" * 70)
    logging.info("核心特点:")
    logging.info("  1. 纯位姿输入，无需历史")
    logging.info("  2. 显式耦合图建模关节关系")
    logging.info("  3. 可解释的耦合强度")
    
    # 加载数据集
    logging.info("\n加载数据集...")
    train_loader, val_loader = create_windowed_dataloaders(config.data_path, config)
    
    # 创建模型
    logging.info(f"\n创建模型: {model_name}")
    model = ExplicitCouplingIK(
        num_joints=config.num_joints,
        num_frames=config.num_frames,
        hidden_dim=config.hidden_dim,
        use_temporal=config.use_temporal
    )
    model = model.to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"模型参数量: {total_params:.2f}M")
    
    # 加载GPU FK模型
    logging.info("\n加载GPU FK模型...")
    gpu_fk = SimpleGPUFK()
    
    # 损失函数
    mse_criterion = nn.MSELoss()
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.initial_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=config.min_lr
    )
    
    # 断点路径
    checkpoint_path = f"/home/bonuli/Pieper/pieper1802/{model_name}.pth"
    
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
    logging.info("-" * 70)
    
    for epoch in range(start_epoch, config.epochs):
        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        train_ik_loss = 0.0
        train_fk_loss = 0.0
        train_coupling_reg = 0.0
        
        for batch_X, batch_y, _ in train_loader:
            # 我们只关心目标位姿和角度，不需要历史X
            batch_y = batch_y.to(config.device, non_blocking=True)
            
            batch_size = batch_y.shape[0]
            
            # 提取目标
            if batch_y.shape[1] == 14:
                # 前7维：人位姿（模型输入）
                human_pose = batch_y[:, :7]
                human_position = human_pose[:, :3]
                human_orientation = human_pose[:, 3:7]
                
                # 后7维：机器人目标角度
                target_angles = batch_y[:, 7:]
                
                # 用机器人目标角度计算机器人目标位姿（用于FK验证）
                with torch.no_grad():
                    target_position = gpu_fk.forward(target_angles)
                target_orientation = None
            else:
                target_angles = batch_y
                target_position = gpu_fk.forward(batch_y)
                human_position = target_position
                human_orientation = None
            
            optimizer.zero_grad()
            
            # 前向传播（纯位姿输入！）
            pred_angles, info = model(
                human_position,
                human_orientation,
                history_frames=None  # 纯位姿，不用历史
            )
            
            # 计算损失
            ik_loss = mse_criterion(pred_angles, target_angles)
            
            # FK验证：用预测角度计算预测位姿
            pred_position = gpu_fk.forward(pred_angles)
            
            # FK loss：预测位姿 vs 机器人目标位姿
            fk_loss = mse_criterion(pred_position, target_position)
            
            # 耦合正则化（鼓励耦合强度在合理范围内）
            coupling_loss = torch.tensor(0.0, device=config.device)
            for key, value in info['coupling_info'].items():
                # 鼓励耦合强度接近0.5（既不太弱也不太强）
                coupling_loss += ((value - 0.5) ** 2)
            
            total_loss = (
                config.ik_weight * ik_loss +
                config.fk_weight * fk_loss +
                config.coupling_reg * coupling_loss
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item() * batch_size
            train_ik_loss += ik_loss.item() * batch_size
            train_fk_loss += fk_loss.item() * batch_size
            train_coupling_reg += coupling_loss.item() * batch_size
        
        train_loss /= len(train_loader.dataset)
        train_ik_loss /= len(train_loader.dataset)
        train_fk_loss /= len(train_loader.dataset)
        
        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0.0
        val_ik_loss = 0.0
        val_fk_loss = 0.0
        
        # 统计耦合强度
        coupling_stats = {key: [] for key in ['elbow_shoulder', 'forearm_elbow', 'wrist_forearm']}
        
        with torch.no_grad():
            for batch_X, batch_y, _ in val_loader:
                batch_y = batch_y.to(config.device, non_blocking=True)
                
                batch_size = batch_y.shape[0]
                
                if batch_y.shape[1] == 14:
                    # 前7维：人位姿（模型输入）
                    human_pose = batch_y[:, :7]
                    human_position = human_pose[:, :3]
                    human_orientation = human_pose[:, 3:7]
                    
                    # 后7维：机器人目标角度
                    target_angles = batch_y[:, 7:]
                    
                    # 用机器人目标角度计算机器人目标位姿（用于FK验证）
                    target_position = gpu_fk.forward(target_angles)
                    target_orientation = None
                else:
                    target_angles = batch_y
                    target_position = gpu_fk.forward(batch_y)
                    human_position = target_position
                    human_orientation = None
                
                # 纯位姿输入
                pred_angles, info = model(human_position, human_orientation)
                
                pred_position = gpu_fk.forward(pred_angles)
                
                ik_loss = mse_criterion(pred_angles, target_angles)
                fk_loss = mse_criterion(pred_position, target_position)
                
                total_loss = ik_loss + 0.5 * fk_loss
                
                val_loss += total_loss.item() * batch_size
                val_ik_loss += ik_loss.item() * batch_size
                val_fk_loss += fk_loss.item() * batch_size
                
                # 收集耦合统计
                for key in coupling_stats:
                    full_key = key + '_coupling'
                    if full_key in info['coupling_info']:
                        coupling_stats[key].append(info['coupling_info'][full_key].item())
        
        val_loss /= len(val_loader.dataset)
        val_ik_loss /= len(val_loader.dataset)
        val_fk_loss /= len(val_loader.dataset)
        
        # 计算平均耦合强度
        avg_coupling = {}
        for key, values in coupling_stats.items():
            if len(values) > 0:
                avg_coupling[key] = np.mean(values)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 日志
        coupling_str = " | ".join([f"{k}: {v:.3f}" for k, v in avg_coupling.items()])
        
        logging.info(
            f"Epoch [{epoch}/{config.epochs}] | "
            f"Val Loss: {val_loss:.6f} | "
            f"IK: {val_ik_loss:.6f} | FK: {val_fk_loss:.6f} | "
            f"Coupling: {coupling_str} | "
            f"LR: {current_lr:.6f}"
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
                'config': config.__dict__
            }, checkpoint_path)
            logging.info(f"  >>> 保存最优模型（验证损失：{val_loss:.6f}）")
    
    logging.info("\n训练完成！")
    logging.info(f"最优验证损失: {best_val_loss:.6f}")
    logging.info(f"模型保存至: {checkpoint_path}")


if __name__ == "__main__":
    train()
