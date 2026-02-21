"""
GPU优化版显式耦合IK训练

优化策略：
1. 自动混合精度(AMP) - 减少显存占用，加速计算
2. 增大batch size - 提高GPU并行度
3. 优化数据加载 - 更多worker，预读取
4. torch.compile - PyTorch 2.0+ 图编译优化
5. 梯度累积 - 小显存也能大batch
6. 减少CPU-GPU同步
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
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
    handlers=[logging.FileHandler("explicit_coupling_train_optimized.log"), logging.StreamHandler()]
)


class Config:
    # 数据配置
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
    num_joints = 7
    num_frames = 10
    
    # 训练参数 - 优化版
    batch_size = 2048        # 增大batch size（原512）
    gradient_accumulation = 2  # 梯度累积步数，等效batch = 2048*2 = 4096
    epochs = 200
    initial_lr = 1e-3 * 2    # 学习率随batch增大而增大（线性缩放）
    min_lr = 1e-6
    
    # 混合精度
    use_amp = True
    
    # torch.compile优化
    use_compile = True
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 模型配置
    hidden_dim = 512
    use_temporal = False
    
    # 损失权重
    ik_weight = 1.0
    fk_weight = 0.5
    coupling_reg = 0.01
    
    # 数据加载优化
    num_workers = 8          # 增加worker（原4）
    pin_memory = True
    prefetch_factor = 4      # 预读取批次
    persistent_workers = True  # 保持worker进程


def get_gpu_memory():
    """获取GPU显存信息"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    return 0


def train():
    config = Config()
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = 'explicit_coupling_ik_optimized'
    
    logging.info("=" * 70)
    logging.info("训练显式耦合IK模型 (GPU优化版)")
    logging.info("=" * 70)
    logging.info("优化策略:")
    logging.info(f"  1. 混合精度(AMP): {config.use_amp}")
    logging.info(f"  2. Batch Size: {config.batch_size} x {config.gradient_accumulation} = {config.batch_size * config.gradient_accumulation}")
    logging.info(f"  3. torch.compile: {config.use_compile}")
    logging.info(f"  4. DataLoader workers: {config.num_workers}")
    logging.info(f"  5. GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logging.info(f"  6. GPU Memory: {get_gpu_memory():.1f} GB")
    
    # 检查CUDA版本是否支持compile
    if config.use_compile:
        if hasattr(torch, 'compile'):
            logging.info("  ✓ PyTorch支持torch.compile")
        else:
            logging.info("  ✗ PyTorch版本不支持torch.compile，已禁用")
            config.use_compile = False
    
    # 加载数据集
    logging.info("\n加载数据集...")
    # 创建临时配置用于数据加载
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
        num_joints=config.num_joints,
        num_frames=config.num_frames,
        hidden_dim=config.hidden_dim,
        use_temporal=config.use_temporal
    )
    model = model.to(config.device)
    
    # torch.compile优化
    if config.use_compile:
        logging.info("应用torch.compile优化...")
        try:
            model = torch.compile(model, mode='default')
            logging.info("✓ torch.compile应用成功")
        except Exception as e:
            logging.info(f"✗ torch.compile失败: {e}")
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"模型参数量: {total_params:.2f}M")
    
    # 加载GPU FK模型
    logging.info("\n加载GPU FK模型...")
    gpu_fk = SimpleGPUFK()
    
    # 将FK模型也移到GPU并编译
    if config.use_compile:
        try:
            # 注意：FK可能是numpy实现，需要适配
            pass
        except:
            pass
    
    # 损失函数
    mse_criterion = nn.MSELoss()
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.initial_lr, weight_decay=1e-4)
    
    # 学习率调度器 - 使用OneCycleLR更快收敛
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.initial_lr,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader) // config.gradient_accumulation,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    # 混合精度scaler
    scaler = GradScaler('cuda') if config.use_amp else None
    
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
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        logging.info(f"从第{start_epoch} epoch继续")
    
    # 训练循环
    logging.info(f"\n开始训练（{config.epochs} epochs）...")
    logging.info("-" * 70)
    
    # 记录训练时间
    epoch_times = []
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        
        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        train_ik_loss = 0.0
        train_fk_loss = 0.0
        
        optimizer.zero_grad()  # 梯度累积需要在循环外清零
        
        for batch_idx, (batch_X, batch_y, _) in enumerate(train_loader):
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
                target_orientation = None  # 如果需要姿态验证，可以在这里添加
            else:
                target_angles = batch_y
                target_position = gpu_fk.forward(batch_y)
                human_position = target_position
                human_orientation = None
            
            # 混合精度前向
            with autocast(device_type='cuda', enabled=config.use_amp):
                # 模型输入：人位姿，输出：预测角度
                pred_angles, info = model(
                    human_position,
                    human_orientation,
                    history_frames=None
                )
                
                # IK loss：预测角度 vs 机器人目标角度
                ik_loss = mse_criterion(pred_angles, target_angles)
                
                # FK验证：用预测角度计算预测位姿
                pred_position = gpu_fk.forward(pred_angles)
                
                # FK loss：预测位姿 vs 机器人目标位姿
                fk_loss = mse_criterion(pred_position, target_position)
                
                # 耦合正则化
                coupling_loss = torch.tensor(0.0, device=config.device)
                for key, value in info['coupling_info'].items():
                    coupling_loss += ((value - 0.5) ** 2)
                
                loss = (
                    config.ik_weight * ik_loss +
                    config.fk_weight * fk_loss +
                    config.coupling_reg * coupling_loss
                ) / config.gradient_accumulation  # 梯度累积需要除以步数
            
            # 混合精度反向
            if config.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积更新
            if (batch_idx + 1) % config.gradient_accumulation == 0:
                if config.use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
            
            # 记录损失（乘以gradient_accumulation恢复真实值）
            train_loss += loss.item() * config.gradient_accumulation * batch_size
            train_ik_loss += ik_loss.item() * batch_size
            train_fk_loss += fk_loss.item() * batch_size
        
        train_loss /= len(train_loader.dataset)
        train_ik_loss /= len(train_loader.dataset)
        train_fk_loss /= len(train_loader.dataset)
        
        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0.0
        val_ik_loss = 0.0
        val_fk_loss = 0.0
        
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
                
                # 验证时也可以用AMP加速
                with autocast(device_type='cuda', enabled=config.use_amp):
                    # 模型输入：人位姿
                    pred_angles, info = model(human_position, human_orientation)
                    
                    # FK验证：用预测角度计算预测位姿
                    pred_position = gpu_fk.forward(pred_angles)
                    
                    ik_loss = mse_criterion(pred_angles, target_angles)
                    fk_loss = mse_criterion(pred_position, target_position)
                    total_loss = ik_loss + 0.5 * fk_loss
                
                val_loss += total_loss.item() * batch_size
                val_ik_loss += ik_loss.item() * batch_size
                val_fk_loss += fk_loss.item() * batch_size
                
                for key in coupling_stats:
                    full_key = key + '_coupling'
                    if full_key in info['coupling_info']:
                        coupling_stats[key].append(info['coupling_info'][full_key].item())
        
        val_loss /= len(val_loader.dataset)
        val_ik_loss /= len(val_loader.dataset)
        val_fk_loss /= len(val_loader.dataset)
        
        avg_coupling = {k: np.mean(v) if len(v) > 0 else 0 for k, v in coupling_stats.items()}
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_time = np.mean(epoch_times[-10:]) if len(epoch_times) >= 10 else epoch_time
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 获取GPU利用率（使用nvidia-smi）
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                util, mem = result.stdout.strip().split(', ')
                gpu_util = int(util)
                gpu_mem = float(mem) / 1024  # MB -> GB
            else:
                gpu_util = 0
                gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        except:
            gpu_util = 0
            gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        coupling_str = " | ".join([f"{k}: {v:.3f}" for k, v in avg_coupling.items()])
        
        logging.info(
            f"Epoch [{epoch}/{config.epochs}] | "
            f"Val Loss: {val_loss:.6f} | "
            f"IK: {val_ik_loss:.6f} | FK: {val_fk_loss:.6f} | "
            f"Coupling: {coupling_str} | "
            f"Time: {epoch_time:.1f}s | "
            f"GPU: {gpu_util}% {gpu_mem:.1f}GB | "
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
    logging.info(f"平均epoch时间: {np.mean(epoch_times):.1f}s")
    logging.info(f"模型保存至: {checkpoint_path}")


if __name__ == "__main__":
    train()
