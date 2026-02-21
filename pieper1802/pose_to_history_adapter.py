"""
位姿到历史特征的适配器训练

核心思想：
1. 冻结已训练好的IK模型
2. 训练一个轻量级适配器网络，将目标位姿映射到历史特征空间
3. 这样推理时只需要位姿输入，不需要历史角度

优势：
- 保留原模型学到的关节耦合关系
- 推理时真正的纯位姿输入
- 训练成本低（只需要小网络）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from causal_ik_model_pieper2 import PieperCausalIK
from dataset_generalized import create_windowed_dataloaders


class PoseToHistoryAdapter(nn.Module):
    """
    将目标位姿映射到历史特征的轻量级适配器
    
    输入: 目标位姿 (位置 + 四元数)
    输出: 模拟的历史特征 [num_frames, num_joints]
    """
    
    def __init__(self, num_frames=10, num_joints=7, hidden_dim=128):
        super().__init__()
        self.num_frames = num_frames
        self.num_joints = num_joints
        
        # 位姿编码器
        self.pose_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim),  # 3+4=7
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 生成多帧历史（模拟自然的关节变化）
        self.history_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, num_frames * num_joints)
        )
        
        # 帧间平滑约束（使生成的历史更自然）
        self.smoothness_factor = 0.1
        
    def forward(self, target_position, target_orientation):
        """
        Args:
            target_position: [batch, 3]
            target_orientation: [batch, 4]
            
        Returns:
            simulated_history: [batch, num_frames, num_joints]
        """
        batch_size = target_position.shape[0]
        
        # 编码位姿
        pose = torch.cat([target_position, target_orientation], dim=-1)
        pose_feat = self.pose_encoder(pose)
        
        # 生成历史
        history_flat = self.history_generator(pose_feat)
        history = history_flat.view(batch_size, self.num_frames, self.num_joints)
        
        # 添加帧间平滑（使历史变化更自然）
        # 使用cumsum模拟连续的关节运动
        history_diff = torch.tanh(history[:, 1:, :] - history[:, :-1, :])
        history_smooth = torch.cat([
            history[:, 0:1, :],
            history[:, 0:1, :] + torch.cumsum(history_diff * self.smoothness_factor, dim=1)
        ], dim=1)
        
        return history_smooth


class AdaptedPoseOnlyIK(nn.Module):
    """
    组合模型：适配器 + 冻结的IK模型
    
    推理时只需要位姿输入
    """
    
    def __init__(self, ik_model_path, num_frames=10, num_joints=7):
        super().__init__()
        
        # 加载并冻结IK模型
        self.ik_model = PieperCausalIK(
            num_joints=num_joints,
            num_frames=num_frames,
            hidden_dim=256,
            num_layers=2
        )
        checkpoint = torch.load(ik_model_path, map_location='cpu')
        self.ik_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 冻结IK模型参数
        for param in self.ik_model.parameters():
            param.requires_grad = False
        self.ik_model.eval()
        
        # 可训练的适配器
        self.adapter = PoseToHistoryAdapter(num_frames, num_joints)
        
    def forward(self, target_position, target_orientation):
        """
        纯位姿输入 -> 关节角度输出
        """
        # 适配器生成模拟历史
        simulated_history = self.adapter(target_position, target_orientation)
        
        # IK模型预测（不计算梯度，节省内存）
        with torch.no_grad():
            pred_angles, info = self.ik_model(
                simulated_history,
                target_position,
                target_orientation
            )
        
        return pred_angles, info, simulated_history


def train_adapter():
    """
    训练适配器网络
    
    训练数据：
    - 输入：目标位姿
    - 目标：使用真实历史帧时IK模型的输出
    
    目标：让适配器生成的历史能产生与真实历史相似的预测结果
    """
    
    # 配置
    class Config:
        data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
        ik_model_path = "/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth"
        batch_size = 256
        epochs = 50
        lr = 1e-3
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_frames = 10
        num_joints = 7
    
    config = Config()
    
    print("=" * 60)
    print("训练位姿->历史适配器")
    print("=" * 60)
    
    # 加载数据
    from types import SimpleNamespace
    data_config = SimpleNamespace(
        data_path=config.data_path,
        num_joints=config.num_joints,
        num_frames=config.num_frames,
        batch_size=config.batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    train_loader, val_loader = create_windowed_dataloaders(config.data_path, data_config)
    
    # 创建组合模型
    model = AdaptedPoseOnlyIK(config.ik_model_path, config.num_frames, config.num_joints)
    model = model.to(config.device)
    
    # 只优化适配器参数
    optimizer = optim.Adam(model.adapter.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(config.epochs):
        # 训练
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y, _ in train_loader:
            batch_X = batch_X.to(config.device)
            
            # 提取目标位姿
            if batch_y.shape[1] == 14:
                target_pose = batch_y[:, :7].to(config.device)
                target_angles = batch_y[:, 7:].to(config.device)
            else:
                continue  # 需要位姿信息
            
            target_pos = target_pose[:, :3]
            target_ori = target_pose[:, 3:7]
            
            # 使用真实历史得到"教师"预测
            with torch.no_grad():
                teacher_angles, _ = model.ik_model(batch_X, target_pos, target_ori)
            
            # 适配器预测
            pred_angles, _, _ = model(target_pos, target_ori)
            
            # 损失：匹配教师输出 + 匹配真实角度
            loss = criterion(pred_angles, teacher_angles) + \
                   0.5 * criterion(pred_angles, target_angles)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y, _ in val_loader:
                batch_X = batch_X.to(config.device)
                
                if batch_y.shape[1] == 14:
                    target_pose = batch_y[:, :7].to(config.device)
                    target_angles = batch_y[:, 7:].to(config.device)
                else:
                    continue
                
                target_pos = target_pose[:, :3]
                target_ori = target_pose[:, 3:7]
                
                # 教师预测
                teacher_angles, _ = model.ik_model(batch_X, target_pos, target_ori)
                
                # 适配器预测
                pred_angles, _, _ = model(target_pos, target_ori)
                
                loss = criterion(pred_angles, teacher_angles) + \
                       0.5 * criterion(pred_angles, target_angles)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # 保存最优
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'adapter_state_dict': model.adapter.state_dict(),
                'epoch': epoch,
                'loss': val_loss
            }, "/home/bonuli/Pieper/pieper1802/pose_to_history_adapter.pth")
            print(f"  >>> 保存最优模型")
    
    print(f"\n训练完成！最优验证损失: {best_loss:.6f}")


class PurePoseIKPredictor:
    """
    纯位姿输入的IK预测器（使用训练好的适配器）
    
    真正的纯位姿输入，无需维护历史缓冲区
    """
    
    def __init__(self, ik_model_path, adapter_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载组合模型
        self.model = AdaptedPoseOnlyIK(ik_model_path)
        
        # 加载适配器权重
        checkpoint = torch.load(adapter_path, map_location=self.device)
        self.model.adapter.load_state_dict(checkpoint['adapter_state_dict'])
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ 纯位姿IK预测器加载成功")
        
    def predict(self, target_position, target_orientation=None):
        """
        纯位姿输入预测
        
        Args:
            target_position: [3] 或 [batch, 3]
            target_orientation: [4] 或 [batch, 4]，可选
            
        Returns:
            joint_angles: [7] 或 [batch, 7]
        """
        # 处理输入
        if isinstance(target_position, np.ndarray):
            target_position = torch.from_numpy(target_position).float()
        if target_orientation is not None and isinstance(target_orientation, np.ndarray):
            target_orientation = torch.from_numpy(target_orientation).float()
        
        single_input = (target_position.dim() == 1)
        if single_input:
            target_position = target_position.unsqueeze(0)
            if target_orientation is not None:
                target_orientation = target_orientation.unsqueeze(0)
        
        # 默认姿态
        if target_orientation is None:
            target_orientation = torch.zeros(target_position.shape[0], 4)
            target_orientation[:, 3] = 1.0  # 单位四元数
        
        target_position = target_position.to(self.device)
        target_orientation = target_orientation.to(self.device)
        
        # 推理
        with torch.no_grad():
            pred_angles, _, _ = self.model(target_position, target_orientation)
        
        if single_input:
            return pred_angles[0].cpu().numpy()
        return pred_angles.cpu().numpy()


if __name__ == "__main__":
    print("位姿到历史适配器训练脚本")
    print("\n使用方法:")
    print("1. 训练适配器:")
    print("   python pose_to_history_adapter.py")
    print("\n2. 使用纯位姿预测:")
    print("""
   from pose_to_history_adapter import PurePoseIKPredictor
   
   predictor = PurePoseIKPredictor(
       ik_model_path="path/to/ik_model.pth",
       adapter_path="path/to/adapter.pth"
   )
   
   # 纯位姿输入！
   angles = predictor.predict(position, orientation)
   """)
    
    # 训练
    # train_adapter()
