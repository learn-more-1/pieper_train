"""
精确的FK模型 - 使用URDF提取的参数

连杆长度（米）:
Link 0 (shoulder_pitch): 0.2673 m = 267.3 mm
Link 1 (shoulder_roll):  0.0404 m = 40.4 mm
Link 2 (shoulder_yaw):   0.1034 m = 103.4 mm
Link 3 (elbow):          0.0821 m = 82.1 mm
Link 4 (wrist_roll):     0.1005 m = 100.5 mm
Link 5 (wrist_pitch):    0.0380 m = 38.0 mm
Link 6 (wrist_yaw):      0.0460 m = 46.0 mm

关节旋转轴:
J0: Y轴 (shoulder_pitch)
J1: X轴 (shoulder_roll)
J2: Z轴 (shoulder_yaw)
J3: Y轴 (elbow)
J4: X轴 (wrist_roll/forearm)
J5: Y轴 (wrist_pitch)
J6: Z轴 (wrist_yaw)
"""

import torch
import torch.nn as nn


class FKWithElbowURDF:
    """
    精确的FK模型 - 基于URDF参数
    """
    
    def __init__(self):
        """初始化精确的FK参数（从URDF提取）"""
        
        # 关节旋转轴 (0=X, 1=Y, 2=Z)
        self.joint_axes = [1, 0, 2, 1, 0, 1, 2]
        
        # 精确的连杆长度（单位：米）- 从URDF提取
        self.link_lengths = [
            0.2673,   # Link 0: shoulder_pitch
            0.0404,   # Link 1: shoulder_roll
            0.1034,   # Link 2: shoulder_yaw
            0.0821,   # Link 3: elbow
            0.1005,   # Link 4: wrist_roll (forearm)
            0.0380,   # Link 5: wrist_pitch
            0.0460,   # Link 6: wrist_yaw
        ]
        
        # 肘部位置索引（在J3之后）
        self.elbow_joint_idx = 4  # elbow在J3之后，即第4个关节后
        
    @staticmethod
    def rot_x(theta):
        """绕X轴旋转"""
        zeros = torch.zeros_like(theta)
        ones = torch.ones_like(theta)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        
        return torch.stack([
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cos, -sin], dim=-1),
            torch.stack([zeros, sin, cos], dim=-1)
        ], dim=-2)  # [batch, 3, 3]
    
    @staticmethod
    def rot_y(theta):
        """绕Y轴旋转"""
        zeros = torch.zeros_like(theta)
        ones = torch.ones_like(theta)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        
        return torch.stack([
            torch.stack([cos, zeros, sin], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sin, zeros, cos], dim=-1)
        ], dim=-2)
    
    @staticmethod
    def rot_z(theta):
        """绕Z轴旋转"""
        zeros = torch.zeros_like(theta)
        ones = torch.ones_like(theta)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        
        return torch.stack([
            torch.stack([cos, -sin, zeros], dim=-1),
            torch.stack([sin, cos, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1)
        ], dim=-2)
    
    def forward(self, joint_angles):
        """计算末端位置"""
        return self.compute_positions(joint_angles, return_elbow=False)
    
    def compute_positions(self, joint_angles, return_elbow=True):
        """
        计算肘部和末端位置
        
        Args:
            joint_angles: [batch, 7] 关节角度（弧度）
            return_elbow: 是否返回肘部位置
            
        Returns:
            if return_elbow: (elbow_pos, end_pos) 都是 [batch, 3]
            else: end_pos [batch, 3]
        """
        batch_size = joint_angles.shape[0]
        device = joint_angles.device
        
        # 初始化变换矩阵和位置
        R = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        t = torch.zeros(batch_size, 3, device=device)
        
        elbow_pos = None
        
        # 逐关节变换
        for i in range(7):
            angle = joint_angles[:, i]
            
            # 根据轴类型选择旋转矩阵
            if self.joint_axes[i] == 0:  # X轴
                R_joint = self.rot_x(angle)
            elif self.joint_axes[i] == 1:  # Y轴
                R_joint = self.rot_y(angle)
            else:  # Z轴
                R_joint = self.rot_z(angle)
            
            # 更新旋转
            R = torch.bmm(R, R_joint)
            
            # 平移（沿当前Z方向移动连杆长度）
            if self.link_lengths[i] > 0:
                # 沿Z轴平移
                t_joint = torch.zeros(batch_size, 3, device=device)
                t_joint[:, 2] = self.link_lengths[i]
                # 转换到世界坐标系
                t = t + torch.bmm(R, t_joint.unsqueeze(-1)).squeeze(-1)
            
            # 记录肘部位置（在J3之后，即i==3时）
            if i == 3 and return_elbow:
                elbow_pos = t.clone()
        
        end_pos = t
        
        if return_elbow:
            return elbow_pos, end_pos
        return end_pos


class MultiPositionLoss(nn.Module):
    """多位置损失：同时约束肘部和末端"""
    
    def __init__(self, elbow_weight=0.3, end_weight=0.7):
        super().__init__()
        self.elbow_weight = elbow_weight
        self.end_weight = end_weight
        self.mse = nn.MSELoss()
    
    def forward(self, pred_elbow, pred_end, target_elbow, target_end):
        """
        Args:
            pred_elbow: [batch, 3] 预测肘部位置
            pred_end: [batch, 3] 预测末端位置
            target_elbow: [batch, 3] 目标肘部位置
            target_end: [batch, 3] 目标末端位置
        """
        elbow_loss = self.mse(pred_elbow, target_elbow)
        end_loss = self.mse(pred_end, target_end)
        
        total_loss = self.elbow_weight * elbow_loss + self.end_weight * end_loss
        
        return total_loss, {
            'elbow_loss': elbow_loss.item(),
            'end_loss': end_loss.item(),
            'total': total_loss.item()
        }


# 测试
def test_fk():
    """测试精确FK"""
    print("=" * 70)
    print("测试精确FK（URDF参数）")
    print("=" * 70)
    
    fk = FKWithElbowURDF()
    
    # 零姿态测试
    zero_angles = torch.zeros(1, 7).cuda()
    elbow_pos, end_pos = fk.compute_positions(zero_angles)
    
    print(f"\n零姿态（所有关节0度）:")
    print(f"  肘部位置: [{elbow_pos[0,0].item():.4f}, {elbow_pos[0,1].item():.4f}, {elbow_pos[0,2].item():.4f}] m")
    print(f"  末端位置: [{end_pos[0,0].item():.4f}, {end_pos[0,1].item():.4f}, {end_pos[0,2].item():.4f}] m")
    
    # 伸展姿态
    extend_angles = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).cuda()
    elbow_pos, end_pos = fk.compute_positions(extend_angles)
    
    print(f"\n期望的肘部位置（大致）:")
    # 累加连杆0-2: 267 + 40 + 103 ≈ 410mm (沿Z)
    print(f"  ~0.41 m (沿Z方向)")
    print(f"\n实际的肘部Z位置: {elbow_pos[0,2].item():.4f} m")
    
    # 测试损失
    print(f"\n测试损失函数...")
    criterion = MultiPositionLoss(elbow_weight=0.3, end_weight=0.7)
    
    target_elbow = elbow_pos + 0.01
    target_end = end_pos + 0.02
    
    loss, loss_dict = criterion(elbow_pos, end_pos, target_elbow, target_end)
    
    print(f"  总损失: {loss_dict['total']:.6f}")
    print(f"    - 肘部: {loss_dict['elbow_loss']:.6f}")
    print(f"    - 末端: {loss_dict['end_loss']:.6f}")
    
    print("\n✓ 测试完成")


if __name__ == "__main__":
    test_fk()
