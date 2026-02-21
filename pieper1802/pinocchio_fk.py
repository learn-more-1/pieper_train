"""
使用Pinocchio的精确FK

安装：pip install pinocchio
"""

import torch
import numpy as np


class PinocchioFK:
    """
    基于Pinocchio的精确正运动学
    """
    
    def __init__(self, urdf_path="/home/bonuli/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"):
        """初始化Pinocchio模型"""
        try:
            import pinocchio as pin
            self.pin = pin
            
            # 加载URDF
            self.model = pin.buildModelFromUrdf(urdf_path)
            self.data = self.model.createData()
            
            # 左臂关节ID
            self.joint_names = [
                'left_shoulder_pitch_joint',
                'left_shoulder_roll_joint', 
                'left_shoulder_yaw_joint',
                'left_elbow_joint',
                'left_wrist_roll_joint',
                'left_wrist_pitch_joint',
                'left_wrist_yaw_joint'
            ]
            
            # 获取关节ID
            self.joint_ids = []
            for name in self.joint_names:
                joint_id = self.model.getJointId(name)
                if joint_id < self.model.njoints:
                    self.joint_ids.append(joint_id)
                else:
                    print(f"⚠ 未找到关节: {name}")
            
            # 肘部和腕部link名称
            self.elbow_link = "left_elbow_link"
            self.wrist_link = "left_wrist_yaw_link"  # 末端
            
            self.elbow_frame_id = self.model.getFrameId(self.elbow_link)
            self.wrist_frame_id = self.model.getFrameId(self.wrist_link)
            
            print(f"✓ Pinocchio模型加载成功")
            print(f"  关节数: {len(self.joint_ids)}")
            print(f"  肘部frame: {self.elbow_link} (ID: {self.elbow_frame_id})")
            print(f"  腕部frame: {self.wrist_link} (ID: {self.wrist_frame_id})")
            
        except ImportError:
            print("❌ 未安装Pinocchio，请运行: pip install pinocchio")
            raise
    
    def forward(self, joint_angles):
        """
        计算末端位置
        
        Args:
            joint_angles: [batch, 7] tensor
            
        Returns:
            positions: [batch, 3] 腕部位置
        """
        return self.compute_positions(joint_angles, return_elbow=False)
    
    def compute_positions(self, joint_angles, return_elbow=True):
        """
        计算肘部和腕部位置
        
        Args:
            joint_angles: [batch, 7] 关节角度
            return_elbow: 是否返回肘部
            
        Returns:
            (elbow_pos, wrist_pos) 或 wrist_pos
        """
        batch_size = joint_angles.shape[0]
        device = joint_angles.device
        
        # 转换为numpy
        if isinstance(joint_angles, torch.Tensor):
            angles_np = joint_angles.cpu().numpy()
        else:
            angles_np = joint_angles
        
        elbow_positions = []
        wrist_positions = []
        
        for i in range(batch_size):
            # 设置关节角度
            q = np.zeros(self.model.nq)
            
            # 填入左臂角度
            for j, joint_id in enumerate(self.joint_ids):
                if j < len(angles_np[i]):
                    # 获取关节的索引
                    joint = self.model.joints[joint_id]
                    idx_q = joint.idx_q
                    if idx_q < self.model.nq:
                        q[idx_q] = angles_np[i][j]
            
            # 正运动学
            self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            
            # 获取肘部位置
            if return_elbow and self.elbow_frame_id < len(self.data.oMf):
                elbow_pos = self.data.oMf[self.elbow_frame_id].translation
                elbow_positions.append(elbow_pos)
            
            # 获取腕部位置
            if self.wrist_frame_id < len(self.data.oMf):
                wrist_pos = self.data.oMf[self.wrist_frame_id].translation
                wrist_positions.append(wrist_pos)
        
        # 转换为tensor
        wrist_tensor = torch.tensor(np.array(wrist_positions), 
                                    dtype=torch.float32, device=device)
        
        if return_elbow:
            elbow_tensor = torch.tensor(np.array(elbow_positions), 
                                        dtype=torch.float32, device=device)
            return elbow_tensor, wrist_tensor
        
        return wrist_tensor


def test_pinocchio_fk():
    """测试Pinocchio FK"""
    print("=" * 70)
    print("测试Pinocchio FK")
    print("=" * 70)
    
    fk = PinocchioFK()
    
    # 测试零姿态
    print("\n测试零姿态...")
    angles = torch.zeros(2, 7).cuda()
    
    elbow_pos, wrist_pos = fk.compute_positions(angles)
    
    print(f"样本1:")
    print(f"  肘部: [{elbow_pos[0,0].item():.4f}, {elbow_pos[0,1].item():.4f}, {elbow_pos[0,2].item():.4f}]")
    print(f"  腕部: [{wrist_pos[0,0].item():.4f}, {wrist_pos[0,1].item():.4f}, {wrist_pos[0,2].item():.4f}]")
    
    # 测试单个关节运动
    print("\n测试J3(Elbow)弯曲...")
    angles = torch.zeros(1, 7).cuda()
    angles[0, 3] = 1.0  # elbow = 1.0 rad
    
    elbow_pos, wrist_pos = fk.compute_positions(angles)
    
    print(f"  肘部: [{elbow_pos[0,0].item():.4f}, {elbow_pos[0,1].item():.4f}, {elbow_pos[0,2].item():.4f}]")
    print(f"  腕部: [{wrist_pos[0,0].item():.4f}, {wrist_pos[0,1].item():.4f}, {wrist_pos[0,2].item():.4f}]")
    print("  (肘部应该基本不变，腕部移动)")
    
    print("\n✓ 测试完成")


if __name__ == "__main__":
    test_pinocchio_fk()
