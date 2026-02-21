"""
修复版纯位姿IK推理 - 解决"手臂不动"问题

与之前版本的关键区别：
1. 必须用机器人实际姿态初始化
2. 必须用机器人实际反馈更新历史
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from causal_ik_model_pieper2 import PieperCausalIK


class FixedPoseOnlyIKPredictor:
    """
    修复版纯位姿IK预测器
    
    使用方法（3步）：
    
    1. 创建预测器
       predictor = FixedPoseOnlyIKPredictor("model.pth")
    
    2. 用机器人实际姿态初始化（关键！）
       predictor.init_with_actual(robot.get_joint_angles())
    
    3. 循环中更新时用实际反馈（关键！）
       for target_pose in poses:
           angles = predictor.predict(target_pose[:3], target_pose[3:7])
           robot.move(angles)
           predictor.update_with_actual(robot.get_joint_angles())  # 用实际更新
    """
    
    def __init__(self, model_path, num_frames=10, device='cuda'):
        self.num_frames = num_frames
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = PieperCausalIK(num_joints=7, num_frames=num_frames).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.history = None
        self.initialized = False
        
    def init_with_actual(self, current_joint_angles):
        """
        用机器人当前实际姿态初始化历史
        
        ⚠️ 这是解决"手臂不动"的关键！必须在第一次预测前调用！
        
        Args:
            current_joint_angles: [7] 机器人当前的7个关节角度
        """
        if isinstance(current_joint_angles, np.ndarray):
            current_joint_angles = torch.from_numpy(current_joint_angles).float()
        
        current_joint_angles = current_joint_angles.to(self.device)
        
        # 用当前姿态重复填充历史窗口
        self.history = current_joint_angles.unsqueeze(0).repeat(self.num_frames, 1)
        self.initialized = True
        
        print(f"✓ 历史已用实际姿态初始化: {current_joint_angles.cpu().numpy().round(3)}")
        
    def update_with_actual(self, actual_joint_angles):
        """
        用机器人实际反馈更新历史
        
        ⚠️ 这是保持同步的关键！不要省略！
        """
        if not self.initialized:
            raise RuntimeError("请先调用 init_with_actual() 初始化")
        
        if isinstance(actual_joint_angles, np.ndarray):
            actual_joint_angles = torch.from_numpy(actual_joint_angles).float()
        actual_joint_angles = actual_joint_angles.to(self.device)
        
        # 滑动窗口
        self.history = torch.cat([
            self.history[1:],
            actual_joint_angles.unsqueeze(0)
        ], dim=0)
        
    def predict(self, target_position, target_orientation=None):
        """
        预测关节角度
        
        Args:
            target_position: [3] 目标位置 (x, y, z)
            target_orientation: [4] 目标姿态四元数 (可选)
        
        Returns:
            angles: [7] 预测的关节角度
        """
        if not self.initialized:
            raise RuntimeError("请先调用 init_with_actual() 初始化！")
        
        # 处理numpy输入
        if isinstance(target_position, np.ndarray):
            target_position = torch.from_numpy(target_position).float()
        if target_orientation is not None and isinstance(target_orientation, np.ndarray):
            target_orientation = torch.from_numpy(target_orientation).float()
        
        # 加batch维度
        if target_position.dim() == 1:
            target_position = target_position.unsqueeze(0)
        if target_orientation is not None and target_orientation.dim() == 1:
            target_orientation = target_orientation.unsqueeze(0)
        
        target_position = target_position.to(self.device)
        if target_orientation is not None:
            target_orientation = target_orientation.to(self.device)
        
        # 推理
        with torch.no_grad():
            pred_angles, _ = self.model(
                self.history.unsqueeze(0),
                target_position,
                target_orientation
            )
        
        return pred_angles[0].cpu().numpy()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("修复版纯位姿IK推理")
    print("=" * 60)
    
    print("""
    # ========== 你的代码只需要改这几行 ==========
    
    # 原来的代码（可能导致手臂不动）：
    
        predictor = PoseOnlyIKPredictor("model.pth")  # 历史初始化为零
        for pose in target_poses:
            angles = predictor.predict(pose)  # 预测与当前状态不符
            robot.move(angles)
            predictor.update_history(angles)  # 用预测值更新，累积误差
    
    # 修复后的代码：
    
        from inference_simple_fixed import FixedPoseOnlyIKPredictor
        
        predictor = FixedPoseOnlyIKPredictor("model.pth")
        
        # 关键1：用实际姿态初始化！
        predictor.init_with_actual(robot.get_joint_angles())
        
        for pose in target_poses:
            angles = predictor.predict(pose[:3], pose[3:7])
            robot.move(angles)
            
            # 关键2：用实际反馈更新！
            predictor.update_with_actual(robot.get_joint_angles())
    
    # ============================================
    """)
    
    print("关键修改点：")
    print("  1. 初始化：predictor.init_with_actual(robot.get_joint_angles())")
    print("  2. 更新：predictor.update_with_actual(robot.get_joint_angles())")
    print("\n这两个调用确保历史缓冲区与机器人实际状态同步！")
