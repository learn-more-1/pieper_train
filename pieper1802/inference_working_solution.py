"""
可用的纯位姿IK推理方案

问题：模型过度依赖历史，零历史导致零输出
解决：用目标位姿生成有意义的引导历史

核心思想：
不是"修复模型"，而是"适应模型的特性"
- 模型需要看到"已经在那里的历史"才会输出对应角度
- 那我们就用启发式生成一个"假装已经在那里"的历史
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from causal_ik_model_pieper2 import PieperCausalIK


class WorkingPoseIKPredictor:
    """
    实际可用的纯位姿IK预测器
    
    关键改进：
    1. 用启发式从目标位姿估计一个近似关节角度
    2. 用这个近似角度填充历史
    3. 模型会输出接近这个近似角度的精细结果
    
    启发式估计（简化版）：
    - 根据目标位置反推大致的shoulder和elbow角度
    - 不需要精确，只要让历史"有意义"即可
    """
    
    def __init__(self, model_path, num_frames=10, device='cuda'):
        self.num_frames = num_frames
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = PieperCausalIK(
            num_joints=7,
            num_frames=num_frames,
            hidden_dim=256,
            num_layers=2
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("✓ 预测器初始化完成")
        print("  使用启发式引导历史方案")
        
    def _estimate_angles_from_pose(self, position, orientation=None):
        """
        从目标位姿启发式估计关节角度
        
        这是一个简化版的逆运动学估计，不需要精确，
        只要产生一个"合理"的初始猜测即可。
        """
        x, y, z = position
        
        # 简化7轴臂的启发式估计
        # 假设： shoulder控制大致方向，elbow控制距离
        
        # 距离
        distance = np.sqrt(x**2 + y**2 + z**2)
        
        # 启发式映射（针对典型7轴臂，需要根据实际机器人调整）
        estimated = np.zeros(7)
        
        # shoulder 1: 水平方向
        estimated[0] = np.arctan2(y, x)
        
        # shoulder 2: 垂直方向（简化）
        horizontal_dist = np.sqrt(x**2 + y**2)
        estimated[1] = -np.arctan2(z - 0.3, horizontal_dist)  # 假设base高度0.3
        
        # shoulder 3: 保持为零或根据姿态调整
        estimated[2] = 0.0
        
        # elbow: 根据距离粗略估计
        # 假设前臂长度约0.3m，上臂长度约0.3m
        # 简化：距离越远，elbow角度越小（伸直）
        if distance > 0.5:
            estimated[3] = 0.2  # 较直
        elif distance > 0.4:
            estimated[3] = 0.5  # 中等
        else:
            estimated[3] = 0.8  # 弯曲
        
        # forearm
        estimated[4] = 0.0
        
        # wrist: 根据orientation调整（简化）
        if orientation is not None:
            # 简化：根据四元数估计wrist角度
            estimated[5] = orientation[0] * 0.5  # 粗略映射
            estimated[6] = orientation[1] * 0.5
        else:
            estimated[5] = 0.0
            estimated[6] = 0.0
        
        return estimated
    
    def predict(self, target_position, target_orientation=None, 
                num_refinement=5):
        """
        预测关节角度
        
        Args:
            target_position: [3] 目标位置 (x, y, z)
            target_orientation: [4] 目标姿态四元数（可选）
            num_refinement: 精化迭代次数
            
        Returns:
            angles: [7] 预测的关节角度
        """
        # 处理输入
        if isinstance(target_position, np.ndarray):
            target_position = target_position.astype(np.float32)
        else:
            target_position = np.array(target_position, dtype=np.float32)
        
        if target_orientation is not None:
            if isinstance(target_orientation, np.ndarray):
                target_orientation = target_orientation.astype(np.float32)
            else:
                target_orientation = np.array(target_orientation, dtype=np.float32)
        
        # 步骤1：启发式估计初始角度
        estimated_angles = self._estimate_angles_from_pose(
            target_position, target_orientation
        )
        
        # 步骤2：用估计角度构建引导历史
        estimated_tensor = torch.from_numpy(estimated_angles).float().to(self.device)
        guided_history = estimated_tensor.unsqueeze(0).repeat(self.num_frames, 1)
        
        # 步骤3：迭代精化
        target_pos_tensor = torch.from_numpy(target_position).float().unsqueeze(0).to(self.device)
        target_ori_tensor = None
        if target_orientation is not None:
            target_ori_tensor = torch.from_numpy(target_orientation).float().unsqueeze(0).to(self.device)
        
        history = guided_history
        for _ in range(num_refinement):
            with torch.no_grad():
                pred, _ = self.model(
                    history.unsqueeze(0),
                    target_pos_tensor,
                    target_ori_tensor
                )
            # 用预测结果更新历史（滑动窗口）
            history = torch.cat([history[1:], pred], dim=0)
        
        return pred[0].cpu().numpy()
    
    def predict_batch(self, target_positions, target_orientations=None,
                      num_refinement=5):
        """
        批量预测
        """
        results = []
        for i in range(len(target_positions)):
            ori = target_orientations[i] if target_orientations is not None else None
            angles = self.predict(target_positions[i], ori, num_refinement)
            results.append(angles)
        return np.array(results)


class SmoothPoseTracker:
    """
    平滑的位姿跟踪器
    
    结合引导历史和时序平滑
    """
    
    def __init__(self, model_path, smooth_factor=0.3):
        self.predictor = WorkingPoseIKPredictor(model_path)
        self.smooth_factor = smooth_factor
        self.prev_angles = None
        
    def update(self, target_position, target_orientation=None):
        """
        更新并获取平滑的关节角度
        """
        # 获取预测
        angles = self.predictor.predict(target_position, target_orientation)
        
        # 平滑处理
        if self.prev_angles is not None:
            smoothed = self.prev_angles * self.smooth_factor + \
                      angles * (1 - self.smooth_factor)
        else:
            smoothed = angles
        
        self.prev_angles = smoothed
        return smoothed
    
    def reset(self):
        """重置平滑状态"""
        self.prev_angles = None


# ==================== 测试 ====================

def test_working_predictor():
    """测试可用的预测器"""
    print("=" * 70)
    print("测试可用的纯位姿预测器")
    print("=" * 70)
    
    model_path = "/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth"
    predictor = WorkingPoseIKPredictor(model_path)
    
    # 测试：不同目标位姿
    print("\n测试：不同目标位置")
    print("-" * 50)
    
    test_positions = [
        np.array([0.4, 0.1, 0.3]),
        np.array([0.5, 0.2, 0.4]),
        np.array([0.6, 0.3, 0.5]),
    ]
    
    for i, pos in enumerate(test_positions):
        angles = predictor.predict(pos, num_refinement=5)
        print(f"目标 {pos.round(2)} -> 角度 {angles[:4].round(3)}...")
    
    # 验证：目标变化，输出变化
    print("\n验证：目标位姿驱动输出")
    angles1 = predictor.predict(np.array([0.4, 0.2, 0.4]))
    angles2 = predictor.predict(np.array([0.6, 0.2, 0.4]))
    
    diff = np.linalg.norm(angles2 - angles1)
    print(f"目标X变化0.2 -> 角度变化: {diff:.4f}")
    
    if diff > 0.1:
        print("✓ 目标变化驱动输出变化，正常工作")
    else:
        print("✗ 输出变化太小，可能需要调整启发式")
    
    return predictor


def test_smooth_tracker():
    """测试平滑跟踪"""
    print("\n" + "=" * 70)
    print("测试平滑位姿跟踪")
    print("=" * 70)
    
    model_path = "/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth"
    tracker = SmoothPoseTracker(model_path, smooth_factor=0.3)
    
    # 模拟连续跟踪
    print("\n模拟人手连续移动：")
    print("-" * 50)
    
    positions = []
    for t in np.linspace(0, 1, 10):
        pos = np.array([0.4 + 0.2*t, 0.2, 0.4])
        positions.append(pos)
    
    all_angles = []
    for i, pos in enumerate(positions):
        angles = tracker.update(pos)
        all_angles.append(angles)
        print(f"  点{i+1} {pos.round(2)} -> {angles[:3].round(3)}...")
    
    all_angles = np.array(all_angles)
    total_change = np.linalg.norm(all_angles[-1] - all_angles[0])
    print(f"\n总角度变化: {total_change:.4f}")


def demo_real_usage():
    """实际使用演示"""
    print("\n" + "=" * 70)
    print("实际使用示例")
    print("=" * 70)
    
    print("""
    # 场景：VR遥操作机器人
    
    from inference_working_solution import WorkingPoseIKPredictor
    
    # 1. 初始化预测器
    predictor = WorkingPoseIKPredictor("model.pth")
    
    # 2. 实时循环
    while True:
        # 获取VR手柄位姿
        vr_pose = vr_controller.get_pose()  # [x, y, z, qx, qy, qz, qw]
        
        # IK求解（内部用启发式生成引导历史）
        joint_angles = predictor.predict(
            vr_pose[:3],      # 位置
            vr_pose[3:7]      # 姿态
        )
        
        # 发送给机器人
        robot.move(joint_angles)
    
    # 如果需要平滑跟踪，用SmoothPoseTracker
    from inference_working_solution import SmoothPoseTracker
    
    tracker = SmoothPoseTracker("model.pth", smooth_factor=0.3)
    
    while True:
        vr_pose = vr_controller.get_pose()
        joint_angles = tracker.update(vr_pose[:3], vr_pose[3:7])
        robot.move(joint_angles)
    """)


if __name__ == "__main__":
    # 运行测试
    predictor = test_working_predictor()
    test_smooth_tracker()
    demo_real_usage()
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
这个方案的核心：
1. 不试图改变模型（它已经训练好了）
2. 而是适应模型的特性（需要有意义的历史）
3. 用启发式从目标位姿生成有意义的历史
4. 模型在这个基础上精化结果

优点：
- 立即生效，无需重新训练
- 响应目标位姿变化
- 关节耦合由模型内部GNN处理

缺点：
- 启发式估计需要针对具体机器人调整
- 不如纯ML方案优雅
    """)
