"""
纯位姿输入的IK推理脚本

核心思想：
1. 维护一个历史角度缓冲区（滑动窗口）
2. 第一次推理时：用默认/零角度初始化历史
3. 后续推理：用上一轮预测的角度更新历史窗口
4. 模型仍能从历史中学到的耦合关系进行预测

使用方法：
    # 方式1: 单次推理（自动维护历史）
    predictor = PoseOnlyIKPredictor(model_path)
    joint_angles = predictor.predict(target_position, target_orientation)
    
    # 方式2: 序列推理（推荐用于实时控制）
    for pose in target_poses:
        angles = predictor.predict(pose[:3], pose[3:7])
        robot.move(angles)
    
    # 方式3: 批量推理（独立样本）
    angles = predictor.predict_batch(positions, orientations)
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from causal_ik_model_pieper2 import PieperCausalIK


class PoseOnlyIKPredictor:
    """
    仅使用目标位姿进行IK预测的包装器
    
    自动维护历史角度缓冲区，使模型能在推理时利用学到的关节耦合关系
    """
    
    def __init__(self, model_path, num_frames=10, num_joints=7, 
                 device='cuda', default_init='zero'):
        """
        Args:
            model_path: 模型权重路径
            num_frames: 历史帧数
            num_joints: 关节数量
            device: 计算设备
            default_init: 初始历史填充方式 ('zero', 'random', 'neutral')
        """
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.default_init = default_init
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 历史角度缓冲区 [num_frames, num_joints]
        self.history_buffer = self._init_history()
        
        # 标记是否是第一次推理
        self.is_first_prediction = True
        
    def _load_model(self, model_path):
        """加载训练好的模型"""
        model = PieperCausalIK(
            num_joints=self.num_joints,
            num_frames=self.num_frames,
            hidden_dim=256,
            num_layers=2
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 模型加载成功: {model_path}")
        return model
    
    def _init_history(self):
        """初始化历史缓冲区"""
        if self.default_init == 'zero':
            return torch.zeros(self.num_frames, self.num_joints, 
                             dtype=torch.float32, device=self.device)
        elif self.default_init == 'neutral':
            # 中性姿态（可根据实际机器人调整）
            neutral = torch.tensor([0.0, 0.0, 0.0,  # shoulder
                                   0.5,            # elbow
                                   0.0,            # forearm  
                                   0.0, 0.0],      # wrist
                                  dtype=torch.float32, device=self.device)
            return neutral.unsqueeze(0).repeat(self.num_frames, 1)
        elif self.default_init == 'random':
            return torch.randn(self.num_frames, self.num_joints, 
                             dtype=torch.float32, device=self.device) * 0.1
        else:
            raise ValueError(f"Unknown init method: {self.default_init}")
    
    def reset_history(self, init_method=None):
        """
        重置历史缓冲区
        
        使用场景：
        - 切换运动序列时
        - 长时间运行后误差累积时
        - 机器人姿态被外部重置时
        """
        if init_method:
            self.default_init = init_method
        self.history_buffer = self._init_history()
        self.is_first_prediction = True
        print(f"✓ 历史缓冲区已重置（方式: {self.default_init}）")
    
    def update_history(self, new_angles):
        """
        更新历史缓冲区（滑动窗口）
        
        Args:
            new_angles: [num_joints] 或 [7] 新预测的关节角度
        """
        # 转换为tensor
        if isinstance(new_angles, np.ndarray):
            new_angles = torch.from_numpy(new_angles).float()
        new_angles = new_angles.to(self.device)
        
        # 滑动窗口：移除最旧的一帧，添加最新的一帧
        self.history_buffer = torch.cat([
            self.history_buffer[1:],  # 移除第一帧
            new_angles.unsqueeze(0)    # 添加新帧
        ], dim=0)
    
    def predict(self, target_position, target_orientation=None):
        """
        单次推理
        
        Args:
            target_position: [3] 或 [batch, 3] 目标位置 (x, y, z)
            target_orientation: [4] 或 [batch, 4] 目标姿态四元数 (qx, qy, qz, qw)，可选
            
        Returns:
            pred_angles: [7] 或 [batch, 7] 预测的关节角度
        """
        # 处理numpy输入
        if isinstance(target_position, np.ndarray):
            target_position = torch.from_numpy(target_position).float()
        if target_orientation is not None and isinstance(target_orientation, np.ndarray):
            target_orientation = torch.from_numpy(target_orientation).float()
        
        # 确保是batch格式
        single_input = (target_position.dim() == 1)
        if single_input:
            target_position = target_position.unsqueeze(0)
            if target_orientation is not None:
                target_orientation = target_orientation.unsqueeze(0)
        
        target_position = target_position.to(self.device)
        if target_orientation is not None:
            target_orientation = target_orientation.to(self.device)
        
        batch_size = target_position.shape[0]
        
        # 扩展历史缓冲区到batch
        history_batch = self.history_buffer.unsqueeze(0).repeat(batch_size, 1, 1)
        
        with torch.no_grad():
            pred_angles, info = self.model(
                history_batch,
                target_position,
                target_orientation
            )
        
        # 更新历史缓冲区（使用第一个样本的角度作为代表）
        if batch_size == 1:
            self.update_history(pred_angles[0])
        else:
            # 对于batch，使用平均值更新历史
            self.update_history(pred_angles.mean(dim=0))
        
        self.is_first_prediction = False
        
        # 返回格式与输入一致
        if single_input:
            return pred_angles[0].cpu().numpy()
        return pred_angles.cpu().numpy()
    
    def predict_batch(self, target_positions, target_orientations=None):
        """
        批量推理（每个样本独立，不更新共享历史）
        
        适用于离线处理，每个样本用自己的历史初始化
        """
        if isinstance(target_positions, np.ndarray):
            target_positions = torch.from_numpy(target_positions).float()
        if target_orientations is not None and isinstance(target_orientations, np.ndarray):
            target_orientations = torch.from_numpy(target_orientations).float()
        
        target_positions = target_positions.to(self.device)
        if target_orientations is not None:
            target_orientations = target_orientations.to(self.device)
        
        batch_size = target_positions.shape[0]
        
        # 为每个样本初始化独立的历史（使用默认初始化）
        history_batch = self.history_buffer.unsqueeze(0).repeat(batch_size, 1, 1)
        
        with torch.no_grad():
            pred_angles, info = self.model(
                history_batch,
                target_positions,
                target_orientations
            )
        
        return pred_angles.cpu().numpy()
    
    def predict_with_iteration(self, target_position, target_orientation=None, 
                               num_iterations=5, lr=0.01):
        """
        迭代优化推理（在纯位姿输入基础上进一步优化）
        
        适用于对精度要求极高的场景
        """
        # 先用模型得到初始猜测
        init_angles = self.predict(target_position, target_orientation)
        
        # 转换为可优化的tensor
        opt_angles = torch.tensor(init_angles, device=self.device, 
                                  dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([opt_angles], lr=lr)
        
        # 目标位姿
        target_pos = torch.tensor(target_position, device=self.device, dtype=torch.float32)
        if target_orientation is not None:
            target_ori = torch.tensor(target_orientation, device=self.device, dtype=torch.float32)
        
        # 迭代优化
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            # 更新历史缓冲区
            self.update_history(opt_angles.detach())
            history_batch = self.history_buffer.unsqueeze(0)
            
            # 前向传播
            pred_angles, _ = self.model(
                history_batch,
                target_pos.unsqueeze(0),
                target_ori.unsqueeze(0) if target_orientation is not None else None
            )
            
            # 计算与初始猜测的差异（正则化，防止偏离太远）
            loss = torch.mean((pred_angles[0] - opt_angles) ** 2)
            
            loss.backward()
            optimizer.step()
        
        return opt_angles.detach().cpu().numpy()


class SequenceIKProcessor:
    """
    序列IK处理器（用于处理连续的运动轨迹）
    
    特点：
    1. 自动维护平滑的历史窗口
    2. 支持位姿插值，使运动更平滑
    3. 可选的平滑滤波
    """
    
    def __init__(self, model_path, smooth_factor=0.3, interpolation_steps=0):
        """
        Args:
            model_path: 模型路径
            smooth_factor: 平滑因子（0-1，越大越平滑但延迟越大）
            interpolation_steps: 位姿间插值步数（0表示不插值）
        """
        self.predictor = PoseOnlyIKPredictor(model_path)
        self.smooth_factor = smooth_factor
        self.interpolation_steps = interpolation_steps
        self.prev_angles = None
        
    def process_sequence(self, target_poses):
        """
        处理位姿序列
        
        Args:
            target_poses: [N, 7] 位姿序列 (x, y, z, qx, qy, qz, qw)
            
        Returns:
            joint_angles: [N, 7] 关节角度序列
        """
        results = []
        
        for i, pose in enumerate(target_poses):
            position = pose[:3]
            orientation = pose[3:7] if len(pose) >= 7 else None
            
            # 插值处理
            if self.interpolation_steps > 0 and i > 0:
                interp_poses = self._interpolate(
                    target_poses[i-1], pose, self.interpolation_steps
                )
                for interp_pose in interp_poses:
                    interp_pos = interp_pose[:3]
                    interp_ori = interp_pose[3:7] if len(interp_pose) >= 7 else None
                    angles = self.predictor.predict(interp_pos, interp_ori)
                    angles = self._smooth(angles)
                    results.append(angles)
            else:
                angles = self.predictor.predict(position, orientation)
                angles = self._smooth(angles)
                results.append(angles)
        
        return np.array(results)
    
    def _interpolate(self, pose1, pose2, steps):
        """位姿插值"""
        alphas = np.linspace(0, 1, steps + 2)[1:-1]  # 不包括端点
        results = []
        for alpha in alphas:
            interp = pose1 * (1 - alpha) + pose2 * alpha
            results.append(interp)
        return results
    
    def _smooth(self, angles):
        """平滑滤波"""
        if self.prev_angles is None:
            self.prev_angles = angles
            return angles
        
        smoothed = self.smooth_factor * self.prev_angles + (1 - self.smooth_factor) * angles
        self.prev_angles = smoothed
        return smoothed


# ==================== 使用示例 ====================

def demo_single_prediction():
    """单次推理示例"""
    print("=" * 60)
    print("示例1: 单次推理")
    print("=" * 60)
    
    # 初始化预测器
    predictor = PoseOnlyIKPredictor(
        model_path="/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth",
        default_init='neutral'  # 使用中性姿态初始化历史
    )
    
    # 模拟目标位姿（实际应用时从传感器/规划器获取）
    target_position = np.array([0.5, 0.2, 0.3])  # x, y, z
    target_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # qx, qy, qz, qw
    
    # 推理
    joint_angles = predictor.predict(target_position, target_orientation)
    print(f"目标位置: {target_position}")
    print(f"预测角度: {joint_angles}")
    
    # 第二次推理（历史已自动更新）
    target_position2 = np.array([0.55, 0.25, 0.35])
    joint_angles2 = predictor.predict(target_position2, target_orientation)
    print(f"\n目标位置2: {target_position2}")
    print(f"预测角度2: {joint_angles2}")


def demo_sequence_prediction():
    """序列推理示例"""
    print("\n" + "=" * 60)
    print("示例2: 序列推理（适合实时控制）")
    print("=" * 60)
    
    predictor = PoseOnlyIKPredictor(
        model_path="/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth"
    )
    
    # 模拟轨迹（10个位姿点）
    trajectory = []
    for t in np.linspace(0, 1, 10):
        pos = np.array([0.4 + 0.2 * t, 0.1 + 0.2 * t, 0.2 + 0.3 * t])
        ori = np.array([0.0, 0.0, 0.0, 1.0])
        trajectory.append(np.concatenate([pos, ori]))
    
    # 顺序推理
    print("轨迹推理:")
    for i, pose in enumerate(trajectory):
        angles = predictor.predict(pose[:3], pose[3:7])
        print(f"  点{i}: pos=[{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}], "
              f"angles=[{angles[0]:.2f}, {angles[1]:.2f}, ...]")


def demo_with_fk_verification():
    """带FK验证的推理示例"""
    print("\n" + "=" * 60)
    print("示例3: 带FK验证的推理（需要GPU FK模型）")
    print("=" * 60)
    
    try:
        from gpu_fk_wrapper import SimpleGPUFK
        
        # 加载FK模型
        gpu_fk = SimpleGPUFK()
        
        predictor = PoseOnlyIKPredictor(
            model_path="/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth"
        )
        
        # 目标位姿
        target_pos = np.array([0.5, 0.1, 0.4])
        target_ori = np.array([0.0, 0.0, 0.0, 1.0])
        
        # IK推理
        pred_angles = predictor.predict(target_pos, target_ori)
        
        # FK验证
        pred_pos = gpu_fk.forward(torch.tensor(pred_angles).unsqueeze(0).cuda())
        pred_pos = pred_pos.cpu().numpy()[0]
        
        error = np.linalg.norm(pred_pos - target_pos)
        print(f"目标位置: {target_pos}")
        print(f"预测角度: {pred_angles}")
        print(f"FK验证位置: {pred_pos}")
        print(f"位置误差: {error:.6f}m")
        
    except Exception as e:
        print(f"FK验证需要GPU FK模型: {e}")


if __name__ == "__main__":
    # 运行示例（请确保模型路径正确）
    # demo_single_prediction()
    # demo_sequence_prediction()
    # demo_with_fk_verification()
    
    print("请根据实际需求修改模型路径后运行示例")
    print("\n使用示例:")
    print("""
    from inference_pose_only import PoseOnlyIKPredictor
    
    # 初始化
    predictor = PoseOnlyIKPredictor("path/to/model.pth")
    
    # 实时循环
    while True:
        # 从传感器获取目标位姿
        target_pose = get_target_pose_from_sensor()
        
        # IK求解
        joint_angles = predictor.predict(
            target_pose[:3],   # 位置
            target_pose[3:7]   # 姿态（可选）
        )
        
        # 发送给机器人
        robot.move(joint_angles)
    """)
