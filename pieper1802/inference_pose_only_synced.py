"""
与机器人实际状态同步的IK推理脚本

解决了原自回归方案"手臂不动"的问题：
1. 用机器人实际当前姿态初始化历史缓冲区
2. 用机器人实际反馈更新历史（而非预测值）
3. 提供开环和闭环两种模式

使用方法：
    # 方式1: 闭环控制（推荐）
    predictor = SyncedIKPredictor(model_path)
    predictor.initialize_with_robot_state(robot.get_joint_angles())
    
    while True:
        target_pose = get_target_pose()
        # 预测时使用内部历史，但更新时用机器人实际状态
        angles = predictor.predict(target_pose)
        robot.move(angles)
        
        # 关键：用机器人实际状态更新历史
        actual_angles = robot.get_joint_angles()
        predictor.update_history_with_actual(actual_angles)
    
    # 方式2: 开环控制（无反馈）
    predictor = SyncedIKPredictor(model_path)
    predictor.initialize_with_robot_state(robot.get_joint_angles())
    
    for target_pose in trajectory:
        angles = predictor.predict(target_pose)
        robot.move(angles)
        # 用预测值更新（标准自回归）
        predictor.update_history_with_prediction(angles)
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from causal_ik_model_pieper2 import PieperCausalIK


class SyncedIKPredictor:
    """
    与机器人状态同步的IK预测器
    
    关键改进：
    1. 初始化时使用机器人实际姿态
    2. 支持用实际反馈更新历史（闭环）
    3. 支持用预测值更新历史（开环）
    4. 预测时加入平滑约束，避免突变
    """
    
    def __init__(self, model_path, num_frames=10, num_joints=7, 
                 device='cuda', max_delta=0.1):
        """
        Args:
            model_path: 模型权重路径
            num_frames: 历史帧数
            num_joints: 关节数量
            device: 计算设备
            max_delta: 单步最大角度变化（弧度），用于安全限制
        """
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_delta = max_delta
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 历史角度缓冲区 [num_frames, num_joints]
        self.history_buffer = None  # 延迟初始化，等待机器人实际状态
        
        # 上一帧预测结果（用于平滑）
        self.last_prediction = None
        
        # 平滑因子
        self.smooth_alpha = 0.3
        
        self.is_initialized = False
        
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
    
    def initialize_with_robot_state(self, current_joint_angles):
        """
        用机器人当前实际姿态初始化历史缓冲区
        
        这是解决"手臂不动"问题的关键！
        用实际状态填充整个历史窗口，而非零/中性
        
        Args:
            current_joint_angles: [7] 机器人当前的7个关节角度
        """
        if isinstance(current_joint_angles, np.ndarray):
            current_joint_angles = torch.from_numpy(current_joint_angles).float()
        
        current_joint_angles = current_joint_angles.to(self.device)
        
        # 用当前姿态重复填充历史窗口
        # 表示"过去一段时间内保持当前姿态"
        self.history_buffer = current_joint_angles.unsqueeze(0).repeat(self.num_frames, 1)
        
        # 初始化上一帧预测
        self.last_prediction = current_joint_angles.clone()
        
        self.is_initialized = True
        
        print(f"✓ 历史缓冲区已用实际姿态初始化")
        print(f"  当前姿态: {current_joint_angles.cpu().numpy()}")
        
    def update_history_with_prediction(self, predicted_angles):
        """
        用预测值更新历史（开环模式）
        
        标准自回归方式，适用于：
        - 机器人无反馈
        - 快速预览
        """
        if not self.is_initialized:
            raise RuntimeError("请先调用 initialize_with_robot_state() 初始化")
        
        if isinstance(predicted_angles, np.ndarray):
            predicted_angles = torch.from_numpy(predicted_angles).float()
        predicted_angles = predicted_angles.to(self.device)
        
        # 滑动窗口
        self.history_buffer = torch.cat([
            self.history_buffer[1:],
            predicted_angles.unsqueeze(0)
        ], dim=0)
        
        self.last_prediction = predicted_angles
        
    def update_history_with_actual(self, actual_joint_angles):
        """
        用机器人实际反馈更新历史（闭环模式）⭐推荐
        
        这是解决"手臂不动"问题的关键！
        用实际状态更新，保持历史缓冲区与机器人真实状态同步
        
        Args:
            actual_joint_angles: [7] 机器人实际的关节角度反馈
        """
        if not self.is_initialized:
            raise RuntimeError("请先调用 initialize_with_robot_state() 初始化")
        
        if isinstance(actual_joint_angles, np.ndarray):
            actual_joint_angles = torch.from_numpy(actual_joint_angles).float()
        actual_joint_angles = actual_joint_angles.to(self.device)
        
        # 滑动窗口
        self.history_buffer = torch.cat([
            self.history_buffer[1:],
            actual_joint_angles.unsqueeze(0)
        ], dim=0)
        
    def predict(self, target_position, target_orientation=None, 
                use_smoothing=True, enforce_limit=True):
        """
        预测关节角度
        
        Args:
            target_position: [3] 目标位置
            target_orientation: [4] 目标姿态（可选）
            use_smoothing: 是否使用平滑约束
            enforce_limit: 是否限制单步变化量
            
        Returns:
            pred_angles: [7] 预测的关节角度
        """
        if not self.is_initialized:
            raise RuntimeError("请先调用 initialize_with_robot_state() 初始化")
        
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
        
        target_position = target_position.to(self.device)
        if target_orientation is not None:
            target_orientation = target_orientation.to(self.device)
        
        # 扩展历史到batch
        history_batch = self.history_buffer.unsqueeze(0)
        
        with torch.no_grad():
            pred_angles, info = self.model(
                history_batch,
                target_position,
                target_orientation
            )
        
        pred_angles = pred_angles[0]  # [7]
        
        # 平滑约束：限制与上一帧的差异
        if use_smoothing and self.last_prediction is not None:
            delta = pred_angles - self.last_prediction
            delta = torch.clamp(delta, -self.max_delta, self.max_delta)
            pred_angles = self.last_prediction + delta * (1 - self.smooth_alpha) + \
                          pred_angles * self.smooth_alpha
        
        # 硬限制：单步最大变化
        if enforce_limit and self.last_prediction is not None:
            delta = pred_angles - self.last_prediction
            delta = torch.clamp(delta, -self.max_delta, self.max_delta)
            pred_angles = self.last_prediction + delta
        
        return pred_angles.cpu().numpy()
    
    def predict_with_warmup(self, target_position, target_orientation=None, 
                           warmup_steps=5):
        """
        带预热的预测（首次使用或重置后）
        
        通过多步小幅度更新让历史缓冲区进入自然状态
        
        Args:
            target_position: [3] 目标位置
            target_orientation: [4] 目标姿态
            warmup_steps: 预热步数
        """
        if not self.is_initialized:
            raise RuntimeError("请先调用 initialize_with_robot_state() 初始化")
        
        current_pos = self.history_buffer[-1, :3].cpu().numpy()  # 近似
        target_pos = target_position if isinstance(target_position, np.ndarray) else target_position.cpu().numpy()
        
        # 插值路径
        for i in range(1, warmup_steps + 1):
            alpha = i / warmup_steps
            interp_pos = current_pos * (1 - alpha) + target_pos * alpha
            angles = self.predict(interp_pos, target_orientation, use_smoothing=True)
            self.update_history_with_prediction(angles)
        
        # 最终预测
        return self.predict(target_position, target_orientation)


class RealtimeIKController:
    """
    实时IK控制器（集成闭环反馈）
    
    专为解决"手臂不动"问题设计
    """
    
    def __init__(self, model_path, robot_interface, control_freq=50):
        """
        Args:
            model_path: 模型路径
            robot_interface: 机器人接口对象，需要实现:
                - get_joint_angles() -> [7]
                - move(joint_angles) -> None
            control_freq: 控制频率(Hz)
        """
        self.predictor = SyncedIKPredictor(model_path)
        self.robot = robot_interface
        self.dt = 1.0 / control_freq
        
        # 初始化
        self._initialize()
        
    def _initialize(self):
        """用机器人实际状态初始化"""
        current_angles = self.robot.get_joint_angles()
        self.predictor.initialize_with_robot_state(current_angles)
        print(f"✓ 控制器已用机器人实际状态初始化")
        
    def move_to_pose(self, target_position, target_orientation=None, 
                     duration=None, use_warmup=True):
        """
        移动到目标位姿
        
        Args:
            target_position: [3] 目标位置
            target_orientation: [4] 目标姿态
            duration: 运动时间（秒），None表示单步到达
            use_warmup: 是否使用预热
        """
        if duration is None:
            # 单步到达
            if use_warmup and self.predictor.is_initialized:
                angles = self.predictor.predict_with_warmup(
                    target_position, target_orientation
                )
            else:
                angles = self.predictor.predict(target_position, target_orientation)
            
            self.robot.move(angles)
            
            # 闭环更新：用实际状态
            actual = self.robot.get_joint_angles()
            self.predictor.update_history_with_actual(actual)
            
        else:
            # 轨迹跟踪
            steps = int(duration / self.dt)
            current_pos = self.robot.get_joint_angles()[:3]  # 简化近似
            
            for i in range(1, steps + 1):
                alpha = i / steps
                interp_pos = current_pos * (1 - alpha) + target_position * alpha
                
                angles = self.predictor.predict(interp_pos, target_orientation)
                self.robot.move(angles)
                
                # 闭环更新
                actual = self.robot.get_joint_angles()
                self.predictor.update_history_with_actual(actual)
                
    def track_trajectory(self, trajectory_poses):
        """
        跟踪轨迹
        
        Args:
            trajectory_poses: [N, 7] 位姿序列
        """
        results = []
        
        for i, pose in enumerate(trajectory_poses):
            position = pose[:3]
            orientation = pose[3:7] if len(pose) >= 7 else None
            
            # 预测
            angles = self.predictor.predict(position, orientation)
            
            # 执行
            self.robot.move(angles)
            
            # 获取实际反馈并更新历史
            actual = self.robot.get_joint_angles()
            self.predictor.update_history_with_actual(actual)
            
            results.append(actual)
            
        return np.array(results)


# ==================== 模拟机器人接口示例 ====================

class MockRobotInterface:
    """
    模拟机器人接口（用于测试）
    
    模拟真实机器人的行为：
    - get_joint_angles() 返回当前状态
    - move() 执行运动（带小延迟和噪声模拟真实系统）
    """
    
    def __init__(self, initial_angles=None):
        if initial_angles is None:
            # 中性姿态
            self.current_angles = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        else:
            self.current_angles = np.array(initial_angles)
            
        self.move_delay = 0.01  # 模拟运动延迟
        
    def get_joint_angles(self):
        """获取当前关节角度"""
        # 添加微小噪声模拟传感器
        noise = np.random.randn(7) * 0.001
        return self.current_angles + noise
    
    def move(self, joint_angles):
        """执行运动"""
        joint_angles = np.array(joint_angles)
        
        # 模拟：机器人不会瞬间到达目标，而是部分跟踪
        tracking_ratio = 0.9  # 跟踪比例
        self.current_angles = self.current_angles * (1 - tracking_ratio) + \
                              joint_angles * tracking_ratio
        
        # 添加执行噪声
        self.current_angles += np.random.randn(7) * 0.005
        
        # 模拟延迟
        import time
        time.sleep(self.move_delay)


# ==================== 测试和演示 ====================

def test_closed_loop_vs_open_loop():
    """
    对比闭环 vs 开环
    
    演示为什么闭环能解决"手臂不动"问题
    """
    print("=" * 70)
    print("测试: 闭环 vs 开环控制")
    print("=" * 70)
    
    model_path = "/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth"
    
    # 创建模拟机器人（初始姿态非零）
    initial_pose = np.array([0.2, -0.1, 0.3, 0.4, 0.1, -0.2, 0.0])
    robot = MockRobotInterface(initial_angles=initial_pose)
    
    print(f"\n机器人初始姿态: {initial_pose}")
    
    # 目标位姿
    target_pos = np.array([0.5, 0.2, 0.3])
    target_ori = np.array([0.0, 0.0, 0.0, 1.0])
    
    # ===== 测试1: 开环（标准自回归）=====
    print("\n" + "-" * 50)
    print("开环控制（标准自回归）")
    print("-" * 50)
    
    predictor_open = SyncedIKPredictor(model_path)
    # 问题所在：用零/中性初始化，而非机器人实际状态！
    predictor_open.initialize_with_robot_state(np.zeros(7))  # 错误方式
    
    print(f"历史初始化: 零姿态")
    print(f"预测序列:")
    
    for i in range(5):
        pred = predictor_open.predict(target_pos, target_ori)
        print(f"  步{i}: 预测={pred[:3].round(3)}..., 机器人实际={robot.get_joint_angles()[:3].round(3)}...")
        
        # 开环：用预测值更新
        predictor_open.update_history_with_prediction(pred)
        
        # 机器人实际运动（模拟）
        robot.move(pred)
    
    # ===== 测试2: 闭环（推荐）=====
    print("\n" + "-" * 50)
    print("闭环控制（用实际反馈更新）")
    print("-" * 50)
    
    # 重置机器人
    robot = MockRobotInterface(initial_angles=initial_pose)
    
    predictor_closed = SyncedIKPredictor(model_path)
    # 关键：用机器人实际状态初始化！
    predictor_closed.initialize_with_robot_state(robot.get_joint_angles())
    
    print(f"历史初始化: 机器人实际姿态")
    print(f"预测序列:")
    
    for i in range(5):
        pred = predictor_closed.predict(target_pos, target_ori)
        actual_before = robot.get_joint_angles()
        
        # 机器人运动
        robot.move(pred)
        actual_after = robot.get_joint_angles()
        
        print(f"  步{i}: 预测={pred[:3].round(3)}..., 实际={actual_after[:3].round(3)}...")
        
        # 闭环：用实际反馈更新
        predictor_closed.update_history_with_actual(actual_after)
    
    print("\n结论：闭环控制使历史缓冲区与机器人状态保持同步")


def test_warmup_effect():
    """测试预热效果"""
    print("\n" + "=" * 70)
    print("测试: 预热效果")
    print("=" * 70)
    
    model_path = "/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth"
    
    robot = MockRobotInterface()
    predictor = SyncedIKPredictor(model_path)
    predictor.initialize_with_robot_state(robot.get_joint_angles())
    
    target_pos = np.array([0.6, 0.3, 0.4])
    target_ori = np.array([0.0, 0.0, 0.0, 1.0])
    
    print(f"\n目标位置: {target_pos}")
    print(f"初始姿态: {robot.get_joint_angles()}")
    
    # 无预热
    print("\n无预热直接预测:")
    pred_no_warmup = predictor.predict(target_pos, target_ori, use_smoothing=False)
    print(f"  预测结果: {pred_no_warmup.round(3)}")
    
    # 重置
    predictor.initialize_with_robot_state(robot.get_joint_angles())
    
    # 有预热
    print("\n有预热预测:")
    pred_warmup = predictor.predict_with_warmup(target_pos, target_ori, warmup_steps=5)
    print(f"  预测结果: {pred_warmup.round(3)}")
    
    print("\n预热使历史缓冲区逐步过渡，预测更平滑")


def demo_realtime_controller():
    """演示实时控制器"""
    print("\n" + "=" * 70)
    print("演示: 实时IK控制器")
    print("=" * 70)
    
    model_path = "/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth"
    
    # 创建模拟机器人
    robot = MockRobotInterface(initial_angles=np.array([0.1, 0.1, 0.1, 0.3, 0.1, 0.0, 0.0]))
    
    # 创建控制器
    controller = RealtimeIKController(model_path, robot, control_freq=50)
    
    # 目标轨迹
    target_poses = []
    for t in np.linspace(0, 1, 10):
        pos = np.array([0.4 + 0.2 * t, 0.1 + 0.15 * t, 0.2 + 0.2 * t])
        ori = np.array([0.0, 0.0, 0.0, 1.0])
        target_poses.append(np.concatenate([pos, ori]))
    
    print(f"\n跟踪轨迹（10个点）...")
    results = controller.track_trajectory(np.array(target_poses))
    
    print(f"\n完成！最终姿态: {results[-1].round(3)}")


if __name__ == "__main__":
    print("同步IK推理测试")
    print("=" * 70)
    
    # 运行测试
    # test_closed_loop_vs_open_loop()
    # test_warmup_effect()
    # demo_realtime_controller()
    
    print("\n请根据实际模型路径运行测试")
    print("\n实际使用示例:")
    print("""
    from inference_pose_only_synced import SyncedIKPredictor
    
    # 初始化预测器
    predictor = SyncedIKPredictor("path/to/model.pth")
    
    # 关键：用机器人实际当前姿态初始化！
    current_angles = robot.get_joint_angles()
    predictor.initialize_with_robot_state(current_angles)
    
    # 实时循环
    while True:
        target_pose = get_target_pose()
        
        # 预测
        angles = predictor.predict(target_pose[:3], target_pose[3:7])
        
        # 发送给机器人
        robot.move(angles)
        
        # 关键：用机器人实际反馈更新历史！
        actual_angles = robot.get_joint_angles()
        predictor.update_history_with_actual(actual_angles)
    """)
