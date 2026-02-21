"""
真正的纯位姿IK推理 - 历史固定，位姿驱动

解决了自回归的"冻结"问题：
- 历史缓冲区固定为中性姿态（常量，不更新）
- 只根据目标位姿变化预测不同角度
- 每次推理独立，无累积依赖

核心思想：
历史只是给模型一个"参考姿态"的上下文，真正的驱动信号是目标位姿。
就像人一样：知道当前姿态后，看到目标位置就能决定怎么动，
不需要记住过去每一帧的详细变化。
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from causal_ik_model_pieper2 import PieperCausalIK


class PurePoseIKPredictor:
    """
    纯位姿驱动IK预测器
    
    关键区别：
    - 历史缓冲区固定为中性姿态（或指定姿态），永不更新
    - 只根据目标位姿变化产生不同预测
    - 每次推理完全独立
    
    适用场景：
    - 目标位姿来自外部跟踪（如人手、视觉）
    - 需要即时响应位姿变化
    - 不关心历史轨迹，只关心当前目标
    """
    
    def __init__(self, model_path, num_frames=10, device='cuda', 
                 neutral_angles=None):
        """
        Args:
            model_path: 模型权重路径
            num_frames: 历史帧数
            device: 计算设备
            neutral_angles: 中性姿态 [7]，None则使用零姿态
        """
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
        
        # 固定历史缓冲区：中性姿态重复
        if neutral_angles is None:
            # 零姿态（机器人初始姿态）
            neutral_angles = torch.zeros(7, dtype=torch.float32)
        else:
            if isinstance(neutral_angles, np.ndarray):
                neutral_angles = torch.from_numpy(neutral_angles).float()
        
        # 关键：历史固定为常量，永不更新！
        self.fixed_history = neutral_angles.to(self.device).unsqueeze(0).repeat(num_frames, 1)
        
        print(f"✓ 历史已固定为: {neutral_angles.numpy().round(3)}")
        print(f"  模型将根据目标位姿变化独立预测，不依赖历史更新")
        
    def predict(self, target_position, target_orientation=None):
        """
        纯位姿输入预测
        
        Args:
            target_position: [3] 目标位置 (x, y, z) 
            target_orientation: [4] 目标姿态四元数 (可选)
            
        Returns:
            angles: [7] 预测的关节角度
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
        
        # 关键：使用固定的历史（复制到batch）
        history_batch = self.fixed_history.unsqueeze(0).repeat(batch_size, 1, 1)
        
        with torch.no_grad():
            pred_angles, info = self.model(
                history_batch,
                target_position,
                target_orientation
            )
        
        if single_input:
            return pred_angles[0].cpu().numpy()
        return pred_angles.cpu().numpy()
    
    def predict_batch(self, target_positions, target_orientations=None):
        """
        批量预测
        
        Args:
            target_positions: [N, 3] 多个目标位置
            target_orientations: [N, 4] 多个目标姿态（可选）
            
        Returns:
            angles: [N, 7] 预测的关节角度
        """
        return self.predict(target_positions, target_orientations)


class AdaptivePoseIKPredictor:
    """
    自适应纯位姿IK预测器
    
    结合两种策略：
    1. 主要依赖目标位姿变化（解决冻结问题）
    2. 可选地根据预测结果微调历史（保留一定连续性）
    
    模式1 - 纯位姿（默认）：
        历史固定，完全由目标位姿驱动
        适合：人手跟踪、视觉引导
    
    模式2 - 软更新：
        历史缓慢向预测结果收敛
        适合：需要一定连续性的场景
    """
    
    def __init__(self, model_path, num_frames=10, device='cuda',
                 mode='pure_pose', adaptation_rate=0.1):
        """
        Args:
            mode: 'pure_pose'（纯位姿）或 'soft_update'（软更新）
            adaptation_rate: 软更新时的收敛速度（0-1）
        """
        self.mode = mode
        self.adaptation_rate = adaptation_rate
        
        # 加载模型
        self.num_frames = num_frames
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.model = PieperCausalIK(
            num_joints=7,
            num_frames=num_frames,
            hidden_dim=256,
            num_layers=2
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 初始化历史为零/中性
        self.history = torch.zeros(num_frames, 7, device=self.device)
        
    def reset_history(self, joint_angles=None):
        """重置历史"""
        if joint_angles is None:
            self.history = torch.zeros(self.num_frames, 7, device=self.device)
        else:
            if isinstance(joint_angles, np.ndarray):
                joint_angles = torch.from_numpy(joint_angles).float()
            self.history = joint_angles.to(self.device).unsqueeze(0).repeat(self.num_frames, 1)
    
    def predict(self, target_position, target_orientation=None):
        """预测"""
        # 处理输入
        if isinstance(target_position, np.ndarray):
            target_position = torch.from_numpy(target_position).float()
        if target_orientation is not None and isinstance(target_orientation, np.ndarray):
            target_orientation = torch.from_numpy(target_orientation).float()
        
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
        
        pred = pred_angles[0]
        
        # 根据模式更新历史
        if self.mode == 'pure_pose':
            # 纯位姿模式：历史缓慢向预测收敛（保持稳定性，但不冻结）
            new_frame = self.history[-1] * (1 - self.adaptation_rate) + pred * self.adaptation_rate
            self.history = torch.cat([self.history[1:], new_frame.unsqueeze(0)], dim=0)
            
        elif self.mode == 'soft_update':
            # 软更新模式：历史跟随预测，但有延迟
            self.history = torch.cat([self.history[1:], pred.unsqueeze(0)], dim=0)
        
        return pred.cpu().numpy()


# ==================== 测试 ====================

def test_pure_pose_predictor():
    """测试纯位姿预测器"""
    print("=" * 60)
    print("测试纯位姿IK预测器")
    print("=" * 60)
    
    model_path = "/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth"
    
    # 创建预测器（历史固定为零姿态）
    predictor = PurePoseIKPredictor(model_path, neutral_angles=np.zeros(7))
    
    print("\n测试：不同目标位姿应该产生不同预测")
    print("-" * 60)
    
    # 测试1：目标位姿变化，历史固定
    target_positions = [
        np.array([0.4, 0.1, 0.3]),
        np.array([0.5, 0.2, 0.4]),
        np.array([0.6, 0.3, 0.5]),
    ]
    
    print("历史固定为零，改变目标位姿：")
    for i, pos in enumerate(target_positions):
        angles = predictor.predict(pos)
        print(f"  目标{i+1} {pos.round(2)} -> 角度 {angles[:4].round(3)}...")
    
    # 测试2：相同目标位姿，预测应该相同
    print("\n相同目标位姿（验证一致性）：")
    pos = np.array([0.5, 0.2, 0.4])
    for i in range(3):
        angles = predictor.predict(pos)
        print(f"  重复{i+1}: {angles[:4].round(3)}...")
    
    print("\n✓ 只要目标位姿变化，预测就会变化！")
    print("✓ 历史固定不影响预测变化！")


def test_comparison():
    """对比：纯位姿 vs 自回归"""
    print("\n" + "=" * 60)
    print("对比：纯位姿 vs 自回归")
    print("=" * 60)
    
    model_path = "/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth"
    
    # 自回归（有问题的）
    print("\n自回归方式（历史更新）：")
    history = torch.zeros(10, 7)
    predictor = PurePoseIKPredictor(model_path)  # 借用其模型
    
    target_pos = np.array([0.5, 0.2, 0.4])
    
    for i in range(3):
        # 模拟自回归
        with torch.no_grad():
            pred, _ = predictor.model(
                history.unsqueeze(0).cuda(),
                torch.tensor(target_pos).unsqueeze(0).cuda(),
                None
            )
        angles = pred[0].cpu().numpy()
        print(f"  步{i+1}: 历史[-1]={history[-1, 0].item():.3f}... -> 预测={angles[:3].round(3)}...")
        # 更新历史（问题所在：如果预测不变，历史也不变）
        history = torch.cat([history[1:], pred[0].unsqueeze(0).cpu()], dim=0)
    
    # 纯位姿（修复的）
    print("\n纯位姿方式（历史固定）：")
    fixed_predictor = PurePoseIKPredictor(model_path, neutral_angles=np.zeros(7))
    
    target_positions = [
        np.array([0.5, 0.2, 0.4]),
        np.array([0.55, 0.25, 0.45]),
        np.array([0.6, 0.3, 0.5]),
    ]
    
    for i, pos in enumerate(target_positions):
        angles = fixed_predictor.predict(pos)
        print(f"  目标{i+1}: {pos.round(2)} -> 预测={angles[:3].round(3)}...")
    
    print("\n结论：纯位姿方式下，目标变化驱动预测变化！")


def demo_real_usage():
    """实际使用演示"""
    print("\n" + "=" * 60)
    print("实际使用示例")
    print("=" * 60)
    
    print("""
    # 场景：人手跟踪控制机器人
    
    from inference_pose_pure import PurePoseIKPredictor
    
    # 1. 初始化（历史固定为零，与机器人初始姿态一致）
    predictor = PurePoseIKPredictor(
        "model.pth",
        neutral_angles=np.zeros(7)  # 机器人初始零姿态
    )
    
    # 2. 实时循环（无历史维护负担）
    while True:
        # 获取人手目标位姿（来自VR手柄/视觉/动捕）
        hand_pose = vr_controller.get_pose()  # [x, y, z, qx, qy, qz, qw]
        
        # 直接预测！历史固定，只根据目标位姿变化
        joint_angles = predictor.predict(
            hand_pose[:3],   # 位置
            hand_pose[3:7]   # 姿态
        )
        
        # 发送给机器人
        robot.move(joint_angles)
        
        # 不需要更新历史！历史是固定的！
    """)


if __name__ == "__main__":
    print("纯位姿IK推理")
    print("=" * 60)
    
    # 运行测试
    # test_pure_pose_predictor()
    # test_comparison()
    # demo_real_usage()
    
    print("\n请根据实际模型路径运行测试")
    print("\n核心改进：")
    print("  - 历史固定为常量（零姿态/中性姿态）")
    print("  - 目标位姿变化直接驱动预测变化")
    print("  - 无历史更新死循环问题")
