"""
引导历史IK推理 - 解决模型过度依赖历史的问题

问题诊断：
- 模型过度依赖历史帧（历史影响 > 目标位姿影响）
- 如果历史固定为零，即使目标位姿变化，预测也接近零
- 导致"手臂不动"

解决方案：
不再试图让模型直接从"零历史+目标位姿"预测正确角度，
而是生成一个引导历史序列，让历史逐渐变化，驱动模型产生正确输出。

就像：
- 不说"直接从这个姿势去那个位置"
- 而是说"先想象你已经在那个位置了，然后直接输出那个位置的姿势"
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from causal_ik_model_pieper2 import PieperCausalIK


class GuidedHistoryIKPredictor:
    """
    引导历史IK预测器
    
    核心思想：
    既然模型看到零历史就输出接近零的角度，
    那我们就不给它零历史，而是给它一个"引导历史"——
    让它认为机器人已经在目标附近了。
    
    实现方式：
    1. 用前向 kinematics 找到一个能产生目标位姿的近似关节角度
    2. 用这个近似角度填充历史
    3. 模型会输出接近这个近似角度的结果
    4. 迭代优化使结果更精确
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
        
        # 迭代优化参数
        self.num_iterations = 10
        self.lr = 0.01
        
    def _compute_guided_history(self, target_position, target_orientation=None):
        """
        计算引导历史
        
        用简单的启发式方法估算一个能到达目标位姿的关节角度，
        然后用这个角度填充历史。
        
        启发式（针对7轴机械臂）：
        - shoulder角度与位置强相关
        - 粗略估算：位置归一化后映射到shoulder角度范围
        """
        # 简单启发式：根据位置粗略估计关节角度
        # 这是一个非常简化的近似，实际可以用Jacobian迭代
        
        # 归一化位置（假设工作空间大致为0.3-0.7m）
        pos = target_position if isinstance(target_position, np.ndarray) else target_position.cpu().numpy()
        
        # 启发式映射（针对典型7轴机械臂）
        # shoulder关节对x,y,z位置都有影响
        estimated_angles = np.zeros(7)
        
        # 基于位置的粗略估计
        x, y, z = pos
        estimated_angles[0] = np.arctan2(y, x)  # shoulder 1
        estimated_angles[1] = -np.arcsin(np.clip(z / np.linalg.norm(pos), -1, 1))  # shoulder 2
        estimated_angles[2] = 0.0  # shoulder 3
        estimated_angles[3] = 0.5  # elbow
        estimated_angles[4] = 0.0  # forearm
        estimated_angles[5] = 0.0  # wrist 1
        estimated_angles[6] = 0.0  # wrist 2
        
        return estimated_angles
    
    def predict_iterative(self, target_position, target_orientation=None, 
                         num_iterations=10):
        """
        迭代预测
        
        1. 先用启发式生成引导历史
        2. 模型预测
        3. 用预测结果更新历史
        4. 重复直到收敛
        
        Args:
            target_position: [3] 目标位置
            target_orientation: [4] 目标姿态（可选）
            num_iterations: 迭代次数
            
        Returns:
            angles: [7] 预测的关节角度
        """
        # 处理输入
        if isinstance(target_position, np.ndarray):
            target_position = torch.from_numpy(target_position).float()
        if target_orientation is not None and isinstance(target_orientation, np.ndarray):
            target_orientation = torch.from_numpy(target_orientation).float()
        
        target_position = target_position.to(self.device)
        if target_orientation is not None:
            target_orientation = target_orientation.to(self.device)
        
        # 步骤1：生成引导历史
        guided_angles = self._compute_guided_history(target_position, target_orientation)
        guided_angles_tensor = torch.from_numpy(guided_angles).float().to(self.device)
        
        # 用引导角度填充历史
        history = guided_angles_tensor.unsqueeze(0).repeat(self.num_frames, 1)
        
        # 步骤2-4：迭代优化
        for i in range(num_iterations):
            with torch.no_grad():
                pred, _ = self.model(
                    history.unsqueeze(0),
                    target_position.unsqueeze(0),
                    target_orientation.unsqueeze(0) if target_orientation is not None else None
                )
            
            # 用预测结果更新历史（滑动窗口）
            history = torch.cat([history[1:], pred], dim=0)
        
        return pred[0].cpu().numpy()
    
    def predict_with_interpolation(self, target_position, target_orientation=None,
                                   current_angles=None, num_steps=5):
        """
        插值预测（从当前姿态平滑过渡到目标）
        
        如果提供了当前姿态，从当前姿态插值到目标，生成中间历史
        """
        # 处理输入
        if isinstance(target_position, np.ndarray):
            target_position = torch.from_numpy(target_position).float()
        if target_orientation is not None and isinstance(target_orientation, np.ndarray):
            target_orientation = torch.from_numpy(target_orientation).float()
        
        target_position = target_position.to(self.device)
        if target_orientation is not None:
            target_orientation = target_orientation.to(self.device)
        
        # 如果没有当前姿态，使用零
        if current_angles is None:
            current_angles = torch.zeros(7, device=self.device)
        elif isinstance(current_angles, np.ndarray):
            current_angles = torch.from_numpy(current_angles).float().to(self.device)
        else:
            current_angles = current_angles.to(self.device)
        
        # 生成目标估计
        target_estimated = self._compute_guided_history(target_position, target_orientation)
        target_estimated = torch.from_numpy(target_estimated).float().to(self.device)
        
        # 插值历史序列
        alphas = torch.linspace(0, 1, self.num_frames, device=self.device)
        history_sequence = torch.stack([
            current_angles * (1 - alpha) + target_estimated * alpha 
            for alpha in alphas
        ])
        
        # 用插值历史预测
        with torch.no_grad():
            pred, _ = self.model(
                history_sequence.unsqueeze(0),
                target_position.unsqueeze(0),
                target_orientation.unsqueeze(0) if target_orientation is not None else None
            )
        
        return pred[0].cpu().numpy()


class SimpleFKApproximator:
    """
    简单的FK近似器，用于验证引导历史的效果
    
    如果没有真实的FK模型，可以用这个近似验证
    """
    
    def __init__(self):
        # 简化的机械臂参数（根据实际机器人调整）
        self.link_lengths = [0.1, 0.1, 0.1, 0.15, 0.1, 0.1, 0.05]
        
    def forward(self, joint_angles):
        """
        简化的前向运动学
        
        Args:
            joint_angles: [7] 或 [batch, 7]
            
        Returns:
            position: [3] 或 [batch, 3]
        """
        if isinstance(joint_angles, np.ndarray):
            joint_angles = torch.from_numpy(joint_angles).float()
        
        single = (joint_angles.dim() == 1)
        if single:
            joint_angles = joint_angles.unsqueeze(0)
        
        batch_size = joint_angles.shape[0]
        
        # 简化的FK计算（仅用于验证）
        # 实际应该使用正确的机器人模型
        x = joint_angles[:, 0] * 0.2 + joint_angles[:, 3] * 0.15
        y = joint_angles[:, 1] * 0.2 + joint_angles[:, 3] * 0.1
        z = joint_angles[:, 2] * 0.15 + 0.3
        
        pos = torch.stack([x, y, z], dim=1)
        
        if single:
            return pos[0]
        return pos


# ==================== 测试 ====================

def test_guided_predictor():
    """测试引导历史预测器"""
    print("=" * 60)
    print("测试引导历史IK预测器")
    print("=" * 60)
    
    model_path = "/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth"
    predictor = GuidedHistoryIKPredictor(model_path)
    
    # 测试：不同目标位姿
    target_positions = [
        np.array([0.4, 0.1, 0.3]),
        np.array([0.5, 0.2, 0.4]),
        np.array([0.6, 0.3, 0.5]),
    ]
    
    print("\n迭代预测（引导历史）：")
    for i, pos in enumerate(target_positions):
        angles = predictor.predict_iterative(pos, num_iterations=10)
        print(f"  目标{i+1} {pos.round(2)} -> 角度 {angles[:4].round(3)}...")
    
    print("\n插值预测（从当前姿态过渡）：")
    current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i, pos in enumerate(target_positions):
        angles = predictor.predict_with_interpolation(pos, current_angles=current, num_steps=5)
        print(f"  目标{i+1} {pos.round(2)} -> 角度 {angles[:4].round(3)}...")


def test_comparison():
    """对比不同方法"""
    print("\n" + "=" * 60)
    print("方法对比")
    print("=" * 60)
    
    model_path = "/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PieperCausalIK(num_joints=7, num_frames=10, hidden_dim=256, num_layers=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    target_pos = torch.tensor([[0.5, 0.2, 0.4]], device=device)
    
    # 方法1：零历史（有问题的）
    print("\n方法1：零历史")
    history_zero = torch.zeros(1, 10, 7, device=device)
    with torch.no_grad():
        pred, _ = model(history_zero, target_pos, None)
    print(f"  预测: {pred[0, :3].cpu().numpy().round(4)}...")
    
    # 方法2：引导历史
    print("\n方法2：引导历史")
    guided = GuidedHistoryIKPredictor(model_path)
    angles = guided.predict_iterative(target_pos[0].cpu().numpy())
    print(f"  预测: {angles[:3].round(4)}...")


if __name__ == "__main__":
    test_guided_predictor()
    test_comparison()
    
    print("\n" + "=" * 60)
    print("使用建议")
    print("=" * 60)
    print("""
    由于模型过度依赖历史，建议使用以下方式：
    
    1. 迭代预测（推荐）
       from inference_guided_history import GuidedHistoryIKPredictor
       
       predictor = GuidedHistoryIKPredictor("model.pth")
       angles = predictor.predict_iterative(target_position, num_iterations=10)
    
    2. 插值预测（如果需要平滑过渡）
       angles = predictor.predict_with_interpolation(
           target_position, 
           current_angles=robot.get_joint_angles()
       )
    
    3. 长期方案：重新训练模型
       训练时降低历史帧的权重，增强目标位姿的影响
    """)
