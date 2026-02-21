"""
对比学习风格 IK 推理示例

展示如何在实际应用中使用训练好的模型
关键：只需要 history_poses，不需要 history_joints！
"""

import torch
import numpy as np
import sys

sys.path.insert(0, '/home/bonuli/Pieper/2103')  # 确保优先使用本目录的 model

from model import ContrastiveStyleIK, NormalizationLayer


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    model = ContrastiveStyleIK(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    norm_layer = NormalizationLayer(
        checkpoint['pose_mean'],
        checkpoint['pose_std'],
        checkpoint['joint_mean'],
        checkpoint['joint_std']
    ).to(device)
    
    return model, norm_layer, config


def predict_ik(model, norm_layer, target_pose, history_poses, device='cuda'):
    """
    预测 IK
    
    Args:
        target_pose: [7] 目标位姿 (xyz + quaternion)
        history_poses: [10, 7] 过去10帧的末端位姿
    
    Returns:
        joint_angles: [7] 预测的关节角度
        style: [style_dim] 推断的个人风格（可用于分析）
    """
    with torch.no_grad():
        # 转 tensor
        target_pose_t = torch.tensor(target_pose, dtype=torch.float32, device=device).unsqueeze(0)
        history_poses_t = torch.tensor(history_poses, dtype=torch.float32, device=device).unsqueeze(0)
        
        # 归一化
        target_pose_norm = norm_layer.normalize_pose(target_pose_t)
        history_poses_norm = norm_layer.normalize_history_poses(history_poses_t)
        
        # 推理（关键：不传 history_joints！）
        pred, aux = model(
            target_pose_norm,
            history_poses_norm,
            history_joints=None,
            mode='inference',
            return_aux=True
        )
        
        # 反归一化
        pred_denorm = norm_layer.denormalize_joint(pred)
        
        return pred_denorm.cpu().numpy()[0], aux['style'].cpu().numpy()[0]


def main():
    """示例：使用模型进行推理"""
    
    # 配置
    checkpoint_path = "/home/bonuli/Pieper/2103/contrastive_ik_2103.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("对比学习风格 IK 推理示例")
    print("=" * 60)
    
    # 加载模型
    print(f"\n加载模型: {checkpoint_path}")
    try:
        model, norm_layer, config = load_model(checkpoint_path, device)
        print(f"✓ 模型加载成功")
        print(f"  风格维度: {config['style_dim']}")
        print(f"  隐藏层: {config['hidden_dim']}")
    except FileNotFoundError:
        print(f"✗ 模型文件不存在，请先训练模型")
        print(f"  运行: python train_contrastive_ik.py")
        return
    
    # 模拟输入数据
    # 在实际应用中，这些数据来自传感器或仿真
    print("\n模拟输入数据...")
    
    # 过去10帧的末端位姿 (xyz + quaternion)
    # 模拟一个向右上方伸手的轨迹
    history_poses = np.array([
        [0.3, -0.2, 0.1, 0, 0, 0, 1],  # t-9
        [0.32, -0.18, 0.12, 0, 0, 0, 1],
        [0.34, -0.16, 0.14, 0, 0, 0, 1],
        [0.36, -0.14, 0.16, 0, 0, 0, 1],
        [0.38, -0.12, 0.18, 0, 0, 0, 1],
        [0.40, -0.10, 0.20, 0, 0, 0, 1],
        [0.42, -0.08, 0.22, 0, 0, 0, 1],
        [0.44, -0.06, 0.24, 0, 0, 0, 1],
        [0.46, -0.04, 0.26, 0, 0, 0, 1],
        [0.48, -0.02, 0.28, 0, 0, 0, 1],  # t-1 (当前)
    ])
    
    # 目标位姿（下一帧想要到达的位置）
    target_pose = np.array([0.50, 0.0, 0.30, 0, 0, 0, 1])
    
    print(f"  历史位姿: {history_poses.shape}")
    print(f"  目标位姿: {target_pose.shape}")
    
    # 预测
    print("\n执行推理...")
    joint_angles, style = predict_ik(model, norm_layer, target_pose, history_poses, device)
    
    print(f"\n预测结果:")
    print(f"  关节角度: {joint_angles}")
    print(f"    - Shoulder Pitch: {joint_angles[0]:.4f} rad ({np.degrees(joint_angles[0]):.2f}°)")
    print(f"    - Shoulder Roll:  {joint_angles[1]:.4f} rad ({np.degrees(joint_angles[1]):.2f}°)")
    print(f"    - Shoulder Yaw:   {joint_angles[2]:.4f} rad ({np.degrees(joint_angles[2]):.2f}°)")
    print(f"    - Elbow:          {joint_angles[3]:.4f} rad ({np.degrees(joint_angles[3]):.2f}°)")
    print(f"    - Forearm Roll:   {joint_angles[4]:.4f} rad ({np.degrees(joint_angles[4]):.2f}°)")
    print(f"    - Wrist Yaw:      {joint_angles[5]:.4f} rad ({np.degrees(joint_angles[5]):.2f}°)")
    print(f"    - Wrist Pitch:    {joint_angles[6]:.4f} rad ({np.degrees(joint_angles[6]):.2f}°)")
    
    print(f"\n推断的个人风格向量 (前10维): {style[:10]}")
    print(f"  风格范数: {np.linalg.norm(style):.4f}")
    
    # 连续推理示例（模拟实时应用）
    print("\n" + "=" * 60)
    print("连续推理示例（模拟实时控制）")
    print("=" * 60)
    
    num_steps = 5
    all_styles = []
    
    for step in range(num_steps):
        # 模拟新的观测
        # 实际应用中，这里是从传感器获取的最新数据
        history_poses = np.roll(history_poses, -1, axis=0)  # 移除最旧的一帧
        history_poses[-1] = target_pose  # 添加最新的位姿
        target_pose = target_pose + np.array([0.02, 0.02, 0.02, 0, 0, 0, 0])  # 新的目标
        
        joint_angles, style = predict_ik(model, norm_layer, target_pose, history_poses, device)
        all_styles.append(style)
        
        print(f"Step {step+1}: 目标位置 ({target_pose[0]:.2f}, {target_pose[1]:.2f}, {target_pose[2]:.2f}) | "
              f"风格范数: {np.linalg.norm(style):.4f}")
    
    # 分析风格一致性（应该保持稳定，因为是同一个人）
    all_styles = np.array(all_styles)
    style_variance = np.var(all_styles, axis=0).mean()
    print(f"\n风格一致性分析:")
    print(f"  风格方差: {style_variance:.6f}")
    print(f"  解释: 方差小表示模型识别为同一个人的运动风格")
    
    print("\n" + "=" * 60)
    print("✓ 推理示例完成")
    print("=" * 60)
    print("\n关键要点:")
    print("  1. 只需要 history_poses (末端轨迹)")
    print("  2. 不需要 history_joints (无自回归)")
    print("  3. 模型自动从末端轨迹推断个人运动风格")
    print("  4. 可用于实时控制")


if __name__ == "__main__":
    main()
