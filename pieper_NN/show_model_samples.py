"""
展示模型在真实数据上的输入输出

从验证集中随机选择样本，展示：
1. 输入数据（历史10帧）
2. 模型预测输出
3. 真实标签
4. 详细对比
"""

import torch
import numpy as np
from causal_ik_model_pieper import PieperCausalIK
from gpu_fk_wrapper import SimpleGPUFK
from dataset_accad_cmu import create_accad_cmu_dataloaders
import sys
sys.path.insert(0, '/home/wsy/Desktop/casual/pieper_NN')


class Config:
    data_path = "/home/wsy/Desktop/casual/ACCAD_CMU_merged_training_data.npz"
    batch_size = 512
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def show_samples():
    """展示模型输入输出样本"""

    # 加载模型
    model = PieperCausalIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=512,
        num_layers=2
    ).to(Config.device)

    checkpoint = torch.load("/home/wsy/Desktop/casual/pieper_NN/pieper_causal_ik_092.pth", map_location=Config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载GPU FK
    gpu_fk = SimpleGPUFK()

    # 加载数据
    _, val_loader = create_accad_cmu_dataloaders(Config.data_path, Config)

    # 获取一个batch
    with torch.no_grad():
        for batch_X, batch_y, batch_last_angle, batch_human_pose in val_loader:
            batch_X = batch_X.to(Config.device)
            batch_y = batch_y.to(Config.device)
            batch_last_angle = batch_last_angle.to(Config.device)
            batch_human_pose = batch_human_pose.to(Config.device)

            # 展示前3个样本
            for i in range(3):
                print("\n" + "=" * 70)
                print(f"样本 {i}")
                print("=" * 70)

                # 1. 历史窗口（10帧）
                print("\n【1. 输入：历史窗口（10帧）】")
                print("帧  |  关节角度 (7维)")
                print("-" * 70)
                for frame in range(10):
                    angles = batch_X[i, frame].cpu().numpy()
                    print(f" {frame:2d}  |  {angles[0]:7.4f} {angles[1]:7.4f} {angles[2]:7.4f} {angles[3]:7.4f} {angles[4]:7.4f} {angles[5]:7.4f} {angles[6]:7.4f}")

                # 最后一帧（当前状态）
                print(f"\n【2. 当前状态（历史最后一帧）】")
                current_angles = batch_X[i, -1].cpu().numpy()
                print(f"  关节角度: {current_angles}")
                current_pos, _ = gpu_fk.forward(torch.from_numpy(current_angles).unsqueeze(0).cuda()), None
                print(f"  当前FK位置: {current_pos[0].cpu().numpy()}")

                # 3. 目标人臂位姿
                print(f"\n【3. 目标人臂位姿（条件输入）】")
                target_pose = batch_human_pose[i].cpu().numpy()
                print(f"  位置 (x,y,z):  {target_pose[:3]}")
                print(f"  姿态 (qx,qy,qz,qw): {target_pose[3:]}")

                # 4. 模型预测
                print(f"\n【4. 模型预测输出】")
                pred_angles, info = model(
                    batch_X[i:i+1],
                    batch_human_pose[i:i+1, :3],
                    batch_human_pose[i:i+1, 3:7]
                )
                pred_angles = pred_angles[0].cpu().numpy()
                print(f"  预测角度: {pred_angles}")

                # 5. 真实目标
                print(f"\n【5. 真实目标（标签）】")
                target_angles = batch_y[i].cpu().numpy()
                print(f"  目标角度: {target_angles}")

                # 6. 对比分析
                print(f"\n【6. 对比分析】")
                angle_diff = np.abs(pred_angles - target_angles)
                print(f"  角度误差（绝对值）: {angle_diff}")
                print(f"  平均角度误差: {np.mean(angle_diff):.6f} rad = {np.rad2deg(np.mean(angle_diff)):.4f}°")

                # FK位置
                pred_pos, _ = gpu_fk.forward(torch.from_numpy(pred_angles).unsqueeze(0).cuda()), None
                target_pos, _ = gpu_fk.forward(torch.from_numpy(target_angles).unsqueeze(0).cuda()), None
                pos_error = np.sqrt(np.sum((pred_pos[0].cpu().numpy() - target_pos[0].cpu().numpy())**2))

                print(f"\n  预测FK位置: {pred_pos[0].cpu().numpy()}")
                print(f"  目标FK位置: {target_pos[0].cpu().numpy()}")
                print(f"  位置误差: {pos_error:.6f} 米 = {pos_error*1000:.3f} mm")

                # 连续性
                gap = np.mean((pred_angles - current_angles)**2)
                print(f"\n  GAP（预测-当前）: {gap:.8f}")

                # Pieper权重
                print(f"\n【7. Pieper权重分析】")
                pos_weights = info['position_weights'][0].cpu().numpy()
                ori_weights = info['orientation_weights'][0].cpu().numpy()

                print(f"  位置影响权重:")
                joint_names = ['Shoulder Pitch', 'Shoulder Roll', 'Shoulder Yaw', 'Elbow', 'Forearm', 'Wrist Yaw', 'Wrist Pitch']
                for j, name in enumerate(joint_names):
                    print(f"    J{j} ({name}): {pos_weights[j]:.6e}")

                print(f"\n  姿态影响权重:")
                for j, name in enumerate(joint_names):
                    print(f"    J{j} ({name}): {ori_weights[j]:.6e}")

            break


if __name__ == '__main__':
    show_samples()
