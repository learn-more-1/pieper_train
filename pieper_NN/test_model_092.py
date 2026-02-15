"""
测试 pieper_causal_ik_092.pth 模型

评估：
1. 模型输入输出
2. 角度预测准确性
3. FK位置准确性
4. 连续性（平滑度）
5. 可视化预测结果
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from causal_ik_model_pieper import PieperCausalIK
from gpu_fk_wrapper import SimpleGPUFK
from dataset_accad_cmu import create_accad_cmu_dataloaders
import sys
sys.path.insert(0, '/home/wsy/Desktop/casual/pieper_NN')


class Config:
    data_path = "/home/wsy/Desktop/casual/ACCAD_CMU_merged_training_data.npz"
    batch_size = 512
    num_workers = 4
    pin_memory = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path):
    """加载模型"""
    model = PieperCausalIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=512,
        num_layers=2
    ).to(Config.device)

    checkpoint = torch.load(checkpoint_path, map_location=Config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ 加载模型: {checkpoint_path}")
    print(f"  训练epoch: {checkpoint['epoch']}")
    print(f"  最优验证损失: {checkpoint['best_val_loss']:.6f}")

    return model, checkpoint


def test_model_outputs(model, val_loader, gpu_fk):
    """测试模型输出"""

    print("\n" + "=" * 70)
    print("测试模型输入输出")
    print("=" * 70)

    with torch.no_grad():
        for batch_X, batch_y, batch_last_angle, batch_human_pose in val_loader:
            batch_X = batch_X.to(Config.device)
            batch_y = batch_y.to(Config.device)
            batch_last_angle = batch_last_angle.to(Config.device)
            batch_human_pose = batch_human_pose.to(Config.device)

            # 只测试前5个样本
            n_samples = 5
            batch_X = batch_X[:n_samples]
            batch_y = batch_y[:n_samples]
            batch_last_angle = batch_last_angle[:n_samples]
            batch_human_pose = batch_human_pose[:n_samples]

            # 模型预测
            pred_angles, info = model(
                batch_X,
                batch_human_pose[:, :3],
                batch_human_pose[:, 3:7]
            )

            print(f"\n输入 (样本0):")
            print(f"  历史窗口形状: {batch_X.shape}")
            print(f"  历史最后一帧角度: {batch_X[0, -1].cpu().numpy()}")
            print(f"  目标人臂位姿:")
            print(f"    位置: {batch_human_pose[0, :3].cpu().numpy()}")
            print(f"    姿态: {batch_human_pose[0, 3:].cpu().numpy()}")

            print(f"\n输出 (样本0):")
            print(f"  预测角度: {pred_angles[0].cpu().numpy()}")
            print(f"  目标角度: {batch_y[0].cpu().numpy()}")
            print(f"  差异: {(pred_angles[0] - batch_y[0]).cpu().numpy()}")

            # 计算FK位置
            pred_pos, _ = gpu_fk.forward(pred_angles), None
            target_pos, _ = gpu_fk.forward(batch_y), None

            print(f"\nFK位置对比 (样本0):")
            print(f"  预测位置: {pred_pos[0].cpu().numpy()}")
            print(f"  目标位置: {target_pos[0].cpu().numpy()}")
            print(f"  位置误差: {torch.sqrt(torch.sum((pred_pos[0] - target_pos[0])**2)).item():.6f} 米")

            # Pieper权重分析
            print(f"\nPieper权重 (样本0):")
            pos_weights = info['position_weights'][0].cpu().numpy()
            ori_weights = info['orientation_weights'][0].cpu().numpy()
            print(f"  位置影响: {pos_weights}")
            print(f"  姿态影响: {ori_weights}")

            break


def evaluate_accuracy(model, val_loader, gpu_fk):
    """评估模型准确性"""

    print("\n" + "=" * 70)
    print("评估模型准确性（验证集）")
    print("=" * 70)

    model.eval()
    all_pred_angles = []
    all_target_angles = []
    all_pred_pos = []
    all_target_pos = []
    all_gap_errors = []

    num_batches = 100  # 测试前100个batch
    count = 0

    with torch.no_grad():
        for batch_X, batch_y, batch_last_angle, batch_human_pose in val_loader:
            if count >= num_batches:
                break

            batch_X = batch_X.to(Config.device)
            batch_y = batch_y.to(Config.device)
            batch_last_angle = batch_last_angle.to(Config.device)
            batch_human_pose = batch_human_pose.to(Config.device)

            # 预测
            pred_angles, _ = model(
                batch_X,
                batch_human_pose[:, :3],
                batch_human_pose[:, 3:7]
            )

            # FK位置
            pred_pos, _ = gpu_fk.forward(pred_angles), None
            target_pos, _ = gpu_fk.forward(batch_y), None

            # 统计
            all_pred_angles.append(pred_angles.cpu().numpy())
            all_target_angles.append(batch_y.cpu().numpy())
            all_pred_pos.append(pred_pos.cpu().numpy())
            all_target_pos.append(target_pos.cpu().numpy())

            # GAP误差
            gap_error = torch.mean((pred_angles - batch_last_angle) ** 2, dim=1)
            all_gap_errors.append(gap_error.cpu().numpy())

            count += 1

    # 合并结果
    all_pred_angles = np.vstack(all_pred_angles)
    all_target_angles = np.vstack(all_target_angles)
    all_pred_pos = np.vstack(all_pred_pos)
    all_target_pos = np.vstack(all_target_pos)
    all_gap_errors = np.concatenate(all_gap_errors)

    # 计算指标
    # 1. 角度误差
    angle_mse = np.mean((all_pred_angles - all_target_angles) ** 2)
    angle_rmse = np.sqrt(angle_mse)
    angle_mae = np.mean(np.abs(all_pred_angles - all_target_angles))

    # 每个关节的角度误差
    angle_errors_per_joint = np.sqrt(np.mean((all_pred_angles - all_target_angles) ** 2, axis=0))

    # 2. FK位置误差
    pos_error = np.sqrt(np.sum((all_pred_pos - all_target_pos) ** 2, axis=1))
    pos_error_mean = np.mean(pos_error)
    pos_error_median = np.median(pos_error)
    pos_error_std = np.std(pos_error)

    # 3. R²分数
    ss_res = np.sum((all_target_angles - all_pred_angles) ** 2)
    ss_tot = np.sum((all_target_angles - np.mean(all_target_angles, axis=0)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # 4. GAP误差（连续性）
    gap_mean = np.mean(all_gap_errors)
    gap_median = np.median(all_gap_errors)

    # 打印结果
    print(f"\n角度预测准确性:")
    print(f"  MSE:  {angle_mse:.8f}")
    print(f"  RMSE: {angle_rmse:.8f} rad = {np.rad2deg(angle_rmse):.4f}°")
    print(f"  MAE:  {angle_mae:.8f} rad = {np.rad2deg(angle_mae):.4f}°")
    print(f"  R²:   {r2:.6f}")

    print(f"\n每个关节的角度RMSE (度):")
    joint_names = ['Shoulder Pitch', 'Shoulder Roll', 'Shoulder Yaw',
                   'Elbow', 'Forearm Roll', 'Wrist Yaw', 'Wrist Pitch']
    for i, name in enumerate(joint_names):
        error_deg = np.rad2deg(angle_errors_per_joint[i])
        print(f"  J{i} ({name}): {error_deg:.4f}°")

    print(f"\nFK位置准确性:")
    print(f"  平均误差: {pos_error_mean:.6f} 米 = {pos_error_mean*1000:.3f} mm")
    print(f"  中位数:   {pos_error_median:.6f} 米 = {pos_error_median*1000:.3f} mm")
    print(f"  标准差:   {pos_error_std:.6f} 米 = {pos_error_std*1000:.3f} mm")

    print(f"\n连续性 (GAP):")
    print(f"  平均GAP: {gap_mean:.8f}")
    print(f"  中位数:   {gap_median:.8f}")

    # 误差分布
    print(f"\n位置误差分布:")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(pos_error * 1000, p)
        print(f"  {p}%: {value:.3f} mm")

    print("\n" + "=" * 70)
    print("结论:")
    if pos_error_mean < 0.01:  # 1cm
        print("  ✓ 位置误差优秀 (< 1cm)")
    elif pos_error_mean < 0.02:  # 2cm
        print("  ✓ 位置误差良好 (< 2cm)")
    elif pos_error_mean < 0.05:  # 5cm
        print("  ○ 位置误差可接受 (< 5cm)")
    else:
        print("  ✗ 位置误差较大 (> 5cm)")
    print("=" * 70)


def visualize_predictions(model, val_loader, gpu_fk):
    """可视化预测结果"""

    print("\n生成可视化图表...")

    # 收集一条轨迹的数据
    trajectory_data = {
        'pred_angles': [],
        'target_angles': [],
        'pred_pos': [],
        'target_pos': []
    }

    with torch.no_grad():
        for batch_X, batch_y, batch_last_angle, batch_human_pose in val_loader:
            batch_X = batch_X.to(Config.device)
            batch_y = batch_y.to(Config.device)
            batch_human_pose = batch_human_pose.to(Config.device)

            pred_angles, _ = model(
                batch_X,
                batch_human_pose[:, :3],
                batch_human_pose[:, 3:7]
            )

            pred_pos, _ = gpu_fk.forward(pred_angles), None
            target_pos, _ = gpu_fk.forward(batch_y), None

            trajectory_data['pred_angles'].append(pred_angles[0].cpu().numpy())
            trajectory_data['target_angles'].append(batch_y[0].cpu().numpy())
            trajectory_data['pred_pos'].append(pred_pos[0].cpu().numpy())
            trajectory_data['target_pos'].append(target_pos[0].cpu().numpy())

            if len(trajectory_data['pred_angles']) >= 100:  # 100帧轨迹
                break

    # 合并数据
    pred_angles = np.array(trajectory_data['pred_angles'])
    target_angles = np.array(trajectory_data['target_angles'])
    pred_pos = np.array(trajectory_data['pred_pos'])
    target_pos = np.array(trajectory_data['target_pos'])

    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # 1. 关节角度预测
    time_steps = np.arange(len(pred_angles))
    joint_names = ['J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6']

    for i in range(7):
        axes[0].plot(time_steps, target_angles[:, i], '--', alpha=0.5, label=f'{joint_names[i]} (true)')
        axes[0].plot(time_steps, pred_angles[:, i], '-', alpha=0.7, label=f'{joint_names[i]} (pred)')

    axes[0].set_ylabel('Joint Angle (rad)')
    axes[0].set_title('Joint Angle Predictions vs Ground Truth')
    axes[0].legend(ncol=7, fontsize='small')
    axes[0].grid(True, alpha=0.3)

    # 2. FK位置 (X, Y, Z)
    axes[1].plot(time_steps, target_pos[:, 0], '--', label='Target X', alpha=0.7)
    axes[1].plot(time_steps, pred_pos[:, 0], '-', label='Pred X', alpha=0.7)
    axes[1].plot(time_steps, target_pos[:, 1], '--', label='Target Y', alpha=0.7)
    axes[1].plot(time_steps, pred_pos[:, 1], '-', label='Pred Y', alpha=0.7)
    axes[1].plot(time_steps, target_pos[:, 2], '--', label='Target Z', alpha=0.7)
    axes[1].plot(time_steps, pred_pos[:, 2], '-', label='Pred Z', alpha=0.7)
    axes[1].set_ylabel('Position (m)')
    axes[1].set_title('End-Effector Position (FK)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 位置误差
    pos_error = np.sqrt(np.sum((pred_pos - target_pos) ** 2, axis=1)) * 1000  # mm
    axes[2].plot(time_steps, pos_error, '-', color='red')
    axes[2].axhline(y=np.mean(pos_error), color='green', linestyle='--', label=f'Mean: {np.mean(pos_error):.2f} mm')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Position Error (mm)')
    axes[2].set_title('End-Effector Position Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/wsy/Desktop/casual/pieper_NN/test_results.png', dpi=150)
    print(f"✓ 可视化图表已保存: /home/wsy/Desktop/casual/pieper_NN/test_results.png")

    # 保存分析报告
    with open('/home/wsy/Desktop/casual/pieper_NN/test_report.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("模型测试报告: pieper_causal_ik_092.pth\n")
        f.write("=" * 70 + "\n\n")

        f.write("角度预测准确性:\n")
        f.write(f"  RMSE: {np.sqrt(np.mean((pred_angles - target_angles)**2)):.6f} rad\n")
        f.write(f"  R²: {1 - np.sum((target_angles - pred_angles)**2) / (np.sum((target_angles - np.mean(target_angles))**2) + 1e-8):.6f}\n\n")

        f.write("FK位置准确性:\n")
        f.write(f"  平均误差: {np.mean(np.sqrt(np.sum((pred_pos - target_pos)**2, axis=1)))*1000:.3f} mm\n")
        f.write(f"  最大误差: {np.max(np.sqrt(np.sum((pred_pos - target_pos)**2, axis=1)))*1000:.3f} mm\n")

    print(f"✓ 测试报告已保存: /home/wsy/Desktop/casual/pieper_NN/test_report.txt")


def main():
    checkpoint_path = "/home/wsy/Desktop/casual/pieper_NN/pieper_causal_ik_092.pth"

    print("=" * 70)
    print("测试 Pieper 因果IK模型")
    print("=" * 70)

    # 加载模型
    model, checkpoint = load_model(checkpoint_path)

    # 加载GPU FK
    gpu_fk = SimpleGPUFK()

    # 加载数据
    print("\n加载测试数据...")
    train_loader, val_loader = create_accad_cmu_dataloaders(Config.data_path, Config)

    # 测试1: 模型输入输出
    test_model_outputs(model, val_loader, gpu_fk)

    # 测试2: 评估准确性
    evaluate_accuracy(model, val_loader, gpu_fk)

    # 测试3: 可视化
    visualize_predictions(model, val_loader, gpu_fk)

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
