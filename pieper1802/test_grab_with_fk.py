"""
测试 explicit_coupling_ik 在 GRAB 数据集上的效果

随机抽取 10000 帧进行测试，重点关注位置误差
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from explicit_coupling_ik import ExplicitCouplingIK


def load_grab_dataset(data_path, num_samples=10000):
    """
    加载 GRAB 数据集
    """
    print(f"加载 GRAB 数据集: {data_path}")
    data = np.load(data_path, allow_pickle=True)

    print(f"数据集 Keys: {list(data.keys())}")
    print(f"X shape: {data['X'].shape}")
    print(f"y shape: {data['y'].shape}")

    X = data['X']  # (N, 10, 14)
    y = data['y']  # (N, 14)

    # 随机选择 num_samples 个样本
    total_samples = X.shape[0]
    if total_samples > num_samples:
        indices = np.random.choice(total_samples, num_samples, replace=False)
        X = X[indices]
        y = y[indices]

    print(f"选择 {X.shape[0]} 个样本进行测试")

    return X, y


class SimpleGPUFK:
    """简化的GPU FK计算"""
    def __init__(self):
        # 这里需要实现或使用实际的FK
        # 暂时用简化的近似
        pass
    
    def forward(self, joint_angles):
        """
        简化的FK计算
        实际使用时应该调用正确的FK模型
        """
        # 简化的近似：假设末端位置主要由 shoulder 和 elbow 决定
        # 实际项目中应该使用 pytorch_kinematics 或 pinocchio
        
        # 这是一个占位符，实际应该替换为正确的FK
        # 从 joint_angles [batch, 7] 计算位置 [batch, 3]
        
        # 简化的线性近似（仅用于演示）
        # 实际使用时需要正确的机器人模型
        x = joint_angles[:, 0] * 0.2 + joint_angles[:, 3] * 0.15
        y = joint_angles[:, 1] * 0.2 + joint_angles[:, 3] * 0.1
        z = joint_angles[:, 2] * 0.15 + 0.3
        
        return torch.stack([x, y, z], dim=1)


def create_dataloader(X, y, batch_size=512):
    """创建 DataLoader"""
    class GRABDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X.astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.float32))

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            # 提取人类位姿（前7维）和机器人角度（后7维）
            human_pose = self.y[idx, :7]  # (7,) 人类位姿
            target_angles = self.y[idx, 7:]  # (7,) 机器人目标角度
            
            return human_pose, target_angles

    dataset = GRABDataset(X, y)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return loader


def test_on_grab_data():
    """在 GRAB 数据集上测试模型"""

    print("=" * 70)
    print("测试 explicit_coupling_ik 在 GRAB 数据集上的效果")
    print("=" * 70)

    # 加载模型
    model_path = "/home/bonuli/Pieper/pieper1802/elbow_constraint_ik.pth"
    print(f"\n加载模型: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = ExplicitCouplingIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        use_temporal=False
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # 处理torch.compile前缀
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("检测到torch.compile权重，移除前缀...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()

    print(f"✓ 模型加载成功")
    if 'epoch' in checkpoint:
        print(f"  训练epoch: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"  训练验证损失: {checkpoint['best_val_loss']:.6f}")

    # 加载 GRAB 数据集
    grab_data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"
    X, y = load_grab_dataset(grab_data_path, num_samples=10000)

    # 创建 DataLoader
    loader = create_dataloader(X, y, batch_size=512)

    # 加载 GPU FK
    try:
        from gpu_fk_wrapper import SimpleGPUFK as RealGPUFK
        gpu_fk = RealGPUFK()
        print("✓ 使用实际GPU FK模型")
    except:
        gpu_fk = SimpleGPUFK()
        print("⚠ 使用简化FK模型（结果可能不准确）")

    # 测试
    print("\n" + "=" * 70)
    print("开始测试 (10000 帧)")
    print("=" * 70)

    all_pred_angles = []
    all_target_angles = []
    all_pred_pos = []
    all_target_pos = []

    with torch.no_grad():
        for batch_idx, (batch_human_pose, batch_target_angles) in enumerate(loader):
            batch_human_pose = batch_human_pose.cuda()
            batch_target_angles = batch_target_angles.cuda()
            
            # 提取人类位置和姿态
            human_pos = batch_human_pose[:, :3]
            human_ori = batch_human_pose[:, 3:7]

            # 预测
            pred_angles, _ = model(
                human_pos,     # 人类位置
                human_ori      # 人类姿态
            )

            # FK位置
            pred_pos = gpu_fk.forward(pred_angles)
            target_pos = gpu_fk.forward(batch_target_angles)

            # 收集结果
            all_pred_angles.append(pred_angles.cpu().numpy())
            all_target_angles.append(batch_target_angles.cpu().numpy())
            all_pred_pos.append(pred_pos.cpu().numpy())
            all_target_pos.append(target_pos.cpu().numpy())

            if (batch_idx + 1) % 5 == 0:
                progress = min((batch_idx + 1) * 512, 10000) / 10000 * 100
                print(f"  处理进度: {progress:.1f}%")

    # 合并结果
    all_pred_angles = np.vstack(all_pred_angles)
    all_target_angles = np.vstack(all_target_angles)
    all_pred_pos = np.vstack(all_pred_pos)
    all_target_pos = np.vstack(all_target_pos)

    # 计算指标
    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)

    # 1. 角度误差
    angle_mse = np.mean((all_pred_angles - all_target_angles) ** 2)
    angle_rmse = np.sqrt(angle_mse)
    angle_mae = np.mean(np.abs(all_pred_angles - all_target_angles))

    # 每个关节的角度RMSE
    angle_errors_per_joint = np.sqrt(np.mean((all_pred_angles - all_target_angles) ** 2, axis=0))

    print(f"\n【角度预测准确性】")
    print(f"  MSE:  {angle_mse:.8f}")
    print(f"  RMSE: {angle_rmse:.8f} rad = {np.rad2deg(angle_rmse):.4f}°")
    print(f"  MAE:  {angle_mae:.8f} rad = {np.rad2deg(angle_mae):.4f}°")

    print(f"\n每个关节的角度RMSE (度):")
    joint_names = ['Shoulder Pitch', 'Shoulder Roll', 'Shoulder Yaw',
                   'Elbow', 'Forearm Roll', 'Wrist Pitch', 'Wrist Yaw']
    for i, name in enumerate(joint_names):
        error_deg = np.rad2deg(angle_errors_per_joint[i])
        print(f"  J{i} ({name}): {error_deg:.4f}°")

    # R²
    ss_res = np.sum((all_target_angles - all_pred_angles) ** 2)
    ss_tot = np.sum((all_target_angles - np.mean(all_target_angles, axis=0)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    print(f"\n  R²: {r2:.6f}")

    # 2. FK位置误差（重点）
    pos_error = np.sqrt(np.sum((all_pred_pos - all_target_pos) ** 2, axis=1))
    pos_error_mean = np.mean(pos_error)
    pos_error_median = np.median(pos_error)
    pos_error_std = np.std(pos_error)
    pos_error_max = np.max(pos_error)
    
    # 各维度误差
    pos_error_per_dim = np.abs(all_pred_pos - all_target_pos)
    pos_error_x = np.mean(pos_error_per_dim[:, 0])
    pos_error_y = np.mean(pos_error_per_dim[:, 1])
    pos_error_z = np.mean(pos_error_per_dim[:, 2])

    print(f"\n【FK位置准确性 - 重点】")
    print(f"  平均误差: {pos_error_mean:.6f} m = {pos_error_mean*1000:.3f} mm")
    print(f"  中位数:   {pos_error_median:.6f} m = {pos_error_median*1000:.3f} mm")
    print(f"  标准差:   {pos_error_std:.6f} m = {pos_error_std*1000:.3f} mm")
    print(f"  最大误差: {pos_error_max:.6f} m = {pos_error_max*1000:.3f} mm")
    
    print(f"\n  各维度平均误差:")
    print(f"    X: {pos_error_x*1000:.3f} mm")
    print(f"    Y: {pos_error_y*1000:.3f} mm")
    print(f"    Z: {pos_error_z*1000:.3f} mm")

    # 3. 误差分布
    print(f"\n【位置误差分布】")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(pos_error * 1000, p)
        print(f"  {p}%: {value:.3f} mm")

    # 4. 与训练集对比
    print(f"\n" + "=" * 70)
    print("与训练集对比 (ACCAD+CMU)")
    print("=" * 70)
    
    # 训练集性能（从训练日志获取）
    train_pos_error = 0.004  # ~4mm (来自训练时的FK loss 0.002)
    train_angle_rmse = 0.090  # ~5.16°
    
    print(f"\n指标                  | 训练集       | GRAB测试集")
    print(f"-" * 70)
    print(f"位置误差 (mm)         | {train_pos_error*1000:.2f}         | {pos_error_mean*1000:.2f}")
    print(f"角度RMSE (°)          | {np.rad2deg(train_angle_rmse):.2f}          | {np.rad2deg(angle_rmse):.2f}")
    print(f"R²                    |        | {r2:.4f}")
    
    # 计算差距
    pos_gap = (pos_error_mean - train_pos_error) / train_pos_error * 100
    angle_gap = (np.rad2deg(angle_rmse) - np.rad2deg(train_angle_rmse)) / np.rad2deg(train_angle_rmse) * 100
    
    print(f"\n相对训练集增加:")
    print(f"  位置误差: +{pos_gap:.1f}%")
    print(f"  角度RMSE: +{angle_gap:.1f}%")

    # 5. 结论
    print(f"\n" + "=" * 70)
    print("结论")
    print("=" * 70)

    if pos_error_mean < 0.005:  # 5mm
        print("  ✓✓ 位置误差优秀 (< 5mm) - 模型泛化能力很强！")
    elif pos_error_mean < 0.01:  # 1cm
        print("  ✓ 位置误差良好 (< 1cm) - 模型泛化良好")
    elif pos_error_mean < 0.02:  # 2cm
        print("  ○ 位置误差可接受 (< 2cm) - 模型基本可用")
    else:
        print("  ✗ 位置误差较大 (> 2cm) - 建议Fine-tune或重新训练")
    
    # 泛化性评级
    if pos_gap < 50 and angle_gap < 50:
        print("  ✓ 泛化性良好")
    elif pos_gap < 100 and angle_gap < 100:
        print("  ○ 泛化性一般")
    else:
        print("  ✗ 泛化性较差")

    print("=" * 70)

    # 保存结果
    output_file = '/home/bonuli/Pieper/pieper1802/grab_test_results.txt'
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GRAB数据集测试结果 (ExplicitCouplingIK)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"测试样本数: {len(all_pred_angles)}\n\n")
        f.write(f"角度误差:\n")
        f.write(f"  RMSE: {angle_rmse:.8f} rad = {np.rad2deg(angle_rmse):.4f}°\n")
        f.write(f"  MAE: {angle_mae:.8f} rad = {np.rad2deg(angle_mae):.4f}°\n")
        f.write(f"  R²: {r2:.6f}\n\n")
        f.write(f"位置误差:\n")
        f.write(f"  平均: {pos_error_mean*1000:.3f} mm\n")
        f.write(f"  中位数: {pos_error_median*1000:.3f} mm\n")
        f.write(f"  标准差: {pos_error_std*1000:.3f} mm\n")
        f.write(f"  最大: {pos_error_max*1000:.3f} mm\n\n")
        f.write("误差分布:\n")
        for p in percentiles:
            value = np.percentile(pos_error * 1000, p)
            f.write(f"  {p}%: {value:.3f} mm\n")

    print(f"\n✓ 结果已保存: {output_file}")


if __name__ == '__main__':
    test_on_grab_data()
