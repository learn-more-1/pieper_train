"""
测试 pieper_causal_ik_092.pth 在 GRAB 数据集上的效果

随机抽取 10000 帧进行测试
"""

import torch
import numpy as np
from causal_ik_model_pieper2 import PieperCausalIK
from gpu_fk_wrapper import SimpleGPUFK
import sys
sys.path.insert(0, '/home/wsy/Desktop/casual/pieper_NN')
# from pieper_sefw import PieperCausalIK   # FiLM版本（shoulder-es-wristyaw）

def load_grab_dataset(data_path, num_samples=10000):
    """
    加载 GRAB 数据集

    格式应该与 ACCAD/CMU 相同：
    - X: (N, 10, 14) - 10帧历史，每帧7维位姿+7维角度
    - y: (N, 14) - 目标帧位姿+角度
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


def create_dataloader(X, y, batch_size=512):
    """创建 DataLoader"""
    class GRABDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X[:, :, 7:].astype(np.float32))  # 只用角度
            self.y = torch.from_numpy(y[:, 7:].astype(np.float32))  # 目标角度
            self.full_y = torch.from_numpy(y.astype(np.float32))  # 完整y（含位姿）
            self.full_X = torch.from_numpy(X.astype(np.float32))  # 完整X（含位姿+角度）

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            # 提取目标帧的人臂位姿（从完整y）
            target_human_pose = self.full_y[idx, :7]  # (7,) 目标帧的人臂位姿

            return (
                self.X[idx],      # (10, 7) 历史角度
                self.y[idx],      # (7,) 目标角度
                self.X[idx, -1],  # (7,) 最后一帧角度
                target_human_pose  # (7,) 目标人臂位姿
            )

    dataset = GRABDataset(X, y)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    return loader


def test_on_grab_data():
    """在 GRAB 数据集上测试模型"""

    print("=" * 70)
    print("测试 pieper_causal_ik_092.pth 在 GRAB 数据集上的效果")
    print("=" * 70)

    # 加载模型
    model = PieperCausalIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        num_layers=2
    ).cuda()

    checkpoint = torch.load("/home/wsy/Desktop/casual/pieper1101/pieper_causal_ik_1101.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # print(f"\n✓ 加载模型: pieper_causal_ik_092.pth")
    print(f"  训练epoch: {checkpoint['epoch']}")
    print(f"  训练验证损失: {checkpoint['best_val_loss']:.6f}")

    # 加载 GRAB 数据集
    grab_data_path = "/home/wsy/Desktop/casual/GRAB_training_data.npz"
    X, y = load_grab_dataset(grab_data_path, num_samples=30000)

    # 创建 DataLoader
    loader = create_dataloader(X, y, batch_size=512)

    # 加载 GPU FK
    gpu_fk = SimpleGPUFK()

    # 测试
    print("\n" + "=" * 70)
    print("开始测试 (10000 帧)")
    print("=" * 70)

    all_pred_angles = []
    all_target_angles = []
    all_pred_pos = []
    all_target_pos = []
    all_gap_errors = []

    with torch.no_grad():
        for batch_idx, (batch_X, batch_y, batch_last_angle, batch_human_pose) in enumerate(loader):
            batch_X = batch_X.cuda()
            batch_y = batch_y.cuda()
            batch_last_angle = batch_last_angle.cuda()
            batch_human_pose = batch_human_pose.cuda()

            # 预测
            pred_angles, _ = model(
                batch_X,
                batch_human_pose[:, :3],  # 人臂位置
                batch_human_pose[:, 3:7]  # 人臂姿态
            )

            # FK位置
            pred_pos, _ = gpu_fk.forward(pred_angles), None
            target_pos, _ = gpu_fk.forward(batch_y), None

            # 收集结果
            all_pred_angles.append(pred_angles.cpu().numpy())
            all_target_angles.append(batch_y.cpu().numpy())
            all_pred_pos.append(pred_pos.cpu().numpy())
            all_target_pos.append(target_pos.cpu().numpy())

            # GAP 误差
            gap_error = torch.mean((pred_angles - batch_last_angle) ** 2, dim=1)
            all_gap_errors.append(gap_error.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  处理进度: {(batch_idx + 1) * 512 / 10000 * 100:.1f}%")

    # 合并结果
    all_pred_angles = np.vstack(all_pred_angles)
    all_target_angles = np.vstack(all_target_angles)
    all_pred_pos = np.vstack(all_pred_pos)
    all_target_pos = np.vstack(all_target_pos)
    all_gap_errors = np.concatenate(all_gap_errors)

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
                   'Elbow', 'Forearm Roll', 'Wrist Yaw', 'Wrist Pitch']
    for i, name in enumerate(joint_names):
        error_deg = np.rad2deg(angle_errors_per_joint[i])
        print(f"  J{i} ({name}): {error_deg:.4f}°")

    # R²
    ss_res = np.sum((all_target_angles - all_pred_angles) ** 2)
    ss_tot = np.sum((all_target_angles - np.mean(all_target_angles, axis=0)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    print(f"\n  R²: {r2:.6f}")

    # 2. FK位置误差
    pos_error = np.sqrt(np.sum((all_pred_pos - all_target_pos) ** 2, axis=1))
    pos_error_mean = np.mean(pos_error)
    pos_error_median = np.median(pos_error)
    pos_error_std = np.std(pos_error)
    pos_error_max = np.max(pos_error)

    print(f"\n【FK位置准确性】")
    print(f"  平均误差: {pos_error_mean:.6f} m = {pos_error_mean*1000:.3f} mm")
    print(f"  中位数:   {pos_error_median:.6f} m = {pos_error_median*1000:.3f} mm")
    print(f"  标准差:   {pos_error_std:.6f} m = {pos_error_std*1000:.3f} mm")
    print(f"  最大误差: {pos_error_max:.6f} m = {pos_error_max*1000:.3f} mm")

    # 3. 连续性
    gap_mean = np.mean(all_gap_errors)
    gap_median = np.median(all_gap_errors)
    print(f"\n【连续性 (GAP)】")
    print(f"  平均: {gap_mean:.8f}")
    print(f"  中位数: {gap_median:.8f}")

    # 4. 误差分布
    print(f"\n【位置误差分布】")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(pos_error * 1000, p)
        print(f"  {p}%: {value:.3f} mm")

    # 5. 与训练集对比
    print(f"\n" + "=" * 70)
    print("与训练集对比")
    print("=" * 70)
    print(f"\n指标          | 训练集 (ACCAD/CMU) | GRAB测试集")
    print(f"-" * 70)
    print(f"位置误差      | 3.38 mm             | {pos_error_mean*1000:.2f} mm")
    print(f"角度RMSE      | 0.45°               | {np.rad2deg(angle_rmse):.2f}°")
    print(f"R²            | 0.9997              | {r2:.4f}")

    # 6. 结论
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
        print("  ✗ 位置误差较大 (> 2cm) - 可能需要重新训练或调整")

    print("=" * 70)

    # 保存结果
    with open('/home/wsy/Desktop/casual/pieper_NN/grab_test_results.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GRAB数据集测试结果\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"测试样本数: {len(all_pred_angles)}\n\n")
        f.write(f"角度误差:\n")
        f.write(f"  RMSE: {angle_rmse:.8f} rad = {np.rad2deg(angle_rmse):.4f}°\n")
        f.write(f"  R²: {r2:.6f}\n\n")
        f.write(f"位置误差:\n")
        f.write(f"  平均: {pos_error_mean*1000:.3f} mm\n")
        f.write(f"  中位数: {pos_error_median*1000:.3f} mm\n")

    print(f"\n✓ 结果已保存: /home/wsy/Desktop/casual/pieper_NN/grab_test_results.txt")


if __name__ == '__main__':
    test_on_grab_data()
