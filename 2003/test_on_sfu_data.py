"""
测试 implicit_ik_2003.pth 在 GRAB 数据集上的效果

2003版本：使用10帧历史位姿，避免自回归误差
"""

import torch
import numpy as np
from model import ImplicitIKWithHistory, ImplicitIKWithHistoryEnsemble, NormalizationLayer
import sys
sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/2003')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')
from pinocchio_fk import PinocchioFK


def load_grab_dataset(data_path, num_samples=10000):
    print(f"加载 GRAB 数据集: {data_path}")
    data = np.load(data_path, allow_pickle=True)

    print(f"数据集 Keys: {list(data.keys())}")
    print(f"X shape: {data['X'].shape}")
    print(f"y shape: {data['y'].shape}")

    X = data['X']
    y = data['y']

    total_samples = X.shape[0]
    if total_samples > num_samples:
        indices = np.random.choice(total_samples, num_samples, replace=False)
        X = X[indices]
        y = y[indices]

    print(f"选择 {X.shape[0]} 个样本进行测试")
    return X, y


def create_dataloader(X, y, batch_size=512):
    class GRABDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X.astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.float32))

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            # X: [10, 14] = 10帧历史，每帧7维位姿+7维角度
            # 返回完整X（包含历史位姿）和y
            return self.X[idx], self.y[idx]

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
    print("=" * 70)
    print("测试 implicit_ik_2003.pth (历史位姿版本)")
    print("=" * 70)

    # 加载模型
    checkpoint = torch.load("/home/bonuli/Pieper/2003/implicit_ik_2003.pth", weights_only=False)

    config = checkpoint.get('config', {})
    use_ensemble = config.get('use_ensemble', True)

    pose_mean = checkpoint['pose_mean']
    pose_std = checkpoint['pose_std']
    joint_mean = checkpoint['joint_mean']
    joint_std = checkpoint['joint_std']

    norm_layer = NormalizationLayer(pose_mean, pose_std, joint_mean, joint_std)
    norm_layer = norm_layer.cuda()

    # 创建模型
    if use_ensemble:
        model = ImplicitIKWithHistoryEnsemble(
            pose_dim=config.get('pose_dim', 7),
            joint_dim=config.get('joint_dim', 7),
            hidden_dim=config.get('hidden_dim', 2000),
            temporal_hidden=config.get('temporal_hidden', 256),
            num_freqs=config.get('num_freqs', 15),
            num_frames=config.get('num_frames', 10),
            num_models=config.get('num_models', 8)
        )
    else:
        model = ImplicitIKWithHistory(
            pose_dim=config.get('pose_dim', 7),
            joint_dim=config.get('joint_dim', 7),
            hidden_dim=config.get('hidden_dim', 2000),
            temporal_hidden=config.get('temporal_hidden', 256),
            num_freqs=config.get('num_freqs', 15),
            num_frames=config.get('num_frames', 10)
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()

    print(f"  训练epoch: {checkpoint['epoch']}")
    print(f"  训练验证损失: {checkpoint['best_val_loss']:.6f}")

    # 加载数据集
    grab_data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"
    X, y = load_grab_dataset(grab_data_path, num_samples=10000)

    loader = create_dataloader(X, y, batch_size=512)

    pinocchio_fk = PinocchioFK()

    print("\n" + "=" * 70)
    print("开始测试 (10000 帧)")
    print("=" * 70)

    all_pred_angles = []
    all_target_angles = []
    all_pred_pos = []
    all_target_pos = []

    with torch.no_grad():
        for batch_idx, (batch_X, batch_y) in enumerate(loader):
            batch_X = batch_X.cuda()
            batch_y = batch_y.cuda()

            # ========== 2003关键：从X提取历史位姿 ==========
            # batch_X: [batch, 10, 14]
            history_poses = batch_X[:, :, :7]  # [batch, 10, 7] 历史位姿

            # 提取目标位姿和目标角度
            if batch_y.shape[1] == 14:
                target_pose = batch_y[:, :7]
                target_angles = batch_y[:, 7:]
            else:
                target_angles = batch_y
                target_pose = batch_X[:, -1, :7]

            # 归一化
            target_pose_norm = norm_layer.normalize_pose(target_pose)
            history_poses_norm = norm_layer.normalize_history_poses(history_poses)

            # 预测
            pred_joint_angles_norm = model(target_pose_norm, history_poses_norm)
            pred_angles = norm_layer.denormalize_joint(pred_joint_angles_norm)

            # FK位置
            pred_pos = pinocchio_fk.forward(pred_angles)
            target_pos = pinocchio_fk.forward(target_angles)

            # 收集结果
            all_pred_angles.append(pred_angles.cpu().numpy())
            all_target_angles.append(target_angles.cpu().numpy())
            all_pred_pos.append(pred_pos.cpu().numpy())
            all_target_pos.append(target_pos.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  处理进度: {(batch_idx + 1) * 512 / 10000 * 100:.1f}%")

    # 合并结果
    all_pred_angles = np.vstack(all_pred_angles)
    all_target_angles = np.vstack(all_target_angles)
    all_pred_pos = np.vstack(all_pred_pos)
    all_target_pos = np.vstack(all_target_pos)

    # 计算指标
    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)

    # 角度误差
    angle_mse = np.mean((all_pred_angles - all_target_angles) ** 2)
    angle_rmse = np.sqrt(angle_mse)

    angle_errors_per_joint = np.sqrt(np.mean((all_pred_angles - all_target_angles) ** 2, axis=0))

    print(f"\n【角度预测准确性】")
    print(f"  RMSE: {angle_rmse:.8f} rad = {np.rad2deg(angle_rmse):.4f}°")

    print(f"\n每个关节的角度RMSE (度):")
    joint_names = ['Shoulder Pitch', 'Shoulder Roll', 'Shoulder Yaw',
                   'Elbow', 'Forearm Roll', 'Wrist Yaw', 'Wrist Pitch']
    for i, name in enumerate(joint_names):
        error_deg = np.rad2deg(angle_errors_per_joint[i])
        print(f"  J{i} ({name}): {error_deg:.4f}°")

    ss_res = np.sum((all_target_angles - all_pred_angles) ** 2)
    ss_tot = np.sum((all_target_angles - np.mean(all_target_angles, axis=0)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    print(f"\n  R²: {r2:.6f}")

    # FK位置误差
    pos_error = np.sqrt(np.sum((all_pred_pos - all_target_pos) ** 2, axis=1))
    pos_error_mean = np.mean(pos_error)
    pos_error_median = np.median(pos_error)

    print(f"\n【FK位置准确性】")
    print(f"  平均误差: {pos_error_mean:.6f} m = {pos_error_mean*1000:.3f} mm")
    print(f"  中位数:   {pos_error_median:.6f} m = {pos_error_median*1000:.3f} mm")

    # 误差分布
    print(f"\n【位置误差分布】")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(pos_error * 1000, p)
        print(f"  {p}%: {value:.3f} mm")

    print("=" * 70)

    # 保存结果
    with open('/home/bonuli/Pieper/2003/grab_test_results.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GRAB数据集测试结果 (2003历史位姿版本)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"测试样本数: {len(all_pred_angles)}\n\n")
        f.write(f"角度误差:\n")
        f.write(f"  RMSE: {angle_rmse:.8f} rad = {np.rad2deg(angle_rmse):.4f}°\n")
        f.write(f"\n每个关节的角度RMSE (度):\n")
        for i, name in enumerate(joint_names):
            error_deg = np.rad2deg(angle_errors_per_joint[i])
            f.write(f"  J{i} ({name}): {error_deg:.4f}°\n")
        f.write(f"\nR²: {r2:.6f}\n\n")
        f.write(f"位置误差:\n")
        f.write(f"  平均: {pos_error_mean*1000:.3f} mm\n")
        f.write(f"  中位数: {pos_error_median*1000:.3f} mm\n")

    print(f"\n✓ 结果已保存: /home/bonuli/Pieper/2003/grab_test_results.txt")


if __name__ == '__main__':
    test_on_grab_data()
