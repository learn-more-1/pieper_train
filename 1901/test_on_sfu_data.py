"""
测试隐式场IK模型在SFU数据集上的泛化性能

测试模型: implicit_ik_1901.pth
测试数据: SFU_G1_training_data.npz
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/1901')

from model import ImplicitIK, NormalizationLayer
from gpu_fk_wrapper import SimpleGPUFK


def load_sfu_dataset(data_path, num_samples=10000):
    """
    加载 SFU 数据集

    格式:
    - X: (N, 10, 14) - 10帧历史，每帧7维位姿+7维角度
    - y: (N, 14) - 目标帧位姿+角度
    """
    print(f"加载 SFU 数据集: {data_path}")
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
    class SFUDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X.astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.float32))

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            # 提取目标位姿和角度
            target_pose = self.y[idx, :7]   # (7,) 目标位姿
            target_angles = self.y[idx, 7:]  # (7,) 目标角度
            last_angle = self.X[idx, -1, 7:]  # (7,) 最后一帧角度（用于GAP计算）

            return (
                target_pose,      # (7,) 目标位姿
                target_angles,    # (7,) 目标角度
                last_angle         # (7,) 最后一帧角度
            )

    dataset = SFUDataset(X, y)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return loader


def test_on_sfu_data():
    """在 SFU 数据集上测试隐式场IK模型"""

    print("=" * 70)
    print("测试隐式场IK模型在SFU数据集上的泛化性能")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ==================== 加载模型 ====================
    model_path = "/home/bonuli/Pieper/1901/implicit_ik_2001.pth"

    print(f"\n加载模型: {model_path}")
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)

    # 获取配置
    config = checkpoint.get('config', {})
    pose_dim = config.get('pose_dim', 7)
    joint_dim = config.get('joint_dim', 7)
    hidden_dim = config.get('hidden_dim', 512)
    num_freqs = config.get('num_freqs', 10)
    use_condition = config.get('use_condition', False)
    use_ensemble = config.get('use_ensemble', False)
    num_models = config.get('num_models', 5)

    print(f"  模型配置:")
    print(f"    pose_dim: {pose_dim}")
    print(f"    joint_dim: {joint_dim}")
    print(f"    hidden_dim: {hidden_dim}")
    print(f"    num_freqs: {num_freqs}")
    print(f"    use_condition: {use_condition}")
    print(f"    use_ensemble: {use_ensemble}")
    print(f"    训练epoch: {checkpoint['epoch']}")
    print(f"    训练验证损失: {checkpoint['best_val_loss']:.6f}")

    # 创建模型
    if use_ensemble:
        from model import ImplicitIKEnsemble
        model = ImplicitIKEnsemble(
            pose_dim=pose_dim,
            joint_dim=joint_dim,
            hidden_dim=hidden_dim,
            num_models=num_models,
            num_freqs=num_freqs
        )
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        from model import ImplicitIK
        model = ImplicitIK(
            pose_dim=pose_dim,
            joint_dim=joint_dim,
            hidden_dim=hidden_dim,
            use_fourier=False,
            num_freqs=num_freqs,
            use_condition=use_condition
        )
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    # 创建归一化层
    norm_layer = NormalizationLayer(
        checkpoint['pose_mean'],
        checkpoint['pose_std'],
        checkpoint['joint_mean'],
        checkpoint['joint_std']
    ).to(device)

    print(f"✓ 模型加载完成")

    # ==================== 加载SFU数据集 ====================
    sfu_data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"
    X, y = load_sfu_dataset(sfu_data_path, num_samples=10000)

    # 创建 DataLoader
    loader = create_dataloader(X, y, batch_size=512)

    # ==================== 加载FK模型 ====================
    print("\n加载FK模型...")

    # 尝试使用本地Pinocchio FK
    try:
        from pinocchio_fk import PinocchioFK as PinoFK
        urdf_path = "/home/bonuli/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
        fk_model = PinoFK(urdf_path)
        fk_model_type = "PinocchioFK"
        print(f"✓ 使用Pinocchio FK (精确)")
    except Exception as e:
        print(f"Pinocchio FK加载失败: {e}")
        print(f"使用SimpleGPU FK...")
        fk_model = SimpleGPUFK()
        fk_model_type = "SimpleGPUFK"

    # ==================== 测试 ====================
    print("\n" + "=" * 70)
    print("开始测试 (10000 帧)")
    print("=" * 70)

    all_pred_angles = []
    all_target_angles = []
    all_pred_pos = []
    all_target_pos = []
    all_gap_errors = []

    with torch.no_grad():
        for batch_idx, (batch_pose, batch_angles, batch_last_angle) in enumerate(loader):
            batch_pose = batch_pose.to(device)
            batch_angles = batch_angles.to(device)
            batch_last_angle = batch_last_angle.to(device)

            # 归一化位姿
            batch_pose_norm = norm_layer.normalize_pose(batch_pose)

            # 预测
            if use_condition:
                pred_angles_norm = model(batch_pose_norm, batch_last_angle)
            else:
                pred_angles_norm = model(batch_pose_norm)

            # 反归一化
            pred_angles = norm_layer.denormalize_joint(pred_angles_norm)

            # FK位置
            if fk_model_type == "PinocchioFK":
                # Pinocchio FK需要numpy输入
                pred_pos = fk_model.forward(pred_angles.cpu())
                target_pos = fk_model.forward(batch_angles.cpu())
            else:
                # SimpleGPU FK
                pred_pos = fk_model.forward(pred_angles)
                target_pos = fk_model.forward(batch_angles)

            # 收集结果
            all_pred_angles.append(pred_angles.cpu().numpy())
            all_target_angles.append(batch_angles.cpu().numpy())
            all_pred_pos.append(pred_pos.cpu().numpy() if isinstance(pred_pos, torch.Tensor) else pred_pos)
            all_target_pos.append(target_pos.cpu().numpy() if isinstance(target_pos, torch.Tensor) else target_pos)

            # GAP 误差
            gap_error = torch.mean((pred_angles - batch_last_angle) ** 2, dim=1)
            all_gap_errors.append(gap_error.cpu().numpy())

            if (batch_idx + 1) % 5 == 0:
                print(f"  处理进度: {(batch_idx + 1) * 512 / 10000 * 100:.1f}%")

    # 合并结果
    all_pred_angles = np.vstack(all_pred_angles)
    all_target_angles = np.vstack(all_target_angles)
    all_pred_pos = np.vstack(all_pred_pos)
    all_target_pos = np.vstack(all_target_pos)
    all_gap_errors = np.concatenate(all_gap_errors)

    # ==================== 计算指标 ====================
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
    train_val_loss = checkpoint['best_val_loss']
    print(f"\n指标          | 训练集验证损失 | SFU测试集")
    print(f"-" * 70)
    print(f"验证损失      | {train_val_loss:.6f}        | {angle_mse:.6f}")
    print(f"位置误差      | -              | {pos_error_mean*1000:.2f} mm")
    print(f"角度RMSE      | -              | {np.rad2deg(angle_rmse):.2f}°")
    print(f"R²            | -              | {r2:.4f}")

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

    # 7. 无历史帧模型特性分析
    print(f"\n" + "=" * 70)
    print("无历史帧模型特性分析")
    print("=" * 70)
    print(f"  - 模型类型: 隐式神经场 (Implicit Neural Field)")
    print(f"  - 输入: 目标位姿 [7] (位置3 + 四元数4)")
    print(f"  - 输出: 关节角度 [7]")
    print(f"  - 特点: 无需历史帧，支持实时推理")
    print(f"  - 编码方式: 位置编码 (Positional Encoding, {num_freqs} 频率)")

    print("=" * 70)

    # ==================== 保存结果 ====================
    result_path = "/home/bonuli/Pieper/1901/sfu_test_results.txt"
    with open(result_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SFU数据集测试结果 - 隐式场IK模型\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"模型: {model_path}\n")
        f.write(f"测试样本数: {len(all_pred_angles)}\n\n")

        f.write(f"模型配置:\n")
        f.write(f"  hidden_dim: {hidden_dim}\n")
        f.write(f"  num_freqs: {num_freqs}\n")
        f.write(f"  use_condition: {use_condition}\n\n")

        f.write(f"角度误差:\n")
        f.write(f"  RMSE: {angle_rmse:.8f} rad = {np.rad2deg(angle_rmse):.4f}°\n")
        f.write(f"  MAE: {angle_mae:.8f} rad = {np.rad2deg(angle_mae):.4f}°\n")
        f.write(f"  R²: {r2:.6f}\n\n")

        f.write(f"每个关节的角度RMSE (度):\n")
        for i, name in enumerate(joint_names):
            error_deg = np.rad2deg(angle_errors_per_joint[i])
            f.write(f"  J{i} ({name}): {error_deg:.4f}°\n")
        f.write("\n")

        f.write(f"位置误差:\n")
        f.write(f"  平均: {pos_error_mean*1000:.3f} mm\n")
        f.write(f"  中位数: {pos_error_median*1000:.3f} mm\n")
        f.write(f"  标准差: {pos_error_std*1000:.3f} mm\n")
        f.write(f"  最大: {pos_error_max*1000:.3f} mm\n\n")

        f.write(f"位置误差分布:\n")
        for p in percentiles:
            value = np.percentile(pos_error * 1000, p)
            f.write(f"  {p}%: {value:.3f} mm\n")

    print(f"\n✓ 结果已保存: {result_path}")


if __name__ == '__main__':
    test_on_sfu_data()
