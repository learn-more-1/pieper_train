"""
用原始模型在 GRAB 数据上测试（支持无历史帧）

使用 pieper_causal_ik_1101.pth 权重
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '/home/bonuli/pieper/casual')
sys.path.insert(0, '/home/bonuli/Pieper/casual/pieper1101')

from causal_ik_model_pieper2 import PieperCausalIK
from gpu_fk_wrapper import SimpleGPUFK


def test_on_GRAB_original(use_history=False):
    """
    用原始模型测试 GRAB 数据

    Args:
        use_history: 是否使用历史帧（False则用None）
    """
    print("=" * 70)
    print(f"GRAB 数据集测试：原始模型 (use_history={use_history})")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载数据
    grab_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"
    print(f"\n加载数据: {grab_path}")
    data = np.load(grab_path)
    y = data['y'].astype(np.float32)
    print(f"  数据形状: {y.shape}")

    # 采样部分数据测试（加速）
    max_samples = 100000
    if len(y) > max_samples:
        indices = np.random.choice(len(y), max_samples, replace=False)
        y = y[indices]
        print(f"  采样 {max_samples} 个样本")

    # 加载模型
    print(f"\n加载原始模型...")
    model = PieperCausalIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        num_layers=2
    ).to(device)

    checkpoint_path = "/home/bonuli/Pieper/pieper1101/pieper_causal_ik_1101.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"✓ 成功加载权重")
    print(f"  原始验证损失: {checkpoint.get('best_val_loss', 'N/A')}")

    model.eval()
    gpu_fk = SimpleGPUFK()

    # 提取数据
    target_pose = torch.from_numpy(y[:, :7]).float().to(device)
    target_angles_gt = torch.from_numpy(y[:, 7:14]).float().to(device)

    target_position = target_pose[:, :3]
    target_orientation = target_pose[:, 3:7]

    # 归一化四元数
    target_orientation = target_orientation / target_orientation.norm(dim=1, keepdim=True)

    # 批量推理
    batch_size = 512
    n_batches = (len(y) + batch_size - 1) // batch_size

    pred_angles_list = []
    gt_angles_list = []
    pred_position_list = []
    gt_position_list = []

    print(f"\n开始推理...")
    print(f"  use_history: {use_history}")
    print(f"  batch_size: {batch_size}, n_batches: {n_batches}")

    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(y))

            batch_pos = target_position[start_idx:end_idx]
            batch_ori = target_orientation[start_idx:end_idx]
            batch_angles_gt = target_angles_gt[start_idx:end_idx]

            # 准备历史帧
            if use_history:
                # 使用当前角度作为历史（实际应用中可能无法获取）
                history_frames = batch_angles_gt.unsqueeze(1).repeat(1, 10, 1)
            else:
                # 无历史帧，使用 default_history
                history_frames = None

            # 推理
            pred_angles, info = model(history_frames, batch_pos, batch_ori)

            # FK 计算位置
            pred_position = gpu_fk.forward(pred_angles)
            gt_position = gpu_fk.forward(batch_angles_gt)

            pred_angles_list.append(pred_angles.cpu().numpy())
            gt_angles_list.append(batch_angles_gt.cpu().numpy())
            pred_position_list.append(pred_position.cpu().numpy())
            gt_position_list.append(gt_position.cpu().numpy())

            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{n_batches}")

    # 合并结果
    pred_angles = np.vstack(pred_angles_list)
    gt_angles = np.vstack(gt_angles_list)
    pred_position = np.vstack(pred_position_list)
    gt_position = np.vstack(gt_position_list)

    # 计算误差
    angle_mae = np.mean(np.abs(pred_angles - gt_angles))
    angle_rmse = np.sqrt(np.mean((pred_angles - gt_angles) ** 2))

    position_error = np.sqrt(np.sum((pred_position - gt_position) ** 2, axis=1))
    position_mae = np.mean(position_error)
    position_rmse = np.sqrt(np.mean(position_error ** 2))

    # R² 分数
    ss_res = np.sum((gt_angles - pred_angles) ** 2)
    ss_tot = np.sum((gt_angles - np.mean(gt_angles, axis=0)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    # 打印结果
    print("\n" + "=" * 70)
    print(f"测试结果 (use_history={use_history})")
    print("=" * 70)

    joint_names = ['J0(shd_pitch)', 'J1(shd_roll)', 'J2(shd_yaw)', 'J3(elbow)', 'J4(forearm)', 'J5(wrist_0)', 'J6(wrist_1)']

    print(f"\n[关节角度误差]")
    print(f"  MAE:  {angle_mae:.6f} rad ({np.degrees(angle_mae):.2f}°)")
    print(f"  RMSE: {angle_rmse:.6f} rad ({np.degrees(angle_rmse):.2f}°)")
    print(f"\n  各关节误差:")
    for i, name in enumerate(joint_names):
        mae = np.mean(np.abs(pred_angles[:, i] - gt_angles[:, i]))
        print(f"    {name}: {mae:.6f} rad ({np.degrees(mae):.2f}°)")

    print(f"\n[末端位置误差]")
    print(f"  MAE:  {position_mae:.6f} m ({position_mae*100:.2f} cm)")
    print(f"  RMSE: {position_rmse:.6f} m ({position_rmse*100:.2f} cm)")
    print(f"  Max:  {np.max(position_error):.6f} m ({np.max(position_error)*100:.2f} cm)")

    print(f"\n[R² 分数]")
    print(f"  总体 R²: {r2:.6f}")

    print(f"\n[角度误差分布]")
    error_per_sample = np.sqrt(np.sum((pred_angles - gt_angles) ** 2, axis=1))
    print(f"  中位数: {np.percentile(error_per_sample, 50):.6f} rad ({np.degrees(np.percentile(error_per_sample, 50)):.2f}°)")
    print(f"  90分位: {np.percentile(error_per_sample, 90):.6f} rad ({np.degrees(np.percentile(error_per_sample, 90)):.2f}°)")
    print(f"  95分位: {np.percentile(error_per_sample, 95):.6f} rad ({np.degrees(np.percentile(error_per_sample, 95)):.2f}°)")

    print(f"\n[位置误差分布]")
    print(f"  中位数: {np.percentile(position_error, 50):.6f} m ({np.percentile(position_error, 50)*100:.2f} cm)")
    print(f"  90分位: {np.percentile(position_error, 90):.6f} m ({np.percentile(position_error, 90)*100:.2f} cm)")
    print(f"  95分位: {np.percentile(position_error, 95):.6f} m ({np.percentile(position_error, 95)*100:.2f} cm)")

    print("\n" + "=" * 70)

    return {
        'angle_mae': angle_mae,
        'position_mae': position_mae,
        'r2': r2,
        'use_history': use_history
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--use-history', action='store_true', help='使用历史帧')
    parser.add_argument('--compare', action='store_true', help='对比有历史和无历史')
    args = parser.parse_args()

    if args.compare:
        # 对比测试
        result_with = test_on_GRAB_original(use_history=True)
        result_without = test_on_GRAB_original(use_history=False)

        print("\n" + "=" * 70)
        print("对比结果")
        print("=" * 70)
        print(f"\n有历史:")
        print(f"  角度 MAE: {result_with['angle_mae']:.6f} rad ({np.degrees(result_with['angle_mae']):.2f}°)")
        print(f"  位置 MAE: {result_with['position_mae']:.6f} m ({result_with['position_mae']*100:.2f} cm)")
        print(f"  R²: {result_with['r2']:.6f}")

        print(f"\n无历史:")
        print(f"  角度 MAE: {result_without['angle_mae']:.6f} rad ({np.degrees(result_without['angle_mae']):.2f}°)")
        print(f"  位置 MAE: {result_without['position_mae']:.6f} m ({result_without['position_mae']*100:.2f} cm)")
        print(f"  R²: {result_without['r2']:.6f}")

        angle_degradation = (result_without['angle_mae'] - result_with['angle_mae']) / result_with['angle_mae'] * 100
        pos_degradation = (result_without['position_mae'] - result_with['position_mae']) / result_with['position_mae'] * 100

        print(f"\n性能下降:")
        print(f"  角度: {angle_degradation:+.1f}%")
        print(f"  位置: {pos_degradation:+.1f}%")
        print("=" * 70)
    else:
        test_on_GRAB_original(use_history=args.use_history)
