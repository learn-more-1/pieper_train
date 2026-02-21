"""
对比测试：蒸馏模型 vs 直接训练模型

分析两个简化模型的误差表现
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '/home/bonuli/pieper/casual')
sys.path.insert(0, '/home/bonuli/Pieper/casual/pieper1101')

from causal_ik_model_pieper_simple import SimplifiedCausalIK
from gpu_fk_wrapper import SimpleGPUFK


def load_model(model_path, device):
    """加载简化模型"""
    model = SimplifiedCausalIK(num_joints=7, hidden_dim=256).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        val_loss = checkpoint.get("best_val_loss", "N/A")
        print(f"  ✓ 加载成功")
        print(f"  验证损失: {val_loss}")
        return model, val_loss
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return None, None


def test_model(model, data, device, batch_size=4096, model_name="Model"):
    """测试模型性能"""
    model.eval()
    gpu_fk = SimpleGPUFK()

    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"{'='*60}")

    # 提取数据
    if isinstance(data, torch.Tensor):
        y = data
    else:
        y = torch.from_numpy(data).float().to(device)

    target_pose = y[:, :7]
    target_angles_gt = y[:, 7:]
    target_position = target_pose[:, :3]
    target_orientation = target_pose[:, 3:7]

    # 归一化四元数
    target_orientation = target_orientation / target_orientation.norm(dim=1, keepdim=True)

    # 批量推理
    n_samples = len(y)
    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"  样本数: {n_samples}")
    print(f"  批次大小: {batch_size}")
    print(f"  批次数: {n_batches}")
    print(f"\n开始推理...")

    pred_angles_list = []
    gt_angles_list = []
    pred_position_list = []
    gt_position_list = []

    with torch.no_grad():
        for i in range(n_batches):
            if (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{n_batches}")

            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            batch_pos = target_position[start_idx:end_idx]
            batch_ori = target_orientation[start_idx:end_idx]
            batch_angles_gt = target_angles_gt[start_idx:end_idx]

            # 推理
            pred_angles, info = model(batch_pos, batch_ori)

            # FK 计算位置
            pred_position = gpu_fk.forward(pred_angles)
            gt_position = gpu_fk.forward(batch_angles_gt)

            pred_angles_list.append(pred_angles.cpu().numpy())
            gt_angles_list.append(batch_angles_gt.cpu().numpy())
            pred_position_list.append(pred_position.cpu().numpy())
            gt_position_list.append(gt_position.cpu().numpy())

    # 合并结果
    pred_angles = np.vstack(pred_angles_list)
    gt_angles = np.vstack(gt_angles_list)
    pred_position = np.vstack(pred_position_list)
    gt_position = np.vstack(gt_position_list)

    # 计算误差
    print(f"\n计算误差...")

    # 1. 关节角度误差
    angle_mae = np.mean(np.abs(pred_angles - gt_angles))
    angle_rmse = np.sqrt(np.mean((pred_angles - gt_angles) ** 2))
    angle_max = np.max(np.abs(pred_angles - gt_angles))

    # 每个关节的误差
    joint_names = ['J0(shd_pitch)', 'J1(shd_roll)', 'J2(shd_yaw)', 'J3(elbow)', 'J4(forearm)', 'J5(wrist_0)', 'J6(wrist_1)']
    joint_maes = [np.mean(np.abs(pred_angles[:, i] - gt_angles[:, i])) for i in range(7)]

    # 2. 位置误差
    pos_error = np.sqrt(np.sum((pred_position - gt_position) ** 2, axis=1))
    position_mae = np.mean(pos_error)
    position_rmse = np.sqrt(np.mean(pos_error ** 2))
    position_max = np.max(pos_error)

    # 3. R² 分数
    ss_res = np.sum((gt_angles - pred_angles) ** 2)
    ss_tot = np.sum((gt_angles - np.mean(gt_angles, axis=0)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    # 4. 各关节 R²
    r2_per_joint = []
    for i in range(7):
        ss_res_j = np.sum((gt_angles[:, i] - pred_angles[:, i]) ** 2)
        ss_tot_j = np.sum((gt_angles[:, i] - np.mean(gt_angles[:, i])) ** 2)
        r2_j = 1 - (ss_res_j / (ss_tot_j + 1e-8))
        r2_per_joint.append(r2_j)

    # 5. 误差分布
    error_per_sample = np.sqrt(np.sum((pred_angles - gt_angles) ** 2, axis=1))

    # 打印结果
    print(f"\n{'='*60}")
    print(f"测试结果: {model_name}")
    print(f"{'='*60}")

    print(f"\n[关节角度误差]")
    print(f"  MAE:  {angle_mae:.6f} rad ({np.degrees(angle_mae):.2f}°)")
    print(f"  RMSE: {angle_rmse:.6f} rad ({np.degrees(angle_rmse):.2f}°)")
    print(f"  Max:  {angle_max:.6f} rad ({np.degrees(angle_max):.2f}°)")
    print(f"\n  各关节误差:")
    for i, name in enumerate(joint_names):
        print(f"    {name}: {joint_maes[i]:.6f} rad ({np.degrees(joint_maes[i]):.2f}°)")

    print(f"\n[末端位置误差]")
    print(f"  MAE:  {position_mae:.6f} m ({position_mae*100:.2f} cm)")
    print(f"  RMSE: {position_rmse:.6f} m ({position_rmse*100:.2f} cm)")
    print(f"  Max:  {position_max:.6f} m ({position_max*100:.2f} cm)")

    print(f"\n[R² 分数]")
    print(f"  总体 R²: {r2:.6f}")
    print(f"  各关节 R²:")
    for i, name in enumerate(joint_names):
        print(f"    {name}: {r2_per_joint[i]:.6f}")

    print(f"\n[角度误差分布]")
    print(f"  25分位: {np.percentile(error_per_sample, 25):.6f} rad ({np.degrees(np.percentile(error_per_sample, 25)):.2f}°)")
    print(f"  50分位: {np.percentile(error_per_sample, 50):.6f} rad ({np.degrees(np.percentile(error_per_sample, 50)):.2f}°)")
    print(f"  75分位: {np.percentile(error_per_sample, 75):.6f} rad ({np.degrees(np.percentile(error_per_sample, 75)):.2f}°)")
    print(f"  90分位: {np.percentile(error_per_sample, 90):.6f} rad ({np.degrees(np.percentile(error_per_sample, 90)):.2f}°)")
    print(f"  95分位: {np.percentile(error_per_sample, 95):.6f} rad ({np.degrees(np.percentile(error_per_sample, 95)):.2f}°)")

    print(f"\n[位置误差分布]")
    print(f"  25分位: {np.percentile(pos_error, 25):.6f} m ({np.percentile(pos_error, 25)*100:.2f} cm)")
    print(f"  50分位: {np.percentile(pos_error, 50):.6f} m ({np.percentile(pos_error, 50)*100:.2f} cm)")
    print(f"  75分位: {np.percentile(pos_error, 75):.6f} m ({np.percentile(pos_error, 75)*100:.2f} cm)")
    print(f"  90分位: {np.percentile(pos_error, 90):.6f} m ({np.percentile(pos_error, 90)*100:.2f} cm)")
    print(f"  95分位: {np.percentile(pos_error, 95):.6f} m ({np.percentile(pos_error, 95)*100:.2f} cm)")

    return {
        'name': model_name,
        'angle_mae': angle_mae,
        'angle_rmse': angle_rmse,
        'position_mae': position_mae,
        'position_rmse': position_rmse,
        'r2': r2,
        'joint_maes': joint_maes,
        'r2_per_joint': r2_per_joint,
    }


def main():
    print("=" * 70)
    print("对比测试：蒸馏模型 vs 直接训练模型")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载数据
    grab_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"
    print(f"\n加载数据: {grab_path}")

    data = np.load(grab_path)
    y = data['y'].astype(np.float32)

    # 采样部分数据测试
    max_samples = 100000
    if len(y) > max_samples:
        indices = np.random.choice(len(y), max_samples, replace=False)
        y = y[indices]
        print(f"  采样 {max_samples} 个样本")

    # 测试蒸馏模型
    distilled_path = "/home/bonuli/Pieper/pieper1702/simplified_causal_ik_distilled.pth"
    print(f"\n[1] 加载蒸馏模型:")
    print(f"  路径: {distilled_path}")
    model_distilled, val_loss_distilled = load_model(distilled_path, device)

    # 测试直接训练模型
    grab_path_model = "/home/bonuli/Pieper/pieper1702/simplified_causal_ik_GRAB.pth"
    print(f"\n[2] 加载直接训练模型:")
    print(f"  路径: {grab_path_model}")
    model_grab, val_loss_grab = load_model(grab_path_model, device)

    # 对比测试
    if model_distilled is not None and model_grab is not None:
        results_distilled = test_model(model_distilled, y, device, model_name="蒸馏模型")
        results_grab = test_model(model_grab, y, device, model_name="直接训练模型")

        # 对比总结
        print("\n" + "=" * 70)
        print("对比总结")
        print("=" * 70)

        print(f"\n| 指标 | 蒸馏模型 | 直接训练 | 差异 |")
        print(f"|------|---------|----------|------|")
        print(f"| 角度 MAE | {results_distilled['angle_mae']:.6f} | {results_grab['angle_mae']:.6f} | {results_distilled['angle_mae']-results_grab['angle_mae']:+.6f} |")
        print(f"| 角度 RMSE | {results_distilled['angle_rmse']:.6f} | {results_grab['angle_rmse']:.6f} | {results_distilled['angle_rmse']-results_grab['angle_rmse']:+.6f} |")
        print(f"| 位置 MAE | {results_distilled['position_mae']:.6f} | {results_grab['position_mae']:.6f} | {results_distilled['position_mae']-results_grab['position_mae']:+.6f} |")
        print(f"| 位置 RMSE | {results_distilled['position_rmse']:.6f} | {results_grab['position_rmse']:.6f} | {results_distilled['position_rmse']-results_grab['position_rmse']:+.6f} |")
        print(f"| R² | {results_distilled['r2']:.6f} | {results_grab['r2']:.6f} | {results_distilled['r2']-results_grab['r2']:+.6f} |")

        print(f"\n各关节 MAE 对比:")
        print(f"| 关节 | 蒸馏模型 | 直接训练 | 差异 |")
        print(f"|------|---------|----------|------|")
        joint_names = ['J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        for i in range(7):
            print(f"| {joint_names[i]} | {results_distilled['joint_maes'][i]:.6f} | {results_grab['joint_maes'][i]:.6f} | {results_distilled['joint_maes'][i]-results_grab['joint_maes'][i]:+.6f} |")

        # 结论
        print(f"\n结论:")
        angle_improve = (results_distilled['angle_mae'] - results_grab['angle_mae']) / results_grab['angle_mae'] * 100
        pos_improve = (results_distilled['position_mae'] - results_grab['position_mae']) / results_grab['position_mae'] * 100

        if angle_improve < 0:
            print(f"  ✓ 蒸馏模型角度误差低 {abs(angle_improve):.1f}%")
        else:
            print(f"  ✗ 直接训练模型角度误差低 {abs(angle_improve):.1f}%")

        if pos_improve < 0:
            print(f"  ✓ 蒸馏模型位置误差低 {abs(pos_improve):.1f}%")
        else:
            print(f"  ✗ 直接训练模型位置误差低 {abs(pos_improve):.1f}%")

    elif model_distilled is not None:
        test_model(model_distilled, y, device, model_name="蒸馏模型")

    elif model_grab is not None:
        test_model(model_grab, y, device, model_name="直接训练模型")

    else:
        print("\n✗ 没有可用的模型进行测试")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
