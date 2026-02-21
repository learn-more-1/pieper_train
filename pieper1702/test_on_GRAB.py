"""
在 GRAB 数据集上测试简化模型

数据格式：
- y 的前 7 个：目标位姿 [3(位置) + 4(四元数)]
- 后 7 个：关节角度
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '/home/bonuli/pieper/casual')
sys.path.insert(0, '/home/bonuli/Pieper/casual/pieper1101')

from causal_ik_model_pieper_simple import SimplifiedCausalIK
from gpu_fk_wrapper import SimpleGPUFK


def load_GRAB_data(data_path):
    """
    加载 GRAB 数据集

    Returns:
        data: dict with keys:
            - X: 历史帧（可能不使用）
            - y: [N, 14] 前7是位姿，后7是角度
    """
    print(f"加载 GRAB 数据集: {data_path}")
    data = np.load(data_path)

    # 打印数据信息
    print(f"  数据键: {list(data.keys())}")
    for key in data.keys():
        print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")

    return data


def test_on_GRAB(model, data, device, num_samples=None):
    """
    在 GRAB 数据上测试模型

    Args:
        model: SimplifiedCausalIK 模型
        data: GRAB 数据
        device: cuda/cpu
        num_samples: 测试样本数量，None 表示全部

    Returns:
        metrics: dict of error metrics
    """
    model.eval()
    gpu_fk = SimpleGPUFK()

    # 获取数据
    if 'y' in data:
        y = data['y']
    elif 'joint_angles' in data:
        y = data['joint_angles']
    else:
        raise ValueError(f"找不到目标数据，可用键: {list(data.keys())}")

    # 如果是字典类型的数据
    if isinstance(y, np.ndarray) and y.ndim == 1:
        # 可能是压缩格式，需要解压
        y = np.load(y[0]) if y[0].endswith('.npz') else y

    # 采样
    if num_samples is not None and num_samples < len(y):
        indices = np.random.choice(len(y), num_samples, replace=False)
        y = y[indices]

    print(f"\n测试数据: {y.shape}")
    print(f"  格式: 前7维位姿 + 后7维角度")

    # 提取数据
    target_pose = torch.from_numpy(y[:, :7]).float().to(device)  # [N, 7]
    target_angles_gt = torch.from_numpy(y[:, 7:14]).float().to(device)  # [N, 7]

    target_position = target_pose[:, :3]  # [N, 3]
    target_orientation = target_pose[:, 3:7]  # [N, 4]

    # 归一化四元数
    target_orientation = target_orientation / target_orientation.norm(dim=1, keepdim=True)

    # 批量推理
    batch_size = 512
    n_batches = (len(y) + batch_size - 1) // batch_size

    pred_angles_list = []
    gt_angles_list = []
    gt_position_list = []
    pred_position_list = []

    print(f"\n开始推理（batch_size={batch_size}, n_batches={n_batches})...")

    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(y))

            batch_pos = target_position[start_idx:end_idx]
            batch_ori = target_orientation[start_idx:end_idx]
            batch_angles_gt = target_angles_gt[start_idx:end_idx]

            # 推理
            pred_angles, info = model(batch_pos, batch_ori)

            # FK 计算位置误差
            pred_position = gpu_fk.forward(pred_angles)
            gt_position = gpu_fk.forward(batch_angles_gt)

            pred_angles_list.append(pred_angles.cpu().numpy())
            gt_angles_list.append(batch_angles_gt.cpu().numpy())
            pred_position_list.append(pred_position.cpu().numpy())
            gt_position_list.append(gt_position.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{n_batches}")

    # 合并结果
    pred_angles = np.vstack(pred_angles_list)
    gt_angles = np.vstack(gt_angles_list)
    pred_position = np.vstack(pred_position_list)
    gt_position = np.vstack(gt_position_list)

    # 计算误差
    print(f"\n计算误差指标...")

    # 1. 关节角度误差
    angle_mae = np.mean(np.abs(pred_angles - gt_angles))
    angle_rmse = np.sqrt(np.mean((pred_angles - gt_angles) ** 2))
    angle_max_error = np.max(np.abs(pred_angles - gt_angles))

    # 每个关节的误差
    joint_names = ['J0(shd_pitch)', 'J1(shd_roll)', 'J2(shd_yaw)', 'J3(elbow)', 'J4(forearm)', 'J5(wrist_0)', 'J6(wrist_1)']

    # 2. 位置误差
    position_mae = np.mean(np.abs(pred_position - gt_position))
    position_rmse = np.sqrt(np.mean((pred_position - gt_position) ** 2))
    position_max_error = np.max(np.abs(pred_position - gt_position))

    # 3. R² 分数
    ss_res = np.sum((gt_angles - pred_angles) ** 2)
    ss_tot = np.sum((gt_angles - np.mean(gt_angles, axis=0)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    # 打印结果
    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)

    print(f"\n[关节角度误差]")
    print(f"  MAE:  {angle_mae:.6f} rad")
    print(f"  RMSE: {angle_rmse:.6f} rad")
    print(f"  Max:  {angle_max_error:.6f} rad")
    print(f"\n  各关节误差:")
    for i, name in enumerate(joint_names):
        mae = np.mean(np.abs(pred_angles[:, i] - gt_angles[:, i]))
        print(f"    {name}: {mae:.6f} rad")

    print(f"\n[末端位置误差]")
    print(f"  MAE:  {position_mae:.6f} m")
    print(f"  RMSE: {position_rmse:.6f} m")
    print(f"  Max:  {position_max_error:.6f} m")

    print(f"\n[R² 分数]")
    print(f"  总体 R²: {r2:.6f}")
    print(f"  各关节 R²:")
    for i, name in enumerate(joint_names):
        ss_res_j = np.sum((gt_angles[:, i] - pred_angles[:, i]) ** 2)
        ss_tot_j = np.sum((gt_angles[:, i] - np.mean(gt_angles[:, i])) ** 2)
        r2_j = 1 - (ss_res_j / (ss_tot_j + 1e-8))
        print(f"    {name}: {r2_j:.6f}")

    # 误差分布
    print(f"\n[角度误差分布]")
    error_per_sample = np.sqrt(np.sum((pred_angles - gt_angles) ** 2, axis=1))
    print(f"  25分位: {np.percentile(error_per_sample, 25):.6f} rad")
    print(f"  50分位(中位数): {np.percentile(error_per_sample, 50):.6f} rad")
    print(f"  75分位: {np.percentile(error_per_sample, 75):.6f} rad")
    print(f"  90分位: {np.percentile(error_per_sample, 90):.6f} rad")
    print(f"  95分位: {np.percentile(error_per_sample, 95):.6f} rad")

    # 位置误差分布
    print(f"\n[位置误差分布]")
    pos_error_per_sample = np.sqrt(np.sum((pred_position - gt_position) ** 2, axis=1))
    print(f"  25分位: {np.percentile(pos_error_per_sample, 25):.6f} m")
    print(f"  50分位(中位数): {np.percentile(pos_error_per_sample, 50):.6f} m")
    print(f"  75分位: {np.percentile(pos_error_per_sample, 75):.6f} m")
    print(f"  90分位: {np.percentile(pos_error_per_sample, 90):.6f} m")
    print(f"  95分位: {np.percentile(pos_error_per_sample, 95):.6f} m")

    print("\n" + "=" * 70)

    return {
        'angle_mae': angle_mae,
        'angle_rmse': angle_rmse,
        'angle_max': angle_max_error,
        'position_mae': position_mae,
        'position_rmse': position_rmse,
        'position_max': position_max_error,
        'r2': r2,
        'pred_angles': pred_angles,
        'gt_angles': gt_angles,
        'pred_position': pred_position,
        'gt_position': gt_position,
    }


def main():
    print("=" * 70)
    print("GRAB 数据集测试：简化 IK 模型")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载 GRAB 数据
    grab_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"
    grab_data = load_GRAB_data(grab_path)

    # 加载模型
    print(f"\n加载简化模型...")
    model = SimplifiedCausalIK(num_joints=7, hidden_dim=256).to(device)

    # 加载权重（如果有）
    checkpoint_path = "/home/bonuli/Pieper/pieper1101/simplified_causal_ik.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ 成功加载模型权重: {checkpoint_path}")
    except:
        print(f"⚠ 未找到权重文件，使用随机初始化")
        print(f"  路径: {checkpoint_path}")

    # 测试
    print(f"\n开始测试...")
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 可以指定测试样本数，None 表示全部
    metrics = test_on_GRAB(model, grab_data, device, num_samples=None)

    print(f"\n测试完成！")
    print(f"  测试样本数: {len(metrics['gt_angles'])}")
    print(f"  关节角度 MAE: {metrics['angle_mae']:.6f} rad")
    print(f"  末端位置 MAE: {metrics['position_mae']:.6f} m")


if __name__ == "__main__":
    main()
