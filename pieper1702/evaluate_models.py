"""
统一评估所有IK模型，确保MAE计算方式一致
"""
import sys
import torch
import numpy as np
from gpu_fk_wrapper import SimpleGPUFK

sys.path.insert(0, '/home/bonuli/Pieper/pieper1101')
sys.path.insert(1, '/home/bonuli/Pieper/pieper1702')
sys.path.insert(2, '/home/bonuli/Pieper/pieper1801')

def load_data(data_path, device):
    """加载数据"""
    data = np.load(data_path)
    # 数据格式: [position(3) + orientation(4) + angles(7)] = 14维
    full_data = torch.from_numpy(data['y']).float().to(device)

    # 划分训练集和验证集
    total_samples = len(full_data)
    train_size = int(total_samples * 0.9)
    train_data = full_data[:train_size]
    val_data = full_data[train_size:]

    return train_data, val_data


def evaluate_model(model_path, model_class, val_data, device, model_name=""):
    """统一评估函数"""
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"{'='*60}")

    # 加载模型
    model = model_class().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 加载FK
    gpu_fk = SimpleGPUFK()

    # 评估
    val_mae = 0.0
    val_mse = 0.0
    val_fk_mse = 0.0

    batch_size = 2048
    n_batches = (len(val_data) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(val_data))

            batch_y = val_data[start_idx:end_idx]

            target_pose = batch_y[:, :7]
            target_angles = batch_y[:, 7:]
            target_position = target_pose[:, :3]
            target_orientation = target_pose[:, 3:7]

            pred_angles, _ = model(target_position, target_orientation)

            # 计算MAE和MSE
            mae = torch.nn.functional.l1_loss(pred_angles, target_angles)
            mse = torch.nn.functional.mse_loss(pred_angles, target_angles)

            # FK验证
            pred_position = gpu_fk.forward(pred_angles)
            fk_mse = torch.nn.functional.mse_loss(pred_position, target_position)

            val_mae += mae.item() * len(batch_y)
            val_mse += mse.item() * len(batch_y)
            val_fk_mse += fk_mse.item() * len(batch_y)

    val_mae /= len(val_data)
    val_mse /= len(val_data)
    val_fk_mse /= len(val_data)

    print(f"验证集样本数: {len(val_data)}")
    print(f"角度 MAE:  {val_mae:.6f}")
    print(f"角度 MSE:  {val_mse:.6f}")
    print(f"位置 MSE:  {val_fk_mse:.6f}")

    return val_mae, val_mse, val_fk_mse


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
    print(f"加载数据: {data_path}")

    train_data, val_data = load_data(data_path, device)
    print(f"验证集: {len(val_data)} 样本")

    results = {}

    # 评估改进模型
    from causal_ik_model_improved import ImprovedSimplifiedCausalIK
    mae, mse, fk_mse = evaluate_model(
        "/home/bonuli/Pieper/pieper1702/improved_simplified_ik_best.pth",
        ImprovedSimplifiedCausalIK,
        val_data,
        device,
        "改进模型 (4.42M)"
    )
    results['改进模型'] = {'mae': mae, 'mse': mse, 'fk_mse': fk_mse}

    # 评估大模型
    from causal_ik_model_large import LargeImprovedSimplifiedCausalIK
    mae, mse, fk_mse = evaluate_model(
        "/home/bonuli/Pieper/pieper1801/large_improved_ik_best.pth",
        LargeImprovedSimplifiedCausalIK,
        val_data,
        device,
        "大模型 (24.19M)"
    )
    results['大模型'] = {'mae': mae, 'mse': mse, 'fk_mse': fk_mse}

    # 汇总对比
    print(f"\n{'='*60}")
    print("模型对比汇总")
    print(f"{'='*60}")
    print(f"{'模型':<15} {'MAE':<12} {'MSE':<12} {'FK MSE':<12}")
    print(f"{'-'*60}")

    for name, metrics in results.items():
        print(f"{name:<15} {metrics['mae']:<12.6f} {metrics['mse']:<12.6f} {metrics['fk_mse']:<12.6f}")

    # 找出最优模型
    best_model = min(results.items(), key=lambda x: x[1]['mae'])
    print(f"\n最优模型: {best_model[0]} (MAE: {best_model[1]['mae']:.6f})")
