"""
对比实验：评估无历史场景下的不同模型方案

目标：解决实际应用时没有历史角度信息的问题

对比方案：
1. Baseline (有历史): 原始 PieperCausalIK 模型，使用真实历史
2. 微调版 (无历史): PieperCausalIK + default_history 微调
3. 简化模型: SimplifiedCausalIK (从一开始就不使用历史)
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import time

# 路径设置
sys.path.insert(0, '/home/wsy/Desktop/casual/pieper1101')
sys.path.insert(1, '/home/bonuli/Pieper/pieper1702')

from gpu_fk_wrapper import SimpleGPUFK
from causal_ik_model_pieper2 import PieperCausalIK
from causal_ik_model_pieper_simple import SimplifiedCausalIK

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)


class CompareConfig:
    # 数据路径
    data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data.npz"

    # 模型路径
    baseline_model_path = "/home/bonuli/Pieper/pieper1101/pieper_causal_ik_1101.pth"
    finetuned_model_path = "/home/bonuli/Pieper/pieper1101/pieper_causal_ik_GRAB_adapted.pth"
    simplified_model_path = "/home/bonuli/Pieper/pieper1702/simplified_causal_ik_GRAB2.pth"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 2048
    num_test_batches = 50  # 测试 batch 数量


def load_data(data_path, num_samples=100000):
    """加载测试数据"""
    logging.info(f"加载数据: {data_path}")
    data = np.load(data_path)
    y = data['y'].astype(np.float32)

    # 使用部分数据进行测试
    test_data = y[:num_samples]
    logging.info(f"  测试样本数: {len(test_data)}")

    return test_data


def load_baseline_model(config):
    """加载原始模型（有历史）"""
    logging.info(f"\n{'='*70}")
    logging.info(f"模型 1: Baseline (有历史)")
    logging.info(f"{'='*70}")

    model = PieperCausalIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        num_layers=2
    ).to(config.device)

    checkpoint = torch.load(config.baseline_model_path, map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)  # 允许 default_history 不匹配
    model.eval()

    logging.info(f"  ✓ 模型加载成功")
    logging.info(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model


def load_finetuned_model(config):
    """加载微调后的模型（使用 default_history）"""
    logging.info(f"\n{'='*70}")
    logging.info(f"模型 2: 微调版 (无历史 + default_history)")
    logging.info(f"{'='*70}")

    model = PieperCausalIK(
        num_joints=7,
        num_frames=10,
        hidden_dim=256,
        num_layers=2
    ).to(config.device)

    checkpoint = torch.load(config.finetuned_model_path, map_location=config.device)

    # 确保有 default_history 参数
    if not hasattr(model, 'default_history'):
        model.default_history = nn.Parameter(
            torch.zeros(1, 10, 7, device=config.device),
            requires_grad=True
        )
    else:
        if not isinstance(model.default_history, nn.Parameter):
            model.default_history = nn.Parameter(
                torch.zeros(1, 10, 7, device=config.device),
                requires_grad=True
            )

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    logging.info(f"  ✓ 模型加载成功")
    logging.info(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 检查 default_history 值
    if hasattr(model, 'default_history') and isinstance(model.default_history, nn.Parameter):
        logging.info(f"  default_history 范围: [{model.default_history.min():.4f}, {model.default_history.max():.4f}]")

    return model


def load_simplified_model(config):
    """加载简化模型（无需历史）"""
    logging.info(f"\n{'='*70}")
    logging.info(f"模型 3: 简化模型 (无需历史)")
    logging.info(f"{'='*70}")

    model = SimplifiedCausalIK(
        num_joints=7,
        hidden_dim=256
    ).to(config.device)

    # 尝试加载预训练权重
    try:
        checkpoint = torch.load(config.simplified_model_path, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"  ✓ 加载预训练权重")
    except FileNotFoundError:
        logging.info(f"  ⚠ 未找到预训练权重，使用随机初始化")

    model.eval()

    logging.info(f"  ✓ 模型加载成功")
    logging.info(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model


def evaluate_model(model, test_data, config, model_type="baseline", gpu_fk=None):
    """评估模型性能"""
    if gpu_fk is None:
        gpu_fk = SimpleGPUFK()

    model.eval()
    all_mae = []
    all_mse = []
    all_pos_error = []
    inference_times = []

    n_batches = min(config.num_test_batches, (len(test_data) + config.batch_size - 1) // config.batch_size)

    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * config.batch_size
            end_idx = min((i + 1) * config.batch_size, len(test_data))

            batch_y = torch.from_numpy(test_data[start_idx:end_idx]).to(config.device)

            target_pose = batch_y[:, :7]
            target_angles = batch_y[:, 7:]
            target_position = target_pose[:, :3]
            target_orientation = target_pose[:, 3:7]

            # 计时
            torch.cuda.synchronize() if config.device.type == 'cuda' else None
            start_time = time.time()

            # 根据模型类型调用不同的 forward
            if model_type == "baseline":
                # 使用真实历史
                history = target_angles.unsqueeze(1).repeat(1, 10, 1)
                pred_angles, _ = model(history, target_position, target_orientation)
            elif model_type == "finetuned":
                # 使用 None 历史（模型会用 default_history）
                pred_angles, _ = model(None, target_position, target_orientation)
            else:  # simplified
                # 简化模型不需要历史
                pred_angles, _ = model(target_position, target_orientation)

            torch.cuda.synchronize() if config.device.type == 'cuda' else None
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # 计算误差
            mae = nn.functional.l1_loss(pred_angles, target_angles).item()
            mse = nn.functional.mse_loss(pred_angles, target_angles).item()

            # 位置误差 (通过 FK 计算)
            pred_position = gpu_fk.forward(pred_angles)
            target_position_fk = gpu_fk.forward(target_angles)
            pos_error = nn.functional.mse_loss(pred_position, target_position_fk).item()

            all_mae.append(mae * len(batch_y))
            all_mse.append(mse * len(batch_y))
            all_pos_error.append(pos_error * len(batch_y))

    # 加权平均
    total_samples = 0
    for i in range(n_batches):
        start_idx = i * config.batch_size
        end_idx = min((i + 1) * config.batch_size, len(test_data))
        total_samples += (end_idx - start_idx)

    avg_mae = sum(all_mae) / total_samples
    avg_mse = sum(all_mse) / total_samples
    avg_pos_error = sum(all_pos_error) / total_samples
    avg_inference_time = sum(inference_times) / len(inference_times) / len(batch_y) * 1000  # ms

    return {
        'mae': avg_mae,
        'mse': avg_mse,
        'pos_error': avg_pos_error,
        'inference_time_ms': avg_inference_time
    }


def main():
    config = CompareConfig()

    logging.info("=" * 70)
    logging.info("对比实验：无历史场景下的模型性能评估")
    logging.info("=" * 70)

    # 加载数据
    test_data = load_data(config.data_path, num_samples=100000)

    # 加载 GPU FK
    try:
        gpu_fk = SimpleGPUFK()
        logging.info(f"\n✓ GPU FK 加载成功")
    except:
        logging.error(f"\n✗ GPU FK 加载失败")
        return

    # 加载三个模型
    baseline_model = load_baseline_model(config)
    finetuned_model = load_finetuned_model(config)
    simplified_model = load_simplified_model(config)

    # 评估三个模型
    results = {}

    logging.info(f"\n{'='*70}")
    logging.info(f"开始评估...")
    logging.info(f"{'='*70}")

    # 1. Baseline (有历史)
    logging.info(f"\n评估 Baseline (有历史)...")
    results['baseline'] = evaluate_model(baseline_model, test_data, config, "baseline", gpu_fk)

    # 2. 微调版 (无历史 + default_history)
    logging.info(f"\n评估微调版 (无历史)...")
    results['finetuned'] = evaluate_model(finetuned_model, test_data, config, "finetuned", gpu_fk)

    # 3. 简化模型 (无需历史)
    logging.info(f"\n评估简化模型...")
    results['simplified'] = evaluate_model(simplified_model, test_data, config, "simplified", gpu_fk)

    # 打印对比结果
    logging.info(f"\n{'='*70}")
    logging.info(f"对比结果汇总")
    logging.info(f"{'='*70}")

    print(f"\n{'模型':<25} {'MAE':<12} {'MSE':<12} {'位置误差':<12} {'推理时间(ms)':<15}")
    print("-" * 80)

    model_names = {
        'baseline': 'Baseline (有历史)',
        'finetuned': '微调版 (无历史)',
        'simplified': '简化模型 (无历史)'
    }

    for key, name in model_names.items():
        r = results[key]
        print(f"{name:<25} {r['mae']:<12.6f} {r['mse']:<12.6f} "
              f"{r['pos_error']:<12.6f} {r['inference_time_ms']:<15.4f}")

    # 计算相对性能
    logging.info(f"\n{'='*70}")
    logging.info(f"相对性能分析 (以 Baseline 为参考)")
    logging.info(f"{'='*70}")

    baseline_mae = results['baseline']['mae']

    for key in ['finetuned', 'simplified']:
        name = model_names[key]
        mae = results[key]['mae']
        ratio = mae / baseline_mae
        logging.info(f"  {name}: MAE 是 Baseline 的 {ratio:.2f}x")

    # 推荐
    logging.info(f"\n{'='*70}")
    logging.info(f"推荐建议")
    logging.info(f"{'='*70}")

    # 找出 MAE 最小的无历史方案
    no_history_options = [(k, v) for k, v in results.items() if k != 'baseline']
    best_no_history = min(no_history_options, key=lambda x: x[1]['mae'])

    if best_no_history[0] == 'finetuned':
        logging.info(f"  ✅ 推荐使用: 微调版 (default_history)")
        logging.info(f"     - MAE: {best_no_history[1]['mae']:.6f}")
        logging.info(f"     - 优势: 可以复用原模型架构，性能较好")
    else:
        logging.info(f"  ✅ 推荐使用: 简化模型")
        logging.info(f"     - MAE: {best_no_history[1]['mae']:.6f}")
        logging.info(f"     - 优势: 模型更简单，推理更快")

    # 性能下降分析
    degradation = (results['finetuned']['mae'] - results['baseline']['mae']) / results['baseline']['mae'] * 100
    logging.info(f"\n  无历史性能下降: {degradation:.1f}%")

    if degradation < 10:
        logging.info(f"  ⭐ 性能下降很小，微调版 default_history 方案可行！")
    elif degradation < 50:
        logging.info(f"  ⚠ 性能下降中等，建议考虑重新训练简化模型")
    else:
        logging.info(f"  ❌ 性能下降严重，建议专门训练无历史模型")


if __name__ == "__main__":
    main()
