"""
检查 GAP 损失（连续性损失）的实际数值

GAP = mean((pred_angles - last_angle)^2)
其中：
- pred_angles: 模型预测的下一帧角度
- last_angle: 历史窗口最后一帧的角度（当前状态）
"""

import torch
import numpy as np
from dataset_generalized import create_windowed_dataloaders

class Config:
    data_path = "/home/wsy/Desktop/casual/merged_training_data.npz"
    batch_size = 512
    num_workers = 4
    pin_memory = True

def check_gap_statistics():
    """检查数据集中 last_angle 和 target 的差异"""
    print("=" * 70)
    print("检查 GAP 损失统计")
    print("=" * 70)

    # 加载数据
    train_loader, val_loader = create_windowed_dataloaders(
        Config.data_path,
        Config
    )

    # 检查前100个batch
    gap_values = []
    count = 0

    for batch_X, batch_y, batch_last_angle in train_loader:
        if count >= 100:
            break

        # 计算真实数据中的 gap (target - last_angle)
        batch_gap = torch.mean((batch_y - batch_last_angle) ** 2, dim=1)  # [batch]
        gap_values.extend(batch_gap.cpu().numpy())
        count += 1

    gap_values = np.array(gap_values)

    print(f"\n统计了 {count} 个 batches (共 {len(gap_values)} 个样本)")
    print(f"\n数据集中 target 与 last_angle 的 MSE 差异分布:")
    print(f"  平均值: {np.mean(gap_values):.8f}")
    print(f"  中位数: {np.median(gap_values):.8f}")
    print(f"  标准差: {np.std(gap_values):.8f}")
    print(f"  最小值: {np.min(gap_values):.8f}")
    print(f"  最大值: {np.max(gap_values):.8f}")
    print(f"  25分位: {np.percentile(gap_values, 25):.8f}")
    print(f"  75分位: {np.percentile(gap_values, 75):.8f}")
    print(f"  95分位: {np.percentile(gap_values, 95):.8f}")
    print(f"  99分位: {np.percentile(gap_values, 99):.8f}")

    # 直方图
    print(f"\n分布区间:")
    bins = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, float('inf')]
    labels = ['0-0.00001', '0.00001-0.0001', '0.0001-0.001', '0.001-0.01', '0.01-0.1', '>0.1']
    for i in range(len(bins)-1):
        count_in_bin = np.sum((gap_values >= bins[i]) & (gap_values < bins[i+1]))
        percentage = count_in_bin / len(gap_values) * 100
        print(f"  {labels[i]}: {count_in_bin:8d} ({percentage:5.2f}%)")

    print("\n" + "=" * 70)
    print("结论:")
    print(f"  如果 95% 以上的样本 gap < 0.001，说明相邻帧之间的角度变化很小")
    print(f"  这是正常的！因为人臂运动是连续的，相邻帧（33ms间隔）差异很小")
    print("=" * 70)


if __name__ == '__main__':
    check_gap_statistics()
