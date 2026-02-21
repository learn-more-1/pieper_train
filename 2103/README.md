# 对比学习风格 IK (2103版本)

## 核心思想

解决 IK 中的**个性化**问题：不同人有不同的运动习惯（喜欢用肩膀 vs 手肘），但传统的纯轨迹模型学不到这些。

**关键创新**：通过对比学习，让模型学会从**末端轨迹**推断**个人运动风格**，而不需要在推理时依赖历史关节角（避免自回归）。

```
训练阶段：
  history_poses ──→ PoseStyleEncoder ──→ pred_style ──┐
                                                      ├── 对比损失 ──→ 对齐
  history_joints ──→ JointStyleEncoder ─→ true_style ─┘

推理阶段（无自回归）：
  history_poses ──→ PoseStyleEncoder ──→ style ──→ IK 预测
  ✓ 只需要末端轨迹
  ✓ 自动推断个人习惯
```

## 文件结构

```
Pieper/2103/
├── model.py                    # 核心模型定义
│   ├── ContrastiveStyleIK      # 主模型
│   ├── PositionalEncoding      # 位置编码
│   ├── TemporalPoseEncoder     # 时序编码
│   └── NormalizationLayer      # 归一化
├── train_contrastive_ik.py     # 训练脚本
├── test_generalization.py      # 泛化性测试
├── inference_example.py        # 推理示例
└── README.md                   # 本文件
```

## 快速开始

### 1. 训练模型

```bash
cd /home/bonuli/Pieper/2103
python train_contrastive_ik.py contrastive_ik
```

关键超参数（在 `Config` 类中调整）：
- `style_dim`: 风格维度（默认 128）
- `contrastive_weight`: 对比损失权重（默认 0.1）
- `temperature`: 对比学习温度（默认 0.07）

### 2. 测试泛化性

```bash
python test_generalization.py --checkpoint contrastive_ik_2103.pth
```

输出指标：
- **风格相似度**：只用 poses 推断的风格 vs 用 joints 提取的风格（目标 > 0.9）
- **IK误差差距**：只用 poses 的误差 vs 用 joints 的误差（目标 < 5%）

### 3. 运行推理

```bash
python inference_example.py
```

## 模型架构

```
输入：
  - target_pose: [B, 7] 目标位姿
  - history_poses: [B, 10, 7] 末端历史轨迹
  - history_joints: [B, 10, 7] 关节历史（仅训练）

编码器：
  1. PoseEncoder: 正弦位置编码 → [B, 217]
  2. TemporalEncoder: 1D-CNN 编码轨迹 → [B, 256]
  
风格提取（核心）：
  3. PoseStyleEncoder: 从 TemporalEncoder 输出提取风格（学生）→ [B, 128]
  4. JointStyleEncoder: 从 history_joints 直接提取风格（教师）→ [B, 128]
  5. 对比损失: 让 (3) 的输出接近 (4) 的输出

生成：
  6. Generator: [217 + 256 + 128] → [B, 7] 关节角
  7. StyleResidual: 风格对关节的直接调整
```

## 关键设计决策

### 1. 为什么用对比学习而不是直接 MSE？

- **InfoNCE Loss**（对比学习标准）：将 batch 内其他样本作为负样本，强迫模型学习更有判别性的风格表示
- **MSE Loss**（备选）：简单直接，但可能收敛到平凡解

在 `model.py` 中可以切换：
```python
# InfoNCE（推荐）
contrastive_loss = model.compute_contrastive_loss(pred_style, true_style)

# 或简单 MSE
contrastive_loss = model.compute_mse_alignment_loss(pred_style, true_style)
```

### 2. 教师网络需要更新吗？

不需要特别处理，正常反向传播即可。因为：
- JointStyleEncoder 和 PoseStyleEncoder 共享梯度
- 对比损失同时优化两者
- 如果担心，可以冻结教师网络（实验显示不必要）

### 3. 风格维度怎么选？

- **64**: 轻量级，适合简单场景
- **128**（推荐）: 平衡性能和效率
- **256**: 复杂场景，但需要更多数据

## 与 2003 版本的区别

| 特性 | 2003版本 | 2103版本（本） |
|------|----------|----------------|
| 历史输入 | history_poses | history_poses + history_joints（训练） |
| 个性化 | 无 | 有（风格编码器） |
| 自回归 | 无 | 无（关键！） |
| 推理依赖 | 只需要 poses | 只需要 poses |
| 适用场景 | 通用 IK | 需要适应个人习惯的 IK |

## 训练技巧

### 1. 监控风格相似度

训练日志中的 `StyleSim` 列：
- `0.95/0.92`: 训练时相似度 / 验证时相似度
- 两者都应 > 0.85，差距应 < 0.1

### 2. 调参建议

如果验证相似度低（< 0.8）：
```python
# 增加对比损失权重
contrastive_weight = 0.3  # 从 0.1 增加

# 或降低温度（让分布更尖锐）
temperature = 0.03  # 从 0.07 降低
```

如果 IK 误差高：
```python
# 降低对比损失权重，优先保证 IK 精度
contrastive_weight = 0.05

# 增加基础隐藏层
hidden_dim = 1500  # 从 1200 增加
```

### 3. 数据增强

对比学习受益于多样化的风格样本。确保：
- 训练数据包含多个不同人的运动
- 每个人有足够多的样本（>1000帧）

## 扩展思路

### 1. 在线适应（Online Adaptation）

如果新人使用系统，可以用前几帧快速微调：
```python
# 冻结主网络，只训练风格适配器
for param in model.generator.parameters():
    param.requires_grad = False

# 用新人的前 N 帧微调 PoseStyleEncoder
```

### 2. 风格插值

混合两个人的风格：
```python
style_person_a = model.extract_pose_style(history_a)
style_person_b = model.extract_pose_style(history_b)
style_mixed = 0.5 * style_person_a + 0.5 * style_person_b
```

### 3. 风格聚类

用 t-SNE 可视化风格向量，自动发现数据中的运动类型：
```python
# 见 test_generalization.py 中的 visualize_styles 函数
```

## 常见问题

### Q: 为什么验证时还要计算 style_sim？

验证时的 style_sim 是**泛化性指标**：
- 用 `history_poses` 通过 PoseStyleEncoder 得到 pred_style
- 用 `history_joints` 通过 JointStyleEncoder 得到 true_style（仅用于评估）
- 两者越接近，说明模型在推理时能准确推断风格

### Q: 如果新人没有历史数据怎么办？

用**零初始化风格**（风格向量设为零），模型会退化为通用 IK：
```python
style = torch.zeros(batch_size, style_dim)
```

### Q: 风格向量有物理意义吗？

没有显式意义，但可以通过分析发现：
- 不同人的风格向量在空间中聚类
- 相似运动风格的人距离更近
- 可以用 PCA 分析主要变化方向

## 参考

- SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
- MOCO: Momentum Contrast for Unsupervised Visual Representation Learning
