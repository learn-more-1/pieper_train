# 提升模型泛化性方案

## 当前问题

GRAB数据集上性能显著下降（误差增加200%+）

## 原因

1. **领域差异**: ACCAD/CMU (全身运动) vs GRAB (手部抓取)
2. **姿态范围**: GRAB的姿态可能超出训练分布
3. **末端执行器**: 手 vs 工具/物体交互

## 解决方案

### 方案1: 混合训练（推荐）

将GRAB数据加入训练集：

```python
# 合并多个数据集
accad_cmu = load_data("ACCAD_CMU_merged_training_data.npz")
grab = load_data("GRAB_training_data.npz")

# 混合
mixed_data = concat([accad_cmu, grab])
```

**优点**: 简单直接，效果通常最好
**缺点**: 需要重新训练

### 方案2: 数据增强

在训练时增强数据分布：

```python
# 添加噪声
noise_std = 0.1

# 随机缩放
scale_range = [0.8, 1.2]

# 姿态扰动
rotation_perturb = 0.2  # rad
```

### 方案3: 域适应 (Domain Adaptation)

使用对抗训练或特征对齐：

```python
# 添加域分类器
# 区分源域(ACCAD/CMU)和目标域(GRAB)
# 让特征提取器学习域不变特征
```

### 方案4: 元学习 (Meta-Learning)

训练模型快速适应新领域：

```python
# MAML或类似方法
# 让模型学会"如何学习新映射"
```

### 方案5: 物理约束强化

增强FK loss权重，让模型更关注物理可达性：

```python
# 增加FK权重
fk_weight = 1.0  # 原来是0.5

# 添加关节限位约束
joint_limit_loss = ...
```

## 推荐策略

### 短期（立即生效）

1. **用GRAB数据 fine-tune**
   - 加载已训练模型
   - 用GRAB数据继续训练几轮
   - 冻结部分层（如耦合图）

```python
# Fine-tune示例
model = load_model("explicit_coupling_ik_optimized.pth")

# 冻结耦合图
for param in model.coupling_graph.parameters():
    param.requires_grad = False

# 只训练编码器和解码器
optimizer = Adam([
    {'params': model.intention_encoder.parameters()},
    {'params': model.angle_decoder.parameters()}
], lr=1e-4)

# 用GRAB数据训练
for epoch in range(20):
    train_on_grab(model, optimizer)
```

### 中期（效果最好）

2. **混合数据集重新训练**
   - 合并ACCAD/CMU + GRAB
   - 重新训练完整模型

### 长期（根本解决）

3. **收集更多样化的数据**
   - 包括不同的物体、任务、姿态
   - 或合成数据增强

## 针对GRAB的特殊处理

由于GRAB是手部抓取数据，与全身运动不同：

```python
# 方案A: 针对GRAB微调末端关节
# (因为观察到手部关节已经泛化好)

# 方案B: 增加GRAB特征提取器
# 专门处理手部姿态

# 方案C: 分层控制
# 肩/肘: 使用预训练 (冻结)
# 腕部: 针对GRAB微调
```

## 快速测试 Fine-tune

```bash
# 1. 创建fine-tune脚本
python finetune_on_grab.py \
    --model explicit_coupling_ik_optimized.pth \
    --data GRAB_training_data.npz \
    --epochs 20 \
    --lr 1e-4

# 2. 评估
python test_grab_simple.py \
    --model explicit_coupling_ik_finetuned.pth
```

## 总结

| 方案 | 效果 | 成本 | 推荐度 |
|------|------|------|--------|
| Fine-tune GRAB | 高 | 低 | ⭐⭐⭐ |
| 混合训练 | 很高 | 中 | ⭐⭐⭐⭐ |
| 数据增强 | 中 | 低 | ⭐⭐ |
| 域适应 | 中 | 高 | ⭐ |

**当前建议**: 先用GRAB数据 fine-tune 20轮，观察效果
