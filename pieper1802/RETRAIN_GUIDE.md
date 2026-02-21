# 重新训练模型指南

## 为什么要重新训练？

原模型的问题：
- 过度依赖历史帧（历史影响 >> 目标位姿影响）
- 零历史输入 → 零输出（无法纯位姿推理）
- 关节耦合关系被"冻结"在历史依赖中

## 训练方案选择

### 方案1：改进训练策略（推荐）
**文件**: `train_pose_focused_ik.py`

**特点**：
- 使用原模型架构
- 改进训练策略弱化历史依赖
- 训练成本较低（可复用现有数据）

**核心改进**：
1. **历史数据增强**：添加噪声、随机mask、时间dropout
2. **课程学习**：逐步减少历史长度（10帧 → 1帧）
3. **位姿重建任务**：辅助任务增强位姿表示学习
4. **纯位姿验证**：每轮测试零历史输入性能

**启动训练**：
```bash
cd /home/bonuli/Pieper/pieper1802
python train_pose_focused_ik.py pose_focused_ik
```

**监控指标**：
- `Pose-Only Err`: 零历史输入的误差（关键指标，应逐渐降低）
- `Frames`: 当前历史长度（课程学习进度）

---

### 方案2：改进模型架构
**文件**: `pose_focused_ik_model.py` + `train_pose_focused_ik_v2.py`

**特点**：
- 重新设计模型架构
- 位姿流权重更高
- 历史压缩为单帧

**架构改进**：
1. **历史压缩器**：多帧 → 单帧表示
2. **深层位姿编码器**：更强的位姿特征提取
3. **自适应融合**：动态调整历史vs位姿比例
4. **位姿重建头**：辅助任务

**训练脚本**（需要额外创建）：
```python
from pose_focused_ik_model import PoseFocusedIK

model = PoseFocusedIK(num_joints=7, num_frames=10, hidden_dim=256)
# 其余训练流程类似
```

---

## 关键训练技巧

### 1. 课程学习调度

```python
# 前50 epoch：历史从10帧逐步减少到1帧
# 后150 epoch：固定1帧历史

epoch 0-10:  10帧历史
epoch 10-20: 8帧历史
epoch 20-30: 6帧历史
epoch 30-40: 4帧历史
epoch 40-50: 2帧历史
epoch 50+:   1帧历史
```

这样模型被迫逐步学习目标位姿到关节角度的直接映射。

### 2. 数据增强强度

```python
# 历史数据增强参数
noise_std = 0.05    # 高斯噪声标准差（弧度）
mask_prob = 0.1     # 随机mask概率
drop_prob = 0.2     # 时间dropout概率
```

**调整建议**：
- 如果模型仍过度依赖历史 → 增大noise_std和mask_prob
- 如果训练不稳定 → 减小增强强度

### 3. 损失权重平衡

```python
# 损失组成
ik_weight = 1.0           # 关节角度损失（主要）
fk_weight = 0.5           # 位置误差损失
pose_recon_weight = 0.3   # 位姿重建损失（辅助）
history_indep_weight = 0.1 # 历史独立性损失
```

**调整建议**：
- 要提高纯位姿性能 → 增大pose_recon_weight
- 要减少历史依赖 → 增大history_indep_weight

---

## 验证指标

### 关键指标：`Pose-Only Error`

```python
# 测试零历史输入的性能
zero_history = torch.zeros(1, 10, 7)
pred = model(zero_history, target_pos, target_ori)
error = torch.norm(pred - target_angles)
```

**目标值**：
- `< 0.1 rad`: 优秀，可纯位姿推理
- `0.1-0.3 rad`: 良好，需要引导历史
- `> 0.3 rad`: 较差，仍需改进

### 其他指标

| 指标 | 说明 | 目标 |
|------|------|------|
| Val Loss | 验证损失 | < 0.01 |
| IK Loss | 关节角度误差 | < 0.005 |
| FK Loss | 位置误差 | < 0.001 |

---

## 训练流程

### 步骤1：准备数据
确保数据文件路径正确：
```python
data_path = "/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz"
```

### 步骤2：开始训练

```bash
cd /home/bonuli/Pieper/pieper1802

# 方案1：改进训练策略
python train_pose_focused_ik.py pose_focused_ik

# 后台运行
nohup python train_pose_focused_ik.py pose_focused_ik > train.log 2>&1 &
```

### 步骤3：监控训练

```bash
# 实时查看日志
tail -f pose_focused_train.log

# 关键指标
grep "Pose-Only Err" pose_focused_train.log
```

### 步骤4：验证模型

```python
import torch
from pose_focused_ik_model import PoseFocusedIK

model = PoseFocusedIK().cuda()
checkpoint = torch.load("pose_focused_ik.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 测试纯位姿输入
zero_hist = torch.zeros(1, 10, 7).cuda()
target_pos = torch.tensor([[0.5, 0.2, 0.4]]).cuda()

with torch.no_grad():
    pred, _ = model(zero_hist, target_pos, None)

print(f"零历史预测: {pred.cpu().numpy()}")
```

---

## 常见问题

### Q1: 训练后纯位姿误差仍很大？

**可能原因**：
- 课程学习速度太快 → 延长warmup_epochs
- 历史增强不够 → 增大noise_std和mask_prob
- 模型容量不够 → 增大hidden_dim

**解决方案**：
```python
# 调整配置
warmup_epochs = 100  # 原来是50
noise_std = 0.1      # 原来是0.05
hidden_dim = 512     # 原来是256
```

### Q2: 关节耦合关系丢失了？

**可能原因**：
- 过度弱化历史 → 保留了时序信息但丢失了结构
- GNN层数不够 → 增加num_layers

**解决方案**：
```python
# 保持GNN层数
num_layers = 3  # 原来是2

# 或者使用混合策略
# 训练前期保留历史，后期弱化
```

### Q3: 训练不稳定？

**解决方案**：
```python
# 降低学习率
initial_lr = 5e-4  # 原来是1e-3

# 减小增强强度
noise_std = 0.02
mask_prob = 0.05

# 增大batch_size
batch_size = 1024  # 原来是512
```

---

## 方案对比

| 方案 | 训练成本 | 纯位姿性能 | 关节耦合 | 实现难度 |
|------|---------|-----------|---------|---------|
| 原模型 | - | ❌ 差 | ✅ 好 | - |
| 方案1: 改进训练 | 中 | ✅ 良好 | ✅ 保留 | 低 |
| 方案2: 改进架构 | 中 | ✅ 优秀 | ⚠️ 需验证 | 中 |
| 引导历史(不训练) | 无 | ✅ 可用 | ✅ 保留 | 无 |

---

## 推荐流程

1. **短期**（立即使用）：
   - 使用 `inference_working_solution.py` 的引导历史方案

2. **中期**（重新训练）：
   - 运行 `train_pose_focused_ik.py`
   - 监控 `Pose-Only Err` 指标
   - 目标: `< 0.1 rad`

3. **长期**（进一步优化）：
   - 尝试 `pose_focused_ik_model.py` 新架构
   - 收集更多纯位姿-角度配对数据
   - 考虑端到端训练（视觉→位姿→角度）

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `train_pose_focused_ik.py` | 改进训练脚本（推荐） |
| `pose_focused_ik_model.py` | 改进模型架构 |
| `inference_working_solution.py` | 引导历史推理（不训练也能用） |
| `RETRAIN_GUIDE.md` | 本指南 |

---

## 快速开始

```bash
# 1. 进入目录
cd /home/bonuli/Pieper/pieper1802

# 2. 开始训练
python train_pose_focused_ik.py

# 3. 监控日志
tail -f pose_focused_train.log

# 4. 等待收敛（约100-200 epochs）
# 关键指标：Pose-Only Err < 0.1

# 5. 使用新模型
from inference_pose_only import PurePoseIKPredictor
predictor = PurePoseIKPredictor("pose_focused_ik.pth")
angles = predictor.predict(target_position)
```

祝训练顺利！有问题随时问我。
