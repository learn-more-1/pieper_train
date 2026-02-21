# 纯位姿输入IK推理指南

## 问题背景

你的模型在训练时使用了**历史关节角度**来学习关节间的耦合关系，但**实际应用时只有目标位姿**，没有历史角度信息。

## 解决方案

提供了两种方案，根据你的实际需求选择：

---

## 方案1: 滑动窗口自回归（推荐 ⭐）

**文件**: `inference_pose_only.py`

**核心思想**: 维护一个历史角度缓冲区，用预测结果自动更新

**优点**: 
- 保留原模型全部能力
- 实现简单，无需重新训练
- 适合实时控制场景

**缺点**:
- 第一次推理时历史是初始化的（零/中性姿态）
- 需要维护状态

### 使用方法

```python
from inference_pose_only import PoseOnlyIKPredictor

# 初始化（只需一次）
predictor = PoseOnlyIKPredictor(
    model_path="/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth",
    default_init='neutral'  # 初始化方式: 'zero', 'neutral', 'random'
)

# 实时循环
while True:
    # 从传感器/规划器获取目标位姿
    target_pos = get_position()      # [x, y, z]
    target_ori = get_orientation()   # [qx, qy, qz, qw] 可选
    
    # IK求解（自动更新内部历史）
    joint_angles = predictor.predict(target_pos, target_ori)
    
    # 发送给机器人
    robot.move(joint_angles)
```

### 关键特性

1. **自动历史更新**: 每次预测后自动用结果更新历史窗口
2. **序列平滑**: 连续调用时自然产生平滑的关节轨迹
3. **可重置**: 切换任务时调用 `predictor.reset_history()`

---

## 方案2: 位姿到历史适配器（真正的纯位姿输入）

**文件**: `pose_to_history_adapter.py`

**核心思想**: 训练一个小网络，将位姿直接映射到历史特征空间

**优点**:
- 真正的纯位姿输入，无状态维护
- 并行推理友好（batch处理）
- 每次推理独立，无累积误差

**缺点**:
- 需要额外训练适配器
- 精度可能略低于方案1

### 使用方法

#### 步骤1: 训练适配器

```bash
cd /home/bonuli/Pieper/pieper1802
python pose_to_history_adapter.py
```

训练约50 epochs，几分钟即可完成。

#### 步骤2: 使用

```python
from pose_to_history_adapter import PurePoseIKPredictor

# 加载模型
predictor = PurePoseIKPredictor(
    ik_model_path="/home/bonuli/Pieper/pieper1802/pieper_causal_ik_1101.pth",
    adapter_path="/home/bonuli/Pieper/pieper1802/pose_to_history_adapter.pth"
)

# 纯位姿输入，无需历史！
joint_angles = predictor.predict(target_position, target_orientation)
```

---

## 方案对比

| 特性 | 方案1: 滑动窗口 | 方案2: 适配器 |
|------|----------------|--------------|
| 是否需要训练 | ❌ 不需要 | ✅ 需要训练适配器 |
| 状态维护 | ✅ 需要 | ❌ 不需要 |
| 首次推理精度 | ⚠️ 依赖初始化 | ✅ 正常 |
| 连续推理精度 | ✅ 高 | ✅ 高 |
| 批量推理 | ⚠️ 需独立初始化 | ✅ 天然支持 |
| 实现复杂度 | 低 | 中 |
| 推荐场景 | 实时控制、轨迹跟踪 | 离线处理、批量求解 |

---

## 实际应用建议

### 场景A: 实时遥操作/视觉伺服
→ **使用方案1（滑动窗口）**

原因：
- 连续帧之间的时间连续性好
- 历史缓冲区自然反映机器人实际运动历史
- 实现简单，计算开销小

```python
predictor = PoseOnlyIKPredictor(model_path)

while running:
    target_pose = camera.get_target_pose()
    angles = predictor.predict(target_pose[:3], target_pose[3:7])
    robot.send_joint_command(angles)
```

### 场景B: 离线轨迹规划/批量求解
→ **使用方案2（适配器）**

原因：
- 轨迹点之间可能不连续
- 需要并行批量处理
- 不希望状态累积误差

```python
predictor = PurePoseIKPredictor(ik_path, adapter_path)

# 批量求解
joint_trajectory = predictor.predict_batch(target_poses[:, :3], 
                                           target_poses[:, 3:7])
```

### 场景C: 单次目标到达（如抓取）
→ **两种方案均可**

如果精度要求高：先用方案2得到初始解，再用数值优化 refinement。

---

## 精度优化技巧

### 1. 方案1的初始化选择

```python
# 如果知道大致姿态范围，用中性姿态初始化
predictor = PoseOnlyIKPredictor(model_path, default_init='neutral')

# 如果完全未知，用零初始化
predictor = PoseOnlyIKPredictor(model_path, default_init='zero')
```

### 2. 迭代优化（方案1支持）

```python
# 对关键帧使用迭代优化
angles = predictor.predict_with_iteration(
    target_pos, target_ori, 
    num_iterations=10
)
```

### 3. 位姿插值

```python
from inference_pose_only import SequenceIKProcessor

# 自动插值使运动平滑
processor = SequenceIKProcessor(
    model_path, 
    smooth_factor=0.3,
    interpolation_steps=3  # 位姿间插入3个中间点
)

trajectory = processor.process_sequence(target_poses)
```

---

## 常见问题

**Q: 第一次推理精度不高怎么办？**
- 方案1：这是正常的，因为历史是初始化的。连续调用会快速收敛。
- 方案2：适配器需要充分训练，确保损失足够低。

**Q: 长时间运行后误差累积？**
- 方案1：定期调用 `predictor.reset_history()` 重置历史
- 方案2：天然无累积误差

**Q: 如何评估推理精度？**

```python
# 使用FK验证
from gpu_fk_wrapper import SimpleGPUFK

gpu_fk = SimpleGPUFK()
pred_pos = gpu_fk.forward(torch.tensor(pred_angles).unsqueeze(0).cuda())
error = torch.norm(pred_pos - target_pos)
print(f"位置误差: {error.item():.6f}m")
```

---

## 总结

| 你的需求 | 推荐方案 |
|---------|---------|
| 实时连续控制 | 方案1 |
| 批量离线计算 | 方案2 |
| 无状态/无记忆要求 | 方案2 |
| 最高精度要求 | 方案1 + 迭代优化 |
| 快速部署（不训练）| 方案1 |
