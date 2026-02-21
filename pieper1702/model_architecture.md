# 简化IK模型框架对比

## 1. 直接训练模型 (SimplifiedCausalIK)

```
                    ┌─────────────────────────────────────────┐
                    │          直接训练简化模型                 │
                    └─────────────────────────────────────────┘

输入:
┌──────────────────┐         ┌──────────────────┐
│  target_position │         │ target_orientation│
│      [batch, 3]  │         │     [batch, 4]   │
└────────┬─────────┘         └────────┬─────────┘
         │                            │
         └────────────┬───────────────┘
                      │
                      ▼

    ┌─────────────────────────────────────────────┐
    │  Step 1: 关节耦合嵌入 (JointCouplingEmbedding)  │
    │                                              │
    │  coupling_prototype: [7, 256] (可学习参数)     │
    │         ↓                                    │
    │  添加关节组偏置                               │
    │         ↓                                    │
    │  coupling_features: [batch, 7, 256]          │
    └──────────────────────┬──────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────┐
    │  Step 2: 关节位置注意力 (JointAttention)     │
    │                                              │
    │  coupling_features → MLP → 权重              │
    │         ↓                                    │
    │  pos_weights: [batch, 7] (位置影响)          │
    │  ori_weights: [batch, 7] (姿态影响)          │
    └──────────────────────┬──────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
    ┌───────────────────┐    ┌─────────────────────┐
    │  Step 3a: FiLM生成器 │    │ Step 3b: 末端位姿编码 │
    │                     │    │                     │
    │ weighted_features → │    │ target_position →   │
    │   MLP →             │    │   Encoder → pos_feat │
    │   γ_pos, β_pos      │    │ target_orientation→ │
    │   γ_ori, β_ori      │    │   Encoder → ori_feat │
    └─────────┬───────────┘    └─────────┬───────────┘
              │                          │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────┐
              │  Step 4: FiLM调制     │
              │                       │
              │ modulated_pos =       │
              │   γ_pos * pos + β_pos │
              │                       │
              │ modulated_ori =       │
              │   γ_ori * ori + β_ori │
              └───────────┬───────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────┐
    │  Step 5: 关节组解码 (因果链)                  │
    │                                              │
    │  融合耦合特征 + 调制位姿特征:                  │
    │                                              │
    │  shoulder (J0-2) → elbow (J3) →             │
    │    forearm (J4) → wrist (J5-6)               │
    │         │                                     │
    │         ▼                                     │
    │  消息传递 + 门控融合 → 预测关节角度              │
    └──────────────────────┬──────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  pred_angles   │
                  │   [batch, 7]   │
                  └─────────────────┘

损失计算:
    pred_angles vs target_angles (IK损失)
    pred_position vs target_position (FK损失)
```

---

## 2. 蒸馏训练模型 (Distillation)

```
                    ┌─────────────────────────────────────────┐
                    │          蒸馏训练框架                     │
                    └─────────────────────────────────────────┘

输入数据:
┌──────────────────────────────────────────────┐
│         batch_y: [batch, 14]                │
│  [target_pose (7) | target_angles (7)]      │
└──────────────────┬───────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼

┌──────────────────┐  ┌──────────────────┐
│   教师模型        │  │   学生模型        │
│ (Teacher)        │  │ (Student)        │
│                  │  │                  │
│ PieperCausalIK   │  │ SimplifiedCausalIK│
│                  │  │                  │
│ 需要: 历史帧      │  │ 需要: 仅位姿       │
│ [batch, 10, 7]   │  │ [batch, 3+4]     │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         │                     │
    ┌────┴────┐           ┌────┴────┐
    │         │           │         │
    ▼         ▼           ▼         ▼
┌──────┐ ┌──────┐    ┌──────┐ ┌──────┐
│history│ │target│    │      │ │target│
│frames │ │ pose │    │      │ │ pose │
│(用真实 │ │      │    │      │ │(输入)│
│角度)  │ │      │    │      │ │      │
└───┬───┘ └───┬──┘    └───┬──┘ └───┬──┘
    │        │            │        │
    └────┬───┘            │        │
         │                │        │
         ▼                │        │
    ┌─────────┐           │        │
    │ Temporal│           │        │
    │ Encoder │           │        │
    │(10帧→特征)│          │        │
    └────┬────┘           │        │
         │                │        │
         ▼                │        │
    ┌─────────┐           │        │
    │Attention│           │        │
    │+ GNN    │           │        │
    └────┬────┘           │        │
         │                │        │
         ▼                │        │
    ┌─────────┐           │        │
    │FiLM +   │           │        │
    │Decoder  │           │        │
    └────┬────┘           │        │
         │                │        │
         ▼                ▼        ▼
    teacher_output    student_output
     [batch, 7]         [batch, 7]
         │                  │
         │                  │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  蒸馏损失计算      │
         │                  │
         │  soft_loss =     │
         │    MSE(          │
         │      student/T,  │
         │      teacher/T   │
         │    ) * T²        │
         │                  │
         │  hard_loss =     │
         │    MSE(          │
         │      student,    │
         │      target      │
         │    )             │
         │                  │
         │  total_loss =    │
         │    α*soft_loss + │
         │    (1-α)*hard_   │
         │    loss          │
         └──────────────────┘

超参数:
    T (temperature) = 3.0  # 温度，控制软标签平滑度
    α (alpha) = 0.7         # soft_loss 权重
```

---

## 3. 关键区别对比

| 特性 | 直接训练模型 | 蒸馏训练模型 |
|------|-------------|-------------|
| **输入** | 仅目标位姿 | 仅目标位姿 |
| **学习来源** | 真实标签 (ground truth) | 教师模型 + 真实标签 |
| **损失函数** | MSE(student, target) | α·MSE(student/T, teacher/T) + (1-α)·MSE(student, target) |
| **教师模型** | 无 | PieperCausalIK (冻结) |
| **训练速度** | 快 | 较慢 (需要教师推理) |
| **知识迁移** | 从数据学习 | 从教师+数据学习 |

---

## 4. 数据流对比

### 直接训练:
```
target_pose → SimplifiedCausalIK → pred_angles → loss(pred_angles, target_angles)
```

### 蒸馏训练:
```
target_pose → SimplifiedCausalIK → student_angles ─┐
target_pose + history → PieperCausalIK → teacher_angles ─┤
                                                           ├→ distillation_loss
target_angles (ground truth) ─────────────────────────────┘
```

---

## 5. 模型参数对比

| 模型 | 参数量 | 需要历史 | 推理速度 |
|------|-------|---------|---------|
| PieperCausalIK (教师) | 3.39M | ✅ (10帧) | 慢 |
| SimplifiedCausalIK (学生) | 1.52M | ❌ | 快 |
| 压缩比 | 2.2x | - | ~2x |
