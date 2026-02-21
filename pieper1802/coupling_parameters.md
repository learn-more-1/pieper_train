# 显式耦合图 - 训练参数详解

## 一句话总结

> **结构固定，参数可学**
> 
> - 哪些关节之间有耦合（边）→ **预定义，不训练**
> - 耦合消息如何传递（MLP权重）→ **训练**
> - 组间如何融合（门控权重）→ **训练**

---

## 1. 耦合图的可训练参数

### 边类型编码 (Edge Embedding)

```python
self.edge_embedding = nn.Embedding(num_edge_types, hidden_dim)
```

**训练什么？**
- 一个嵌入矩阵 `[num_edge_types, hidden_dim]`
- 将边的类型（运动链/功能耦合）编码为向量

**形状示例：**
```
边类型: 0=J0-J1, 1=J1-J2, 2=J2-J3... (运动链)
        10=J0-J3, 11=J1-J3... (功能耦合)
        
嵌入矩阵: [14, 128]  (14种边，每种128维)
```

---

### 消息传递MLPs (核心训练参数)

```python
self.message_mlps = nn.ModuleList([
    nn.Sequential(
        nn.Linear(hidden_dim * 3, hidden_dim),  # 输入: 源+目标+边类型
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim)       # 输出: 消息
    ) for _ in range(num_edge_types)
])
```

**训练什么？**
- **每条边有独立的MLP** (不是共享的！)
- 每个MLP包含：**权重矩阵 W1, W2 + 偏置 b1, b2**

**为什么要独立？**
```
J0→J1 (shoulder内部协调):  学习"shoulder关节如何配合"
J2→J3 (shoulder到elbow):  学习"手臂伸展时如何配合"
J3→J5 (elbow到wrist):     学习"姿态调整时如何配合"

不同的物理关系 → 不同的消息传递方式
```

**参数数量：**
```
边数: 14
每个MLP: 
  - Layer1: (384, 128) + (128,) = 49,280
  - Layer2: (128, 128) + (128,) = 16,512
  - 每个MLP: ~66k 参数

总参数: 14 × 66k ≈ 924k (约90万)
```

---

### 节点更新MLPs

```python
self.node_update = nn.ModuleList([
    nn.Sequential(
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU()
    ) for _ in range(7)  # 每个关节一个
])
```

**训练什么？**
- 每个关节独立的更新网络
- 聚合所有入边消息 + 自身特征 → 更新后的特征

**参数数量：**
```
7个关节 × [(256×128) + 128] ≈ 230k
```

---

## 2. 耦合约束解码器的可训练参数

### 分组解码器

```python
self.group_decoders = nn.ModuleDict({
    'shoulder': nn.Sequential(nn.Linear(128, 128), ..., nn.Linear(128, 3)),
    'elbow':    nn.Sequential(nn.Linear(128, 64), ..., nn.Linear(64, 1)),
    'forearm':  nn.Sequential(nn.Linear(128, 64), ..., nn.Linear(64, 1)),
    'wrist':    nn.Sequential(nn.Linear(128, 128), ..., nn.Linear(128, 2)),
})
```

**训练什么？**
- 各组关节的解码网络权重
- 从耦合特征映射到具体角度

---

### 组间门控融合 (关键！)

```python
self.gates = nn.ModuleDict({
    'elbow_from_shoulder': nn.Sequential(
        nn.Linear(128 * 2, 128),  # 输入: [elbow_feat, shoulder_feat]
        nn.Sigmoid()               # 输出: 门控值 0~1
    ),
    'forearm_from_elbow': nn.Sequential(...),
    'wrist_from_forearm': nn.Sequential(...),
})
```

**训练什么？**
- 门控网络的权重矩阵
- **决定组间耦合的强度**

**门控如何工作？**
```python
gate = Sigmoid(MLP([elbow_feat, shoulder_feat]))  # 0~1之间的值
elbow_output = elbow_feat * gate + shoulder_feat * (1 - gate)

# gate ≈ 1: elbow主要听自己的
# gate ≈ 0: elbow主要听shoulder的
# gate ≈ 0.5: 两者平衡
```

**这就是"耦合强度"的来源！**

---

## 3. 训练参数汇总

```
总参数量: ~1.47M

├─ 显式耦合图: ~1.15M (78%)
│   ├─ 边类型嵌入: ~2k
│   ├─ 消息传递MLPs: ~924k (核心)
│   └─ 节点更新MLPs: ~230k
│
├─ 运动意图编码: ~200k (14%)
│   ├─ 位置编码器: ~100k
│   └─ 姿态编码器: ~100k
│
├─ 耦合约束解码: ~120k (8%)
│   ├─ 分组解码器: ~80k
│   └─ 组间门控: ~40k
│
└─ 其他: ~10k
```

---

## 4. 训练 vs 预定义 的对比

### 预定义的（不训练）

```python
# 图的拓扑结构 - 代码写死的
self.chain_edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6)]
self.coupling_edges = [(0,3), (1,3), (2,5), (3,5)]

# 哪些关节之间有边 → 预定义
# 边的物理含义 → 预定义（运动链/Pieper准则）
```

### 训练的（学习）

```python
# 1. 消息如何传递 (边MLPs的权重)
message = MLP_edge_i([source_feat, target_feat, edge_type_feat])
#        ↑ 这些权重是训练出来的

# 2. 节点如何更新 (节点MLPs的权重)
updated = MLP_node_i([self_feat, aggregated_messages])
#         ↑ 这些权重是训练出来的

# 3. 组间如何融合 (门控网络的权重)
gate = Sigmoid(MLP_gate([child_feat, parent_feat]))
#      ↑ 这些权重是训练出来的
```

---

## 5. 直观理解

### 比喻：社交网络

```
预定义的（不训练）:
- 谁和谁是朋友 → 固定的社交关系图
- 小明-小红是同事
- 小红-小刚是家人

训练的（学习）:
- 消息如何传递 → "同事之间怎么交流"
- 信息如何融合 → "听谁的更多一些"

门控就像:
- 小红做决定时，听同事小明的多少 (0~1)
- 这个"多少"是训练出来的
```

### 比喻：机械臂控制

```
预定义的（机械结构）:
- Shoulder和Elbow物理连接
- Elbow弯曲会影响Wrist姿态（Pieper准则）

训练的（控制策略）:
- Shoulder如何告诉Elbow"我需要你这样配合"
- Elbow听Shoulder的多少（门控值）
- 这个"配合方式"和"听从程度"是训练出来的
```

---

## 6. 关键洞察

### 为什么这种设计有效？

**传统GNN的问题：**
```python
# 所有边共享相同的MLP
message = Shared_MLP([source, target])  # 所有边一样

结果: J0→J1 和 J2→J3 用相同的传递方式
      但它们的物理关系不同！
```

**本模型的改进：**
```python
# 每种边类型有独立的MLP
message_j0_j1 = MLP_type_0([J0, J1])  # shoulder内部协调
message_j2_j3 = MLP_type_2([J2, J3])  # shoulder到elbow

结果: 不同的物理关系 → 不同的消息传递方式
```

### 耦合强度是如何学习的？

```python
# 训练前: 随机初始化
gate_elbow = 0.5  # 随机

# 训练后: 根据数据学习
# 如果发现elbow经常需要跟随shoulder
gate_elbow = 0.3  # 更听shoulder的

# 这个值会显示在输出中:
elbow_shoulder_coupling: 0.30
```

---

## 总结

| 组件 | 预定义 | 训练 | 作用 |
|------|--------|------|------|
| 图拓扑（哪些关节连边） | ✅ | ❌ | 保证物理约束 |
| 边类型编码 | ❌ | ✅ | 区分关系类型 |
| 消息传递MLPs | ❌ | ✅ | 学习如何传递信息 |
| 节点更新MLPs | ❌ | ✅ | 学习如何更新状态 |
| 组间门控 | ❌ | ✅ | 学习耦合强度 |
| 分组解码器 | ❌ | ✅ | 学习映射到角度 |

**核心创新：结构预定义 + 参数可学 = 既有物理约束，又有数据驱动的灵活性**
