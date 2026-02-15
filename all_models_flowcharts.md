# Pieper 系列模型流程图合集

## 目录
1. [GNN_Film/causal_ik_model_film.py](#1-gnn_film模型)
2. [pieper0902/causal_ik_model_pieper2.py](#2-pieper0902模型)
3. [pieper1002/causal_ik_model_pieper.py](#3-pieper1002模型)
4. [pieper1003/pieper_direct.py](#4-pieper1003-direct模型)
5. [pieper1003/pieper_sefw.py](#5-pieper1003-sefw模型)

---

## 1. GNN_Film 模型

### 1.1 CausalIKGNN (基础GNN因果IK)

```mermaid
flowchart TD
    A[Input wrist_pose Bx21] --> B[Input Encoder Linear+ReLU]
    B --> C[Global Features Bx256]
    
    C --> D1[Shoulder Embedding]
    C --> D2[Elbow Embedding]
    C --> D3[Forearm Embedding]
    C --> D4[Wrist Embedding]
    
    D1 --> E1[Shoulder Node]
    D2 --> E2[Elbow Node]
    D3 --> E3[Forearm Node]
    D4 --> E4[Wrist Node]
    
    E1 --> F1[shoulder_to_elbow]
    F1 --> E2
    
    E1 & E2 --> F2[elbow_to_forearm Concat+Linear]
    F2 --> E3
    
    E3 --> F3[forearm_to_wrist]
    F3 --> E4
    
    E1 --> G1[Output Head 256->3]
    E2 --> G2[Output Head 256->1]
    E3 --> G3[Output Head 256->1]
    E4 --> G4[Output Head 256->2]
    
    G1 & G2 & G3 & G4 --> H[Concat Bx7]
```

### 1.2 CausalIKGNNv2 (改进版+FiLM)

```mermaid
flowchart TD
    A[Input x history] --> B[Input Encoder GELU+LayerNorm]
    Y[Input y target_pose] --> C[Condition Encoder]
    
    B --> D[Global Features]
    C --> E[Condition Features]
    
    D --> F1[Shoulder Embedding]
    D --> F2[Elbow Embedding]
    D --> F3[Forearm Embedding]
    D --> F4[Wrist Embedding]
    
    F1 --> G1[Shoulder Node]
    F2 --> G2[Elbow Node]
    F3 --> G3[Forearm Node]
    F4 --> G4[Wrist Node]
    
    subgraph MessagePassingLayer
        G1 --> H1[shoulder_to_elbow]
        E --> I1[FiLM Generator]
        I1 --> J1[gamma beta]
        J1 --> K1[Apply FiLM]
        H1 --> K1
        K1 --> G2
        
        G1 & G2 --> H2[to_forearm Concat]
        H2 --> L1[MessagePassingBlock]
        E --> I2[FiLM Generator]
        I2 --> J2[gamma beta]
        J2 --> K2[Apply FiLM]
        L1 --> K2
        G3 & K2 --> M1[Gated Fusion]
        M1 --> G3
        
        G2 & G3 --> H3[forearm_to_wrist Concat]
        H3 --> L2[MessagePassingBlock]
        E --> I3[FiLM Generator]
        I3 --> J3[gamma beta]
        J3 --> K3[Apply FiLM]
        L2 --> K3
        G4 & K3 --> M2[Gated Fusion]
        M2 --> G4
    end
    
    G1 --> N1[Output Head 256->3]
    G2 --> N2[Output Head 256->1]
    G3 --> N3[Output Head 256->1]
    G4 --> N4[Output Head 256->2]
    
    N1 & N2 & N3 & N4 --> O[Concat Bx7]
```

### 1.3 PhysicsAwareCausalIKWithHistory (物理感知+历史)

```mermaid
flowchart TD
    A[history_frames BxTx7] --> B[JointwiseTemporalEncoder]
    
    subgraph JointwiseTemporalEncoder
        B1[Shoulder Conv1d 3->256]
        B2[Elbow Conv1d 1->256]
        B3[Forearm Conv1d 1->256]
        B4[Wrist Conv1d 2->256]
    end
    
    B --> B1 & B2 & B3 & B4
    
    B1 --> C1[Shoulder Features]
    B2 --> C2[Elbow Features]
    B3 --> C3[Forearm Features]
    B4 --> C4[Wrist Features]
    
    C1 & C2 & C3 & C4 --> D[Init Nodes]
    
    D --> E[CausalIKGNNv2]
    Y[target_pose] --> E
    
    E --> F[joint_angles Bx7]
    
    F --> G[FK Approximator]
    G --> H[FK Loss]
    
    D --> I[Joint Coupling Loss]
```

---

## 2. pieper0902 模型 (PieperCausalIK)

**核心特点**: 使用当前关节角度计算Pieper权重，FiLM调制末端位姿

```mermaid
flowchart TD
    A[history_frames BxTx7] --> B[JointwiseTemporalEncoder]
    
    subgraph JointwiseTemporalEncoder
        B1[Shoulder Conv1d]
        B2[Elbow Conv1d]
        B3[Forearm Conv1d]
        B4[Wrist Conv1d]
    end
    
    A --> B1 & B2 & B3 & B4
    B1 --> C1[Shoulder Features]
    B2 --> C2[Elbow Features]
    B3 --> C3[Forearm Features]
    B4 --> C4[Wrist Features]
    
    A --> D[Current Joint Angles last frame]
    D --> E[PieperAttention]
    
    subgraph PieperAttention
        E1[Joint Encoder Linear]
        E2[position_attention Softmax]
        E3[orientation_attention Softmax]
    end
    
    E --> E1 --> E2 & E3
    E2 --> F1[pos_weights Bx7]
    E3 --> F2[ori_weights Bx7]
    
    D & F1 & F2 --> G[PieperFiLMGenerator]
    
    subgraph PieperFiLMGenerator
        G1[Weighted Joints x pos_weights]
        G2[Weighted Joints x ori_weights]
        G3[pos_gamma_net 7->3]
        G4[pos_beta_net 7->3]
        G5[ori_gamma_net 7->4]
        G6[ori_beta_net 7->4]
    end
    
    G1 --> G3 & G4
    G2 --> G5 & G6
    
    P1[end_position Bx3] --> H1[Modulate pos]
    P2[end_orientation Bx4] --> H2[Modulate ori]
    
    G3 & G4 --> H1
    G5 & G6 --> H2
    
    H1 & H2 --> I[EndPoseEncoder]
    I --> J[endeff_feat Bx256]
    
    C1 & C2 & C3 & C4 --> K[Init Nodes]
    J --> K4[Wrist Node]
    
    K --> L[GNN Message Passing]
    
    subgraph GNN
        K1[Shoulder] --> L1[shoulder_to_elbow]
        L1 --> K2[Elbow]
        K1 & K2 --> L2[to_forearm Concat]
        L2 --> M1[forearm_gate Sigmoid]
        M1 --> K3[Forearm]
        K2 & K3 --> L3[forearm_to_wrist Concat]
        L3 --> M2[wrist_gate Sigmoid]
        M2 --> K4
    end
    
    K1 --> N1[Output 256->3]
    K2 --> N2[Output 256->1]
    K3 --> N3[Output 256->1]
    K4 --> N4[Output 256->2]
    
    N1 & N2 & N3 & N4 --> O[pred_angles Bx7]
```

---

## 3. pieper1002 模型 (PieperCausalIK 修正版)

**核心修正**: FiLM调制改为从目标位姿生成参数，调制历史关节特征

```mermaid
flowchart TD
    A[history_frames BxTx7] --> B[JointwiseTemporalEncoder]
    
    subgraph JointwiseTemporalEncoder
        B1[Shoulder Conv1d 3->256]
        B2[Elbow Conv1d 1->256]
        B3[Forearm Conv1d 1->256]
        B4[Wrist Conv1d 2->256]
    end
    
    A --> B1 & B2 & B3 & B4
    B1 --> C1[Shoulder Features]
    B2 --> C2[Elbow Features]
    B3 --> C3[Forearm Features]
    B4 --> C4[Wrist Features]
    
    P1[end_position Bx3] --> D[EndPoseEncoder]
    P2[end_orientation Bx4] --> D
    
    subgraph EndPoseEncoder
        D1[position_encoder 3->256]
        D2[orientation_encoder 4->256]
        D3[Add pos_feat + ori_feat]
    end
    
    D --> D1 & D2 --> D3 --> E[target_feat Bx256]
    
    C1 & C2 & C3 & C4 --> F[Joint Features Dict]
    F & E --> G[TargetConditionedFiLM]
    
    subgraph TargetConditionedFiLM
        G1[For each joint]
        G2[gamma_net target->hidden]
        G3[beta_net target->hidden]
        G4[modulated = gamma x feat + beta]
    end
    
    G --> H1[Modulated Shoulder]
    G --> H2[Modulated Elbow]
    G --> H3[Modulated Forearm]
    G --> H4[Modulated Wrist]
    
    H1 & H2 & H3 & H4 --> I[Init Nodes]
    
    I --> J[GNN Message Passing]
    
    subgraph GNN
        I1[Shoulder] --> J1[shoulder_to_elbow]
        J1 --> I2[Elbow]
        I1 & I2 --> J2[to_forearm Concat]
        J2 --> K1[forearm_gate]
        K1 --> I3[Forearm]
        I2 & I3 --> J3[forearm_to_wrist Concat]
        J3 --> K2[wrist_gate]
        K2 --> I4[Wrist]
    end
    
    I1 --> L1[Output 256->3]
    I2 --> L2[Output 256->1]
    I3 --> L3[Output 256->1]
    I4 --> L4[Output 256->2]
    
    L1 & L2 & L3 & L4 --> M[pred_angles Bx7]
```

---

## 4. pieper1003/pieper_direct.py (直接相乘版)

**核心特点**: 两阶段架构，直接相乘交互（而非FiLM）

```mermaid
flowchart TD
    A[history_frames BxTx7] --> B[JointwiseAttentionEncoder]
    
    subgraph Stage1_PieperAttention
        B1[Shoulder Attention input=3]
        B2[ES Attention input=3]
        B3[WristYaw Attention input=1]
    end
    
    A --> B1 & B2 & B3
    
    B1 --> C1[Shoulder Attn Feat Bx256]
    B2 --> C2[ES Attn Feat Bx256]
    B3 --> C3[WristYaw Attn Feat Bx256]
    
    C1 & C2 & C3 --> D[Joint Features Dict]
    
    P1[end_position Bx3] --> E[DirectInteractionModule]
    P2[end_orientation Bx4] --> E
    D --> E
    
    subgraph Stage2_DirectInteraction
        E1[For each joint]
        E2[pos_interact = feat x pos unsqueeze]
        E3[ori_interact = feat x ori unsqueeze]
        E4[Concat feat + pos_interact + ori_interact]
        E5[Projection Linear 8H->H]
    end
    
    E --> F1[Interacted Shoulder]
    E --> F2[Interacted ES]
    E --> F3[Interacted WristYaw]
    
    F1 --> G1[MLP 256->256->3]
    F2 --> G2[MLP 256->256->3]
    F3 --> G3[MLP 256->256->1]
    
    G1 --> H1[pred_shoulder Bx3]
    G2 --> H2[pred_es Bx3]
    G3 --> H3[pred_wristyaw Bx1]
    
    H1 & H2 & H3 --> I[pred_angles Bx7]
```

**数据维度说明**:
- `pos_interact`: [Bx256x3] -> [Bx768]
- `ori_interact`: [Bx256x4] -> [Bx1024]
- `combined`: [Bx256 + 768 + 1024] = [Bx2048] for shoulder/es

---

## 5. pieper1003/pieper_sefw.py (注意力+FiLM两阶段，无GNN)

**核心特点**: 两阶段架构，移除GNN，直接从FiLM调制特征预测

```mermaid
flowchart TD
    A[history_frames BxTx7] --> B[JointwiseAttentionEncoder]
    
    subgraph Stage1_PieperAttention
        B1[Shoulder MultiHeadAttn input=3 heads=4]
        B2[Elbow MultiHeadAttn input=1 heads=4]
        B3[Forearm MultiHeadAttn input=1 heads=4]
        B4[Wrist MultiHeadAttn input=2 heads=4]
    end
    
    A --> B1 & B2 & B3 & B4
    
    B1 --> C1[Shoulder Attn Feat Bx256]
    B2 --> C2[Elbow Attn Feat Bx256]
    B3 --> C3[Forearm Attn Feat Bx256]
    B4 --> C4[Wrist Attn Feat Bx256]
    
    C1 & C2 & C3 & C4 --> D[Joint Features Dict]
    
    P1[end_position Bx3] --> E[EndPoseEncoder]
    P2[end_orientation Bx4] --> E
    
    subgraph EndPoseEncoder
        E1[position_encoder 3->256]
        E2[orientation_encoder 4->256]
        E3[Add pos_feat + ori_feat]
    end
    
    E --> E1 & E2 --> E3 --> F[target_feat Bx256]
    
    D & F --> G[TargetConditionedFiLM]
    
    subgraph Stage2_FiLM
        G1[For each joint]
        G2[gamma_net Linear+GELU]
        G3[beta_net Linear+GELU]
        G4[modulated = gamma x feat + beta]
    end
    
    G --> H1[Modulated Shoulder]
    G --> H2[Modulated Elbow]
    G --> H3[Modulated Forearm]
    G --> H4[Modulated Wrist]
    
    H1 --> I1[Output Linear 256->3]
    H2 --> I2[Output Linear 256->1]
    H3 --> I3[Output Linear 256->1]
    H4 --> I4[Output Linear 256->2]
    
    I1 --> J1[pred_shoulder Bx3]
    I2 --> J2[pred_elbow Bx1]
    I3 --> J3[pred_forearm Bx1]
    I4 --> J4[pred_wrist Bx2]
    
    J1 & J2 & J3 & J4 --> K[pred_angles Bx7]
```

---

## 模型对比总结

| 模型 | 历史编码 | 目标交互方式 | GNN | 特点 |
|------|----------|--------------|-----|------|
| CausalIKGNN | 无 | 无 | 有 | 基础GNN因果链 |
| CausalIKGNNv2 | 无 | FiLM调制消息 | 有 | FiLM调制消息传递 |
| PhysicsAwareCausalIKWithHistory | Conv1d关节级 | FiLM | 有 | 物理约束+历史 |
| pieper0902 | Conv1d关节级 | Pieper权重+FiLM | 有 | 从当前关节生成FiLM |
| pieper1002 | Conv1d关节级 | FiLM调制特征 | 有 | **修正版**:目标生成FiLM |
| pieper_direct | MultiHeadAttn | 直接相乘 | 无 | 直接交互，无GNN |
| pieper_sefw | MultiHeadAttn | FiLM调制特征 | 无 | 两阶段，无GNN |
| pieper1101 | Conv1d关节级 | Pieper权重+FiLM | 有 | 时序注意力+多特征融合 |

### FiLM公式对比

| 模型 | FiLM公式 | 说明 |
|------|----------|------|
| pieper0902 | `modulated_pose = gamma * end_pose + beta` | 调制末端位姿 |
| pieper1002/pieper_sefw | `modulated_feat = gamma * history_feat + beta` | 调制历史特征 |
| CausalIKGNNv2 | `modulated_msg = gamma * msg + beta` | 调制消息 |

### 历史编码对比

| 模型 | 编码方式 | 说明 |
|------|----------|------|
| Conv1d版本 | Conv1d时序编码 | 轻量、快速 |
| Attention版本 | MultiHeadAttn | 捕捉长程依赖 |
