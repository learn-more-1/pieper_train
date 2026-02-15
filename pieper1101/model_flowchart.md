# PieperCausalIK 模型流程图

## 简化版流程图

```mermaid
flowchart TD
    subgraph Input
        A1[history_frames]
        A2[end_position]
        A3[end_orientation]
    end

    subgraph TemporalEncoder
        B1[Shoulder Encoder]
        B2[Elbow Encoder]
        B3[Forearm Encoder]
        B4[Wrist Encoder]
    end

    subgraph PieperAttention
        C1[Conv1d Temporal Encoding]
        C2[Joint Fusion]
        C3[position_weights]
        C4[orientation_weights]
    end

    subgraph FiLMGenerator
        D1[Weighted Joint Angles]
        D2[Generate FiLM Params]
        D3[pos_gamma_beta]
        D4[ori_gamma_beta]
    end

    subgraph FiLModulation
        E1[Modulate Position]
        E2[Modulate Orientation]
        E3[EndPoseEncoder]
        E4[endeff_feat]
    end

    subgraph MessagePassingGNN
        G1[Init Nodes]
        G2[Shoulder to Elbow]
        G3[SE to Forearm with Gate]
        G4[EF to Wrist with Gate]
    end

    subgraph FeatureFusion
        H1[Gate Fusion GNN + FiLM]
    end

    subgraph OutputHeads
        I1[shoulder 3D]
        I2[elbow 1D]
        I3[forearm 1D]
        I4[wrist 2D]
        I5[concat 7D]
    end

    A1 --> B1 & B2 & B3 & B4
    A1 --> C1
    B1 & B2 & B3 & B4 --> G1
    
    C1 --> C2 --> C3 & C4
    C3 & C4 --> D1 --> D2 --> D3 & D4
    
    D3 --> E1
    D4 --> E2
    A2 --> E1
    A3 --> E2
    E1 & E2 --> E3 --> E4
    
    G1 --> G2 --> G3 --> G4
    G4 --> H1
    E4 --> H1 --> I1 & I2 & I3 & I4 --> I5
```

## 详细版流程图

```mermaid
flowchart TD
    %% 输入层
    A[Input: history_frames batch x num_frames x 7]
    B[Input: end_position batch x 3]
    C[Input: end_orientation batch x 4]

    %% 时序编码
    subgraph JointwiseTemporalEncoder
        D1[Shoulder Conv1d 3->256]
        D2[Elbow Conv1d 1->256]
        D3[Forearm Conv1d 1->256]
        D4[Wrist Conv1d 2->256]
    end

    %% Pieper注意力
    subgraph PieperAttention
        E[Conv1d Joint Encoding]
        F[Linear Joint Fusion]
        G[position_weights Softmax 7]
        H[orientation_weights Softmax 7]
    end

    %% FiLM生成
    subgraph PieperFiLMGenerator
        I1[Weighted Joints x pos_weights]
        I2[Weighted Joints x ori_weights]
        J1[pos_gamma_net Linear 7->3]
        J2[pos_beta_net Linear 7->3]
        J3[ori_gamma_net Linear 7->4]
        J4[ori_beta_net Linear 7->4]
    end

    %% FiLM调制
    subgraph Modulation
        K1[pos_gamma x end_position + pos_beta]
        K2[ori_gamma x end_orientation + ori_beta]
    end

    %% 末端编码
    subgraph EndPoseEncoder
        L1[Position Encoder 3->256]
        L2[Orientation Encoder 4->256]
        L3[Add: endeff_feat]
    end

    %% GNN因果链
    subgraph GNN_Layer
        M1[Node Init from Temporal Features]
        M2[shoulder_to_elbow Linear]
        M3[to_forearm Concat+Linear+Sigmoid Gate]
        M4[forearm_to_wrist Concat+Linear+Sigmoid Gate]
    end

    %% 特征分配
    subgraph FiLM_Feature_Assignment
        N1[shoulder gets pos_feat]
        N2[elbow gets avg feat]
        N3[forearm gets avg feat]
        N4[wrist gets ori_feat]
    end

    %% 融合
    subgraph MultiFeatureFusion
        O1[Gate Fusion for shoulder]
        O2[Gate Fusion for elbow]
        O3[Gate Fusion for forearm]
        O4[Gate Fusion for wrist with endeff_feat]
    end

    %% 输出
    subgraph OutputHeads
        P1[Linear 256->3 shoulder]
        P2[Linear 256->1 elbow]
        P3[Linear 256->1 forearm]
        P4[Linear 256->2 wrist]
        P5[Concat 7]
    end

    %% 连接
    A --> D1 & D2 & D3 & D4
    A --> E
    
    D1 & D2 & D3 & D4 --> M1
    
    E --> F --> G & H
    G --> I1
    H --> I2
    
    I1 --> J1 & J2
    I2 --> J3 & J4
    
    J1 & J2 --> K1
    J3 & J4 --> K2
    
    B --> K1
    C --> K2
    
    K1 --> L1
    K2 --> L2
    L1 & L2 --> L3
    
    L1 --> N1
    L3 --> N2 & N3
    L2 --> N4
    
    M1 --> M2 --> M3 --> M4
    M4 --> O1 & O2 & O3 & O4
    N1 --> O1
    N2 --> O2
    N3 --> O3
    N4 --> O4
    L3 --> O4
    
    O1 --> P1
    O2 --> P2
    O3 --> P3
    O4 --> P4
    P1 & P2 & P3 & P4 --> P5
```

## 数据流说明

| 模块 | 输入 | 输出 | 功能 |
|------|------|------|------|
| JointwiseTemporalEncoder | history_frames (B,T,7) | joint_features dict | 为每个关节组独立编码时序特征 |
| PieperAttention | history_frames (B,T,7) | pos_weights, ori_weights (B,7) | 学习关节对末端位姿的影响权重 |
| PieperFiLMGenerator | weighted_joints | gamma, beta (B,3/4) | 生成FiLM调制参数 |
| EndPoseEncoder | modulated pos/ori | pos_feat, ori_feat (B,256) | 编码调制后的末端位姿 |
| GNN Message Passing | joint_features | updated nodes dict | 沿因果链传播信息 |
| MultiFeatureFusion | GNN+FiLM features | fused_features dict | 门控融合多源特征 |
| OutputHeads | fused_features | pred_angles (B,7) | 预测各关节角度 |
