# Pieper 系列模型 ASCII 流程图

---

## 1. GNN_Film/causal_ik_model_film.py

### 1.1 CausalIKGNN (基础GNN因果IK)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CausalIKGNN                                  │
└─────────────────────────────────────────────────────────────────────┘

输入: wrist_pose [B, 21]
    │
    ▼
┌─────────────────────┐
│   Input Encoder     │
│  Linear + ReLU x2   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Global Features    │
│     [B, 256]        │
└──────────┬──────────┘
           │
    ┌──────┴──────┬───────────────┬───────────────┐
    ▼             ▼               ▼               ▼
┌────────┐   ┌────────┐     ┌──────────┐    ┌────────┐
│Shoulder│   │ Elbow  │     │ Forearm  │    │ Wrist  │
│Embed   │   │ Embed  │     │  Embed   │    │ Embed  │
└───┬────┘   └───┬────┘     └────┬─────┘    └───┬────┘
    │            │               │              │
    ▼            ▼               ▼              ▼
┌────────┐   ┌────────┐     ┌──────────┐    ┌────────┐
│Shoulder│   │ Elbow  │     │ Forearm  │    │ Wrist  │
│ Node   │   │ Node   │     │  Node    │    │ Node   │
│[B,256] │   │[B,256] │     │ [B,256]  │    │[B,256] │
└───┬────┘   └───┬────┘     └────┬─────┘    └───┬────┘
    │            │               │              │
    │            │               │              │
    │            ▼               │              │
    │    ┌───────────────┐       │              │
    └───►│shoulder_to_   │       │              │
         │   elbow       │       │              │
         └───────┬───────┘       │              │
                 │               │              │
                 └───────────────┘              │
                                 │              │
                                 ▼              │
                         ┌───────────────┐      │
                         │  Elbow Node   │◄─────┤
                         │   [B,256]     │      │
                         └───────┬───────┘      │
                                 │              │
    ┌────────────────────────────┘              │
    │    ┌──────────────────────────────────────┘
    │    │
    ▼    ▼
┌─────────────────────────┐
│   elbow_to_forearm      │
│    Concat + Linear      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│      Forearm Node       │
│       [B,256]           │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   forearm_to_wrist      │
│       Linear            │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│       Wrist Node        │
│       [B,256]           │
└─────────────────────────┘

    ┌───────────┬───────────┬───────────┬───────────┐
    ▼           ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌──────────┐ ┌────────┐
│ Output │ │ Output │ │  Output  │ │ Output │
│ 256->3 │ │ 256->1 │ │  256->1  │ │ 256->2 │
└───┬────┘ └───┬────┘ └────┬─────┘ └───┬────┘
    │          │           │           │
    └──────────┴───────────┴───────────┘
                   │
                   ▼
            ┌─────────────┐
            │ pred_angles │
            │   [B, 7]    │
            └─────────────┘
```

### 1.2 CausalIKGNNv2 (改进版+FiLM)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CausalIKGNNv2                                  │
│              (改进: GELU + LayerNorm + FiLM调制消息)                │
└─────────────────────────────────────────────────────────────────────┘

输入 x (历史)          输入 y (目标位姿)
    │                       │
    ▼                       ▼
┌─────────────────┐   ┌─────────────────┐
│  Input Encoder  │   │ Condition Enc   │
│  GELU+LayerNorm │   │  GELU+LayerNorm │
└────────┬────────┘   └────────┬────────┘
         │                     │
         ▼                     ▼
   Global Features      Condition Features
      [B,256]               [B,256]
         │                     │
         │                     │ (用于每层FiLM)
         ▼                     ▼
    [初始化关节节点]
         │
    ┌────┴────┬────────────┬────────────┐
    ▼         ▼            ▼            ▼
 Shoulder    Elbow      Forearm       Wrist
  Node       Node        Node          Node

每层消息传递 (重复 num_layers 次):

┌─────────────────────────────────────────────────────────────────┐
│                     Message Passing Layer                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Shoulder ──────┐                                               │
│    Node         │                                               │
│   [B,256]       ▼                                               │
│              ┌─────────────────┐                                │
│              │shoulder_to_elbow│                                │
│              │MessagePassingBlock                              │
│              └────────┬────────┘                                │
│                       │                                         │
│  Condition ───────►┌──┴──┐                                      │
│  Features          │FiLM │                                      │
│  [B,256]           │Gen  │──► gamma, beta                      │
│                    └──┬──┘                                      │
│                       │                                         │
│                       ▼                                         │
│              ┌─────────────────┐                                │
│              │  Apply FiLM     │  gamma * msg + beta            │
│              │  modulated_msg  │                                │
│              └────────┬────────┘                                │
│                       │                                         │
│                       ▼                                         │
│  Elbow ◄────────── [Add]                                       │
│   Node                                                          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Shoulder + Elbow ──────┐                                       │
│    Concat [B,512]       ▼                                       │
│              ┌─────────────────────────┐                        │
│              │  elbow_to_forearm_msg   │                        │
│              │    MessagePassingBlock  │                        │
│              └─────────────┬───────────┘                        │
│                            │                                    │
│  Condition ────────────►┌──┴──┐                                 │
│  Features               │FiLM │                                 │
│                         │Gen  │──► gamma, beta                 │
│                         └──┬──┘                                 │
│                            │                                    │
│              ┌─────────────┴─────────────┐                     │
│              │      Apply FiLM           │                     │
│              │   modulated_msg           │                     │
│              └─────────────┬─────────────┘                     │
│                            │                                    │
│              ┌─────────────┴─────────────┐                     │
│              │   GatedMessageFusion      │                     │
│              │  Forearm_node * (1-gate)  │                     │
│              │    + msg * gate           │                     │
│              └─────────────┬─────────────┘                     │
│                            │                                    │
│  Forearm ◄─────────────────┘                                    │
│   Node                                                          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Elbow + Forearm ───────┐                                       │
│    Concat [B,512]       ▼                                       │
│              ┌─────────────────────────┐                        │
│              │  forearm_to_wrist_msg   │                        │
│              │    MessagePassingBlock  │                        │
│              └─────────────┬───────────┘                        │
│                            │                                    │
│  Condition ────────────►┌──┴──┐                                 │
│  Features               │FiLM │                                 │
│                         │Gen  │──► gamma, beta                 │
│                         └──┬──┘                                 │
│                            │                                    │
│              ┌─────────────┴─────────────┐                     │
│              │   GatedMessageFusion      │                     │
│              └─────────────┬─────────────┘                     │
│                            │                                    │
│  Wrist ◄───────────────────┘                                    │
│   Node                                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

输出头 (带LayerNorm):
  Shoulder Node ──► LayerNorm ──► Linear 256->3
  Elbow Node    ──► LayerNorm ──► Linear 256->1
  Forearm Node  ──► LayerNorm ──► Linear 256->1
  Wrist Node    ──► LayerNorm ──► Linear 256->2
                      │
                      ▼
                [Concat Bx7]
```

### 1.3 PhysicsAwareCausalIKWithHistory

```
┌─────────────────────────────────────────────────────────────────────┐
│              PhysicsAwareCausalIKWithHistory                        │
│                 (物理感知 + 历史编码)                                │
└─────────────────────────────────────────────────────────────────────┘

输入: history_frames [B, T, 7]
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  JointwiseTemporalEncoder                       │
├─────────────────┬─────────────────┬───────────────┬─────────────┤
│  Shoulder       │     Elbow       │   Forearm     │   Wrist     │
│  Conv1d(3->256) │   Conv1d(1->256)│ Conv1d(1->256)│ Conv1d(2->256)│
│  [B,T,3]->[B,256]│ [B,T,1]->[B,256]│[B,T,1]->[B,256]│[B,T,2]->[B,256]│
└────────┬────────┴────────┬────────┴───────┬───────┴──────┬──────┘
         │                 │                │              │
         ▼                 ▼                ▼              ▼
   Shoulder Feat      Elbow Feat      Forearm Feat     Wrist Feat
      [B,256]          [B,256]          [B,256]         [B,256]
         │                 │                │              │
         └─────────────────┴────────────────┴──────────────┘
                              │
                              ▼
                    [Initialize Nodes Dict]
                              │
                              ▼
              ┌───────────────────────────┐
              │   CausalIKGNNv2 Forward   │
              │   (with condition y)      │
              │                           │
              │   target_pose ────┐       │
              │                   │       │
              │   [Nodes] ────────┼───────┤
              │                   ▼       │
              │           [Message Passing│
              │            with FiLM]     │
              └───────────┬───────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    joint_angles       │
              │       [B,7]           │
              └───────────┬───────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
   ┌────────────┐  ┌────────────┐  ┌────────────┐
   │  FK Approx │  │ Joint Coup │  │ Joint Limit│
   │   Network  │  │  ling Loss │  │   Penalty  │
   │  7->256->3 │  │            │  │            │
   └──────┬─────┘  └────────────┘  └────────────┘
          │
          ▼
   FK Consistency Loss
```

---

## 2. pieper0902/causal_ik_model_pieper2.py

```
┌─────────────────────────────────────────────────────────────────────┐
│                   pieper0902 PieperCausalIK                         │
│         (使用当前关节角度计算Pieper权重，FiLM调制末端位姿)           │
└─────────────────────────────────────────────────────────────────────┘

输入: history_frames [B, T, 7]
    │
    ├──► [JointwiseTemporalEncoder] ──► joint_features dict
    │
    └──► [Take Last Frame] ──► current_joint_angles [B,7]
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       PieperAttention        │
                    │                              │
                    │  Joint Encoder: Linear+GELU  │
                    │         [B,7]->[B,256]       │
                    │              │               │
                    │      ┌───────┴───────┐       │
                    │      ▼               ▼       │
                    │  position_att    orientation_att│
                    │  Softmax->7      Softmax->7  │
                    │      │               │       │
                    │      ▼               ▼       │
                    │  pos_weights     ori_weights │
                    │    [B,7]           [B,7]     │
                    └──────────┬─────────┬─────────┘
                               │         │
                               ▼         ▼
    current_joint_angles ──► [PieperFiLMGenerator]
    pos_weights, ori_weights ──┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
pos_gamma_net   pos_beta_net   ori_gamma_net   ori_beta_net
[7->256->3]     [7->256->3]    [7->256->4]     [7->256->4]
    │               │               │               │
    ▼               ▼               ▼               ▼
pos_gamma       pos_beta        ori_gamma       ori_beta
[B,3]           [B,3]           [B,4]           [B,4]
    │               │               │               │
    └───────┬───────┘               └───────┬───────┘
            │                               │
            ▼                               ▼
    ┌───────────────┐               ┌───────────────┐
    │ end_position  │               │end_orientation│
    │    [B,3]      │               │    [B,4]      │
    └───────┬───────┘               └───────┬───────┘
            │                               │
            ▼                               ▼
    ┌───────────────┐               ┌───────────────┐
    │  pos_gamma *  │               │  ori_gamma *  │
    │  end_pos +    │               │  end_ori +    │
    │  pos_beta     │               │  ori_beta     │
    └───────┬───────┘               └───────┬───────┘
            │                               │
            ▼                               ▼
      modulated_pos                   modulated_ori
          [B,3]                           [B,4]
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │   EndPoseEncoder  │
                  │                   │
                  │  Position Encoder │
                  │     3->256        │
                  │       +           │
                  │ Orientation Enc   │
                  │     4->256        │
                  │       │           │
                  │      Add          │
                  │       ▼           │
                  │  endeff_feat      │
                  │    [B,256]        │
                  └─────────┬─────────┘
                            │
    ┌───────────────────────┴───────────────────────┐
    │                                               │
    ▼                                               ▼
[Init Nodes from joint_features]          [Add to Wrist Node]
    │                                               │
    ▼                                               ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Shoulder │  │  Elbow   │  │ Forearm  │  │  Wrist   │
│  Node    │  │   Node   │  │   Node   │  │   Node   │
│[B,256]   │  │ [B,256]  │  │ [B,256]  │  │ [B,256]  │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │             │
     │             │             │             │
     │             ▼             │             │
     │    ┌───────────────┐      │             │
     └───►│shoulder_to_   │      │             │
          │   elbow       │      │             │
          └───────┬───────┘      │             │
                  │              │             │
                  └──────────────┘             │
                                 │             │
                                 ▼             │
                         ┌───────────────┐     │
                         │  Elbow Node   │◄────┤
                         │   [B,256]     │     │
                         └───────┬───────┘     │
                                 │             │
    ┌────────────────────────────┘             │
    │    ┌─────────────────────────────────────┘
    │    │
    ▼    ▼
┌─────────────────────────┐
│  Shoulder+Elbow Concat  │
│        [B,512]          │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│      to_forearm         │
│   Linear 512->256       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│    forearm_gate         │
│      Sigmoid            │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│  forearm_node = forearm * (1-gate)      │
│                 + forearm_msg * gate    │
└─────────────────────────────────────────┘
            │
            ▼
    ┌───────────────┐
    │ Forearm Node  │
    │   [B,256]     │
    └───────┬───────┘
            │
            ▼
┌─────────────────────────┐
│  Elbow+Forearm Concat   │
│        [B,512]          │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│    forearm_to_wrist     │
│   Linear 512->256       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│    wrist_gate           │
│      Sigmoid            │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│  wrist_node = wrist * (1-gate)          │
│               + wrist_msg * gate        │
└─────────────────────────────────────────┘

输出预测:
  Shoulder Node ──► Linear 256->3
  Elbow Node    ──► Linear 256->1
  Forearm Node  ──► Linear 256->1
  Wrist Node    ──► Linear 256->2
                      │
                      ▼
                pred_angles [B,7]
```

---

## 3. pieper1002/causal_ik_model_pieper.py

```
┌─────────────────────────────────────────────────────────────────────┐
│                   pieper1002 PieperCausalIK                         │
│           (修正版: 从目标位姿生成FiLM，调制历史特征)                 │
└─────────────────────────────────────────────────────────────────────┘

关键修正:
  pieper0902: FiLM(当前关节) -> 调制末端位姿
  pieper1002: FiLM(目标位姿) -> 调制历史特征

输入: history_frames [B, T, 7]
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  JointwiseTemporalEncoder                       │
│  Shoulder  Elbow  Forearm  Wrist                                │
│  Conv1d    Conv1d  Conv1d  Conv1d                               │
│    3->256    1->256  1->256  2->256                             │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │       Joint Features         │
    │  {shoulder, elbow, forearm,  │
    │   wrist} each [B,256]        │
    └──────────────┬───────────────┘
                   │
                   │◄───────────────────────────────────────┐
                   │                                        │
输入: end_position [B,3], end_orientation [B,4]            │
    │                                                      │
    ▼                                                      │
┌────────────────────────────────┐                         │
│        EndPoseEncoder          │                         │
│                                │                         │
│  position_encoder +            │                         │
│  orientation_encoder           │                         │
│     3->256       4->256        │                         │
│       │            │           │                         │
│       └──► Add ◄───┘           │                         │
│            │                   │                         │
│            ▼                   │                         │
│      target_feat [B,256]       │                         │
└────────────┬───────────────────┘                         │
             │                                             │
             ▼                                             │
    ┌──────────────────────────────────────────┐           │
    │       TargetConditionedFiLM              │           │
    │                                          │           │
    │  For each joint in [shoulder, elbow,     │           │
    │           forearm, wrist]:               │           │
    │                                          │           │
    │  ┌────────────────────────────────────┐  │           │
    │  │  gamma_net: Linear+GELU+Linear     │  │           │
    │  │         [B,256] -> [B,256]         │  │           │
    │  │  beta_net:  Linear+GELU+Linear     │  │           │
    │  │         [B,256] -> [B,256]         │  │           │
    │  └───────────┬───────────┬───────────┘  │           │
    │              │           │              │           │
    │              ▼           ▼              │           │
    │           gamma       beta              │           │
    │          [B,256]     [B,256]            │           │
    │              │           │              │           │
    │              └─────┬─────┘              │           │
    │                    │                    │           │
    │                    ▼                    │           │
    │  modulated_feat = gamma * original_feat │           │
    │                   + beta                │           │
    │                                          │           │
    └──────────────────┬───────────────────────┘           │
                       │                                   │
    ┌──────────────────┼───────────────────┐               │
    ▼                  ▼                   ▼               │
Modulated          Modulated          Modulated            │
Shoulder           Elbow              Forearm              │
[B,256]            [B,256]            [B,256]              │
    │                  │                   │               │
    └──────────────────┼───────────────────┘               │
                       │                                   │
                       ▼                                   │
              Modulated Wrist                              │
                [B,256]                                    │
                       │                                   │
                       └───────────────────────────────────┘

[Init Nodes with Modulated Features]
    │
    ▼
[GNN Message Passing - same as pieper0902]
    │
    ▼
[Output Heads]
    │
    ▼
pred_angles [B,7]
```

---

## 4. pieper1003/pieper_direct.py

```
┌─────────────────────────────────────────────────────────────────────┐
│                   pieper_direct PieperCausalIKDirect                │
│              (两阶段: 注意力 + 直接相乘交互，无GNN)                  │
└─────────────────────────────────────────────────────────────────────┘

与FiLM的区别:
  FiLM:    modulated = gamma * feat + beta
  Direct:  interact = feat * pose (element-wise multiply)

输入: history_frames [B, T, 7]
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 1: Pieper Attention                          │
│                   (JointwiseAttentionEncoder)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   history_shoulder [B,T,3]                                      │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────────────┐                                       │
│   │   PieperAttention   │                                       │
│   │                     │                                       │
│   │  Input Projection   │                                       │
│   │    3 -> 256         │                                       │
│   │         │           │                                       │
│   │         ▼           │                                       │
│   │  MultiHeadAttention │                                       │
│   │    heads=4          │                                       │
│   │         │           │                                       │
│   │  Residual + LayerNorm                                       │
│   │         │           │                                       │
│   │  FFN (256->1024->256)                                       │
│   │         │           │                                       │
│   │  Residual + LayerNorm                                       │
│   │         │           │                                       │
│   │  Mean Pooling       │                                       │
│   │         │           │                                       │
│   │         ▼           │                                       │
│   │  shoulder_feat      │                                       │
│   │     [B,256]         │                                       │
│   └─────────┬───────────┘                                       │
│             │                                                   │
│   history_es [B,T,3]                                            │
│   (elbow+forearm+wrist_pitch)                                   │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────────────┐                                       │
│   │   PieperAttention   │                                       │
│   │     (same flow)     │                                       │
│   └─────────┬───────────┘                                       │
│             │                                                   │
│        es_feat                                                  │
│          [B,256]                                                │
│             │                                                   │
│   history_wristyaw [B,T,1]                                      │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────────────┐                                       │
│   │   PieperAttention   │                                       │
│   │     (same flow)     │                                       │
│   └─────────┬───────────┘                                       │
│             │                                                   │
│     wristyaw_feat                                               │
│          [B,256]                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

输入: end_position [B,3], end_orientation [B,4]
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│          Stage 2: Direct Interaction                            │
│              (DirectInteractionModule)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each joint in [shoulder, es, wristyaw]:                    │
│                                                                 │
│  joint_feat [B,256]                                             │
│       │                                                         │
│       ├────────────────────────────────────┐                    │
│       │                                    │                    │
│       ▼                                    ▼                    │
│  ┌────────────────────┐          ┌────────────────────┐         │
│  │ Position Interact  │          │ Orientation Interact│        │
│  │                    │          │                    │         │
│  │ feat [B,256,1]     │          │ feat [B,256,1]     │         │
│  │   x                │          │   x                │         │
│  │ pos [B,1,3]        │          │ ori [B,1,4]        │         │
│  │   =                │          │   =                │         │
│  │ [B,256,3]          │          │ [B,256,4]          │         │
│  │   │                │          │   │                │         │
│  │ Reshape            │          │ Reshape            │         │
│  │   ▼                │          │   ▼                │         │
│  │ [B,768]            │          │ [B,1024]           │         │
│  └────────┬───────────┘          └────────┬───────────┘         │
│           │                               │                     │
│           └───────────────┬───────────────┘                     │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Concat [joint_feat, pos_interact, ori_interact]         │    │
│  │         [B,256]   +   [B,768]    +   [B,1024]           │    │
│  │                    = [B,2048]                           │    │
│  │                                                         │    │
│  │ Projection Linear: 2048 -> 256                          │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                    │
│                            ▼                                    │
│                  interacted_feat [B,256]                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

输出预测 (无GNN):

  interacted_shoulder [B,256]
         │
         ▼
  ┌─────────────────────┐
  │      MLP            │
  │  Linear 256->256    │
  │  GELU               │
  │  Linear 256->3      │
  └─────────┬───────────┘
            │
            ▼
     pred_shoulder [B,3]

  interacted_es [B,256]
         │
         ▼
  ┌─────────────────────┐
  │      MLP            │
  │  Linear 256->256    │
  │  GELU               │
  │  Linear 256->3      │
  └─────────┬───────────┘
            │
            ▼
        pred_es [B,3]

  interacted_wristyaw [B,256]
         │
         ▼
  ┌─────────────────────┐
  │      MLP            │
  │  Linear 256->256    │
  │  GELU               │
  │  Linear 256->1      │
  └─────────┬───────────┘
            │
            ▼
     pred_wristyaw [B,1]

            │
            ▼
  ┌─────────────────────┐
  │      Concat         │
  │  [3 + 3 + 1] = 7    │
  └─────────────────────┘
            │
            ▼
    pred_angles [B,7]
```

---

## 5. pieper1003/pieper_sefw.py

```
┌─────────────────────────────────────────────────────────────────────┐
│                   pieper_sefw PieperCausalIK                        │
│         (两阶段: 注意力 + FiLM，移除GNN)                             │
└─────────────────────────────────────────────────────────────────────┘

核心架构:
  Stage 1: MultiHeadAttention 编码历史
  Stage 2: FiLM调制 (无GNN消息传递)

输入: history_frames [B, T, 7]
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 1: Pieper Attention                          │
│                   (JointwiseAttentionEncoder)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┬─────────────────┬───────────────┬─────────┤
│  │ Shoulder        │ Elbow           │ Forearm       │ Wrist   │
│  │ input_dim=3     │ input_dim=1     │ input_dim=1   │input=2  │
│  │ num_heads=4     │ num_heads=4     │ num_heads=4   │heads=4  │
│  └────────┬────────┴────────┬────────┴───────┬───────┴────┬────┘
│           │                 │                │            │
│           ▼                 ▼                ▼            ▼
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              PieperAttention (each joint)                  │ │
│  │                                                            │ │
│  │  Input Projection: input_dim -> 256                        │ │
│  │                     │                                      │ │
│  │  MultiHeadAttn: heads=4, batch_first                       │ │
│  │                     │                                      │ │
│  │  Residual + LayerNorm                                      │ │
│  │                     │                                      │ │
│  │  FFN: 256 -> 1024 -> 256                                   │ │
│  │                     │                                      │ │
│  │  Residual + LayerNorm                                      │ │
│  │                     │                                      │ │
│  │  Mean Pooling over T                                       │ │
│  │                     │                                      │ │
│  │                     ▼                                      │ │
│  │              [B, 256]                                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│           │                 │                │            │
│           ▼                 ▼                ▼            ▼
│    shoulder_feat      elbow_feat      forearm_feat   wrist_feat
│       [B,256]          [B,256]          [B,256]       [B,256]
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

输入: end_position [B,3], end_orientation [B,4]
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 2: FiLM Modulation                           │
│              (TargetConditionedFiLM)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────┐                    │
│  │         EndPoseEncoder                  │                    │
│  │                                         │                    │
│  │  Position Encoder: 3 -> 128 -> 256      │                    │
│  │  Orientation Enc:  4 -> 128 -> 256      │                    │
│  │                    +                    │                    │
│  │              target_feat                │                    │
│  │               [B,256]                   │                    │
│  └───────────────────┬─────────────────────┘                    │
│                      │                                          │
│  For each joint:     │                                          │
│                      │                                          │
│  ┌───────────────────┴─────────────────────┐                    │
│  │  ┌──────────────────────────────────┐   │                    │
│  │  │ gamma_net:                       │   │                    │
│  │  │   Linear(256,256) + LayerNorm    │   │                    │
│  │  │   GELU                           │   │                    │
│  │  │   Linear(256,256)                │   │                    │
│  │  └───────────┬──────────────────────┘   │                    │
│  │              │                          │                    │
│  │  ┌───────────┴──────────────────┐       │                    │
│  │  │ beta_net:                    │       │                    │
│  │  │   (same structure)           │       │                    │
│  │  └───────────┬──────────────────┘       │                    │
│  │              │                          │                    │
│  │              ▼                          │                    │
│  │  ┌─────────────────────────────────┐    │                    │
│  │  │ modulated_feat =                │    │                    │
│  │  │   gamma * joint_feat + beta     │    │                    │
│  │  └─────────────────────────────────┘    │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                 │
│       │           │           │           │                     │
│       ▼           ▼           ▼           ▼                     │
│  Modulated   Modulated   Modulated   Modulated                  │
│  Shoulder    Elbow       Forearm     Wrist                      │
│  [B,256]     [B,256]     [B,256]     [B,256]                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

输出预测 (直接从调制特征，无GNN):

  Modulated Shoulder [B,256] ──► Linear 256->3 ──► pred_shoulder [B,3]

  Modulated Elbow    [B,256] ──► Linear 256->1 ──► pred_elbow [B,1]

  Modulated Forearm  [B,256] ──► Linear 256->1 ──► pred_forearm [B,1]

  Modulated Wrist    [B,256] ──► Linear 256->2 ──► pred_wrist [B,2]

                                           │
                                           ▼
                                    ┌─────────────┐
                                    │   Concat    │
                                    │  [3+1+1+2]  │
                                    │     = 7     │
                                    └─────────────┘
                                           │
                                           ▼
                                    pred_angles [B,7]
```

---

## 6. pieper1101/causal_ik_model_pieper2.py (参考)

```
┌─────────────────────────────────────────────────────────────────────┐
│                   pieper1101 PieperCausalIK                         │
│      (时序注意力 + Pieper权重 + FiLM + GNN + 多特征融合)             │
└─────────────────────────────────────────────────────────────────────┘

核心创新:
  1. 时序编码器使用全部历史帧
  2. PieperAttention 从全部历史计算权重
  3. FiLM调制 + GNN + 多特征融合

输入: history_frames [B, T, 7]
    │
    ├──► [JointwiseTemporalEncoder] ──────┐
    │                                      │
    └──► [PieperAttention] ──► pos/ori weights ──┐
           (使用全部历史)                         │
                                                  │
输入: end_position, end_orientation               │
    │                                             │
    ├──► [PieperFiLMGenerator] ◄────────────────┤
    │       (使用加权历史关节)                     │
    │                                             │
    └──► [EndPoseEncoder] ◄─── FiLM调制后的位姿 ◄┘
                                  │
                                  ▼
                      ┌─────────────────────┐
                      │   MultiFeatureFusion │
                      │  (GNN + FiLM融合)    │
                      └─────────────────────┘
                                  │
                                  ▼
                         [Output Heads]
                                  │
                                  ▼
                          pred_angles [B,7]
```
