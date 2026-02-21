"""
计算显式耦合IK模型中耦合关系的参数数量
"""

import torch
import torch.nn as nn


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters())


def analyze_coupling_params():
    """详细分析耦合关系参数"""
    
    print("=" * 70)
    print("显式耦合IK模型 - 耦合关系参数统计")
    print("=" * 70)
    
    hidden_dim = 128  # KinematicCouplingGraph的hidden_dim
    
    # 1. 边类型编码 (Edge Embedding)
    print("\n【1. 边类型编码 (Edge Embedding)】")
    print("-" * 50)
    
    chain_edges = 6   # J0-J1, J1-J2, J2-J3, J3-J4, J4-J5, J5-J6
    coupling_edges = 4  # J0-J3, J1-J3, J2-J5, J3-J5
    num_edge_types = chain_edges + coupling_edges  # = 10
    
    edge_embedding_params = num_edge_types * hidden_dim
    
    print(f"边类型数: {num_edge_types} (运动链: {chain_edges}, 功能耦合: {coupling_edges})")
    print(f"嵌入维度: {hidden_dim}")
    print(f"参数: {num_edge_types} × {hidden_dim} = {edge_embedding_params:,}")
    
    # 2. 消息传递MLPs (核心耦合参数)
    print("\n【2. 消息传递MLPs (核心耦合参数)】")
    print("-" * 50)
    
    # 每个MLP的结构
    input_dim = hidden_dim * 3  # 源特征 + 目标特征 + 边类型 = 128*3 = 384
    
    # 第一层 Linear
    linear1_weights = input_dim * hidden_dim  # 384 * 128
    linear1_bias = hidden_dim  # 128
    linear1_params = linear1_weights + linear1_bias
    
    # LayerNorm
    layernorm_params = 2 * hidden_dim  # gamma + beta = 2 * 128
    
    # 第二层 Linear
    linear2_weights = hidden_dim * hidden_dim  # 128 * 128
    linear2_bias = hidden_dim  # 128
    linear2_params = linear2_weights + linear2_bias
    
    # 每个MLP的总参数
    mlp_params_per_edge = linear1_params + layernorm_params + linear2_params
    
    print(f"MLP输入维度: {input_dim} (源{hidden_dim} + 目标{hidden_dim} + 边类型{hidden_dim})")
    print(f"\n每个MLP的参数:")
    print(f"  第一层 Linear: {linear1_weights:,} + {linear1_bias:,} = {linear1_params:,}")
    print(f"  LayerNorm: {layernorm_params:,}")
    print(f"  第二层 Linear: {linear2_weights:,} + {linear2_bias:,} = {linear2_params:,}")
    print(f"  每个MLP总计: {mlp_params_per_edge:,}")
    
    # 所有边的MLP
    total_message_mlp_params = num_edge_types * mlp_params_per_edge
    
    print(f"\n边类型数: {num_edge_types}")
    print(f"消息传递MLPs总参数: {num_edge_types} × {mlp_params_per_edge:,} = {total_message_mlp_params:,}")
    
    # 3. 节点更新MLPs
    print("\n【3. 节点更新MLPs】")
    print("-" * 50)
    
    num_joints = 7
    node_input_dim = hidden_dim * 2  # 自身特征 + 聚合消息 = 256
    
    # 每个关节的MLP: Linear(256, 128) + LayerNorm(128) + GELU
    node_linear_weights = node_input_dim * hidden_dim  # 256 * 128
    node_linear_bias = hidden_dim  # 128
    node_linear_params = node_linear_weights + node_linear_bias
    
    node_layernorm_params = 2 * hidden_dim  # 256
    
    node_mlp_params_per_joint = node_linear_params + node_layernorm_params
    
    print(f"每个关节的MLP:")
    print(f"  输入维度: {node_input_dim} (自身{hidden_dim} + 消息{hidden_dim})")
    print(f"  Linear: {node_linear_weights:,} + {node_linear_bias:,} = {node_linear_params:,}")
    print(f"  LayerNorm: {node_layernorm_params:,}")
    print(f"  每个关节总计: {node_mlp_params_per_joint:,}")
    
    total_node_update_params = num_joints * node_mlp_params_per_joint
    
    print(f"\n关节数: {num_joints}")
    print(f"节点更新总参数: {num_joints} × {node_mlp_params_per_joint:,} = {total_node_update_params:,}")
    
    # 4. 组间门控融合 (Gates)
    print("\n【4. 组间门控融合 (Gates)】")
    print("-" * 50)
    
    num_gates = 3  # elbow_from_shoulder, forearm_from_elbow, wrist_from_forearm
    gate_input_dim = hidden_dim * 2  # 子组特征 + 父组特征 = 256
    gate_output_dim = hidden_dim  # 128 (每个特征维度一个门控值)
    
    # 每个门控: Linear(256, 128) + Sigmoid
    gate_linear_weights = gate_input_dim * gate_output_dim  # 256 * 128
    gate_linear_bias = gate_output_dim  # 128
    gate_params_per_gate = gate_linear_weights + gate_linear_bias
    
    print(f"门控网络数: {num_gates}")
    print(f"每个门控:")
    print(f"  输入: {gate_input_dim} (子组{hidden_dim} + 父组{hidden_dim})")
    print(f"  输出: {gate_output_dim} (每个维度一个门控值)")
    print(f"  Linear: {gate_linear_weights:,} + {gate_linear_bias:,} = {gate_params_per_gate:,}")
    
    total_gate_params = num_gates * gate_params_per_gate
    
    print(f"\n门控总参数: {num_gates} × {gate_params_per_gate:,} = {total_gate_params:,}")
    
    # 5. 分组解码器中的耦合部分
    print("\n【5. 分组解码器 (耦合相关部分)】")
    print("-" * 50)
    
    # 这里只计算与耦合相关的部分，不是整个解码器
    # 实际上group_decoders大部分是独立的，门控已经计算过了
    
    # 耦合相关的输入变换
    coupling_related_params = 0
    
    # 汇总
    print("\n" + "=" * 70)
    print("耦合关系参数汇总")
    print("=" * 70)
    
    total_coupling_params = (
        edge_embedding_params +
        total_message_mlp_params +
        total_node_update_params +
        total_gate_params
    )
    
    print(f"\n1. 边类型编码:          {edge_embedding_params:>12,} ({edge_embedding_params/total_coupling_params*100:.1f}%)")
    print(f"2. 消息传递MLPs:        {total_message_mlp_params:>12,} ({total_message_mlp_params/total_coupling_params*100:.1f}%)")
    print(f"3. 节点更新MLPs:        {total_node_update_params:>12,} ({total_node_update_params/total_coupling_params*100:.1f}%)")
    print(f"4. 组间门控融合:        {total_gate_params:>12,} ({total_gate_params/total_coupling_params*100:.1f}%)")
    print(f"{'─' * 50}")
    print(f"耦合关系总参数:         {total_coupling_params:>12,} (100.0%)")
    
    # 对比模型总参数
    print("\n" + "=" * 70)
    print("与模型总参数对比")
    print("=" * 70)
    
    # 创建实际模型计算总参数
    from explicit_coupling_ik import ExplicitCouplingIK
    model = ExplicitCouplingIK(num_joints=7, num_frames=10, hidden_dim=256, use_temporal=False)
    total_model_params = count_parameters(model)
    
    print(f"\n模型总参数:    {total_model_params:>12,}")
    print(f"耦合关系参数:  {total_coupling_params:>12,}")
    print(f"耦合占比:      {total_coupling_params/total_model_params*100:>11.1f}%")
    
    # 其他参数
    other_params = total_model_params - total_coupling_params
    print(f"\n其他参数 (意图编码+解码+意图分发):")
    print(f"  {other_params:,} ({other_params/total_model_params*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print(f"\n耦合关系参数: {total_coupling_params:,} ({total_coupling_params/1e6:.2f}M)")
    print(f"占模型总参数的 {total_coupling_params/total_model_params*100:.1f}%")
    print("\n其中消息传递MLPs是核心，占耦合参数的绝大部分")
    print("这些参数学习'关节间如何配合'的策略")


if __name__ == "__main__":
    analyze_coupling_params()
