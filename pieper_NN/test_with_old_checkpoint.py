"""
使用旧版本checkpoint测试

为了兼容 pieper_causal_ik_092.pth (旧结构)，我们需要动态创建旧模型
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/wsy/Desktop/casual/pieper_NN')

# 加载checkpoint看看结构
checkpoint = torch.load('/home/wsy/Desktop/casual/pieper_NN/pieper_causal_ik_092.pth', map_location='cpu')

print("=" * 70)
print("分析旧checkpoint结构")
print("=" * 70)
print(f"\nEpoch: {checkpoint['epoch']}")
print(f"Val Loss: {checkpoint['best_val_loss']:.6f}")
print(f"\n模型参数量: {sum(p.numel() for p in checkpoint['model_state_dict'].values()):,}")

# 分析模块结构
modules = {}
for key in checkpoint['model_state_dict'].keys():
    parts = key.split('.')
    module_name = parts[0]
    if module_name not in modules:
        modules[module_name] = []
    modules[module_name].append(key)

print(f"\n模块结构:")
for module_name, keys in sorted(modules.items()):
    print(f"  {module_name}: {len(keys)} 个参数")

# 看看pieper_attention的结构
if 'pieper_attention' in modules:
    print(f"\npieper_attention 详细结构:")
    for key in sorted(modules['pieper_attention']):
        print(f"  {key}")

# 看看joint_embeddings的结构
if 'joint_embeddings' in modules:
    print(f"\njoint_embeddings 详细结构:")
    for key in sorted(modules['joint_embeddings']):
        print(f"  {key}")

print("\n" + "=" * 70)
print("结论：需要旧版本的 causal_ik_model_pieper.py")
print("=" * 70)
