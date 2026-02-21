"""
GPU利用率诊断工具

分析GPU利用率低的原因并提供优化建议
"""

import torch
import time
import subprocess


def diagnose_gpu_utilization():
    """诊断GPU利用率问题"""
    print("=" * 70)
    print("GPU利用率诊断")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    # 基本信息
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"\nGPU: {device_name}")
    print(f"总显存: {total_memory:.2f} GB")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查PyTorch是否支持compile
    print(f"\nPyTorch 2.0 compile: {'✓ 支持' if hasattr(torch, 'compile') else '✗ 不支持'}")
    
    # 检查cuDNN
    print(f"cuDNN: {'✓ 已启用' if torch.backends.cudnn.enabled else '✗ 未启用'}")
    if torch.backends.cudnn.enabled:
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"cuDNN基准测试: {'✓ 开启' if torch.backends.cudnn.benchmark else '✗ 关闭'}")
    
    # 显存分析
    current_mem = torch.cuda.memory_allocated() / 1024**3
    reserved_mem = torch.cuda.memory_reserved() / 1024**3
    
    print(f"\n显存状态:")
    print(f"  已分配: {current_mem:.2f} GB")
    print(f"  保留:   {reserved_mem:.2f} GB")
    print(f"  可用:   {total_memory - current_mem:.2f} GB")
    
    # 常见问题和建议
    print("\n" + "=" * 70)
    print("常见GPU利用率低的原因及解决方案")
    print("=" * 70)
    
    suggestions = []
    
    # 1. Batch Size
    print("\n1. Batch Size太小")
    print("   症状: GPU显存占用<50%，但利用率低")
    print("   解决: 增大batch_size（建议2048-8192）")
    suggestions.append(("batch_size", "当前可能太小，建议2048+"))
    
    # 2. 数据加载
    print("\n2. 数据加载瓶颈（CPU→GPU）")
    print("   症状: GPU利用率波动大，频繁掉到0%")
    print("   解决:")
    print("      - 增大num_workers（建议8-16）")
    print("      - 开启pin_memory=True")
    print("      - 使用prefetch_factor=4")
    suggestions.append(("num_workers", "建议8-16，当前可能太少"))
    
    # 3. 模型太小
    print("\n3. 模型计算量太小")
    print("   症状: 显存占用低，GPU利用率持续低")
    print("   解决:")
    print("      - 增大hidden_dim（如256→512）")
    print("      - 增加网络层数")
    print("      - 使用torch.compile优化")
    suggestions.append(("model_size", "显式耦合模型1.47M参数量，可以增加"))
    
    # 4. 混合精度
    print("\n4. 未使用混合精度")
    print("   症状: 显存占用高，但利用率低")
    print("   解决: 使用torch.cuda.amp")
    print("      - 节省显存（可增大batch）")
    print("      - Tensor Core加速")
    suggestions.append(("amp", "建议使用混合精度训练"))
    
    # 5. CPU-GPU同步
    print("\n5. 频繁的CPU-GPU同步")
    print("   症状: GPU利用率不稳定")
    print("   解决:")
    print("      - 减少.item()调用")
    print("      - 使用torch.cuda.synchronize()只在必要时")
    suggestions.append(("sync", "减少CPU-GPU同步点"))
    
    # 6. 数据预处理
    print("\n6. 数据预处理在CPU上太慢")
    print("   症状: GPU等待数据")
    print("   解决:")
    print("      - 预处理放到GPU")
    print("      - 使用DALI等加速数据加载")
    suggestions.append(("preprocessing", "考虑GPU预处理"))
    
    # 优化后的配置建议
    print("\n" + "=" * 70)
    print("优化后的配置建议")
    print("=" * 70)
    
    print("""
当前推荐配置（针对你的显式耦合模型）：

class Config:
    # 数据加载优化
    batch_size = 2048          # 从512增大到2048
    num_workers = 8            # 从4增加到8
    pin_memory = True
    prefetch_factor = 4
    persistent_workers = True
    
    # 训练优化
    use_amp = True             # 启用混合精度
    gradient_accumulation = 2  # 等效batch = 4096
    
    # 模型优化
    use_compile = True         # PyTorch 2.0+编译优化
    
    # 学习率（随batch增大）
    initial_lr = 2e-3          # 从1e-3增大到2e-3
""")
    
    # 快速测试
    print("\n" + "=" * 70)
    print("快速GPU测试")
    print("=" * 70)
    
    # 测试矩阵乘法性能
    print("\n测试矩阵乘法性能...")
    a = torch.randn(4096, 4096).cuda()
    b = torch.randn(4096, 4096).cuda()
    
    # 预热
    for _ in range(10):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # 测试
    start = time.time()
    for _ in range(100):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"  100次 4096x4096 矩阵乘法: {elapsed:.3f}s")
    print(f"  单次: {elapsed*10:.3f}ms")
    
    if elapsed < 5:
        print("  ✓ GPU计算性能正常")
    else:
        print("  ⚠ GPU计算性能可能受限")
    
    # 清理
    del a, b
    torch.cuda.empty_cache()
    
    # 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"""
诊断结果：
GPU: {device_name}
显存: {total_memory:.1f} GB

主要优化方向：
1. 增大batch_size到2048+（当前可能太小）
2. 启用混合精度训练（AMP）
3. 使用torch.compile（PyTorch 2.0+）
4. 增加num_workers到8+

预计优化后：
- GPU利用率: 10% → 70-90%
- 训练速度: 提升3-5倍

运行优化版训练：
  python train_explicit_coupling_ik_optimized.py

监控GPU利用率：
  python monitor_gpu.py monitor
""")


if __name__ == "__main__":
    diagnose_gpu_utilization()
