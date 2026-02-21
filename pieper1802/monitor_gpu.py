"""
GPU监控工具

实时监控GPU利用率、显存使用等信息
"""

import torch
import time
import subprocess
import sys


def get_gpu_info():
    """获取GPU信息"""
    if not torch.cuda.is_available():
        print("CUDA不可用")
        return
    
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"GPU: {device_name}")
    print(f"总显存: {total_memory:.2f} GB")
    print("-" * 60)
    
    # 使用nvidia-smi获取更详细信息
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
            print(f"利用率: {util}%")
            print(f"显存使用: {float(mem_used)/1024:.2f} GB / {float(mem_total)/1024:.2f} GB")
            print(f"温度: {temp}°C")
    except:
        pass


def monitor_training(interval=2):
    """
    实时监控GPU（在训练时运行）
    
    用法:
        终端1: python train_explicit_coupling_ik_optimized.py
        终端2: python monitor_gpu.py
    """
    print("=" * 60)
    print("GPU实时监控 (按Ctrl+C停止)")
    print("=" * 60)
    
    get_gpu_info()
    print()
    
    history = []
    
    try:
        while True:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    util, mem_used = result.stdout.strip().split(', ')
                    util = int(util)
                    mem_gb = float(mem_used) / 1024
                    
                    history.append(util)
                    if len(history) > 30:
                        history.pop(0)
                    
                    avg_util = sum(history) / len(history)
                    
                    bar = "█" * (util // 5) + "░" * (20 - util // 5)
                    print(f"\rGPU: [{bar}] {util:3d}% | 显存: {mem_gb:.2f}GB | 平均: {avg_util:.1f}%", end='')
                    sys.stdout.flush()
                    
            except Exception as e:
                print(f"\n获取GPU信息失败: {e}")
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n监控停止")
        if history:
            print(f"平均利用率: {sum(history)/len(history):.1f}%")


def benchmark_model(model, input_shape, iterations=100):
    """
    基准测试模型推理速度
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状
        iterations: 迭代次数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    
    # 创建随机输入
    dummy_input = torch.randn(input_shape).to(device)
    
    # 预热
    print("预热中...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # 同步
    torch.cuda.synchronize()
    
    # 正式测试
    print(f"运行基准测试 ({iterations} iterations)...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    
    avg_time = elapsed_ms / iterations
    throughput = input_shape[0] * iterations / (elapsed_ms / 1000)
    
    print(f"\n结果:")
    print(f"  平均推理时间: {avg_time:.3f} ms")
    print(f"  吞吐量: {throughput:.0f} samples/sec")
    
    return avg_time, throughput


def suggest_batch_size(model_fn, input_shapes, max_batch=8192):
    """
    自动寻找最优batch size
    
    逐步增大batch size直到OOM，然后回退
    """
    device = torch.device('cuda')
    
    print("寻找最优batch size...")
    print("-" * 60)
    
    low, high = 1, max_batch
    optimal = 1
    
    while low <= high:
        mid = (low + high) // 2
        try:
            # 清理缓存
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 创建模型和输入
            model = model_fn().to(device)
            inputs = [torch.randn(mid, *shape).to(device) for shape in input_shapes]
            
            # 前向传播
            with torch.no_grad():
                _ = model(*inputs)
            
            # 检查显存使用
            mem_used = torch.cuda.max_memory_allocated() / 1024**3
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if mem_used < total_mem * 0.9:  # 留10%余量
                optimal = mid
                low = mid + 1
                print(f"  Batch {mid}: OK (显存 {mem_used:.2f}GB)")
            else:
                high = mid - 1
                print(f"  Batch {mid}: 显存占用过高 ({mem_used:.2f}GB)")
                
            del model, inputs
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                high = mid - 1
                print(f"  Batch {mid}: OOM")
                torch.cuda.empty_cache()
            else:
                raise
    
    print(f"\n建议batch size: {optimal}")
    return optimal


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "info":
            get_gpu_info()
        elif cmd == "monitor":
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 2
            monitor_training(interval)
        elif cmd == "benchmark":
            # 导入模型进行基准测试
            from explicit_coupling_ik import ExplicitCouplingIK
            model = ExplicitCouplingIK()
            benchmark_model(model, (512, 3))
        else:
            print(f"未知命令: {cmd}")
    else:
        print("GPU监控工具")
        print("=" * 60)
        print("用法:")
        print("  python monitor_gpu.py info       # 显示GPU信息")
        print("  python monitor_gpu.py monitor    # 实时监控利用率")
        print("  python monitor_gpu.py benchmark  # 模型基准测试")
