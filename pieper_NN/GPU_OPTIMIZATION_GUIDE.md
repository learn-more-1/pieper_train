# GPU利用率优化方案

## 当前问题
- GPU利用率只有 15%
- GPU大部分时间在等待CPU准备数据
- 单batch时间: 176ms（但实际计算只需要几ms）

## 优化措施

### 1. 多进程数据加载 ⭐⭐⭐⭐⭐
**当前**: num_workers=0 (单进程，CPU串行)
**优化**: num_workers=4 (4个进程并行准备数据)

```python
# dataset_generalized.py 中修改DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=512,
    num_workers=4,  # 关键！多进程数据加载
    pin_memory=True,  # 加速CPU->GPU传输
    prefetch_factor=2  # 预取2个batch
)
```

**预期提升**: 2-3x加速

---

### 2. 使用 pin_memory=True ⭐⭐⭐⭐
**作用**: 将数据锁定在内存中，加速CPU->GPU传输

```python
batch_X = batch_X.to(device, non_blocking=True)  # 异步传输
```

**预期提升**: 1.2-1.5x加速

---

### 3. 混合精度训练 (FP16) ⭐⭐⭐⭐⭐
**作用**: 使用FP16代替FP32，减少计算量和内存

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    pred = model(x)
    loss = criterion(pred, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**预期提升**: 1.5-2x加速

---

### 4. torch.compile() ⭐⭐⭐⭐
**作用**: PyTorch 2.0+ JIT编译，优化计算图

```python
model = torch.compile(model, mode='max-autotune')
```

**预期提升**: 1.2-1.8x加速

---

### 5. 增大 batch size ⭐⭐⭐
**当前**: batch_size=512
**优化**: batch_size=1024 (如果GPU内存允许)

**预期提升**: 1.3-1.5x加速（更充分利用GPU）

---

## 实施步骤

### 步骤1: 修改 dataset_generalized.py

在 `create_windowed_dataloaders` 函数中添加参数：

```python
def create_windowed_dataloaders(data_path, config, num_workers=0, pin_memory=False):
    """
    添加 num_workers 和 pin_memory 参数
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,      # 新增
        pin_memory=pin_memory,          # 新增
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,        # 新增
        pin_memory=pin_memory,          # 新增
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader
```

### 步骤2: 修改 train_pieper_model.py

```python
# 在 train() 函数中，添加优化参数
train_loader, val_loader = create_windowed_dataloaders(
    config.data_path,
    config,
    num_workers=4,      # 多进程
    pin_memory=True      # 加速传输
)

# 在数据移到GPU时，使用 non_blocking=True
batch_X = batch_X.to(config.device, non_blocking=True)
batch_y = batch_y.to(config.device, non_blocking=True)
batch_last_angle = batch_last_angle.to(config.device, non_blocking=True)
```

### 步骤3: 添加混合精度训练

```python
# 在 train() 函数开头
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 在训练循环中
with autocast():
    pred_joint_angles, info = model(...)
    pred_position, _ = forward_kinematics_with_pose(gpu_fk, pred_joint_angles)
    fk_position, _ = forward_kinematics_with_pose(gpu_fk, target_angles)

    loss = criterion(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 步骤4: 编译模型

```python
# 在创建模型后
try:
    model = torch.compile(model, mode='max-autotune')
except Exception as e:
    print(f"torch.compile失败: {e}")
```

---

## 预期效果

### 优化前
- GPU利用率: 15%
- Epoch时间: ~100秒
- 300 epochs: 8.2小时

### 优化后（所有措施）
- GPU利用率: 60-80%
- Epoch时间: ~20-30秒
- 300 epochs: 1.7-2.5小时
- **加速比: 3-5x**

---

## 快速测试

先测试 num_workers=4 的效果：

```python
# 在 train_pieper_model.py 中修改
train_loader, val_loader = create_windowed_dataloaders(
    config.data_path,
    config,
    num_workers=4,
    pin_memory=True
)
```

观察GPU利用率：
```bash
watch -n 1 nvidia-smi
```

如果GPU利用率提升到50%+，说明优化有效！
