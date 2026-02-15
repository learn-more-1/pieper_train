#!/usr/bin/env python3
"""验证 Pinocchio 模型中的 frame_id"""

import pinocchio as pin

# 加载模型
urdf_path = "/home/wsy/Desktop/casual/unitree_g1/g1_custom_collision_29dof.urdf"
model = pin.buildModelFromUrdf(urdf_path)

print("=" * 70)
print("验证 wrist_frame_id = 57")
print("=" * 70)

# 检查 frame_id = 57
target_id = 57
print(f"\nFrame ID {target_id}:")
try:
    frame_name = model.frames[target_id].name
    print(f"  名称: {frame_name}")
    print(f"  关节 ID: {model.frames[target_id].parent}")
    print(f"  关节名称: {model.names[model.frames[target_id].parent]}")
except Exception as e:
    print(f"  错误: {e}")

# 查找所有与手掌相关的 frame
print("\n" + "=" * 70)
print("查找所有包含 'hand' 或 'palm' 或 'wrist' 的 frame:")
print("=" * 70)

hand_frames = []
for i in range(len(model.frames)):
    frame_name = model.frames[i].name
    # if any(keyword in frame_name.lower() for keyword in ['hand', 'palm', 'wrist']):
    parent_joint = model.frames[i].parent
    parent_name = model.names[parent_joint]
    hand_frames.append((i, frame_name, parent_joint, parent_name))

for frame_id, frame_name, parent_id, parent_name in hand_frames:
    print(f"  Frame ID {frame_id:2d}: {frame_name:<40} (父关节: {parent_id:2d} - {parent_name})")

# 特别查找左手相关的
print("\n" + "=" * 70)
print("查找所有包含 'left' 的 frame:")
print("=" * 70)

left_frames = []
for i in range(len(model.frames)):
    frame_name = model.frames[i].name
    if 'left' in frame_name.lower():
        parent_joint = model.frames[i].parent
        parent_name = model.names[parent_joint]
        left_frames.append((i, frame_name, parent_joint, parent_name))

for frame_id, frame_name, parent_id, parent_name in left_frames:
    print(f"  Frame ID {frame_id:2d}: {frame_name:<40} (父关节: {parent_id:2d} - {parent_name})")

# 建议的左臂末端 frame
print("\n" + "=" * 70)
print("建议的左臂末端 frame:")
print("=" * 70)

suggested_frames = [
    "left_rubber_hand",
    "left_hand_palm_link",
    "left_wrist_yaw_link",
    "left_palm_link"
]

for name in suggested_frames:
    try:
        frame_id = model.getFrameId(name)
        print(f"  {name:<30} -> Frame ID: {frame_id}")
    except:
        print(f"  {name:<30} -> 未找到")

# 如果 frame_id = 57 确实是 left_rubber_hand，打印一些额外的信息
if target_id < len(model.frames):
    frame = model.frames[target_id]
    print("\n" + "=" * 70)
    print(f"Frame {target_id} 的详细信息:")
    print("=" * 70)
    print(f"  名称: {frame.name}")
    print(f"  类型: {frame.type}")
    print(f"  父关节 ID: {frame.parent}")
    print(f"  父关节名称: {model.names[frame.parent]}")
    print(f"  放置变换 (相对于父关节):")
    print(f"    旋转:\n{frame.placement.rotation}")
    print(f"    平移: {frame.placement.translation.T}")
