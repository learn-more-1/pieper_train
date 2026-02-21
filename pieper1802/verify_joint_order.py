"""
验证关节顺序和名称对应关系
"""

import torch
import numpy as np
import sys

sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/pieper1802')

from fk_with_elbow_urdf import FKWithElbowURDF


def verify_joint_order():
    """验证关节顺序"""
    
    print("=" * 70)
    print("关节顺序验证")
    print("=" * 70)
    
    fk = FKWithElbowURDF()
    
    print("\n代码中的joint_axes:")
    axis_names = {0: 'X', 1: 'Y', 2: 'Z'}
    for i, axis in enumerate(fk.joint_axes):
        print(f"  J{i}: {axis_names[axis]}轴")
    
    print("\n从URDF提取的实际关节顺序:")
    print("  J0: left_shoulder_pitch_joint - Y轴")
    print("  J1: left_shoulder_roll_joint - X轴")
    print("  J2: left_shoulder_yaw_joint - Z轴")
    print("  J3: left_elbow_joint - Y轴")
    print("  J4: left_wrist_roll_joint - X轴")
    print("  J5: left_wrist_pitch_joint - Y轴")
    print("  J6: left_wrist_yaw_joint - Z轴")
    
    print("\n" + "=" * 70)
    print("对应关系检查")
    print("=" * 70)
    
    expected_axes = [1, 0, 2, 1, 0, 1, 2]  # Y, X, Z, Y, X, Y, Z
    actual_axes = fk.joint_axes
    
    match = all(a == b for a, b in zip(expected_axes, actual_axes))
    
    if match:
        print("✓ joint_axes与URDF匹配")
    else:
        print("✗ joint_axes与URDF不匹配！")
        print(f"  期望: {expected_axes}")
        print(f"  实际: {actual_axes}")
    
    print("\n" + "=" * 70)
    print("当前代码中的joint_names")
    print("=" * 70)
    
    joint_names = ['Shoulder Pitch', 'Shoulder Roll', 'Shoulder Yaw',
                   'Elbow', 'Forearm Roll', 'Wrist Pitch', 'Wrist Yaw']
    
    print("\n当前名称列表:")
    for i, name in enumerate(joint_names):
        axis = axis_names[fk.joint_axes[i]]
        print(f"  J{i}: {name:<20s} ({axis}轴)")
    
    print("\n" + "=" * 70)
    print("问题检查")
    print("=" * 70)
    
    # 检查手腕关节
    print("\n手腕关节检查:")
    print(f"  J5: 当前名称='{joint_names[5]}', 实际轴={axis_names[fk.joint_axes[5]]}")
    print(f"      URDF中是wrist_pitch_joint (Y轴)")
    if "Pitch" in joint_names[5] and fk.joint_axes[5] == 1:
        print("      ✓ 匹配 (Pitch对应Y轴)")
    else:
        print("      ✗ 不匹配！")
    
    print(f"\n  J6: 当前名称='{joint_names[6]}', 实际轴={axis_names[fk.joint_axes[6]]}")
    print(f"      URDF中是wrist_yaw_joint (Z轴)")
    if "Yaw" in joint_names[6] and fk.joint_axes[6] == 2:
        print("      ✓ 匹配 (Yaw对应Z轴)")
    else:
        print("      ✗ 不匹配！")
    
    # 测试单一关节运动
    print("\n" + "=" * 70)
    print("单一关节运动测试")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # J0运动 - 应该主要影响某个方向
    angles = torch.zeros(1, 7).to(device)
    angles[0, 0] = 0.5  # J0 = 0.5 rad
    _, end_pos = fk.compute_positions(angles)
    print(f"\nJ0 (Shoulder Pitch) = 0.5 rad:")
    print(f"  末端位置: [{end_pos[0,0].item():.4f}, {end_pos[0,1].item():.4f}, {end_pos[0,2].item():.4f}]")
    
    # J3运动 - Elbow
    angles = torch.zeros(1, 7).to(device)
    angles[0, 3] = 0.5  # J3 = 0.5 rad
    _, end_pos = fk.compute_positions(angles)
    print(f"\nJ3 (Elbow) = 0.5 rad:")
    print(f"  末端位置: [{end_pos[0,0].item():.4f}, {end_pos[0,1].item():.4f}, {end_pos[0,2].item():.4f}]")
    
    # J5运动 - Wrist Pitch
    angles = torch.zeros(1, 7).to(device)
    angles[0, 5] = 0.5  # J5 = 0.5 rad
    elbow_pos, end_pos = fk.compute_positions(angles)
    print(f"\nJ5 (Wrist Pitch) = 0.5 rad:")
    print(f"  肘部位置: [{elbow_pos[0,0].item():.4f}, {elbow_pos[0,1].item():.4f}, {elbow_pos[0,2].item():.4f}]")
    print(f"  末端位置: [{end_pos[0,0].item():.4f}, {end_pos[0,1].item():.4f}, {end_pos[0,2].item():.4f}]")
    print(f"  注意: 肘部应该不变，只有末端变")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    verify_joint_order()
