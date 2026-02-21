"""
从URDF提取精确的FK参数
"""

import numpy as np
from xml.etree import ElementTree as ET


def extract_left_arm_fk(urdf_path):
    """提取左臂的正向运动学参数"""
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # 左臂关节链
    joint_chain = [
        'left_shoulder_pitch_joint',
        'left_shoulder_roll_joint',
        'left_shoulder_yaw_joint',
        'left_elbow_joint',
        'left_wrist_roll_joint',
        'left_wrist_pitch_joint',
        'left_wrist_yaw_joint'
    ]
    
    print("=" * 70)
    print("G1机器人左臂FK参数")
    print("=" * 70)
    
    origins = []
    axes = []
    
    for joint_name in joint_chain:
        joint = root.find(f".//joint[@name='{joint_name}']")
        if joint is not None:
            origin = joint.find('origin')
            axis = joint.find('axis')
            
            if origin is not None:
                xyz_str = origin.get('xyz', '0 0 0')
                xyz = [float(x) for x in xyz_str.split()]
                origins.append(xyz)
            else:
                origins.append([0, 0, 0])
            
            if axis is not None:
                axis_str = axis.get('xyz', '0 0 1')
                axis_vec = [float(x) for x in axis_str.split()]
                axes.append(axis_vec)
            else:
                axes.append([0, 0, 1])
            
            print(f"\n{joint_name}:")
            print(f"  origin xyz: {origins[-1]}")
            print(f"  axis: {axes[-1]}")
        else:
            print(f"⚠ 未找到关节: {joint_name}")
            origins.append([0, 0, 0])
            axes.append([0, 0, 1])
    
    # 计算连杆长度（欧几里得距离）
    print("\n" + "=" * 70)
    print("连杆长度计算")
    print("=" * 70)
    
    link_lengths = []
    for i, origin in enumerate(origins):
        length = np.linalg.norm(origin)
        link_lengths.append(length)
        print(f"Link {i} ({joint_chain[i]}): {length:.6f} m = {length*1000:.2f} mm")
    
    # 确定旋转轴
    print("\n" + "=" * 70)
    print("关节旋转轴")
    print("=" * 70)
    
    axis_map = {'0 0 1': 'Z', '1 0 0': 'X', '0 1 0': 'Y'}
    joint_axes = []
    
    for i, axis in enumerate(axes):
        axis_str = f"{axis[0]} {axis[1]} {axis[2]}"
        axis_name = axis_map.get(axis_str, axis_str)
        joint_axes.append(axis_name)
        print(f"J{i} ({joint_chain[i]}): {axis_name}")
    
    # 肘部位置（在J3之后，即elbow_joint之后）
    print("\n" + "=" * 70)
    print("关键位置索引")
    print("=" * 70)
    print("肘部位置: J3 (left_elbow_joint) 之后")
    print("腕部位置: J6 (left_wrist_yaw_joint) 之后")
    
    return {
        'link_lengths': link_lengths,
        'joint_axes': joint_axes,
        'origins': origins,
        'axes': axes
    }


def generate_fk_code(params):
    """生成Python FK代码"""
    
    print("\n" + "=" * 70)
    print("生成的FK代码")
    print("=" * 70)
    
    code = f'''# 从URDF提取的精确参数
LINK_LENGTHS = {params['link_lengths']}
JOINT_AXES = {params['axes']}  # 0=X, 1=Y, 2=Z
'''
    print(code)
    
    return code


if __name__ == "__main__":
    urdf_path = "/home/bonuli/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
    
    params = extract_left_arm_fk(urdf_path)
    generate_fk_code(params)
