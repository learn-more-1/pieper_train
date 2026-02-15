#!/usr/bin/env python3
"""
泛化增强的数据集类

特性：
1. 数据增强（旋转、噪声、时间偏移）
2. 物理约束（关节角度限制）
3. 领域自适应（不同数据集加权）
4. 在线增强（动态生成变体）
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.spatial.transform import Rotation as R
import logging
import os

class GeneralizedIKDataset(Dataset):
    """泛化增强的IK数据集"""

    def __init__(self,
                 pose_path,
                 joint_path,
                 hand='left',
                 use_velocity=True,
                 use_augmentation=True,
                 augmentation_level='light',  # 'light', 'moderate', 'aggressive'
                 physics_constraints=True,
                 training_mode=False):
        """
        Args:
            augmentation_level:
                - light: 仅添加噪声
                - moderate: 噪声 + 时间偏移
                - aggressive: 噪声 + 时间偏移 + 旋转增强 + mixup
        """
        super().__init__()
        self.pose_path = pose_path
        self.hand = hand
        self.use_velocity = use_velocity
        self.use_augmentation = use_augmentation
        self.augmentation_level = augmentation_level
        self.physics_constraints = physics_constraints
        self.training_mode = False  # 初始化为评估模式

        # 加载数据
        logging.info(f"加载位姿数据: {pose_path}")
        pose_data = np.load(pose_path, allow_pickle=True)
        logging.info(f"加载关节数据: {joint_path}")
        joint_data = np.load(joint_path, allow_pickle=True)

        # 获取数据
        if "data" in pose_data.files:
            full_pose = pose_data["data"]
        else:
            keys = [k for k in pose_data.files if k not in ["num_files", "total_frames", "feature_dim"]]
            full_pose = np.stack([pose_data[k] for k in keys])

        if "data" in joint_data.files:
            full_joint = joint_data["data"]
        else:
            keys = [k for k in joint_data.files if k not in ["num_files", "total_frames", "feature_dim"]]
            full_joint = np.stack([joint_data[k] for k in keys])

        # 根据hand参数选择
        if hand == 'left':
            self.pose_array = full_pose[:, :7]
            self.joint_array = full_joint[:, :7]
        else:
            self.pose_array = full_pose[:, 7:] if full_pose.shape[1] > 7 else full_pose
            self.joint_array = full_joint[:, 7:] if full_joint.shape[1] > 7 else full_joint

        # 计算速度
        if use_velocity:
            logging.info("计算速度...")
            velocity_array = np.zeros_like(self.pose_array)
            velocity_array[1:] = self.pose_array[1:] - self.pose_array[:-1]
            velocity_array[0] = velocity_array[1]

            # 计算加速度（新增特征）
            acceleration_array = np.zeros_like(velocity_array)
            acceleration_array[1:] = velocity_array[1:] - velocity_array[:-1]
            acceleration_array[0] = acceleration_array[1]
            print("acceleration_array:", acceleration_array.shape)
            print("velocity_array:", velocity_array.shape)
            # 合并：位置 + 速度 + 加速度
            self.input_array = np.concatenate([
                self.pose_array,       # 7维：位置+四元数
                velocity_array,        # 7维：速度
                acceleration_array     # 7维：加速度
            ], axis=1)  # 总共 21维

            logging.info(f"输入维度: {self.input_array.shape[1]} (位姿7 + 速度7 + 加速度7)")
        else:
            self.input_array = self.pose_array

        self.total_frames = self.input_array.shape[0]
        logging.info(f"数据加载完成：{self.total_frames}帧")

        # 物理约束（Unitree G1关节角度限制）
        if physics_constraints:
            self.joint_limits = [
                (-3.0892, 1.1490),  # Joint 0
                (-0.6000, 2.2515),  # Joint 1
                (-1.4000, 2.0000),  # Joint 2
                (-1.0472, 1.7000),  # Joint 3
                (-1.9722, 1.9722),  # Joint 4
                (-1.4347, 1.6144),  # Joint 5
                (-1.6144, 1.6144),  # Joint 6
            ]
        else:
            self.joint_limits = None

    def augment_rotation(self, pose):
        """旋转变换：模拟不同的手腕朝向"""
        # 提取位置和四元数
        position = pose[:3]
        quaternion = pose[3:7]

        # 随机旋转
        if self.augmentation_level == 'aggressive':
            # 大幅度旋转
            euler = np.random.uniform(-np.pi/4, np.pi/4, 3)  # ±45度
        elif self.augmentation_level == 'moderate':
            # 中等旋转
            euler = np.random.uniform(-np.pi/8, np.pi/8, 3)  # ±22.5度
        else:
            # 轻微旋转
            euler = np.random.uniform(-np.pi/160, np.pi/160, 3)  # ±11.25度

        rotation = R.from_euler('xyz', euler)
        original_rot = R.from_quat(quaternion)
        rotated_quat = (rotation * original_rot).as_quat()

        return np.concatenate([position, rotated_quat])


    def add_noise(self, data, is_target=False):
        """
    为【无历史帧的独立数据序列】添加高斯噪声（50%概率加/50%概率不加，混合策略）
    适配：一维/多维数值序列，独立处理，平衡多样性与原始分布
    :param data: 输入序列，支持np.ndarray/Python列表
    :param is_target: 是否为目标序列，目标序列噪声强度更弱
    :return: 原始序列 或 叠加噪声后的序列（与输入同类型/同形状）
        """
    # 核心新增：50%随机概率决定是否加噪声，可灵活调整prob（如0.4/0.6）
        add_noise_prob = 0.5  # 固定50%概率，也可设为类属性灵活配置
        if np.random.random() > add_noise_prob:
            return data  # 不添加噪声，直接返回原始数据序列
    
    # 以下为原有成熟逻辑，无任何修改
    # 1. 数据类型快速转换与校验（适配序列）
        # if not isinstance(data, np.ndarray):
        #     data = np.array(data, dtype=np.float32)
        # if data.ndim == 0:
        #     raise ValueError("输入必须是数据序列（一维/多维），不支持单值数据")
    
    # 2. 增强等级合法性校验
        # valid_levels = ['aggressive', 'moderate', 'none']
        # if self.augmentation_level not in valid_levels:
        #     raise ValueError(f"augmentation_level仅支持{valid_levels}，当前值：{self.augmentation_level}")
    
    # 3. 计算序列自身标准差（自适应量纲，无历史帧依赖）
        data_std = data.std()
        data_std = max(data_std, 1e-8)  # 防止全零序列
    
    # 4. 相对噪声强度系数（区分目标/非目标）
        if self.augmentation_level == 'aggressive':
            coeff = 0.02 if not is_target else 0.01
        elif self.augmentation_level == 'moderate':
            coeff = 0.01 if not is_target else 0.005
        else:  # none：强制无噪声，返回原始数据
            return data
    
    # 5. 生成自适应噪声并叠加
        noise = np.random.normal(loc=0.0, scale=data_std * coeff, size=data.shape)
        noised_data = data + noise
    
    # 可选：序列值域保护（根据业务需求开启，如非负/归一化序列）
    # noised_data = np.clip(noised_data, a_min=0.0, a_max=None)  # 非负保护
    # noised_data = np.clip(noised_data, a_min=0.0, a_max=1.0)  # 归一化[0,1]序列保护

        return noised_data

    # def add_noise(self, data, is_target=False):
    #     """添加高斯噪声"""
    #     if self.augmentation_level == 'aggressive':
    #         noise_level = 0.02 if not is_target else 0.01
    #     elif self.augmentation_level == 'moderate':
    #         noise_level = 0.01 if not is_target else 0.005
    #     else:
    #         noise_level = 0.0005 if not is_target else 0.0002

    #     noise = np.random.normal(0, noise_level, data.shape)
    #     return data + noise

    def temporal_shift(self, data, shift_range=3):
        """时间偏移增强（仅对位置+四元数部分）"""
        if not hasattr(self, 'current_idx'):
            return data

        shift = np.random.randint(-shift_range, shift_range + 1)
        new_idx = np.clip(self.current_idx + shift, 0, self.total_frames - 1)

        # 获取偏移后的位置+四元数（前7维）
        shifted_pose = self.input_array[new_idx, :7]

        # 组合：偏移后的位姿 + 原速度和加速度
        if self.use_velocity:
            if self.input_array.shape[1] >= 21:  # 有加速度
                return np.concatenate([shifted_pose, data[7:14], data[14:21]])
            elif self.input_array.shape[1] >= 14:  # 有速度
                return np.concatenate([shifted_pose, data[7:]])
            else:
                return shifted_pose
        else:
            return shifted_pose

    def mixup(self, x1, y1):
        """Mixup增强"""
        if np.random.random() > 0.2:  # 20%概率
            return x1, y1

        # 随机选择另一个样本
        idx2 = np.random.randint(0, self.total_frames)
        x2 = self.input_array[idx2]
        y2 = self.joint_array[idx2]

        # 混合系数
        lam = np.random.beta(0.2, 0.2)

        return lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2

    def apply_physics_constraints(self, joints):
        """应用物理约束（裁剪到合法范围）"""
        if self.joint_limits is None:
            return joints

        constrained = joints.copy()
        for i, (min_val, max_val) in enumerate(self.joint_limits):
            constrained[i] = np.clip(constrained[i], min_val, max_val)

        return constrained

    def __getitem__(self, idx):
        # 设置当前索引（用于temporal_shift）
        self.current_idx = idx

        # 读取原始数据
        input_data = self.input_array[idx].copy()
        joint = self.joint_array[idx].copy()

        # 数据增强
        if self.use_augmentation and hasattr(self, 'training_mode') and self.training_mode:
            # print("Applying augmentation...")
            # 1. 旋转增强（仅对位置+四元数部分）
            if self.use_velocity:
                pose_part = input_data[:7]  # 只对前7维（位置+四元数）旋转
                pose_part = self.augment_rotation(pose_part)
                input_data[:7] = pose_part
            else:
                input_data = self.augment_rotation(input_data)

            # 2. 添加噪声
            input_data = self.add_noise(input_data, is_target=False)
            joint = self.add_noise(joint, is_target=True)

            # 3. 时间偏移（仅moderate和aggressive）
            if self.augmentation_level in ['moderate', 'aggressive']:
                if np.random.random() < 0.3:  # 30%概率
                    input_data = self.temporal_shift(input_data)

            # 4. Mixup（仅aggressive）
            if self.augmentation_level == 'aggressive':
                input_data, joint = self.mixup(input_data, joint)

            # 5. 应用物理约束
            joint = self.apply_physics_constraints(joint)

        return torch.tensor(input_data, dtype=torch.float32), \
               torch.tensor(joint, dtype=torch.float32)

    def __len__(self):
        return self.total_frames

    def train(self):
        """设置为训练模式"""
        self.training_mode = True

    def eval(self):
        """设置为评估模式"""
        self.training_mode = False


class WindowedIKDataset(Dataset):
    """
    窗口化IK数据集（用于SFU数据）

    数据格式：
    - X: (N, window_size, 14) - 前7维是位姿，后7维是角度
    - y: (N, 7) - 下一帧的关节角度

    模型输入: (window_size, 7) - 只使用位姿
    模型输出: (7,) - 下一帧的关节角度
    """

    def __init__(self, data_path, use_augmentation=False, augmentation_level='light'):
        """
        Args:
            data_path: SFU数据集路径 (包含X, y的npz文件)
            use_augmentation: 是否使用数据增强
            augmentation_level: 增强级别
        """
        super().__init__()
        self.data_path = data_path
        self.use_augmentation = use_augmentation
        self.augmentation_level = augmentation_level

        # 加载数据
        logging.info(f"加载窗口化数据集: {data_path}")
        data = np.load(data_path, allow_pickle=True)

        self.X_full = data['X']  # (N, window_size, 14) - 包含位姿和角度
        self.y = data['y']  # (N, 7) - 下一帧的角度

        # 只使用后7维（位姿）作为输入
        self.X = self.X_full[:, :, 7:]  # (N, window_size, 7)

        self.window_size = self.X.shape[1]
        self.input_dim = self.X.shape[2]  # 7维（只有位姿）
        self.output_dim = self.y.shape[1]  # 7维（角度）
        self.num_samples = self.X.shape[0]

        logging.info(f"  样本数: {self.num_samples}")
        logging.info(f"  窗口大小: {self.window_size}")
        logging.info(f"  输入维度（位姿）: {self.input_dim}")
        logging.info(f"  输出维度（角度）: {self.output_dim}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        返回:
            X: (window_size, 7) - 位姿
            y: (7,) - 下一帧的关节角度
            last_angle: (7,) - 最后一帧的关节角度（用于连续性约束）
        """
        X = self.X[idx].astype(np.float32)  # (window_size, 7)
        y = self.y[idx].astype(np.float32)  # (7,)
        last_angle = self.X_full[idx, -1, 7:].astype(np.float32)  # (7,) - 最后一帧的角度

        # 数据增强（可选）
        if self.use_augmentation and self.augmentation_level != 'none':
            X, y = self._apply_augmentation(X, y)

        return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(last_angle)

    def _apply_augmentation(self, X, y):
        """应用数据增强"""
        if self.augmentation_level == 'light':
            # 添加高斯噪声
            noise = np.random.normal(0, 0.01, X.shape).astype(np.float32)
            X = X + 0.1 * noise
        elif self.augmentation_level == 'moderate':
            # 时序抖动
            if np.random.rand() < 0.3:
                # 随机选择一个时间步进行小幅度偏移
                t = np.random.randint(1, self.window_size - 1)
                X[t] = X[t] + 0.05 * np.random.randn(*X[t].shape).astype(np.float32)

        return X, y


def create_windowed_dataloaders(data_path, config):
    """
    创建窗口化数据的数据加载器

    Args:
        data_path: SFU数据集路径
        config: 配置对象

    Returns:
        train_loader, val_loader
    """
    # 创建训练集（启用数据增强）
    train_dataset = WindowedIKDataset(
        data_path,
        use_augmentation=getattr(config, 'use_augmentation', False),
        augmentation_level=getattr(config, 'augmentation_level', 'light')
    )

    # 创建验证集（不启用数据增强）
    val_dataset = WindowedIKDataset(
        data_path,
        use_augmentation=False,  # 验证集不使用数据增强
        augmentation_level='none'
    )

    # 分割训练集和验证集（使用随机采样，避免时序偏差）
    from torch.utils.data import random_split

    total = len(train_dataset)
    train_size = int(0.9 * total)
    val_size = total - train_size

    torch.manual_seed(42)

    # 使用 random_split 进行随机划分
    # 这样可以确保训练集和验证集的数据分布一致
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=getattr(config, 'num_workers', 8),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, 'num_workers', 8),
        pin_memory=True
    )

    logging.info(f"训练集样本数: {train_size}")
    logging.info(f"验证集样本数: {val_size}")
    logging.info(f"训练集数据增强: {getattr(config, 'use_augmentation', False)}")
    logging.info(f"验证集数据增强: False（始终关闭）")

    return train_loader, val_loader


def create_generalized_dataloaders(config):
    """
    创建泛化增强的数据加载器

    Returns:
        train_loader, val_loader
    """
    # 合并所有数据集
    datasets_to_merge = [
        # ('/home/wsy/Desktop/casual/cmu/cmu_wrists_left.npz',
        #  '/home/wsy/Desktop/casual/cmu/cmu_arm_joints_left.npz'),
        # ('/home/wsy/Desktop/casual/sfu/sfu_wrists_left.npz',
        #  '/home/wsy/Desktop/casual/sfu/sfu_arm_joints_left.npz'),
        # # ('/home/wsy/Desktop/casual/grab/grab_wrists_left.npz',
        #  '/home/wsy/Desktop/casual/grab/grab_joints_left.npz'),
        # ('/home/wsy/Desktop/MultiFeatureNet/kit/kit_wrists_left.npz',
        #  '/home/wsy/Desktop/MultiFeatureNet/kit/kit_arm_joints_left.npz'),  # 添加KIT数据集
    ]

    all_inputs = []
    all_joints = []

    for wrists_path, joints_path in datasets_to_merge:
        try:
            data = np.load(wrists_path, allow_pickle=True)
            inputs = data['data']
            joints = np.load(joints_path, allow_pickle=True)['data']

            all_inputs.append(inputs)
            all_joints.append(joints)
            logging.info(f"加载 {wrists_path}: {len(inputs)} 样本")
        except Exception as e:
            logging.warning(f"跳过 {wrists_path}: {e}")

    if len(all_inputs) == 0:
        raise ValueError("没有可用的数据集")

    # 合并数据
    merged_inputs = np.vstack(all_inputs)
    merged_joints = np.vstack(all_joints)

    logging.info(f"合并后总样本数: {len(merged_inputs)}")

    # 保存合并数据（临时）
    temp_dir = '/home/wsy/Desktop/casual/combined'
    os.makedirs(temp_dir, exist_ok=True)

    temp_wrists = os.path.join(temp_dir, 'temp_wrists.npz')
    temp_joints = os.path.join(temp_dir, 'temp_joints.npz')

    np.savez_compressed(temp_wrists, data=merged_inputs)
    np.savez_compressed(temp_joints, data=merged_joints)

    # 创建泛化增强数据集
    full_dataset = GeneralizedIKDataset(
        temp_wrists,
        temp_joints,
        hand=config.hand,
        use_velocity=config.use_velocity,
        use_augmentation=True,  # 启用数据增强
        augmentation_level='light',  # 轻度增强
        physics_constraints=False
    )

    # 分割训练集和验证集（不增强验证集）
    from torch.utils.data import random_split

    total = len(full_dataset)
    train_size = int(0.9 * total)
    val_size = total - train_size

    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 训练集：启用增强
    train_dataset.dataset.train()

    # 验证集：创建不增强的版本
    val_dataset_clean = GeneralizedIKDataset(
        temp_wrists,
        temp_joints,
        hand=config.hand,
        use_velocity=config.use_velocity,
        use_augmentation=False,
        physics_constraints=True
    )

    # 提取验证集索引
    val_indices = val_dataset.indices
    val_dataset_clean = torch.utils.data.Subset(val_dataset_clean, val_indices)
    val_dataset_clean.dataset.eval()

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset_clean,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    logging.info(f"训练集: {len(train_dataset)} 样本")
    logging.info(f"验证集: {len(val_dataset)} 样本")

    return train_loader, val_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class Config:
        hand = 'left'
        use_velocity = True
        batch_size = 256

    config = Config()

    train_loader, val_loader = create_generalized_dataloaders(config)

    # 测试
    for i, (X, y) in enumerate(train_loader):
        print(f"Batch {i}: X shape={X.shape}, y shape={y.shape}")
        if i >= 2:
            break
