"""
Pinocchio FK 模块

基于URDF的精确FK计算
"""

import numpy as np
import logging


# Pinocchio FK模型（全局变量）
_pinocchio_model = None
_pinocchio_data = None
_joint_ids = []
_wrist_frame_id = None


class PinocchioFK:
    """Pinocchio FK封装类"""

    def __init__(self, urdf_path):
        """
        加载Pinocchio模型

        Args:
            urdf_path: URDF文件路径
        """
        global _pinocchio_model, _pinocchio_data, _joint_ids, _wrist_frame_id

        try:
            import pinocchio as pin

            # 加载URDF
            _pinocchio_model = pin.buildModelFromUrdf(urdf_path)
            _pinocchio_data = _pinocchio_model.createData()

            # 左臂关节ID
            joint_names = [
                'left_shoulder_pitch_joint',
                'left_shoulder_roll_joint',
                'left_shoulder_yaw_joint',
                'left_elbow_joint',
                'left_wrist_roll_joint',
                'left_wrist_pitch_joint',
                'left_wrist_yaw_joint'
            ]

            # 获取关节ID
            _joint_ids = []
            for name in joint_names:
                joint_id = _pinocchio_model.getJointId(name)
                if joint_id < _pinocchio_model.njoints:
                    _joint_ids.append(joint_id)

            # 腕部link名称
            wrist_link = "left_wrist_yaw_link"
            _wrist_frame_id = _pinocchio_model.getFrameId(wrist_link)

            logging.info(f"✓ Pinocchio模型加载成功")
            logging.info(f"  URDF: {urdf_path}")
            logging.info(f"  关节数: {len(_joint_ids)}")
            logging.info(f"  腕部frame: {wrist_link} (ID: {_wrist_frame_id})")

        except ImportError:
            logging.error("❌ 未安装Pinocchio，请运行: pip install pinocchio")
            raise
        except Exception as e:
            logging.error(f"❌ 加载Pinocchio模型失败: {e}")
            raise

    def forward(self, joint_angles):
        """
        使用Pinocchio批量计算FK

        Args:
            joint_angles: [batch, 7] numpy/tensor 关节角度

        Returns:
            positions: [batch, 3] numpy tensor 腕部位置
        """
        global _pinocchio_model, _pinocchio_data, _joint_ids, _wrist_frame_id

        import torch

        batch_size = joint_angles.shape[0]

        # 转换为numpy
        if isinstance(joint_angles, torch.Tensor):
            angles_np = joint_angles.detach().cpu().numpy()
        else:
            angles_np = joint_angles

        import pinocchio as pin

        wrist_positions = []

        for i in range(batch_size):
            # 设置关节角度（零位配置）
            q = np.zeros(_pinocchio_model.nq)

            # 填入左臂角度
            for j, joint_id in enumerate(_joint_ids):
                if j < len(angles_np[i]):
                    joint = _pinocchio_model.joints[joint_id]
                    idx_q = joint.idx_q
                    if idx_q < _pinocchio_model.nq:
                        q[idx_q] = angles_np[i][j]

            # 正运动学
            pin.forwardKinematics(_pinocchio_model, _pinocchio_data, q)
            pin.updateFramePlacements(_pinocchio_model, _pinocchio_data)

            # 获取腕部位置
            wrist_pos = _pinocchio_data.oMf[_wrist_frame_id].translation
            wrist_positions.append(wrist_pos.copy())

        # 转换为tensor
        wrist_tensor = torch.tensor(
            np.array(wrist_positions),
            dtype=torch.float32
        )

        return wrist_tensor
