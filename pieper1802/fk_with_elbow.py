"""
增强版FK - 使用Pinocchio精确计算肘部和末端位置
"""

import torch
import torch.nn as nn
import numpy as np
import logging

# Pinocchio模型（全局单例）
_pinocchio_model = None
_pinocchio_data = None
_joint_ids = []
_elbow_frame_id = None
_wrist_frame_id = None


def init_pinocchio(urdf_path="/home/bonuli/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"):
    """
    初始化Pinocchio模型（全局单例）

    Args:
        urdf_path: URDF文件路径
    """
    global _pinocchio_model, _pinocchio_data, _joint_ids, _elbow_frame_id, _wrist_frame_id

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

        # 肘部和腕部link名称
        elbow_link = "left_elbow_link"
        wrist_link = "left_wrist_yaw_link"

        _elbow_frame_id = _pinocchio_model.getFrameId(elbow_link)
        _wrist_frame_id = _pinocchio_model.getFrameId(wrist_link)

        logging.info(f"✓ Pinocchio模型初始化成功")
        logging.info(f"  URDF: {urdf_path}")
        logging.info(f"  关节数: {len(_joint_ids)}")
        logging.info(f"  肘部frame: {elbow_link} (ID: {_elbow_frame_id})")
        logging.info(f"  腕部frame: {wrist_link} (ID: {_wrist_frame_id})")

        return True

    except ImportError:
        logging.error("❌ 未安装Pinocchio，请运行: pip install pinocchio")
        return False
    except Exception as e:
        logging.error(f"❌ 初始化Pinocchio模型失败: {e}")
        return False


class FKWithElbow:
    """
    使用Pinocchio计算肘部和末端位置的FK

    Unitree G1左臂结构：
    - J0-J2: Shoulder (肩)
    - J3: Elbow (肘)
    - J4: Forearm (前臂)
    - J5-J6: Wrist (腕)
    """

    def __init__(self):
        """初始化FK - 使用Pinocchio"""
        # 确保Pinocchio已初始化
        global _pinocchio_model, _joint_ids, _elbow_frame_id, _wrist_frame_id

        if _pinocchio_model is None:
            # 自动初始化
            if not init_pinocchio():
                raise RuntimeError("无法初始化Pinocchio模型")

    def forward(self, joint_angles):
        """
        计算末端位置

        Args:
            joint_angles: [batch, 7] 关节角度

        Returns:
            end_pos: [batch, 3] 末端位置
        """
        return self.compute_positions(joint_angles, return_elbow=False)
    
    def forward(self, joint_angles):
        """
        计算末端位置
        
        Args:
            joint_angles: [batch, 7] 关节角度
            
        Returns:
            end_pos: [batch, 3] 末端位置
        """
        return self.compute_positions(joint_angles, return_elbow=False)
    
    def compute_positions(self, joint_angles, return_elbow=True):
        """
        使用Pinocchio计算肘部和末端位置

        Args:
            joint_angles: [batch, 7] 关节角度
            return_elbow: 是否返回肘部位置

        Returns:
            if return_elbow:
                (elbow_pos, end_pos)
            else:
                end_pos
        """
        global _pinocchio_model, _pinocchio_data, _joint_ids, _elbow_frame_id, _wrist_frame_id

        if _pinocchio_model is None:
            raise RuntimeError("Pinocchio模型未初始化，请先调用init_pinocchio()")

        batch_size = joint_angles.shape[0]
        device = joint_angles.device

        # 转换为numpy
        if isinstance(joint_angles, torch.Tensor):
            angles_np = joint_angles.detach().cpu().numpy()
        else:
            angles_np = joint_angles

        elbow_positions = []
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
            import pinocchio as pin
            pin.forwardKinematics(_pinocchio_model, _pinocchio_data, q)
            pin.updateFramePlacements(_pinocchio_model, _pinocchio_data)

            # 获取肘部位置
            if return_elbow and _elbow_frame_id < len(_pinocchio_data.oMf):
                elbow_pos = _pinocchio_data.oMf[_elbow_frame_id].translation
                elbow_positions.append(elbow_pos.copy())

            # 获取腕部位置
            if _wrist_frame_id < len(_pinocchio_data.oMf):
                wrist_pos = _pinocchio_data.oMf[_wrist_frame_id].translation
                wrist_positions.append(wrist_pos.copy())

        # 转换为tensor
        wrist_tensor = torch.tensor(np.array(wrist_positions),
                                    dtype=torch.float32, device=device)

        if return_elbow:
            elbow_tensor = torch.tensor(np.array(elbow_positions),
                                        dtype=torch.float32, device=device)
            return elbow_tensor, wrist_tensor

        return wrist_tensor


class MultiPositionLoss(nn.Module):
    """
    多位置损失：同时约束肘部和末端
    """
    
    def __init__(self, elbow_weight=0.3, end_weight=0.7):
        super().__init__()
        self.elbow_weight = elbow_weight
        self.end_weight = end_weight
        self.mse = nn.MSELoss()
    
    def forward(self, pred_elbow, pred_end, target_elbow, target_end):
        """
        Args:
            pred_elbow: [batch, 3] 预测肘部位置
            pred_end: [batch, 3] 预测末端位置
            target_elbow: [batch, 3] 目标肘部位置
            target_end: [batch, 3] 目标末端位置
        """
        elbow_loss = self.mse(pred_elbow, target_elbow)
        end_loss = self.mse(pred_end, target_end)
        
        total_loss = self.elbow_weight * elbow_loss + self.end_weight * end_loss
        
        return total_loss, {
            'elbow_loss': elbow_loss.item(),
            'end_loss': end_loss.item(),
            'total': total_loss.item()
        }


# 测试
def test_fk_with_elbow():
    """测试增强版FK"""
    print("=" * 60)
    print("测试增强版FK")
    print("=" * 60)
    
    fk = FKWithElbow()
    
    # 测试数据
    joint_angles = torch.tensor([
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],  # 简单姿态
        [0.2, 0.1, 0.0, 0.8, 0.1, 0.2, 0.0],  # 弯曲姿态
    ]).cuda()
    
    # 计算位置
    elbow_pos, end_pos = fk.compute_positions(joint_angles)
    
    print(f"\n样本1:")
    print(f"  肘部位置: {elbow_pos[0].cpu().numpy().round(4)}")
    print(f"  末端位置: {end_pos[0].cpu().numpy().round(4)}")
    
    print(f"\n样本2:")
    print(f"  肘部位置: {elbow_pos[1].cpu().numpy().round(4)}")
    print(f"  末端位置: {end_pos[1].cpu().numpy().round(4)}")
    
    # 测试损失
    print(f"\n测试多位置损失...")
    criterion = MultiPositionLoss(elbow_weight=0.3, end_weight=0.7)
    
    target_elbow = elbow_pos + 0.01  # 模拟误差
    target_end = end_pos + 0.02
    
    loss, loss_dict = criterion(elbow_pos, end_pos, target_elbow, target_end)
    
    print(f"  损失: {loss_dict['total']:.6f}")
    print(f"    - 肘部: {loss_dict['elbow_loss']:.6f}")
    print(f"    - 末端: {loss_dict['end_loss']:.6f}")
    
    print("\n✓ 测试完成")


if __name__ == "__main__":
    test_fk_with_elbow()
