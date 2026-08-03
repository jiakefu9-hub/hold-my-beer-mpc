"""预测运动学后端工厂。"""

from .mujoco_backend import MujocoPredictionBackend
from .pinocchio_backend import PinocchioPredictionBackend


def create_prediction_backend(
    backend,
    *,
    mujoco_model,
    joint_names,
    mjcf_path=None,
    ee_name="right_grasp_site",
    imu_name="imu_in_torso",
):
    """按名称创建后端；两者返回完全相同的数据结构。"""

    name = str(backend).strip().lower()
    if name == "mujoco":
        return MujocoPredictionBackend(
            mujoco_model,
            joint_names,
            ee_site_name=ee_name,
            imu_site_name=imu_name,
        )
    if name == "pinocchio":
        if mjcf_path is None:
            raise ValueError("Pinocchio 后端必须提供 mjcf_path。")
        return PinocchioPredictionBackend(
            mujoco_model,
            mjcf_path,
            joint_names,
            ee_frame_name=ee_name,
            imu_frame_name=imu_name,
        )
    raise ValueError(
        f"robot model backend={backend!r} 无效，只能是 mujoco/pinocchio。"
    )
