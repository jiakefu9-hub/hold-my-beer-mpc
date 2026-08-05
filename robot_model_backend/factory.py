"""预测运动学后端工厂。"""

from .cpp_rnea_backend import (
    EE_FRAME_NAME,
    IMU_FRAME_NAME,
    RIGHT_ARM_JOINT_NAMES,
    CppRightArmRneaBackend,
)
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
    """按名称创建后端；各实现返回完全相同的数据结构。"""

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
    if name == "cpp_pinocchio":
        if mjcf_path is None:
            raise ValueError("C++ Pinocchio 后端必须提供 mjcf_path。")
        if tuple(joint_names) != RIGHT_ARM_JOINT_NAMES:
            raise ValueError(
                "C++ Pinocchio ABI v2 当前只支持固定顺序的右臂 "
                "5 关节。"
            )
        if ee_name != EE_FRAME_NAME or imu_name != IMU_FRAME_NAME:
            raise ValueError(
                "C++ Pinocchio ABI v2 当前只支持 "
                "right_grasp_site/imu_in_torso。"
            )
        result = CppRightArmRneaBackend(mjcf_path)
        if result.nq != int(mujoco_model.nq) or result.nv != int(
            mujoco_model.nv
        ):
            result.close()
            raise ValueError(
                "C++ Pinocchio 与当前 MuJoCo 模型 nq/nv 不一致。"
            )
        return result
    raise ValueError(
        f"robot model backend={backend!r} 无效，只能是 "
        "mujoco/pinocchio/cpp_pinocchio。"
    )
