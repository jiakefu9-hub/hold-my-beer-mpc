"""Pinocchio 预测运动学后端。"""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .base import PredictionKinematics, PredictionKinematicsBackend

try:
    import pinocchio as pin
except ImportError as error:  # pragma: no cover - 由运行环境决定
    pin = None
    _PINOCCHIO_IMPORT_ERROR = error
else:
    _PINOCCHIO_IMPORT_ERROR = None


def resolve_robot_mjcf_path(mjcf_path):
    """Pinocchio 不解析 scene.xml 的 include，因此定位其中的机器人 MJCF。"""

    path = Path(mjcf_path).expanduser().resolve()
    root = ET.parse(path).getroot()
    includes = root.findall("include")
    if not includes:
        return path
    if len(includes) != 1 or not includes[0].get("file"):
        raise ValueError(
            "Pinocchio 后端只能自动解析含单个 include 的 scene MJCF。"
        )
    included = (path.parent / includes[0].get("file")).resolve()
    if not included.is_file():
        raise FileNotFoundError(f"MJCF include 不存在: {included}")
    return included


class PinocchioPredictionBackend(PredictionKinematicsBackend):
    """从与 MuJoCo 仿真相同的 MJCF 构造 Pinocchio 模型。"""

    backend_name = "pinocchio"

    def __init__(
        self,
        mujoco_model,
        mjcf_path,
        joint_names,
        *,
        ee_frame_name="right_grasp_site",
        imu_frame_name="imu_in_torso",
    ):
        if pin is None:
            raise ImportError(
                "Pinocchio 后端需要安装 pinocchio 包。"
            ) from _PINOCCHIO_IMPORT_ERROR
        self.mujoco_model = mujoco_model
        self.mjcf_path = resolve_robot_mjcf_path(mjcf_path)
        # 【核心】直接读取仿真所用 MJCF，避免 URDF/MJCF 负载与 frame 偏置漂移。
        self.model = pin.buildModelFromMJCF(str(self.mjcf_path))
        self.data = self.model.createData()
        self.joint_names = tuple(joint_names)
        self.ee_frame_id = self._frame_id(ee_frame_name)
        self.imu_frame_id = self._frame_id(imu_frame_name)
        self._joint_mapping = self._build_joint_mapping()
        (
            self._mj_q_for_pin_q,
            self._pin_scalar_v_indices,
            self._mj_scalar_v_indices,
            self._pin_free_q_index,
            self._mj_free_q_index,
            self._pin_free_v_index,
            self._mj_free_v_index,
        ) = self._build_full_state_mapping()
        self.pin_arm_v_indices = np.asarray(
            [item["pin_v"] for item in self._joint_mapping], dtype=np.int32
        )
        self.pin_arm_q_indices = np.asarray(
            [item["pin_q"] for item in self._joint_mapping], dtype=np.int32
        )
        self.mj_arm_q_indices = np.asarray(
            [item["mj_q"] for item in self._joint_mapping], dtype=np.int32
        )
        self.mj_arm_v_indices = np.asarray(
            [item["mj_v"] for item in self._joint_mapping], dtype=np.int32
        )
        # 预测和 RNEA 各自复用 data；工作向量也只在初始化时分配。
        self.inverse_data = self.model.createData()
        self._q_work = np.empty(self.model.nq, dtype=np.float64)
        self._v_work = np.zeros(self.model.nv, dtype=np.float64)
        self._a_work = np.zeros(self.model.nv, dtype=np.float64)

    def _frame_id(self, name):
        frame_id = int(self.model.getFrameId(name))
        if frame_id >= len(self.model.frames) or self.model.frames[frame_id].name != name:
            raise ValueError(f"Pinocchio MJCF 模型缺少 frame: {name!r}")
        return frame_id

    def _build_joint_mapping(self):
        mapping = []
        for name in self.joint_names:
            mj_id = mujoco.mj_name2id(
                self.mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            pin_id = int(self.model.getJointId(name))
            if mj_id < 0 or pin_id == 0 or self.model.names[pin_id] != name:
                raise ValueError(f"MuJoCo/Pinocchio 关节名不匹配: {name!r}")
            pin_joint = self.model.joints[pin_id]
            if pin_joint.nq != 1 or pin_joint.nv != 1:
                raise ValueError(f"右臂关节 {name!r} 不是单自由度。")
            mapping.append(
                {
                    "name": name,
                    "mj_q": int(self.mujoco_model.jnt_qposadr[mj_id]),
                    "mj_v": int(self.mujoco_model.jnt_dofadr[mj_id]),
                    "pin_q": int(pin_joint.idx_q),
                    "pin_v": int(pin_joint.idx_v),
                }
            )
        return tuple(mapping)

    def _build_full_state_mapping(self):
        """预计算整机 q/v 索引；实时路径不再逐关节查名字。"""

        mj_q_for_pin_q = np.full(self.model.nq, -1, dtype=np.int32)
        pin_scalar_v = []
        mj_scalar_v = []
        free_mapping = None
        for pin_id in range(1, self.model.njoints):
            name = self.model.names[pin_id]
            mj_id = mujoco.mj_name2id(
                self.mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if mj_id < 0:
                raise ValueError(f"MuJoCo 模型缺少 Pinocchio 关节 {name!r}")
            joint = self.model.joints[pin_id]
            pin_q = int(joint.idx_q)
            pin_v = int(joint.idx_v)
            mj_q = int(self.mujoco_model.jnt_qposadr[mj_id])
            mj_v = int(self.mujoco_model.jnt_dofadr[mj_id])
            mj_type = int(self.mujoco_model.jnt_type[mj_id])
            if joint.nq == 1 and joint.nv == 1:
                mj_q_for_pin_q[pin_q] = mj_q
                pin_scalar_v.append(pin_v)
                mj_scalar_v.append(mj_v)
            elif joint.nq == 7 and joint.nv == 6 and mj_type == int(
                mujoco.mjtJoint.mjJNT_FREE
            ):
                # Pin q: xyz + xyzw；MuJoCo q: xyz + wxyz。
                mj_q_for_pin_q[pin_q : pin_q + 7] = np.asarray(
                    [mj_q, mj_q + 1, mj_q + 2, mj_q + 4, mj_q + 5,
                     mj_q + 6, mj_q + 3],
                    dtype=np.int32,
                )
                if free_mapping is not None:
                    raise ValueError("当前后端只支持一个 floating base。")
                free_mapping = (pin_q, mj_q, pin_v, mj_v)
            elif joint.nq == 4 and joint.nv == 3 and mj_type == int(
                mujoco.mjtJoint.mjJNT_BALL
            ):
                mj_q_for_pin_q[pin_q : pin_q + 4] = np.asarray(
                    [mj_q + 1, mj_q + 2, mj_q + 3, mj_q],
                    dtype=np.int32,
                )
                # 本项目没有 ball joint；若未来加入，需要明确角速度坐标约定。
                raise ValueError("当前后端尚未支持 ball joint 速度映射。")
            else:
                raise ValueError(
                    f"暂不支持关节 {name!r} 的 nq/nv={joint.nq}/{joint.nv}。"
                )
        if np.any(mj_q_for_pin_q < 0) or free_mapping is None:
            raise ValueError("Pinocchio/MuJoCo 整机状态映射不完整。")
        return (
            mj_q_for_pin_q,
            np.asarray(pin_scalar_v, dtype=np.int32),
            np.asarray(mj_scalar_v, dtype=np.int32),
            *free_mapping,
        )

    def mujoco_qpos_to_pinocchio(self, qpos_mujoco):
        """按关节名转换整机 q，显式处理 free-joint 四元数顺序。"""

        q_mj = np.asarray(qpos_mujoco, dtype=np.float64)
        if q_mj.shape != (self.mujoco_model.nq,):
            raise ValueError(
                f"MuJoCo qpos 必须为 {(self.mujoco_model.nq,)}，"
                f"当前 {q_mj.shape}"
            )
        return q_mj[self._mj_q_for_pin_q].copy()

    def evaluate(
        self,
        qpos_reference,
        q_arm,
        dq_arm,
        *,
        acceleration_required=True,
    ):
        qpos_reference, q_arm, dq_arm = self._validate_inputs(
            qpos_reference, q_arm, dq_arm
        )
        q_pin = self._q_work
        q_pin[:] = qpos_reference[self._mj_q_for_pin_q]
        q_pin[self.pin_arm_q_indices] = q_arm
        v_pin = self._v_work
        v_pin.fill(0.0)
        v_pin[self.pin_arm_v_indices] = dq_arm

        # 【核心】LOCAL_WORLD_ALIGNED 与 mj_jacSite/mj_jacDot 的世界系表达对齐。
        if acceleration_required:
            pin.computeJointJacobiansTimeVariation(
                self.model, self.data, q_pin, v_pin
            )
        else:
            pin.computeJointJacobians(self.model, self.data, q_pin)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(
            self.model,
            self.data,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        if acceleration_required:
            dJ = pin.getFrameJacobianTimeVariation(
                self.model,
                self.data,
                self.ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
        else:
            dJ = np.zeros_like(J)

        ee = self.data.oMf[self.ee_frame_id]
        imu = self.data.oMf[self.imu_frame_id]
        columns = self.pin_arm_v_indices
        # Pinocchio Motion/Jacobian 的前 3 行是线速度，后 3 行是角速度。
        return PredictionKinematics(
            ee_position_world=np.asarray(ee.translation).copy(),
            ee_rotation_world=np.asarray(ee.rotation).copy(),
            imu_position_world=np.asarray(imu.translation).copy(),
            imu_rotation_world=np.asarray(imu.rotation).copy(),
            J_v_world=np.asarray(J[:3, columns]).copy(),
            J_w_world=np.asarray(J[3:, columns]).copy(),
            dJ_v_world=np.asarray(dJ[:3, columns]).copy(),
            dJ_w_world=np.asarray(dJ[3:, columns]).copy(),
        )

    def compute_right_arm_rnea(
        self,
        qpos_mujoco,
        qvel_mujoco,
        desired_arm_qacc,
        reference_qacc=None,
    ):
        """计算整机 RNEA 后取右臂 5 维；不包含被动力和接触约束。"""

        qpos = np.asarray(qpos_mujoco, dtype=np.float64)
        qvel = np.asarray(qvel_mujoco, dtype=np.float64)
        desired = np.asarray(desired_arm_qacc, dtype=np.float64)
        if qpos.shape != (self.mujoco_model.nq,):
            raise ValueError("RNEA qpos 维度不正确。")
        if qvel.shape != (self.mujoco_model.nv,):
            raise ValueError("RNEA qvel 维度不正确。")
        if desired.shape != (len(self.joint_names),):
            raise ValueError("RNEA 右臂 qacc 维度不正确。")
        reference = (
            np.zeros(self.mujoco_model.nv, dtype=np.float64)
            if reference_qacc is None
            else np.asarray(reference_qacc, dtype=np.float64)
        )
        if reference.shape != (self.mujoco_model.nv,):
            raise ValueError("RNEA reference_qacc 维度不正确。")
        if not all(
            np.all(np.isfinite(x)) for x in (qpos, qvel, desired, reference)
        ):
            raise ValueError("RNEA 输入包含 NaN 或 Inf。")

        q_pin = self._q_work
        q_pin[:] = qpos[self._mj_q_for_pin_q]
        v_pin = self._v_work
        v_pin.fill(0.0)
        v_pin[self._pin_scalar_v_indices] = qvel[self._mj_scalar_v_indices]

        w, x, y, z = qpos[
            self._mj_free_q_index + 3 : self._mj_free_q_index + 7
        ]
        R_world_base = np.asarray(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        pin_free_v = self._pin_free_v_index
        mj_free_v = self._mj_free_v_index
        v_pin[pin_free_v : pin_free_v + 3] = (
            R_world_base.T @ qvel[mj_free_v : mj_free_v + 3]
        )
        # MuJoCo free-joint 角速度与 Pinocchio free-flyer 都在 body 系表达。
        v_pin[pin_free_v + 3 : pin_free_v + 6] = qvel[
            mj_free_v + 3 : mj_free_v + 6
        ]

        a_pin = self._a_work
        a_pin.fill(0.0)
        a_pin[self._pin_scalar_v_indices] = reference[
            self._mj_scalar_v_indices
        ]
        # MuJoCo free-joint 平动加速度是世界系导数；Pinocchio 使用 body
        # 空间加速度，因此还需旋转并减去 omega x v 的坐标导数项。
        a_pin[pin_free_v : pin_free_v + 3] = (
            R_world_base.T @ reference[mj_free_v : mj_free_v + 3]
            - np.cross(
            v_pin[pin_free_v + 3 : pin_free_v + 6],
            v_pin[pin_free_v : pin_free_v + 3],
            )
        )
        a_pin[pin_free_v + 3 : pin_free_v + 6] = reference[
            mj_free_v + 3 : mj_free_v + 6
        ]
        a_pin[self.pin_arm_v_indices] = desired
        tau = pin.rnea(
            self.model, self.inverse_data, q_pin, v_pin, a_pin
        )
        return np.asarray(tau[self.pin_arm_v_indices]).copy()

    def _validate_inputs(self, qpos_reference, q_arm, dq_arm):
        qpos = np.asarray(qpos_reference, dtype=np.float64)
        q = np.asarray(q_arm, dtype=np.float64)
        dq = np.asarray(dq_arm, dtype=np.float64)
        expected = (len(self.joint_names),)
        if qpos.shape != (self.mujoco_model.nq,):
            raise ValueError(
                f"qpos_reference 必须为 {(self.mujoco_model.nq,)}，"
                f"当前 {qpos.shape}"
            )
        if q.shape != expected or dq.shape != expected:
            raise ValueError(
                f"q_arm/dq_arm 必须为 {expected}，当前 {q.shape}/{dq.shape}"
            )
        if not all(np.all(np.isfinite(x)) for x in (qpos, q, dq)):
            raise ValueError("预测运动学输入包含 NaN 或 Inf。")
        return qpos, q, dq
