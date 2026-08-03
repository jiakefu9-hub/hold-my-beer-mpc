"""MuJoCo 参考预测运动学后端。"""

import mujoco
import numpy as np

from .base import PredictionKinematics, PredictionKinematicsBackend


class MujocoPredictionBackend(PredictionKinematicsBackend):
    """沿用当前 scratch ``MjData`` 语义的参考实现。"""

    backend_name = "mujoco"

    def __init__(
        self,
        model,
        joint_names,
        *,
        ee_site_name="right_grasp_site",
        imu_site_name="imu_in_torso",
    ):
        self.model = model
        self.joint_names = tuple(joint_names)
        self.joint_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                for name in self.joint_names
            ],
            dtype=np.int32,
        )
        if np.any(self.joint_ids < 0):
            missing = [
                name
                for name, joint_id in zip(self.joint_names, self.joint_ids)
                if joint_id < 0
            ]
            raise ValueError(f"MuJoCo 模型缺少关节: {missing}")
        self.qpos_indices = model.jnt_qposadr[self.joint_ids].astype(
            np.int32
        )
        self.qvel_indices = model.jnt_dofadr[self.joint_ids].astype(
            np.int32
        )
        if any(
            int(model.jnt_type[joint_id])
            not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            )
            for joint_id in self.joint_ids
        ):
            raise ValueError("预测关节必须全是单自由度 hinge/slide。")

        self.ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name
        )
        self.imu_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, imu_site_name
        )
        if self.ee_site_id < 0 or self.imu_site_id < 0:
            raise ValueError(
                f"MuJoCo 模型缺少 site: {ee_site_name!r} "
                f"或 {imu_site_name!r}"
            )
        self._scratch = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv), dtype=np.float64)
        self._jacr = np.zeros((3, model.nv), dtype=np.float64)
        self._djacp = np.zeros((3, model.nv), dtype=np.float64)
        self._djacr = np.zeros((3, model.nv), dtype=np.float64)

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
        scratch = self._scratch
        scratch.qpos[:] = qpos_reference
        scratch.qvel[:] = 0.0
        scratch.qpos[self.qpos_indices] = q_arm
        scratch.qvel[self.qvel_indices] = dq_arm

        # 【核心】预测只需位置和速度阶段，不求解接触动力学。
        mujoco.mj_fwdPosition(self.model, scratch)
        mujoco.mj_fwdVelocity(self.model, scratch)
        self._jacp.fill(0.0)
        self._jacr.fill(0.0)
        mujoco.mj_jacSite(
            self.model,
            scratch,
            self._jacp,
            self._jacr,
            self.ee_site_id,
        )

        if acceleration_required:
            self._djacp.fill(0.0)
            self._djacr.fill(0.0)
            mujoco.mj_jacDot(
                self.model,
                scratch,
                self._djacp,
                self._djacr,
                scratch.site_xpos[self.ee_site_id],
                int(self.model.site_bodyid[self.ee_site_id]),
            )
            dJ_v = self._djacp[:, self.qvel_indices].copy()
            dJ_w = self._djacr[:, self.qvel_indices].copy()
        else:
            shape = (3, len(self.joint_names))
            dJ_v = np.zeros(shape, dtype=np.float64)
            dJ_w = np.zeros(shape, dtype=np.float64)

        return PredictionKinematics(
            ee_position_world=scratch.site_xpos[self.ee_site_id].copy(),
            ee_rotation_world=scratch.site_xmat[self.ee_site_id]
            .reshape(3, 3)
            .copy(),
            imu_position_world=scratch.site_xpos[self.imu_site_id].copy(),
            imu_rotation_world=scratch.site_xmat[self.imu_site_id]
            .reshape(3, 3)
            .copy(),
            J_v_world=self._jacp[:, self.qvel_indices].copy(),
            J_w_world=self._jacr[:, self.qvel_indices].copy(),
            dJ_v_world=dJ_v,
            dJ_w_world=dJ_w,
        )

    def _validate_inputs(self, qpos_reference, q_arm, dq_arm):
        qpos = np.asarray(qpos_reference, dtype=np.float64)
        q = np.asarray(q_arm, dtype=np.float64)
        dq = np.asarray(dq_arm, dtype=np.float64)
        expected = (len(self.joint_names),)
        if qpos.shape != (self.model.nq,):
            raise ValueError(
                f"qpos_reference 必须为 {(self.model.nq,)}，当前 {qpos.shape}"
            )
        if q.shape != expected or dq.shape != expected:
            raise ValueError(
                f"q_arm/dq_arm 必须为 {expected}，当前 {q.shape}/{dq.shape}"
            )
        if not all(np.all(np.isfinite(x)) for x in (qpos, q, dq)):
            raise ValueError("预测运动学输入包含 NaN 或 Inf。")
        return qpos, q, dq
