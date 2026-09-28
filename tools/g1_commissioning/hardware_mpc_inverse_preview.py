"""Offline MPC -> inverse-dynamics feedforward candidate, never live output.

Retains the existing reference-trajectory QP; adds the missing physical torque
feedforward using measured q/dq and current estimated torso motion. Firmware
PD remains a separate tracking correction, not a second copy of feedforward.
This does not claim exact acceleration tracking or a calibrated servo model.
"""
from __future__ import annotations

import time

import numpy as np

from hardware_arm_inverse_dynamics import RightArmInverseDynamics, finite_vector
from hardware_mpc_control import RightArmHardwareMpc, HardwareMpcError

ACTUATION = "inverse_dynamics_preview"


class RightArmInversePreviewMpc(RightArmHardwareMpc):
    offline_only = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse = RightArmInverseDynamics(self.model, self.backend)
        self.metadata.update({
            "model": "reference_trajectory_mpc_with_inverse_dynamics_preview",
            "actuation": "OFFLINE tau_ff=RNEA(measured q,dq,governed ddq,torso motion); firmware PD once",
            "offline_only": True,
            "inverse_dynamics": self.inverse.metadata,
            "field_blockers": [
                "payload/inertia and torque estimate not independently calibrated",
                "no verified tau sign/response or acceleration tracking on this robot",
                "torque entry/release/fault envelopes not yet commissioned",
            ],
        })

    def step(self, arm_slots, imu_quaternion_wxyz, yaw0_rad, dt):
        start = time.perf_counter_ns()
        if self._horizon is None:
            raise HardwareMpcError("a fresh disturbance horizon is required")
        # Only the current node drives this feedforward. Never substitute a
        # future template prediction for an already measured base condition.
        current_base = self._horizon.nodes[0]
        qref, dqref, diagnostics = super().step(arm_slots, imu_quaternion_wxyz, yaw0_rad, dt)
        measured_q = np.asarray(arm_slots, dtype=float)[5:10]
        measured_dq = self._measured_dq.copy()
        inverse = self.inverse.compute(
            measured_q, measured_dq,
            diagnostics["governed_ddq_reference_rad_s2"], current_base,
        )
        tau_ff = inverse["tau_model_nm"]
        # Audit only: the Arm SDK must add this PD once, not receive it twice.
        pd = 20.*(qref-measured_q) + 1.*(dqref-measured_dq)
        diagnostics.update({
            "controller_kind": "inverse_dynamics_preview",
            "inverse_dynamics": inverse,
            "tau_ff_candidate_nm": tau_ff.tolist(),
            "tau_pd_at_feedback_nm": pd.tolist(),
            "tau_total_estimated_at_feedback_nm": (tau_ff+pd).tolist(),
            "torque_output_authorized": False,
            "torque_estimate_feedback_used_for_control": False,
            "controller_core_ms": (time.perf_counter_ns()-start)*1e-6,
        })
        return qref, dqref, diagnostics


def make_offline_preview_message(frame, state, constructor, crc):
    """Construct/serialize a candidate packet locally; not used by run_device.

    Non-active stages retain zero torque: this is deliberately not a complete
    live takeover/hand-back plan. The CLI and run_device reject this backend.
    """
    from g1_walk_pid import make_arm_message
    message = make_arm_message(frame, state, constructor, crc)
    diagnostics = frame.get("diagnostics", {})
    if diagnostics.get("controller_kind") == ACTUATION:
        torque = finite_vector(diagnostics["tau_ff_candidate_nm"], 5, "candidate torque")
        for motor, value in zip(range(22, 27), torque):
            message.motor_cmd[motor].tau = float(value)
        message.crc = crc.Crc(message)
    return message
