"""Packet and feedback evidence, not an actuator calibration or output permit."""
import numpy as np

from hardware_pid_control import ARM_MOTOR_INDICES, WEIGHT_MOTOR_INDEX


def command_evidence(packet, low):
    """Snapshot IDL float32 values and the selected (earlier) state separately.

    The selected feedback precedes this Write. It is NOT a response to the new
    command. Offline analysis must join later feedback to prior successful
    writes. Casting matches the float32 IDL fields without another serialization.
    """
    motors = [packet.motor_cmd[i] for i in ARM_MOTOR_INDICES]
    fields = {name: np.asarray([getattr(m, name) for m in motors], dtype=np.float32)
              .astype(float).tolist() for name in ("q", "dq", "kp", "kd", "tau")}
    result = dict(
        execution_record_version=1,
        packet_arm_motor_indices=list(ARM_MOTOR_INDICES),
        packet_q_rad=fields["q"], packet_dq_rad_s=fields["dq"],
        packet_kp=fields["kp"], packet_kd=fields["kd"], tau_ff=fields["tau"],
        packet_weight=float(np.float32(packet.motor_cmd[WEIGHT_MOTOR_INDEX].q)),
        feedback_precedes_this_write=True,
        feedback_crc_valid=bool(low.crc_valid),
        torque_feedback_kind="robot_estimate_not_independent_shaft_sensor",
    )
    for source, name in (("tau_est", "tau_est_at_feedback_nm"),
                         ("ddq_raw", "ddq_raw_at_feedback_rad_s2")):
        values = getattr(low, source, None)
        result[name] = (None if values is None else
                       [float(values[i]) if np.isfinite(values[i]) else None
                        for i in ARM_MOTOR_INDICES])
    return result
