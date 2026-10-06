"""Field packet hand-back, without SDK initialization or model recomputation.

These are software limits and continuity checks, not calibrated torque limits
or a replacement for the robot's independent damping/stop mechanism.
"""
import numpy as np

from hardware_arm_inverse_dynamics import finite_vector
from hardware_mpc_control import HardwareMpcError
from hardware_mpc_torque_control import ACTUATION
from hardware_pid_control import ARM_MOTOR_INDICES, WeightReleaseRamp


class TorqueHandback:
    """Freeze the last successfully sent packet while weight ramps to zero.

    Transfer kd*dq_ref into feedforward when setting dq_ref=0. This preserves
    the packet's complete PD+feedforward law at ANY unchanged measured state,
    not just at the last feedback. No fresh QP or inverse dynamics is needed.
    It is a bounded-duration hand-back, not an indefinite gravity controller.
    """
    def __init__(self):
        self.last = None
        self.release = None

    def accept(self, packet):
        motors = [packet.motor_cmd[i] for i in ARM_MOTOR_INDICES]
        result = {key: np.asarray([getattr(m,key) for m in motors], dtype=np.float32).astype(float)
                  for key in ("q","dq","kp","kd","tau")}
        weight = float(np.float32(packet.motor_cmd[29].q))
        if not all(np.isfinite(x).all() for x in result.values()) or not 0 <= weight <= 1:
            raise HardwareMpcError("invalid successful packet snapshot")
        self.last = dict(**result, weight=weight)

    def apply(self, frame):
        if self.last is None:
            raise HardwareMpcError("no successful torque packet to hand back")
        last = self.last
        ff = last['tau'].copy()
        ff[5:10] += last['kd'][5:10]*last['dq'][5:10]
        frame.update(q_rad=last['q'].copy(), dq_rad_s=np.zeros(13),
                     kp=last['kp'].copy(), kd=last['kd'].copy())
        diag = frame.setdefault('diagnostics', {})
        diag.update(controller_kind=ACTUATION, mpc_active=False,
                    torque_handover=True, torque_output_authorized=False,
                    handover_rule='last_successful_packet_PD_law_preserved_then_weight_release',
                    tau_ff_candidate_nm=(ff[5:10] if frame['weight'] > 0 else np.zeros(5)).tolist(),
                    expected_kp=last['kp'][5:10].tolist(), expected_kd=last['kd'][5:10].tolist())
        return frame

    def normal_release(self, elapsed_s):
        if self.last is None:
            raise HardwareMpcError('release without a successful packet')
        if self.release is None:
            self.release = WeightReleaseRamp(self.last['weight'], .006)
        weight, terminal = self.release.sample(elapsed_s)
        return self.apply(dict(stage='arm_ramp_out' if not terminal else 'complete',
                               weight=weight, terminal=terminal, diagnostics={}))


def check_field_packet(frame, low, config):
    """Check finite emitted values and total-torque estimate at current feedback.

    Emergency/mode interlocks are separate and remain in the shared runner.
    Estimate != independently measured actuator output.
    """
    q = finite_vector(frame['q_rad'],13,'command q')
    dq = finite_vector(frame['dq_rad_s'],13,'command dq')
    kp = finite_vector(frame['kp'],13,'command kp')
    kd = finite_vector(frame['kd'],13,'command kd')
    ff = finite_vector(frame['diagnostics']['tau_ff_candidate_nm'],5,'command feedforward')
    actual_q = finite_vector(low.q[list(ARM_MOTOR_INDICES)],13,'feedback q')
    actual_dq = finite_vector(low.dq[list(ARM_MOTOR_INDICES)],13,'feedback dq')
    total = ff+kp[5:10]*(q[5:10]-actual_q[5:10])+kd[5:10]*(dq[5:10]-actual_dq[5:10])
    frame['diagnostics']['field_total_torque_estimate_at_latest_feedback_nm'] = total.tolist()
    active = bool(frame['diagnostics'].get('mpc_active'))
    # Ramp-in/settle/hand-back are position-PD transitions with support FF.
    # Unlike the PID baseline this includes FF; physical behavior is NOT proven
    # by PID success. The MPC total envelope applies only to the active phase.
    # Feedforward remains bounded at every nonzero ownership weight; once MPC
    # is active, both feedforward and the full PD+feedforward estimate apply.
    if frame['weight'] > 0 and np.any(abs(ff)>np.asarray(config['tau_ff_abs_nm'])+1e-6):
        raise HardwareMpcError('field feedforward torque envelope exceeded; hand back')
    if active and np.any(abs(total)>np.asarray(config['tau_abs_nm'])+1e-6):
        raise HardwareMpcError('field MPC total torque envelope exceeded; hand back')
    frame['diagnostics']['field_total_torque_envelope_applied'] = active
    if active:
        lo, hi = np.deg2rad(config['q_min_deg']), np.deg2rad(config['q_max_deg'])
        if np.any(actual_q[5:10]<lo) or np.any(actual_q[5:10]>hi):
            raise HardwareMpcError('measured arm outside field MPC position envelope; hand back')
