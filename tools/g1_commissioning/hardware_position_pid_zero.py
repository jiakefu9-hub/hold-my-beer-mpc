"""Explicit zero-neutral position PID preparation; no SDK or network imports."""
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'configs/hardware_position_pid_zero_20261013.yaml'
RELAXED_CONFIG = ROOT/'configs/hardware_position_pid_zero_relaxed_20261013.yaml'
MODEL = ROOT/'configs/g1_payload75.yaml'
EXPECTED = dict(schema='g1_position_pid_zero_v1',
    q_reference_min_deg=[-9., -5., -5., -12.8, -5.],
    q_reference_max_deg=[5., 6., 5., 5., 5.])
REVIEWED_CONFIGS = {
    CONFIG: EXPECTED,
    RELAXED_CONFIG: dict(EXPECTED, hardware_pid_max_dq=.14, hardware_pid_max_ddq=.40),
}


def prepare_profile(profile, config, model_config):
    config = Path(config).resolve()
    if config not in REVIEWED_CONFIGS or Path(model_config).resolve() != MODEL.resolve():
        raise ValueError('zero position PID requires a reviewed 20261013 config and payload75 model')
    values = yaml.safe_load(config.read_text())
    if values != REVIEWED_CONFIGS[config]:
        raise ValueError('zero position PID settings differ from the prepared experiment')
    effective = dict(profile)
    effective.pop('position_pid_parameter_overrides', None)
    for key in ('target_q_array', 'kp_array', 'kd_array', 'q_offset_limit_deg_array',
                'pid_q_offset_limit_deg_array'):
        effective[key] = np.asarray(profile[key], dtype=float).copy()
    effective['target_q_array'][:10] = 0.
    effective['position_pid_reference_bounds_rad'] = np.deg2rad([
        values['q_reference_min_deg'], values['q_reference_max_deg']])
    effective['zero_arm_neutral_experiment'] = dict(
        enabled=True, target_q_rad=effective['target_q_array'].tolist(),
        left_uses_body_imu=False, right_pid_uses_measured_body_imu=True,
        reviewed_profile_file_unchanged=True, gains_unchanged=True)
    effective['position_pid_reference_experiment'] = dict(
        **values, config_path=str(config),
        definition='absolute command-reference box; union of old nonzero +/-5 deg and zero +/-5 deg',
        measured_joint_bound=False, nominal_posture_deg=[0.]*5,
        feedback_gains_unchanged=True, reference_governor_unchanged=config == CONFIG)
    if config == RELAXED_CONFIG:
        effective['position_pid_parameter_overrides'] = {
            key: values[key] for key in ('hardware_pid_max_dq', 'hardware_pid_max_ddq')}
    return effective


def preflight(profile, parameters, model_config):
    """Exercise real PID/geometry/governor/plan, without DDS or state propagation."""
    from endpoint_pose import EndpointModel
    from hardware_pid_control import RightArmHardwarePid, HardwarePidPlan
    bounds = profile['position_pid_reference_bounds_rad']
    controller = RightArmHardwarePid(profile['target_q_array'][5:10], parameters,
        model=EndpointModel(model_config), reference_bounds_rad=bounds)
    slots = profile['target_q_array'].copy()
    plan = HardwarePidPlan(slots, slots, profile['kp_array'], profile['kd_array'], controller)
    previous_dq = None
    active = 0
    for i in range(3502):
        t = i*.006
        # Fixed synthetic feedback with a modest body pitch; no simulated plant.
        angle = np.deg2rad(3.)/2
        frame = plan.sample(t, slots, np.zeros(13), [np.cos(angle), 0., np.sin(angle), 0.], 0., .006)
        if not np.isfinite(np.r_[frame['q_rad'], frame['dq_rad_s'], frame['weight']]).all():
            raise ValueError('nonfinite zero-position PID preflight output')
        if frame['diagnostics']['pid_active']:
            active += 1
            q, dq = frame['q_rad'][5:10], frame['dq_rad_s'][5:10]
            if (np.any(q < bounds[0]-1e-12) or np.any(q > bounds[1]+1e-12)
                    or np.max(np.abs(dq)) > parameters.hardware_max_dq+1e-10):
                raise ValueError('PID reference envelope violated')
            if previous_dq is not None and np.max(np.abs(dq-previous_dq)) > parameters.hardware_max_ddq*.006+1e-10:
                raise ValueError('PID reference acceleration limit violated')
            previous_dq = dq.copy()
            np.testing.assert_array_equal(frame['q_rad'][:5], np.zeros(5))
    if frame['weight'] != 0. or not frame['terminal']:
        raise ValueError('offline PID plan did not complete its release')
    return dict(passed=True, hardware_output=False, dds_initialized=False,
        control_period_ms=6., synthetic_feedback=True, active_samples=active,
        final_weight=frame['weight'], zero_arm_neutral=profile['zero_arm_neutral_experiment'],
        reference_max_dq_rad_s=parameters.hardware_max_dq,
        reference_max_ddq_rad_s2=parameters.hardware_max_ddq,
        reference_box=profile['position_pid_reference_experiment'],
        limitations='offline compatibility and reference limits only; no closed-loop or field performance claim')
