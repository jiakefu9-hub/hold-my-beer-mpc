"""One authoritative field schedule. Historical controller defaults stay separate."""
import hashlib
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_MPC_CONFIG = ROOT / 'configs/hardware_mpc_learned_acc001_alpha0005_omega1.yaml'
MODEL_CONFIG = ROOT / 'configs/g1_payload75.yaml'
CONFIG_DIR = ROOT / 'configs/hardware_control_comparison_20261012'
ORIGINAL_TORQUE_CONFIG = ROOT / 'configs/hardware_mpc_torque_learned.yaml'
BASELINE_CONFIG = ORIGINAL_MPC_CONFIG
TORQUE_CONFIG = ORIGINAL_TORQUE_CONFIG
ORIGINAL_TORQUE_SHA256 = '41e153ca138147e2dc006bb7f15a511872a69e5299f925c401322234f703b879'
CONFIG_PATH = ROOT / 'configs/hardware_acceleration_pid_20261012.yaml'
BASELINE_VALUES = dict(
    schema='g1_hardware_mpc_config_v1', q_ee_acc=.01, q_ee_alpha=.0005,
    q_ee_omega=1., mpc_start_s=3.3, mpc_handoff_duration_s=1.5,
    q_gravity=[30., 30.], q_posture=[4., 4., 1.5, .2, .2],
    q_vel=.08, r_ddq=.0025, terminal_scale=2.)

TRIALS = {
    'T01': dict(name='mpc_original_timing', aligned=False, config=BASELINE_CONFIG,
                pid_config=None, purpose='MPC 原权重、时间对齐关闭；当天对照'),
    'T02': dict(name='mpc_aligned', aligned=True, config=BASELINE_CONFIG,
                pid_config=None, purpose='MPC 原权重、只开时间对齐；对照 T01'),
    'T03': dict(name='c_aligned', aligned=True, config=BASELINE_CONFIG,
                pid_config=CONFIG_PATH, purpose='C 前馈 PID、时间对齐开启；对照 T02'),
    'T04': dict(name='mpc_aligned_repeat', aligned=True, config=BASELINE_CONFIG,
                pid_config=None, purpose='重复 T02；检查 MPC 重复性'),
    'T05': dict(name='c_aligned_repeat', aligned=True, config=BASELINE_CONFIG,
                pid_config=CONFIG_PATH, purpose='重复 T03；检查 C 重复性'),
    'T06': dict(name='mpc_acc002_optional', aligned=True,
                config=CONFIG_DIR/'mpc_acc002.yaml',
                pid_config=None, purpose='可选：MPC 仅线加速度权重 0.02；对照 T02/T04'),
}

COMMON_VALUES = dict(BASELINE_VALUES)
CANDIDATE_CONFIGS = frozenset((BASELINE_CONFIG.resolve(), TRIALS['T06']['config'].resolve()))
ZERO_NEUTRAL_MPC_CONFIGS = CANDIDATE_CONFIGS


def validate_trial_config(config):
    path = Path(config).resolve()
    if path in CANDIDATE_CONFIGS:
        expected = dict(COMMON_VALUES, q_ee_acc=.02) if path == TRIALS['T06']['config'].resolve() else COMMON_VALUES
        if yaml.safe_load(path.read_text()) != expected:
            raise ValueError('T series configuration differs from its reviewed overlay')


def validate_comparison_torque(config):
    if not isinstance(config, (str, Path)) or Path(config).resolve() != TORQUE_CONFIG.resolve():
        return  # historical paths retain their original admission rules
    actual = hashlib.sha256(TORQUE_CONFIG.read_bytes()).hexdigest()
    if actual != ORIGINAL_TORQUE_SHA256:
        raise ValueError('comparison torque config differs from the successful field baseline')


def validate_series():
    if yaml.safe_load(ORIGINAL_MPC_CONFIG.read_text()) != BASELINE_VALUES:
        raise ValueError('164546 baseline changed; comparison requires revalidation')
    for config in CANDIDATE_CONFIGS:
        validate_trial_config(config)
    validate_comparison_torque(TORQUE_CONFIG)
