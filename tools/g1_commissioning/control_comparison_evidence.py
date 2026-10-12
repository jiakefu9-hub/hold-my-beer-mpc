"""Shared recorded-input provenance for offline field-comparison validation."""
import hashlib
from pathlib import Path
import numpy as np
import orjson
from hardware_control_comparison import ROOT

DEFAULT_SOURCES = tuple(
    ROOT / "evaluation/hardware_shadow/commissioning" / name / "raw.jsonl"
    for name in (
        "mpc_zero_leftpd100_nopulse_right164546_repeat_walk_20261009_174504",
        "mpc_zero_leftpd100_nopulse_right164546_repeat2_walk_20261009_180354",
    )
)
LIMITATIONS = [
    "Recorded arm and torso states do not respond to new commands: no physical closed-loop validation",
    "Historical source captures used the old payload; replay uses the explicitly selected payload75 model",
    "No network, DDS initialization/output, live compute worker IPC, FIFO scheduling or audit IO timing",
    "Local replay compute times are not a live 6 ms deadline certificate",
    "Math-only, if enabled, raises replay QP allowance to 100 ms and is not a timing pass",
    "T04/T05 are live repeats; duplicate offline inputs are not independent evidence",
    "Measured joint/IMU feedback is preserved; no state extrapolation or new attitude correction",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_paths(paths):
    result = {}
    for path in sorted({Path(p).resolve() for p in paths}):
        try:
            name = str(path.relative_to(ROOT))
        except ValueError:
            name = str(path)
        result[name] = sha256_file(path)
    return result


def source_session(path):
    with path.open("rb") as stream:
        for line in stream:
            row = orjson.loads(line)
            if row.get("schema") == "g1_mpc_session_v1" and row.get("event") == "session_start":
                return row
    raise ValueError("source lacks an MPC session_start record")


def validate_source_session(session):
    zero = session.get("zero_arm_neutral_experiment", {})
    targets = np.asarray(zero.get("target_q_rad", []), dtype=float)
    if not zero.get("enabled") or targets.size < 10 or not np.array_equal(targets[:10], np.zeros(10)):
        raise ValueError("source must explicitly record both arms at zero neutral")
    left = session.get("left_pd_gain_experiment", {})
    if (left.get("enabled", True) or left.get("scale") != 1.0
            or not np.array_equal(np.asarray(left.get("kp", [])), np.full(5, 20.0))
            or not np.array_equal(np.asarray(left.get("kd", [])), np.ones(5))):
        raise ValueError("source must record normal left PD Kp=20, Kd=1 at scale 1")
    for key in ("left_reference_disturbance", "left_acceleration_probe", "left_pulse_experiment"):
        if session.get(key, {}).get("enabled", False):
            raise ValueError(f"source has an enabled artificial left-arm intervention: {key}")
    if (session.get("task") != "walk" or session.get("control_nominal_period_ms") != 6.0
            or session.get("predictor_mode") != "learned_filtered"
            or session.get("assumed_command_delay_ms") is not None):
        raise ValueError("source must be a measured-state learned 6 ms walking trial")
    return dict(zero_arm_neutral=True, left_kp=[20.0] * 5, left_kd=[1.0] * 5,
                artificial_left_disturbance=False, original_session=session)
