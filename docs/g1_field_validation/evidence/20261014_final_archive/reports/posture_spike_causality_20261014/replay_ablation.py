"""Recorded-state objective ablation; no DDS, network, or robot output."""
import json
from pathlib import Path
import shutil
import socket
import sys
from unittest.mock import patch

import numpy as np
import orjson

ROOT = Path('/home/fjk/g1_ws/hold-my-beer-mpc')
sys.path.insert(0, str(ROOT / 'tools/g1_commissioning'))
import hardware_mpc_control
from hardware_mpc_control import load_mpc_config
from validate_mpc_learned import capture, replay

SOURCE = ROOT / ('evaluation/hardware_shadow/commissioning/'
    'morning_M01_mpc_learned_zero_payload75_walk_20261014_125333_831147_ee9f/raw.jsonl')
OUT = Path(__file__).resolve().parent / 'replay'
BASE = load_mpc_config(ROOT / 'configs/hardware_mpc_learned_acc001_alpha0005_omega1.yaml')
VARIANTS = {
    'baseline': {},
    'gravity_20': {'q_gravity': [20., 20.]},
    'gravity_15': {'q_gravity': [15., 15.]},
    'gravity_10': {'q_gravity': [10., 10.]},
    'no_gravity': {'q_gravity': [0., 0.]},
    'no_omega': {'q_ee_omega': 0.},
    'no_alpha': {'q_ee_alpha': 0.},
    'no_linear_acc': {'q_ee_acc': 0.},
    'no_joint_posture': {'q_posture': [0., 0., 0., 0., 0.]},
    'no_angular_objectives': {'q_gravity': [0., 0.], 'q_ee_omega': 0., 'q_ee_alpha': 0.},
}


def read_output(path):
    rows = []
    with path.open('rb') as stream:
        for line in stream:
            r = orjson.loads(line)
            if 5 <= r.get('task_s', -1) < 15 and 'diagnostics' in r:
                d = r['diagnostics']
                if d.get('mpc_active'):
                    m = d['mpc']
                    rows.append(dict(t=float(r['task_s']),
                        ddq=np.asarray(d['raw_mpc_ddq_rad_s2'], float),
                        tau=np.asarray(d['tau_total_estimated_at_feedback_nm'], float),
                        gravity=np.asarray(m['gravity_error'], float),
                        costs=m['one_step_prediction']['cost_terms']))
    return rows


def metrics(rows, events):
    t = np.asarray([r['t'] for r in rows])
    ddq = np.asarray([r['ddq'] for r in rows])
    tau = np.asarray([r['tau'] for r in rows])
    dtau = np.r_[0., np.linalg.norm(np.diff(tau, axis=0), axis=1)]
    dddq = np.r_[0., np.linalg.norm(np.diff(ddq, axis=0), axis=1)]
    event_max_ddq, event_max_step, event_max_ddq_step = [], [], []
    for t0 in events:
        mask = (t >= t0) & (t <= t0 + .060)
        event_max_ddq.append(float(np.max(np.linalg.norm(ddq[mask], axis=1))))
        event_max_step.append(float(np.max(dtau[mask])))
        event_max_ddq_step.append(float(np.max(dddq[mask])))
    costs = {k: np.asarray([r['costs'].get(k, 0.) for r in rows]) for k in rows[0]['costs']}
    return dict(
        ddq_norm_rms=float(np.sqrt(np.mean(np.sum(ddq * ddq, axis=1)))),
        ddq_step_norm_rms=float(np.sqrt(np.mean(dddq * dddq))),
        torque_step_norm_rms=float(np.sqrt(np.mean(dtau * dtau))),
        event_max_ddq_norm_median=float(np.median(event_max_ddq)),
        event_max_ddq_step_median=float(np.median(event_max_ddq_step)),
        event_max_torque_step_median=float(np.median(event_max_step)),
        saturation_fraction=float(np.mean(np.any(np.isclose(abs(ddq), 15., atol=1e-5), axis=1))),
        baseline_cost_term_medians={k: float(np.median(v)) for k, v in costs.items()},
        baseline_cost_term_p90={k: float(np.percentile(v, 90)) for k, v in costs.items()},
    )


if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
data = capture(SOURCE)
event_data = json.loads((Path(__file__).resolve().parent / 'event_timing.json').read_text())
latest = next(r for r in event_data['runs'] if '125333' in r['source'])
events = [e['impact_task_s'] for e in latest['events']]
reports, output = {}, {}
with patch.object(socket, 'socket', side_effect=RuntimeError('offline ablation forbids network')):
    for name, overrides in VARIANTS.items():
        config = dict(BASE); config.update(overrides)
        # Keep the reviewed baseline path admission contract, while replacing
        # only the in-memory weights for this offline counterfactual.
        with patch.object(hardware_mpc_control, 'load_mpc_config', return_value=config):
            report, _ = replay(data, OUT, name, 2, math_only=True,
                               config=ROOT / 'configs/hardware_mpc_learned_acc001_alpha0005_omega1.yaml',
                               predictor_mode='learned_filtered', zero_arm_neutral=True)
        reports[name] = {key: report[key] for key in (
            'status', 'errors', 'active_commands', 'planned_vs_final_acceleration_max_error_rad_s2')}
        output[name] = metrics(read_output(OUT / f'{name}.jsonl'), events)
        print(name, report['status'], output[name], flush=True)

base = output['baseline']
for name, values in output.items():
    values['relative_to_baseline_percent'] = {
        key: 100. * (values[key] / base[key] - 1.)
        for key in ('ddq_norm_rms', 'ddq_step_norm_rms', 'torque_step_norm_rms',
                    'event_max_ddq_norm_median', 'event_max_ddq_step_median',
                    'event_max_torque_step_median')
    }
result = {'source': str(SOURCE.relative_to(ROOT)), 'recorded_state_counterfactual': True,
          'limitation': 'States are held to the recorded baseline trajectory, so this identifies immediate command sensitivity, not closed-loop performance.',
          'event_count': len(events), 'replay_reports': reports, 'metrics': output}
(Path(__file__).resolve().parent / 'ablation.json').write_text(json.dumps(result, indent=2) + '\n')
