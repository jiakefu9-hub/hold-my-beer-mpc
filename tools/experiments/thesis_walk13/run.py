"""Record the matched 10 s walk + 3 s settle comparison, with a fresh 11+4 template."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import argparse
import re
import json
import os
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/g1-walk13-mpl'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from protocol import PROTOCOL, ROOT, STUDY, EVIDENCE
from full_template import build_original_design
from metric_packager import package
import numpy as np
import yaml

BASE = ROOT / 'docs/simulation/experiments/20261117_walk10_stop3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    global STUDY, EVIDENCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='')
    args = parser.parse_args()
    if args.variant:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.variant):
            parser.error('variant must use letters, numbers, underscores or hyphens')
        STUDY = ROOT / 'evaluation' / ('thesis_walk10_stop3_' + args.variant)
        EVIDENCE = BASE / 'variants' / args.variant
        os.environ['THESIS_WALK13_VARIANT'] = args.variant
    final = EVIDENCE / 'manifest.json'
    if final.exists():
        raise FileExistsError('Existing result manifest will not be overwritten')
    baseline = json.loads((BASE / 'manifest.json').read_text())
    for relative, digest in baseline['model_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    for folder in ('configs', 'logs'):
        (STUDY / folder).mkdir(parents=True, exist_ok=True)
    collection = yaml.safe_load((BASE / 'configs/template_collection.yaml').read_text())
    collection_path = STUDY / 'configs/template_collection.yaml'
    collection_path.write_text(yaml.safe_dump(collection, sort_keys=False))
    asset, asset_manifest = build_original_design(collection, collection_path, STUDY / 'template')
    template_manifest = json.loads(asset_manifest.read_text())
    template_manifest['collection']['protocol'].update(
        name=PROTOCOL.protocol_name, version=PROTOCOL.protocol_version)
    save(asset_manifest, template_manifest)
    manifest = dict(simulation_only=True, task=dict(walk=[0., 10.], stop=[10., 13.],
        record_end=13.06, physics_dt=.002, control_dt=.006, horizon=9,
        no_hardware_arm_raise_or_takeover_sequence=True, numerical_startup_fixed_pd_s=.024),
        protocol=asdict(PROTOCOL), cases={}, failures=[],
        model_sha256=baseline['model_sha256'],
        template=str(asset.relative_to(ROOT)), template_sha256=sha(asset),
        template_manifest=str(asset_manifest.relative_to(ROOT)),
        template_manifest_sha256=sha(asset_manifest),
        template_design='original11x4',
        configuration_basis=str((BASE / 'manifest.json').relative_to(ROOT)),
        baseline_definition='hold current a/omega/alpha; integrate current world omega for future posture',
        change_scope='10 s forward plus 3 s settle; fresh full-task template; control settings unchanged',
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in HERE.glob('*.py')})
    configs = {}
    for case, label in (('D01', 'PID'), ('D02', 'NO-PREVIEW MPC'),
                        ('D03', 'DISTURBANCE-PREVIEW MPC')):
        config = yaml.safe_load((BASE / 'configs' / (case + '.yaml')).read_text())
        config.update(full_task_template_path=str(asset.relative_to(ROOT)),
            full_task_template_manifest_path=str(asset_manifest.relative_to(ROOT)),
            full_task_template_sha256=sha(asset),
            full_task_template_manifest_sha256=sha(asset_manifest),
            research_hold_future_disturbance=False,
            research_current_omega_pose=(case == 'D02'))
        configs[case] = config
        path = STUDY / 'configs' / (case + '.yaml')
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        env = dict(os.environ, MPC_CONTROL_CPU='2',
            DISTURBANCE_LAB_FORMAL_LAUNCHER='disturbance_lab_run_sh',
            OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE', PYTHONHASHSEED='0')
        for key, directory, filename in (
            ('RIGHT_ARM_RNEA_LIBRARY', 'right-arm-rnea', 'libright_arm_rnea.so'),
            ('RIGHT_ARM_EXECUTOR_LIBRARY', 'right-arm-executor', 'libright_arm_executor.so'),
            ('DDQ_TORQUE_MAPPER_LIBRARY', 'ddq-torque-mapper', 'libddq_torque_mapper.so'),
            ('RIGHT_ARM_SIM_RUNTIME_WORKER', 'right-arm-sim-runtime', 'right_arm_sim_runtime_worker')):
            env[key] = '/tmp/hold-my-beer-mpc-' + directory + '-build/' + filename
            assert Path(env[key]).is_file(), env[key]
        run_label = 'walk13_' + case.lower()
        command = ['taskset', '-c', '2', sys.executable, str(HERE / 'simulation_entry.py'),
            str(path), '--headless', '--no-video', '--full-task-smoke',
            '--run-label', run_label, '--evaluation-group', STUDY.name]
        started = time.monotonic()
        print('RUN', case, flush=True)
        with (STUDY / 'logs' / (case + '.log')).open('w') as log:
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=300)
        if proc.returncode:
            manifest['failures'].append(dict(case=case, returncode=proc.returncode,
                log=str((STUDY / 'logs' / (case + '.log')).relative_to(ROOT))))
            print('FAILED', case, flush=True)
        else:
            run_dir = sorted(STUDY.glob('20*_' + run_label))[-1]
            entry = package(case, run_dir, config, EVIDENCE)
            entry.update(config=str(path.relative_to(ROOT)), config_sha256=sha(path),
                title=label, controller_label=label, wall_s=time.monotonic() - started,
                gates=json.loads((EVIDENCE / (case + '_metrics.json')).read_text())['gates'],
                adapter_audit=json.loads(path.with_name(case + '_adapter_audit.json').read_text()))
            manifest['cases'][case] = entry
        save(EVIDENCE / 'manifest.partial.json', manifest)
    delta = [k for k in configs['D02'].keys() | configs['D03'].keys()
             if configs['D02'].get(k) != configs['D03'].get(k)]
    assert delta == ['research_current_omega_pose'], delta
    manifest['mpc_pair_configuration_difference'] = delta
    manifest['complete'] = len(manifest['cases']) == 3 and not manifest['failures']
    manifest['all_acceptance_gates_pass'] = manifest['complete'] and all(
        case['gates']['status'] == 'PASS' for case in manifest['cases'].values())
    save(final, manifest)
    print('COMPLETE', manifest['complete'], 'all gates pass',
          manifest['all_acceptance_gates_pass'], flush=True)


if __name__ == '__main__':
    main()
