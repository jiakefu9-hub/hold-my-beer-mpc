"""Run reproducible, isolated 8-second full-task controller comparisons."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

import yaml

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'mpc_comprehensive'))
from metrics import summarize

ROOT = REPO/'evaluation/simple_controller_comparison_20261011'
BASE = yaml.safe_load((REPO/'configs/simulation_optimization_20261010/conservative.yaml').read_text())


def source_files():
    names = ['main_sim.py', 'arm_mpc.py', 'arm_pid.py', 'sim_support.py',
             'right_arm_control_setup.py', 'kinematics_helper.py', 'disturbance_predictor.py',
             'disturbance_template/preview_correction.py', 'right_arm_runtime/sim_mpc_latency.py',
             'tools/g1_commissioning/hardware_mpc_braking.py']
    return [REPO/name for name in names] + list(HERE.glob('*.py'))


def run_case(case, cpu, keep=False):
    name = case['name']
    saved = ROOT/'results'/f'{name}.json'
    if saved.exists():
        return json.loads(saved.read_text())
    if shutil.disk_usage(ROOT).free < 2*1024**3:
        raise RuntimeError('less than 2 GiB free')
    base = yaml.safe_load((REPO/case['base_config']).read_text()) if 'base_config' in case else BASE
    config = dict(base, **case.get('overrides', {}))
    cfg = ROOT/'configs'/f'{name}.yaml'
    cfg.write_text(yaml.safe_dump(config, sort_keys=False))
    env = dict(os.environ, MPC_CONTROL_CPU=str(cpu), PYTHONDONTWRITEBYTECODE='1',
               DISTURBANCE_LAB_FORMAL_LAUNCHER='disturbance_lab_run_sh',
               OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE', PYTHONHASHSEED='0',
               MPLCONFIGDIR='/tmp/hold-my-beer-mpc-matplotlib')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        env[key] = '1'
    for key, directory, filename in (
        ('RIGHT_ARM_RNEA_LIBRARY', 'right-arm-rnea', 'libright_arm_rnea.so'),
        ('RIGHT_ARM_EXECUTOR_LIBRARY', 'right-arm-executor', 'libright_arm_executor.so'),
        ('DDQ_TORQUE_MAPPER_LIBRARY', 'ddq-torque-mapper', 'libddq_torque_mapper.so'),
        ('RIGHT_ARM_SIM_RUNTIME_WORKER', 'right-arm-sim-runtime', 'right_arm_sim_runtime_worker')):
        env[key] = '/tmp/hold-my-beer-mpc-'+directory+'-build/'+filename
    cmd = ['taskset', '-c', str(cpu), sys.executable, str(globals().get('ENTRY', HERE/'simulation_entry.py')), str(cfg),
           '--headless', '--no-video', '--full-task-smoke', '--run-label', name,
           '--evaluation-group', ROOT.name]
    if case.get('latency_ms'):
        cmd += ['--experimental-mpc-latency-mode', 'fixed',
                '--experimental-mpc-latency-ms', str(case['latency_ms'])]
    if case.get('scenario', 'nominal') != 'nominal':
        manifest = REPO/'disturbance_template/data/full_task_template_v2/20260815_162850/episodes'/case['scenario']/'episode_manifest.json'
        cmd += ['--full-task-initial-manifest', str(manifest)]
    started = time.monotonic()
    source_hashes = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}
    with (ROOT/'logs'/f'{name}.log').open('w') as log:
        try:
            process = subprocess.run(cmd, cwd=REPO, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=240)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
    # main_sim removes punctuation from labels before creating directories.
    paths = sorted(ROOT.glob('20*_'+re.sub(r'[^a-zA-Z0-9_-]', '', name)))
    result = dict(complete=False)
    if paths and (paths[-1]/'full_task_smoke_summary.json').exists():
        result = summarize(paths[-1], xml_path=config['xml_path'])
        import numpy as np
        with np.load(paths[-1]/'trajectory.npz') as tr:
            timing = np.asarray(tr['right_mpc_policy_core_time']) if 'right_mpc_policy_core_time' in tr else np.array([])
            timing = timing[np.isfinite(timing) & (timing > 0)]
            if len(timing):
                result['policy_compute_ms'] = dict(zip(('p50', 'p95', 'p99', 'max'),
                    (1000*np.percentile(timing, [50, 95, 99, 100])).tolist()))
    result.update(case, returncode=returncode, cpu=cpu, wall_s=time.monotonic()-started, config=str(cfg))
    result['source_sha256'] = source_hashes
    saved.write_text(json.dumps(result, indent=2)+'\n')
    if not keep and paths and (paths[-1]/'compact_metrics.npz').exists():
        # Only this new study's reproducible bulky artifacts are pruned.
        # Compact physical trajectories, JSON diagnostics and exact config remain.
        for path in paths[-1].iterdir():
            if path.name != 'compact_metrics.npz' and path.suffix in ('.npz', '.csv', '.png', '.mp4'):
                path.unlink()
    print(json.dumps(dict(name=name, complete=result['complete'], returncode=returncode,
                          right=result.get('windows', {}).get('full', {}).get('right'))), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cases', type=Path)
    parser.add_argument('--cpus', default='0,2,4')
    parser.add_argument('--keep', action='store_true')
    args = parser.parse_args()
    # A cleared /tmp must not turn one missing build into a batch of apparent
    # controller failures. These are the libraries used by run_case below.
    required_builds = {
        'right-arm-rnea': 'libright_arm_rnea.so',
        'right-arm-executor': 'libright_arm_executor.so',
        'ddq-torque-mapper': 'libddq_torque_mapper.so',
        'right-arm-sim-runtime': 'right_arm_sim_runtime_worker',
    }
    missing = [str(Path('/tmp')/('hold-my-beer-mpc-'+name+'-build')/filename)
               for name, filename in required_builds.items()
               if not (Path('/tmp')/('hold-my-beer-mpc-'+name+'-build')/filename).is_file()]
    if missing:
        parser.error('build the simulation C++ backends first; missing: '+', '.join(missing))
    for directory in ('results', 'logs', 'configs', 'source_before'):
        (ROOT/directory).mkdir(parents=True, exist_ok=True)
    if not (ROOT/'provenance.json').exists():
        for path in source_files():
            target = ROOT/'source_before'/path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        (ROOT/'provenance.json').write_text(json.dumps(dict(
            simulation_only=True, physical_window='0 <= t < 8 s', physics_dt=.002, control_dt=.006,
            fixed='left PD, legs, task, neutral, payload, physical actuator limits',
            timing_not_hardware_realtime_proof=True,
            head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            **globals().get('PROVENANCE_EXTRA', {})), indent=2))
        (ROOT/'preexisting.diff').write_bytes(subprocess.check_output(['git', 'diff'], cwd=REPO))
    cases = json.loads(args.cases.read_text())
    cpus = [int(x) for x in args.cpus.split(',')]
    def worker(index):
        return [run_case(c, cpus[index], args.keep) for c in cases[index::len(cpus)]]
    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        list(pool.map(worker, range(len(cpus))))


if __name__ == '__main__':
    main()
