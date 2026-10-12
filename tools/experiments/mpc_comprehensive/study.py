"""Reproducible adaptive simulation batches with bounded disk use.

Use the g1_mpc Python, not system Python. Build C++ once with cpp/build_runtime.sh.
Input is a JSON list of {name, overrides, scenario?}; each complete rollout is
saved with all physical metrics and the exact resolved YAML. Only this study's
large outputs are pruned, after compact trajectories have been written.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import yaml
from metrics import summarize

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'evaluation/mpc_comprehensive_20261010'
BASE = yaml.safe_load((REPO / 'configs/g1.yaml').read_text())
HERE = Path(__file__).resolve().parent


def initialize():
    ROOT.mkdir(exist_ok=True)
    for name in ('configs', 'results', 'logs'):
        (ROOT / name).mkdir(exist_ok=True)
    if (ROOT / 'provenance.json').exists():
        return
    names = ['arm_mpc.py', 'main_sim.py', 'sim_support.py', 'configs/g1.yaml',
             'right_arm_control_setup.py', 'kinematics_helper.py',
             'resources/g1_description/scene.xml',
             'resources/g1_description/g1_23dof_rev_1_0.xml']
    hashes = {}
    for name in names:
        target = ROOT / 'source_before' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    (ROOT / 'provenance.json').write_text(json.dumps(dict(
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        source_sha256=hashes, simulation_only=True,
        evaluation_window='0 <= t < 8 s', physics_dt_s=.002, control_dt_s=.006,
        batch_timing_is_not_realtime_evidence=True,
        fixed='left PD; legs policy/command; neutral; payload; contacts; physical actuator limits'), indent=2))
    (ROOT / 'preexisting.diff').write_bytes(subprocess.check_output(['git', 'diff'], cwd=REPO))


def run_case(case, cpu, keep=False):
    name = case['name']
    saved = ROOT / 'results' / (name + '.json')
    if saved.exists():
        return json.loads(saved.read_text())
    if shutil.disk_usage(ROOT).free < 2 * 1024**3:
        raise RuntimeError('Less than 2 GiB free; refusing further rollouts')
    config = dict(BASE, **case.get('overrides', {}))
    cfg = ROOT / 'configs' / (name + '.yaml')
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
        env[key] = '/tmp/hold-my-beer-mpc-' + directory + '-build/' + filename
    cmd = ['taskset', '-c', str(cpu), sys.executable, str(HERE / 'simulation_entry.py'),
           str(cfg), '--headless', '--no-video', '--full-task-smoke',
           '--run-label', name, '--evaluation-group', ROOT.name]
    if 'latency_ms' in case:
        cmd += ['--experimental-mpc-latency-mode', 'fixed',
                '--experimental-mpc-latency-ms', str(case['latency_ms'])]
    scenario = case.get('scenario', 'nominal')
    if scenario != 'nominal':
        manifest = REPO / 'disturbance_template/data/full_task_template_v2/20260815_162850/episodes' / scenario / 'episode_manifest.json'
        if (ROOT / 'manifests' / (scenario + '.json')).exists():
            manifest = ROOT / 'manifests' / (scenario + '.json')
        cmd += ['--full-task-initial-manifest', str(manifest)]
    started = time.monotonic()
    with (ROOT / 'logs' / (name + '.log')).open('w') as log:
        process = subprocess.run(cmd, cwd=REPO, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=300)
    paths = sorted(ROOT.glob('20*_' + name))
    result = summarize(paths[-1], xml_path=config['xml_path']) if paths and (paths[-1] / 'full_task_smoke_summary.json').exists() else dict(complete=False)
    result.update(case, returncode=process.returncode, cpu=cpu,
                  wall_s=time.monotonic() - started, config=str(cfg))
    result['source_sha256'] = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [REPO/'main_sim.py', REPO/'arm_mpc.py', REPO/'disturbance_predictor.py',
                     REPO/'disturbance_template/preview_correction.py',
                     REPO/'right_arm_runtime/sim_mpc_latency.py',
                     HERE/'preview.py', HERE/'strategy.py', HERE/'simulation_entry.py']}
    saved.write_text(json.dumps(result, indent=2) + '\n')
    if not keep and paths and (paths[-1] / 'compact_metrics.npz').exists():
        # Keep full numeric/kinematic evidence, exact config and diagnostics.
        # Heavy redundant prediction histories are reproducible from the config.
        for f in paths[-1].iterdir():
            if f.name != 'compact_metrics.npz' and f.suffix in ('.npz', '.csv', '.png', '.mp4'):
                f.unlink()
    print(json.dumps(dict(name=name, complete=result['complete'], returncode=process.returncode,
        right=result.get('windows', {}).get('full', {}).get('right'), gates=result.get('gates'))), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cases', type=Path)
    parser.add_argument('--cpus', default='0,2,4,7')
    parser.add_argument('--keep', action='store_true')
    args = parser.parse_args()
    initialize()
    cases = json.loads(args.cases.read_text())
    cpus = [int(x) for x in args.cpus.split(',')]
    # Each single-core worker processes its own queue sequentially.
    def worker(index, cpu):
        return [run_case(case, cpu, args.keep) for case in cases[index::len(cpus)]]
    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        futures = [pool.submit(worker, i, cpu) for i, cpu in enumerate(cpus)]
        for future in as_completed(futures):
            future.result()


if __name__ == '__main__':
    main()
