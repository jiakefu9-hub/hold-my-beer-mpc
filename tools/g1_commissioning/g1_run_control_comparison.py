#!/usr/bin/env python3
"""Print or run ONE T01..T06 field experiment. No automatic series execution."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import secrets
import shlex

from hardware_control_comparison import ROOT, TRIALS, MODEL_CONFIG, TORQUE_CONFIG, validate_series
from hardware_acceleration_pid import load_pid_config


PYTHON = Path('/home/fjk/miniforge3/envs/g1_mpc/bin/python')
PROFILE = ROOT / 'evaluation/hardware_shadow/commissioning/g1_pid_6ms_20261005_132052/arm_profile.conf'

def build_command(trial_id, *, execute=False, nic='enx6c1ff701509c', output_dir=None, cpu=7):
    validate_series()
    trial = TRIALS[trial_id]
    command = ['/usr/bin/taskset', '-c', '0-17', str(PYTHON),
               str(ROOT/'tools/g1_commissioning/g1_walk_mpc_learned.py')]
    if execute:
        command.append(nic)
    command += ['--execute' if execute else '--preflight', '--task', 'walk',
                '--predictor', 'learned_filtered', '--zero-arm-neutral', '--left-pd-gain-scale', '1',
                '--controller-config', str(MODEL_CONFIG), '--torque-config', str(TORQUE_CONFIG),
                '--mpc-config', str(trial['config']), '--cpu', str(cpu),
                '--rt-priority', '20', '--compute-process']
    if trial['aligned']:
        command.append('--align-accel-forecast')
    if trial['pid_config']:
        load_pid_config(trial['pid_config'])
        command += ['--acceleration-pid-config', str(trial['pid_config'])]
    if execute:
        if output_dir is None:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            output_dir = ROOT/('evaluation/hardware_shadow/commissioning/'
                f'comparison_{trial_id}_{trial["name"]}_zero_payload75_walk_{stamp}_{secrets.token_hex(2)}')
        output_dir = Path(output_dir).resolve()
        if output_dir.exists():
            raise FileExistsError(f'output already exists: {output_dir}')
        command += ['--profile', str(PROFILE), '--output-dir', str(output_dir),
                    '--pid-6ms-validated', '--torque-stationary-validated',
                    '--permit-real-output', 'MPC_WALK_H0_CAPTURE', '--allow-first-torque-field-trial']
    return command


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trial', nargs='?', choices=tuple(TRIALS))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    p.add_argument('--nic', default='enx6c1ff701509c')
    p.add_argument('--cpu', type=int, default=7)
    args = p.parse_args(argv)
    if args.trial is None:
        if args.execute or args.preflight:
            p.error('select one experiment; no automatic batch execution')
        for key, spec in TRIALS.items():
            print(f'{key}: {spec["purpose"]}')
        return 0
    command = build_command(args.trial, execute=args.execute, nic=args.nic, cpu=args.cpu)
    print(f'{args.trial}: {TRIALS[args.trial]["purpose"]}', flush=True)
    print('共同：零中立位、75 mm/250 g、左PD 20/1、6 ms；保留成功基线的力矩、关节边界和 1.5 s 接管时序。', flush=True)
    print(shlex.join(command), flush=True)
    if not (args.execute or args.preflight):
        print('只显示命令；加 --execute 运行一组，原现场确认和停车退权保持。')
        return 0
    if args.execute and not PROFILE.is_file():
        p.error(f'field profile missing: {PROFILE}')
    os.chdir(ROOT)
    os.execv(command[0], command)


if __name__ == '__main__':
    raise SystemExit(main())
