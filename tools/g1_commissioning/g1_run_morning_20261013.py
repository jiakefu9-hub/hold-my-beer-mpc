#!/usr/bin/env python3
"""Select ONE morning run; default only prints its command, never starts DDS.

M01/M06: learned MPC; M02/M05: current-disturbance-hold MPC;
M03/M04: zero-neutral position PID. M07: double PID reference speed/acceleration.
M08: M01 with all five left-arm packet PD gains doubled.
M10: M01 with SPACE walk/pause; 20 seconds accumulated forward requests.
All use the 6 ms field lifecycle; original M01..M07 remain unchanged.
"""
import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import secrets
import shlex

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path('/home/fjk/miniforge3/envs/g1_mpc/bin/python')
PROFILE = ROOT/'evaluation/hardware_shadow/commissioning/g1_pid_6ms_20261005_132052/arm_profile.conf'
MODEL = ROOT/'configs/g1_payload75.yaml'
MPC_CONFIG = ROOT/'configs/hardware_mpc_learned_acc001_alpha0005_omega1.yaml'
TORQUE_CONFIG = ROOT/'configs/hardware_mpc_torque_learned.yaml'
POSITION_PID_CONFIG = ROOT/'configs/hardware_position_pid_zero_20261013.yaml'
RELAXED_PID_CONFIG = ROOT/'configs/hardware_position_pid_zero_relaxed_20261013.yaml'
FROZEN_FILES = {
    MPC_CONFIG: '2d2da5ee5ce12f173af4cd76a445acfb77c06f644fa330aa3aa28a60a7852cb0',
    TORQUE_CONFIG: '41e153ca138147e2dc006bb7f15a511872a69e5299f925c401322234f703b879',
    MODEL: '7c7e7995612c2a441c901745102ec9b57fa09d40b0a81e6c488f74be8cda29cb',
}
TRIALS = {
    'M01': dict(family='mpc', predictor='learned_filtered', name='mpc_learned',
                purpose='原成功 MPC，学习预测；关闭时间对齐，当天基线'),
    'M02': dict(family='mpc', predictor='hold_current', name='mpc_hold',
                purpose='相同 MPC，只改为当前扰动保持；对照 M01'),
    'M03': dict(family='position_pid', predictor=None, name='position_pid_zero',
                purpose='6 ms 位置 PID，双臂零中立位复验'),
    'M04': dict(family='position_pid', predictor=None, name='position_pid_zero_repeat',
                purpose='重复 M03，检查位置 PID 的重复性'),
    'M05': dict(family='mpc', predictor='hold_current', name='mpc_hold_repeat',
                purpose='重复 M02，当前扰动保持'),
    'M06': dict(family='mpc', predictor='learned_filtered', name='mpc_learned_repeat',
                purpose='重复 M01，检查上午前后 MPC 波动'),
    'M07': dict(family='position_pid', predictor=None, name='position_pid_zero_relaxed',
                pid_config=RELAXED_PID_CONFIG,
                purpose='录像V04：位置PID参考限速0.14 rad/s、限加速度0.4 rad/s²；对照M03'),
    'M08': dict(family='mpc', predictor='learned_filtered', name='mpc_learned_left_pd2x',
                left_pd_gain_scale=2.,
                purpose='M01学习MPC，仅将左臂五关节发包PD增益加倍：Kp 40、Kd 2'),
    'M10': dict(family='mpc', predictor='learned_filtered', name='mpc_learned_space_pause20',
                space_pause_walk=True,
                purpose='M01控制配置；空格暂停/继续，暂停保持手臂控制，累计前进20秒后停车退权'),
}


def validate_baselines():
    for path, expected in FROZEN_FILES.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'morning baseline changed; review before running: {path}')


def build_command(trial_id, *, execute=False, nic='enx6c1ff701509c', cpu=7,
                  output_dir=None):
    validate_baselines()
    trial = TRIALS[trial_id]
    pid = trial['family'] == 'position_pid'
    entry = 'g1_walk_pid.py' if pid else 'g1_walk_mpc_learned.py'
    command = ['/usr/bin/taskset', '-c', '0-17', str(PYTHON),
               str(ROOT/'tools/g1_commissioning'/entry)]
    if execute:
        command.append(nic)
    command += ['--controller-config', str(MODEL), '--cpu', str(cpu), '--zero-arm-neutral']
    if pid:
        command += ['--position-pid-config', str(trial.get('pid_config', POSITION_PID_CONFIG)),
                    '--profile', str(PROFILE)]
        if not execute:
            command.append('--preflight')
    else:
        left_pd_gain_scale = float(trial.get('left_pd_gain_scale', 1.))
        command += ['--execute' if execute else '--preflight', '--task', 'walk',
                    '--predictor', trial['predictor'], '--left-pd-gain-scale',
                    f'{left_pd_gain_scale:g}',
                    '--mpc-config', str(MPC_CONFIG), '--torque-config', str(TORQUE_CONFIG),
                    '--rt-priority', '20', '--compute-process']
        if trial.get('space_pause_walk', False):
            command.append('--space-pause-walk')
    if execute:
        if output_dir is None:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            output_dir = ROOT/'evaluation/hardware_shadow/commissioning'/(
                f'morning_{trial_id}_{trial["name"]}_zero_payload75_walk_{stamp}_{secrets.token_hex(2)}')
        output_dir = Path(output_dir).resolve()
        if output_dir.exists():
            raise FileExistsError(f'output already exists: {output_dir}')
        command += ['--output-dir', str(output_dir)]
        if pid:
            command += ['--permit-real-output', 'PID_WALK_H0_CAPTURE',
                        '--allow-first-6ms-field-trial']
        else:
            command += ['--profile', str(PROFILE), '--pid-6ms-validated',
                        '--torque-stationary-validated', '--permit-real-output',
                        'MPC_WALK_H0_CAPTURE', '--allow-first-torque-field-trial']
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', nargs='?', choices=tuple(TRIALS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    parser.add_argument('--nic', default='enx6c1ff701509c')
    parser.add_argument('--cpu', type=int, default=7)
    args = parser.parse_args(argv)
    if args.trial is None:
        if args.execute or args.preflight:
            parser.error('select ONE trial; no automatic series execution')
        for name, trial in TRIALS.items():
            print(f'{name}: {trial["purpose"]}')
        return 0
    try:
        command = build_command(args.trial, execute=args.execute, nic=args.nic, cpu=args.cpu)
        if (args.execute or args.preflight) and not PROFILE.is_file():
            raise FileNotFoundError(f'field profile missing: {PROFILE}')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'{args.trial}: {TRIALS[args.trial]["purpose"]}', flush=True)
    left_scale = float(TRIALS[args.trial].get('left_pd_gain_scale', 1.))
    print(f'共同：双臂零中立位、75 mm/250 g、左 PD '
          f'{20*left_scale:g}/{1*left_scale:g}、6 ms；时间对齐全部关闭。', flush=True)
    print(shlex.join(command), flush=True)
    if not (args.execute or args.preflight):
        print('仅显示离线检查命令；--preflight 只做离线检查，--execute 单独运行本组。')
        return 0
    os.chdir(ROOT)
    os.execv(command[0], command)


if __name__ == '__main__':
    raise SystemExit(main())
