#!/usr/bin/env python3
"""Opt-in stationary BOTH-arm world-Z alignment. Default: offline validation.

Uses measured torso IMU orientation and arm encoder feedback. No learned
predictor, no walk command, no robot mode switch, no rt/lowcmd.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import threading

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'

from hardware_dual_upright import (ROOT, CONFIG, MODEL, DualUprightRuntime, effective_profile)
from g1_walk_pid import (load_profile, run_device, Journal, json_safe,
                         CONTROL_PERIOD_S, WALK_START_S)

PERMIT='DUAL_UPRIGHT_STATIONARY'
PROFILE=ROOT/'evaluation/hardware_shadow/commissioning/g1_pid_6ms_20261005_132052/arm_profile.conf'


class DiscardJournal:
    """Journal interface for an explicitly requested no-capture field run."""
    def __init__(self):
        self.failed=threading.Event()
        self.dropped=0
        self.failure_reason=None

    def record(self,row):
        pass

    def close(self,timeout_s=5.):
        return True


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('nic',nargs='?')
    g=p.add_mutually_exclusive_group()
    g.add_argument('--execute',action='store_true')
    g.add_argument('--preflight',action='store_true')
    p.add_argument('--permit-real-output',choices=[PERMIT])
    p.add_argument('--profile',type=Path,default=PROFILE)
    p.add_argument('--controller-config',type=Path,default=MODEL)
    p.add_argument('--upright-config',type=Path,default=CONFIG)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--no-capture',action='store_true',
                   help='do not create an output directory or save raw field data')
    p.add_argument('--cpu',type=int,default=7)
    return p


def main(argv=None):
    args=build_parser().parse_args(argv)
    journal=None
    try:
        if args.execute and (not args.nic or args.permit_real_output!=PERMIT):
            raise ValueError('real output requires NIC, --execute and --permit-real-output '+PERMIT)
        if args.no_capture and args.output_dir is not None:
            raise ValueError('--no-capture cannot be combined with --output-dir')
        if not args.execute:
            from validate_dual_upright import validate
            result=validate(args.upright_config,args.controller_config)
            print(json.dumps(result,indent=2,allow_nan=False))
            return 0 if result['passed'] else 1
        profile=effective_profile(load_profile(args.profile))
        runtime=DualUprightRuntime(args.upright_config,args.controller_config)
        if args.output_dir is None and not args.no_capture:
            stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            args.output_dir=ROOT/'evaluation/hardware_shadow/commissioning'/f'dual_upright_stationary_{stamp}_{secrets.token_hex(2)}'
        if args.output_dir is not None and args.output_dir.exists():
            raise FileExistsError('output directory already exists: '+str(args.output_dir))
        from field_performance import prepare
        performance=prepare(args.cpu)
        journal=DiscardJournal() if args.no_capture else Journal(args.output_dir)
        if not args.no_capture:
            for source,name in ((args.profile,'arm_profile.conf'),(args.controller_config,'controller_config.yaml'),
                                (args.upright_config,'upright_config.yaml')):
                shutil.copy2(source,args.output_dir/name)
            (args.output_dir/'effective_profile.json').write_text(json.dumps(json_safe(profile),indent=2)+'\n')
        paths=[Path(__file__),Path(__file__).with_name('hardware_dual_upright.py'),
               Path(__file__).with_name('g1_walk_pid.py'),Path(__file__).with_name('hardware_pid_control.py'),
               Path(__file__).with_name('endpoint_pose.py')]
        journal.record(dict(schema='g1_pid_session_v1',event='session_start',
            controller_kind='dual_upright_reference_servo',program=Path(__file__).name,
            task='stationary',publisher_created=False,mode_setter_registered=False,
            lowcmd_topic_created=False,network_interface=args.nic,required_fsm=500,
            control_nominal_period_ms=CONTROL_PERIOD_S*1000,
            primary_metric_window_s=[WALK_START_S,None],
            primary_metric_scope='stationary alignment until operator stop',forward_speed_m_s=0.,
            release_trigger='one Ctrl+C',minimum_release_duration_s=3.,
            profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
            controller_config_sha256=hashlib.sha256(args.controller_config.read_bytes()).hexdigest(),
            upright_config_sha256=hashlib.sha256(args.upright_config.read_bytes()).hexdigest(),
            control_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            performance_setup=performance,core=runtime.controller.metadata,
            dual_upright_experiment=profile['dual_upright_experiment']))
        print('STATIONARY BOTH ARMS: 3 s entry toward zero references, IMU-based upward-axis alignment '
              'continues until one Ctrl+C, then >=3 s gradual release. '
              'No recurring Loco velocity RPC is used.',flush=True)
        if args.no_capture:
            print('NO CAPTURE: raw field data and input copies will not be written.',flush=True)
        result=run_device(args,profile,None,{},journal,runtime=runtime)
        journal.record(dict(schema='g1_pid_event_v1',event='capture_drained',queue_dropped=journal.dropped))
        journal.close()
        if journal.failed.is_set():
            raise RuntimeError('capture failed: '+str(journal.failure_reason))
        if args.no_capture:
            print('Dual upright stopped; no capture was written.')
        else:
            print('Saved dual upright capture:',args.output_dir/'raw.jsonl')
        return result
    except Exception as exc:
        print('Dual upright refused/stopped:',exc,file=sys.stderr)
        return 1
    finally:
        if journal is not None:
            journal.close()


if __name__=='__main__':
    raise SystemExit(main())
