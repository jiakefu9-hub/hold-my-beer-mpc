#!/usr/bin/env python3
"""Summarize failed field starts without inferring unmeasured motor response."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter


def analyze(path):
    commands, faults, details, cycles = [], [], [], []
    host, shutdown, drops = None, None, None
    schemas = Counter()
    for line in path.open():
        r = json.loads(line); schemas[r.get('schema')] += 1
        if r.get('schema') == 'g1_hardware_mpc_command_v1': commands.append(r)
        if r.get('event') == 'session_fault': faults.append(r)
        if r.get('event') == 'controller_fault_detail': details.append(r)
        if r.get('schema') == 'g1_mpc_cycle_complete_v1': cycles.append(r)
        if r.get('event') == 'control_runtime': host = r
        if r.get('event') == 'sdk_shutdown': shutdown = r
        if r.get('event') == 'capture_drained': drops = r.get('queue_dropped')
    active = [r for r in commands if r.get('mpc_active')]
    normal = [r for r in commands if r.get('task_elapsed_s') is not None]
    def compact(r):
        keys = ('sequence','task_elapsed_s','stage','weight','controller_compute_us',
                'control_prewrite_us','controller_core_ms','controller_thread_cpu_ms',
                'state_prediction_ms','delay_lifecycle_ms','field_prewrite_timing')
        result = {k:r.get(k) for k in keys if k in r}
        for key in ('q_measured_rad','dq_measured_rad_s'):
            if key in r: result['right_'+key] = r[key][5:10]
        result['mpc_timing_ms'] = {k:v*1000. for k,v in r.get('mpc',{}).items()
                                  if k.endswith('_time') and isinstance(v,(int,float))}
        mapper = r.get('mapper',{})
        result['mapper'] = {k:mapper.get(k) for k in ('elapsed_ms','fallback',
            'tracking_within_limit','max_joint_error','checked_ddq_rad_s2','bounded_affine_rescue')}
        result['requested_ddq_rad_s2'] = r.get('raw_mpc_ddq_rad_s2')
        return result
    return dict(raw_path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        command_rows=len(commands),active_mpc_writes=len(active),
        first_command=compact(commands[0]) if commands else None,
        last_normal=compact(normal[-1]) if normal else None,
        last_command=compact(commands[-1]) if commands else None,
        active_commands=[compact(r) for r in active],
        faults=faults,fault_diagnostics=[compact(r['diagnostics']) for r in details],
        host=host,shutdown=shutdown,journal_dropped=drops,schema_counts=dict(schemas),
        limitations=['dq is measured; mapper ddq is a model calculation, not measured acceleration',
                     'one or zero active writes cannot identify torque gain or controller performance',
                     'successful DDS Write does not prove receipt or exact motor application time'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    results=[analyze(p) for p in args.raw]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: json.dump(results,f,indent=2)
    for r in results:
        print(Path(r['raw_path']).parent.name,'active writes:',r['active_mpc_writes'],
              'fault:',[f['reason'] for f in r['faults']])


if __name__=='__main__': main()
