#!/usr/bin/env python3
"""Independent measured-state reference/torque arithmetic audit; no SDK/DDS.

Checks recorded feedback against one-step references and the local packet,
not the physical accuracy of the forward model. Do not use the legacy
persistent-reference audit on this controller.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def check_active(row, config):
    q=np.asarray(row['q_measured_rad'])[5:10]
    dq=np.asarray(row['dq_measured_rad_s'])[5:10]
    ddq=np.asarray(row['raw_mpc_ddq_rad_s2'])
    qr=np.asarray(row['q_command_rad'])[5:10]
    dqr=np.asarray(row['dq_command_rad_s'])[5:10]
    qp=row['mpc']
    if not qp['solved'] or qp['fallback_used'] or qp['max_constraint_violation']>1e-6:
        raise ValueError('unsuccessful QP in command record')
    def close(a,b,name,tolerance=1e-7):
        if not np.allclose(a,b,rtol=0,atol=tolerance):raise ValueError(name)
    close(row['mpc_initial_state'],np.r_[q,dq],'initial state differs from feedback')
    close(row['command_integration_dt_s'],.006,'wrong MPC period')
    close(qr,q+.006*dq+.5*.006**2*ddq,'one-step q reference')
    close(dqr,dq+.006*ddq,'one-step dq reference')
    pd=np.asarray(config['kp'])*(qr-q)+np.asarray(config['kd'])*(dqr-dq)
    close(row['tau_pd_at_feedback_nm'],pd,'PD arithmetic')
    total=np.asarray(row['tau_total_estimated_at_feedback_nm'])
    close(np.asarray(row['offline_packet_right_tau_nm'])+pd,total,'packet duplicates/omits PD',1e-6)
    close(row['mapper']['tau_total_nm'],total,'final output changed after mapper')
    close(row['mapper']['checked_ddq_rad_s2'],row['post_transition_ddq_rad_s2'],'final forward check')
    if (not np.isfinite(np.r_[q,dq,ddq,qr,dqr,total]).all()
            or np.max(np.abs(ddq))>config['max_ddq_rad_s2']+1e-6
            or np.max(np.abs(dqr))>config['max_dq_rad_s']+1e-6
            or np.any(qr<np.deg2rad(config['q_min_deg'])-1e-7)
            or np.any(qr>np.deg2rad(config['q_max_deg'])+1e-7)
            or np.max(np.abs(row['post_transition_ddq_rad_s2']))>config['max_abs_qacc_rad_s2']+1e-6):
        raise ValueError('invalid active command envelope')


def audit_run(directory):
    directory=Path(directory)
    meta=json.loads((directory/'summary.json').read_text())
    if (meta['actuation']!='measured_torque_preview' or meta['status']!='complete'
            or meta['hardware_output'] or meta['dds_initialized'] or meta['publisher_created']
            or meta['journal_dropped'] or meta['journal_failed'] or meta['final_weight']!=0):
        raise ValueError('not a complete output-absent torque replay')
    config=meta['core']['torque_config'];previous=None;count=active=0;times=[]
    raw=directory/'raw.jsonl'
    with raw.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('schema')!='g1_mpc_offline_command_v1':continue
            t=float(row['task_elapsed_s']);total=np.asarray(row['tau_total_estimated_at_feedback_nm'])
            if not np.isfinite(total).all() or np.any(np.abs(total)>np.asarray(config['tau_abs_nm'])+1e-6):
                raise ValueError('total torque envelope')
            if previous:
                if row['sequence']!=previous['sequence']+1 or t<=previous['task_elapsed_s']:
                    raise ValueError('missing/reordered commands')
                change=np.abs(total-np.asarray(previous['tau_total_estimated_at_feedback_nm']))
                bound=np.asarray(config['transition_rate_nm_s'])*min(row['feedback_dt_s'],.006)
                if np.any(change>bound+1e-6):raise ValueError('torque rate envelope')
            if 3<=t<18:
                check_active(row,config);active+=1;times.append(t)
            if row['weight']==0 and np.max(np.abs(row['offline_packet_right_tau_nm']))!=0:
                raise ValueError('residual feedforward after release')
            previous=row;count+=1
    if not times or times[0]>3.05 or times[-1]<17.95 or previous['weight']!=0:
        raise ValueError('missing active window or final release packet')
    return dict(commands=count,active_commands=active,final_weight=previous['weight'],
        checks='measured state, one-step references, single PD, total torque, slew, final zero',
        physical_model_accuracy_certified=False,
        raw_jsonl_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=audit_run(args.run_directory)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text,end='')
