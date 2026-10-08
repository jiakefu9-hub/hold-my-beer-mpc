#!/usr/bin/env python3
"""Offline three-run identification review; no SDK or controller parameter writes.

Run 1 fits and run 2 selects delay; runs 1+2 then fit the final coefficients,
and run 3 evaluates them unchanged. No run-ID intercepts or test-run centering.
This is a measured-acceleration regression, not a causal future forecaster.
The centered velocity derivative is an offline target with a 66 ms window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter

from analyze_arm_system_identification import JOINTS, NOMINAL, delayed, load_run, ridge_fit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rmse(prediction, target):
    return np.sqrt(np.mean((prediction-target)**2, axis=0))


def audit(path):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream]
    commands = [r for r in rows if r.get('event') == 'dds_write'
                and r.get('schema') == 'g1_hardware_pid_command_v1']
    active = [r for r in commands if r.get('stage') == 'arm_identification']
    by_event = {}
    for r in rows:
        by_event.setdefault(r.get('event'), []).append(r)
    ends, drains = by_event.get('session_end', []), by_event.get('capture_drained', [])
    if len(ends) != 1 or ends[0].get('outcome') != 'normal_release_completed' or ends[0]['final_weight'] != 0:
        raise ValueError(f'{path}: incomplete release')
    if len(drains) != 1 or drains[0]['queue_dropped'] != 0:
        raise ValueError(f'{path}: incomplete journal')
    if not commands or commands[-1]['packet_weight'] != 0:
        raise ValueError(f'{path}: missing final zero-weight packet')
    if any(not r['feedback_crc_valid'] for r in commands):
        raise ValueError(f'{path}: invalid command feedback CRC')
    q = np.asarray([r['q_measured_rad'][5:10] for r in active])
    dq = np.asarray([r['dq_measured_rad_s'][5:10] for r in active])
    ff = np.asarray([r['tau_ff'][5:10] for r in active])
    sequence = [r['sequence'] for r in commands]
    if np.any(np.diff(sequence) != 1) or any(r['packet_weight'] != 1 for r in active):
        raise ValueError(f'{path}: discontinuous command sequence or excitation weight != 1')
    config_sha = sha(Path(path).parent/'identification_config.yaml')
    start = by_event['session_start'][0]
    if config_sha != start['config_sha256']:
        raise ValueError(f'{path}: configuration checksum mismatch')
    timing = by_event['control_timing_summary'][0]['primary_5_18']
    return dict(path=str(path), raw_sha256=sha(path), config_sha256=config_sha,
        profile_sha256=start['profile_sha256'], control_source_sha256=start['control_source_sha256'],
        active_samples=len(active), total_commands=len(commands),
        q_peak_to_peak_deg=np.rad2deg(np.ptp(q, axis=0)).tolist(),
        mean_q_deg=np.rad2deg(q.mean(axis=0)).tolist(),
        max_offset_deg=np.rad2deg(abs(q-NOMINAL).max(axis=0)).tolist(),
        max_dq_rad_s=abs(dq).max(axis=0).tolist(),
        peak_ff_nm=abs(ff).max(axis=0).tolist(),
        normal_release_completed=True, final_weight=0., queue_dropped=0,
        primary_timing=timing, operator_observation='smooth, reported in conversation')


def features(run, delay_ms):
    return np.c_[delayed(run, 'tau_ff', delay_ms*.001),
        run['q_measured_rad']-NOMINAL, run['dq_measured_rad_s'],
        np.tanh(run['dq_measured_rad_s']/.05), run['imu_uncentered']]


def select_delay(first, second):
    """Only these two runs influence parameter selection."""
    n = len(first['time']); scan = []
    y = np.vstack((first['qdd'], second['qdd']))
    for delay in range(31):
        x = np.vstack((features(first, delay), features(second, delay)))
        fit = ridge_fit(x, y, np.arange(n), np.arange(n, len(x)))
        scan.append(dict(delay_ms=delay, mean_rmse_rad_s2=float(fit['rmse'].mean())))
    return min(scan, key=lambda r:r['mean_rmse_rad_s2'])['delay_ms'], scan


def cross_run_review(runs):
    delay, scan = select_delay(runs[0], runs[1])
    x = np.vstack([features(r, delay) for r in runs])
    y = np.vstack([r['qdd'] for r in runs])
    n = sum(len(r['time']) for r in runs[:2])
    train, test = np.arange(n), np.arange(n, len(x))
    full = ridge_fit(x, y, train, test)
    state = ridge_fit(x[:,5:], y, train, test)
    diagonal_prediction = np.zeros_like(y)
    for j in range(5):
        cols = np.r_[j, np.arange(5, x.shape[1])]
        fit = ridge_fit(x[:,cols], y[:,j:j+1], train, test)
        diagonal_prediction[:,j] = fit['prediction'][:,0]
    diagonal = rmse(diagonal_prediction[test], y[test])
    gain = full['coef'][:5].T
    # A physical rigid inertia is symmetric positive definite. Inverting an
    # unconstrained closed-loop regression does not establish such an inertia.
    mass = np.linalg.pinv(gain)
    physical = dict(symmetry_relative_error=float(np.linalg.norm(mass-mass.T)/np.linalg.norm(mass)),
        symmetric_part_eigenvalues=np.linalg.eigvalsh(.5*(mass+mass.T)).tolist(),
        warning='This inverse effective input matrix is NOT an identified physical inertia')
    result = dict(selection='fit run1, select delay using run2; fit run1+run2; evaluate run3',
        selected_delay_ms=delay, development_delay_scan=scan,
        full_rmse_rad_s2=full['rmse'].tolist(), diagonal_rmse_rad_s2=diagonal.tolist(),
        state_only_rmse_rad_s2=state['rmse'].tolist(),
        zero_acceleration_rmse_rad_s2=rmse(np.zeros_like(y[test]),y[test]).tolist(),
        improvement_over_state_percent=(100*(state['rmse']-full['rmse'])/state['rmse']).tolist(),
        full_coefficients=full['coef'].tolist(), full_intercept=full['intercept'].tolist(),
        effective_input_matrix=gain.tolist(), inverse_effective_input_matrix=mass.tolist(),
        physical_checks=physical,
        feature_order=['ff[5]', 'q-NOMINAL[5]', 'dq[5]', 'tanh(dq/0.05)[5]', 'raw_local_acc_gyro[6]'])
    arrays=dict(test_qdd=y[test], test_full=full['prediction'][test],
        test_state_only=state['prediction'][test], test_diagonal=diagonal_prediction[test])
    return result, arrays


def derivative_sensitivity(runs):
    """Offline sensitivity checks, NOT a filter selection using held-out error.

    These centered derivatives use future samples and must not be copied into
    the live controller. Small motion plus noisy dq can dominate a short-window
    derivative. Test two fixed alternative target definitions, keeping the
    measured-state inputs and the train/validation split unchanged.
    """
    results = []
    for field, window, order in [('dq_measured_rad_s', 41, 1),
                                 ('q_measured_rad', 61, 2)]:
        variants = [dict(r, qdd=savgol_filter(r[field], window, 3,
                    deriv=order, delta=.006, axis=0)) for r in runs]
        result, _ = cross_run_review(variants)
        results.append(dict(source=field, window_samples=window,
            nominal_window_ms=window*6,
            selected_delay_ms=result['selected_delay_ms'],
            full_rmse_rad_s2=result['full_rmse_rad_s2'],
            zero_acceleration_rmse_rad_s2=result['zero_acceleration_rmse_rad_s2'],
            improvement_over_state_percent=result['improvement_over_state_percent'],
            effective_input_matrix=result['effective_input_matrix'],
            physical_checks=result['physical_checks']))
    return results


def simple_telemetry_fit(runs):
    """Fit tau_est ~ gain * (tau_ff + PD) + offset independently per axis.

    Uses current measured q/dq for firmware-side PD; no additional q/dq
    regressors that could absorb PD gain. Still telemetry, not shaft torque.
    """
    def requested(run, delay):
        return (delayed(run,'tau_ff',delay*.001)
            + run['kp_command']*(delayed(run,'q_command_rad',delay*.001)-run['q_measured_rad'])
            + run['kd_command']*(delayed(run,'dq_command_rad_s',delay*.001)-run['dq_measured_rad_s']))
    def fit_one(train, test, delay):
        x = np.concatenate([requested(r,delay) for r in train])
        y = np.concatenate([r['tau_est_at_feedback_nm'] for r in train])
        xt = requested(test, delay); yt = test['tau_est_at_feedback_nm']
        coefficients=np.zeros((5,2)); pred=np.zeros_like(yt)
        for j in range(5):
            coefficients[j]=np.linalg.lstsq(np.c_[x[:,j],np.ones(len(x))],y[:,j],rcond=None)[0]
            pred[:,j]=coefficients[j,0]*xt[:,j]+coefficients[j,1]
        return coefficients,rmse(pred,yt),rmse(xt,yt)
    scans=[]
    for delay in range(31):
        _,err,_=fit_one(runs[:1],runs[1],delay)
        scans.append(dict(delay_ms=delay,mean_rmse_nm=float(err.mean())))
    delay=min(scans,key=lambda x:x['mean_rmse_nm'])['delay_ms']
    coef,err,identity=fit_one(runs[:2],runs[2],delay)
    return dict(development_delay_scan=scans,selected_delay_ms=delay,
        gain=coef[:,0].tolist(),offset_nm=coef[:,1].tolist(),
        heldout_rmse_nm=err.tolist(),identity_heldout_rmse_nm=identity.tolist(),
        scope='Consistency with reported tau_est only; not motor execution delay or absolute calibration')


def model_predictions(path, run, inverse):
    from hardware_mpc_predictor import HardwareMpcPredictor
    predictor=HardwareMpcPredictor(mode='hold_current')
    low=[];imu=[]
    with Path(path).open() as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('schema')=='g1_lowstate_raw_v1':
                motors=row['motors']
                low.append((row['received_monotonic_ns'],
                    np.asarray([m['q_rad'] for m in motors]),np.asarray([m['dq_rad_s'] for m in motors])))
            elif row.get('schema')=='g1_torso_imu_raw_v1':
                imu.append((row['received_monotonic_ns'],row['quaternion_wxyz'],
                    row['gyroscope_rad_s'],row['accelerometer_raw_m_s2']))
    li=ii=0;mass=[];bias=[]
    for k,t in enumerate(run['time']):
        stamp=round(t*1e9)
        while li<len(low) and low[li][0]<=stamp:
            predictor.observe_low(*low[li]);li+=1
        while ii<len(imu) and imu[ii][0]<=stamp:
            predictor.observe_imu(*imu[ii]);ii+=1
        base=predictor.query(stamp,0.,use_learned=False).horizon.nodes[0]
        m,b=inverse.linear_dynamics(run['q_measured_rad'][k],run['dq_measured_rad_s'][k],base)
        mass.append(m);bias.append(b)
    mass=np.asarray(mass);bias=np.asarray(bias)
    return mass,bias,np.einsum('nij,nj->ni',mass,run['qdd'])+bias


def friction_ablation(runs, nominal, actual):
    """Keep the rigid model fixed; separate constant bias from friction benefit.

    A single constant offset can hide torque-report/gravity mismatch. Friction
    must improve beyond that comparison, with coefficients stable across the
    first two runs. Offsets are diagnostic, not proposed motor commands.
    """
    dq = np.vstack([r['dq_measured_rad_s'] for r in runs])
    counts = [len(r['time']) for r in runs]
    n = sum(counts[:2]); residual = actual-nominal
    indices = [np.arange(counts[0]), np.arange(counts[0], n), np.arange(n)]
    # Constant-only fit has no speed or direction compensation.
    constant = residual[:n].mean(axis=0)
    candidates = []
    for velocity_scale in [.02, .05, .10]:
        coef = np.zeros((3, 5, 2))
        correction = np.zeros_like(actual)
        for j in range(5):
            x = np.c_[np.ones(len(dq)), np.tanh(dq[:, j]/velocity_scale)]
            for k, train in enumerate(indices):
                coef[k, j] = lsq_linear(x[train], residual[train, j],
                    bounds=([-2., 0.], [2., 1.])).x
            correction[:, j] = x@coef[2, j]
        candidates.append(dict(velocity_scale_rad_s=velocity_scale,
            run1_coulomb_nm=coef[0, :, 1].tolist(),
            run2_coulomb_nm=coef[1, :, 1].tolist(),
            train12_coulomb_nm=coef[2, :, 1].tolist(),
            train12_bias_nm=coef[2, :, 0].tolist(),
            heldout_bias_and_friction_rmse_nm=rmse((nominal+correction)[n:], actual[n:]).tolist(),
            # Do NOT silently also add a fitted constant to live force output.
            heldout_friction_only_rmse_nm=rmse(
                (nominal+correction-coef[2, :, 0])[n:], actual[n:]).tolist()))
    return dict(heldout_bias_only_rmse_nm=rmse((nominal+constant)[n:],actual[n:]).tolist(),
        bias_only_nm=constant.tolist(), candidates=candidates,
        scope='Fixed M/bias; all velocity scales reported, no held-out selection or live parameter update')


def rigid_residual_review(paths,runs):
    """Explore bounded nonnegative dissipative terms against measured torque estimates."""
    from endpoint_pose import EndpointModel,ROOT
    from hardware_arm_inverse_dynamics import RightArmInverseDynamics
    from robot_model_backend.cpp_rnea_backend import CppRightArmRneaBackend
    model=EndpointModel();backend=CppRightArmRneaBackend(model.xml,
        library_path=ROOT/'build/right_arm_rnea/libright_arm_rnea.so')
    inverse=RightArmInverseDynamics(model,backend)
    try:
        dynamics=[model_predictions(p,r,inverse) for p,r in zip(paths,runs)]
    finally:
        backend.close()
    nominal=np.vstack([r[2] for r in dynamics]);mass=np.vstack([r[0] for r in dynamics])
    bias=np.vstack([r[1] for r in dynamics]);actual=np.vstack([r['tau_est_at_feedback_nm'] for r in runs])
    dq=np.vstack([r['dq_measured_rad_s'] for r in runs]);target=np.vstack([r['qdd'] for r in runs])
    n=sum(len(r['time']) for r in runs[:2]); coefficients=np.zeros((5,3));correction=np.zeros_like(actual)
    for j in range(5):
        # Fixed constraints are exploratory numerical bounds, not hardware limits.
        x=np.c_[np.ones(len(dq)),dq[:,j],np.tanh(dq[:,j]/.05)]
        fit=lsq_linear(x[:n],(actual-nominal)[:n,j],bounds=([-2,0,0],[2,5,1]))
        coefficients[j]=fit.x;correction[:,j]=x@fit.x
    predicted_acc=np.linalg.solve(mass,(actual-bias)[...,None])[...,0]
    corrected_acc=np.linalg.solve(mass,(actual-bias-correction)[...,None])[...,0]
    # The noisier small-signal speed channels may not independently support
    # viscous/Coulomb separation; retain that correlation in the evidence.
    correlations=[float(np.corrcoef(dq[:n,j],np.tanh(dq[:n,j]/.05))[0,1]) for j in range(5)]
    return dict(model_xml_sha256=model.xml_hashes(),inverse_model_metadata=inverse.metadata,
        fixed_rigid_model_friction_ablation=friction_ablation(runs,nominal,actual),
        candidate_bias_nm=coefficients[:,0].tolist(),candidate_viscous=coefficients[:,1].tolist(),
        candidate_coulomb_nm=coefficients[:,2].tolist(),
        speed_vs_friction_regressor_correlation=correlations,
        nominal_tau_est_rmse_nm=rmse(nominal[n:],actual[n:]).tolist(),
        corrected_tau_est_rmse_nm=rmse((nominal+correction)[n:],actual[n:]).tolist(),
        nominal_acc_from_tau_est_rmse_rad_s2=rmse(predicted_acc[n:],target[n:]).tolist(),
        corrected_acc_from_tau_est_rmse_rad_s2=rmse(corrected_acc[n:],target[n:]).tolist(),
        scope='Offline residual screening conditional on tau_est accuracy; no independent physical torque truth',
        integration_approved=False),dict(test_tau_est=actual[n:],test_rigid_tau=nominal[n:],
            test_corrected_tau=(nominal+correction)[n:],test_rigid_acc=predicted_acc[n:],
            test_corrected_acc=corrected_acc[n:])


def plots(output,runs,arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,2,figsize=(12,12))
    for j,name in enumerate(JOINTS):
        for k,r in enumerate(runs):
            t=r['time']-r['time'][0]
            axes[j,0].plot(t,np.rad2deg(r['q_measured_rad'][:,j]),label=f'run {k+1}',lw=.9)
        axes[j,0].set_ylabel(name+' (deg)')
        t=runs[2]['time']-runs[2]['time'][0]
        for key,label in [('test_qdd','measured, offline derivative'),('test_full','full 5x5 regression'),
                          ('test_state_only','state only')]:
            axes[j,1].plot(t,arrays[key][:,j],label=label,lw=.7,alpha=.8)
        axes[j,1].set_ylabel('acceleration (rad/s2)')
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    for a in axes[-1]:a.set_xlabel('time from trimmed excitation start (s)')
    fig.suptitle('Stationary identification: three repeats and held-out run 3')
    fig.tight_layout();fig.savefig(output/'joint_response_and_holdout.png',dpi=150)
    fig.savefig(output/'joint_response_and_holdout.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw',type=Path,nargs=3);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if len(set(path.resolve() for path in args.raw))!=3:
        raise ValueError('three distinct captures are required')
    audits=[audit(path) for path in args.raw]
    if len({a['raw_sha256'] for a in audits})!=3 or len({a['config_sha256'] for a in audits})!=1:
        raise ValueError('captures must differ and configurations must match')
    if len({a['profile_sha256'] for a in audits})!=1 or any(a['control_source_sha256']!=audits[0]['control_source_sha256'] for a in audits):
        raise ValueError('profiles and executed source versions must match')
    runs=[load_run(path) for path in args.raw]
    args.output_dir.mkdir(parents=True,exist_ok=False)
    cross,arrays=cross_run_review(runs)
    sensitivity=derivative_sensitivity(runs)
    telemetry=simple_telemetry_fit(runs)
    rigid,rigid_arrays=rigid_residual_review(args.raw,runs);arrays.update(rigid_arrays)
    report=dict(schema='g1_arm_identification_three_run_review_v1',
        capture_audit=audits,acceleration_cross_run=cross,reported_torque=telemetry,
        offline_derivative_sensitivity=sensitivity,
        rigid_model_residual=rigid,
        model_parameter_changes_applied=False,
        limitations=['Third run is held out from fitting and delay selection; same repeated excitation, not a new gait/task',
            'qdd is an offline centered dq derivative, not a direct sensor or causal prediction',
            'Host receive/write timestamps do not establish physical actuator delay',
            'Identification yaw PD 6/0.5 differs from MPC 2/0.2',
            'Raw dq noise, stiction and weak high-frequency motion limit inertia/friction separation',
            'Reported torque consistency cannot establish absolute motor gain or justify gain compensation'],
        source_sha256={str(Path(__file__)):sha(__file__),
            'analyze_arm_system_identification.py':sha(Path(__file__).with_name('analyze_arm_system_identification.py'))})
    (args.output_dir/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.output_dir/'heldout_predictions.npz',**arrays)
    plots(args.output_dir,runs,arrays)
    print(json.dumps(dict(output=str(args.output_dir),
        heldout_full_rmse_rad_s2=cross['full_rmse_rad_s2'],
        heldout_tau_est_rmse_nm=telemetry['heldout_rmse_nm'],
        corrected_tau_est_rmse_nm=rigid['corrected_tau_est_rmse_nm'],
        model_parameter_changes_applied=False),indent=2))


if __name__=='__main__':main()
