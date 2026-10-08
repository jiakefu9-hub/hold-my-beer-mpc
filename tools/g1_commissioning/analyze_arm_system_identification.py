#!/usr/bin/env python3
"""Offline preliminary identification from stationary Arm SDK excitation.

This tool never imports Unitree SDK2 and never changes controller parameters.
It estimates a closed-loop *effective* local model.  Robot ``tau_est`` is used
only as reported telemetry, not as independent absolute torque ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


RIGHT = slice(5, 10)
NOMINAL = np.deg2rad([-4., 1., 0., -7.8, 0.])
DT = .006
JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")


def _array(row, key, size=13):
    value = np.asarray(row.get(key), dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{key} must contain {size} finite values")
    return value


def _interp(t, source_t, values):
    values = np.asarray(values,dtype=float)
    return np.column_stack([np.interp(t,source_t,values[:,i]) for i in range(values.shape[1])])


def _zoh(t, source_t, values):
    """Sample commands as values held from their actual DDS write time."""
    values=np.asarray(values,dtype=float)
    indices=np.searchsorted(source_t,t,side="right")-1
    if np.any(indices<0) or np.any(indices>=len(source_t)):
        raise ValueError("zero-order-hold query lies outside command timestamps")
    return values[indices]


def _keep_last_duplicate_time(times, fields, path):
    """LowState may be reused for several control frames; keep one copy."""
    if np.any(np.diff(times)<0):
        raise ValueError(f"{path}: feedback timestamps move backwards")
    keep=np.r_[times[1:]!=times[:-1],True]
    return times[keep],{key:value[keep] for key,value in fields.items()}


def load_run(path):
    path=Path(path);commands=[];imus=[];ends=[];starts=[]
    with path.open() as stream:
        for line_number,line in enumerate(stream,1):
            try: row=json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if row.get("schema")=="g1_arm_identification_session_v1" and row.get("event")=="session_start":
                starts.append(row)
            elif row.get("schema")=="g1_hardware_pid_command_v1" and row.get("event")=="dds_write" \
                    and row.get("stage")=="arm_identification":
                commands.append(row)
            elif row.get("schema")=="g1_torso_imu_raw_v1": imus.append(row)
            elif row.get("schema")=="g1_pid_event_v1" and row.get("event")=="session_end": ends.append(row)
    if len(starts)!=1 or len(ends)!=1 or ends[0].get("outcome")!="normal_release_completed" \
            or float(ends[0].get("final_weight",-1))!=0:
        raise ValueError(f"{path}: requires one identification start and normal weight-zero completion")
    if len(commands)<1200 or len(imus)<500:
        raise ValueError(f"{path}: insufficient complete excitation/IMU samples")
    command_t=np.asarray([int(r["write_begin_monotonic_ns"])*1e-9 for r in commands])
    state_t=np.asarray([int(r["state_received_monotonic_ns"])*1e-9 for r in commands])
    if np.any(np.diff(command_t)<=0): raise ValueError(f"{path}: command timestamps not strictly increasing")
    fields={key:np.asarray([_array(r,key) for r in commands])[:,RIGHT] for key in
            ("q_measured_rad","dq_measured_rad_s","q_command_rad","dq_command_rad_s",
             "kp_command","kd_command","tau_ff","tau_est_at_feedback_nm")}
    measured={key:fields[key] for key in
              ("q_measured_rad","dq_measured_rad_s","tau_est_at_feedback_nm")}
    commanded={key:fields[key] for key in
               ("q_command_rad","dq_command_rad_s","kp_command","kd_command","tau_ff")}
    state_t,measured=_keep_last_duplicate_time(state_t,measured,path)
    imu_t=np.asarray([int(r["received_monotonic_ns"])*1e-9 for r in imus])
    accel=np.asarray([r["accelerometer_raw_m_s2"] for r in imus],dtype=float)
    gyro=np.asarray([r["gyroscope_rad_s"] for r in imus],dtype=float)
    if (np.any(np.diff(imu_t)<=0) or accel.shape!=(len(imu_t),3) or gyro.shape!=(len(imu_t),3)
            or not np.isfinite(np.r_[accel.ravel(),gyro.ravel()]).all()):
        raise ValueError(f"{path}: invalid IMU stream")
    begin=max(command_t[0]+.30,state_t[0]+.30,imu_t[0])
    end=min(command_t[-1]-.30,state_t[-1]-.30,imu_t[-1])
    grid=np.arange(begin,end,DT)
    if len(grid)<1200: raise ValueError(f"{path}: usable resampled window is too short")
    result={key:_interp(grid,state_t,value) for key,value in measured.items()}
    result.update({key:_zoh(grid,command_t,value) for key,value in commanded.items()})
    result["imu"] = np.c_[_interp(grid,imu_t,accel),_interp(grid,imu_t,gyro)]
    result["imu_uncentered"] = result["imu"].copy()
    result["imu"] -= np.mean(result["imu"],axis=0)
    result["time"],result["command_time"],result["source_ff"] = grid,command_t,commanded["tau_ff"]
    result["source_qcmd"],result["source_dqcmd"] = commanded["q_command_rad"],commanded["dq_command_rad_s"]
    result["qdd"] = savgol_filter(result["dq_measured_rad_s"],11,3,deriv=1,delta=DT,axis=0)
    result["path"] = str(path)
    return result


def _split_indices(runs):
    train=[];valid=[];offset=0
    for run in runs:
        n=len(run["time"]);cut=int(.7*n)
        train.extend(range(offset,offset+cut));valid.extend(range(offset+cut,offset+n));offset+=n
    return np.asarray(train),np.asarray(valid)


def ridge_fit(x,y,train,valid,ridge=1e-3):
    mean=x[train].mean(axis=0);scale=x[train].std(axis=0);scale[scale<1e-8]=1.
    xs=(x-mean)/scale
    design=np.c_[xs,np.ones(len(xs))]
    penalty=np.eye(design.shape[1])*ridge;penalty[-1,-1]=0.
    beta=np.linalg.solve(design[train].T@design[train]+penalty,design[train].T@y[train])
    prediction=design@beta
    coef=beta[:-1]/scale[:,None]
    intercept=beta[-1]-mean@coef
    rmse=np.sqrt(np.mean((prediction[valid]-y[valid])**2,axis=0))
    return dict(coef=coef,intercept=intercept,prediction=prediction,rmse=rmse)


def delayed(run,key,delay_s):
    source={"tau_ff":"source_ff","q_command_rad":"source_qcmd","dq_command_rad_s":"source_dqcmd"}[key]
    return _zoh(run["time"]-delay_s,run["command_time"],run[source])


def acceleration_features(runs,delay_s,full=True,include_torque=True):
    rows=[];targets=[]
    for run_index,run in enumerate(runs):
        parts=[]
        if include_torque: parts.append(delayed(run,"tau_ff",delay_s))
        parts.extend((run["q_measured_rad"]-NOMINAL,run["dq_measured_rad_s"],
                      np.tanh(run["dq_measured_rad_s"]/.05),run["imu"]))
        dummy=np.zeros((len(run["time"]),len(runs)));dummy[:,run_index]=1.
        parts.append(dummy);rows.append(np.concatenate(parts,axis=1));targets.append(run["qdd"])
    return np.vstack(rows),np.vstack(targets)


def diagonal_acceleration_rmse(runs,delay_s,train,valid):
    x,y=acceleration_features(runs,delay_s,include_torque=True);values=[]
    # Feature 0..4 is feedforward torque; keep one local input plus all nuisance terms.
    for joint in range(5):
        columns=np.r_[joint,np.arange(5,x.shape[1])]
        values.append(ridge_fit(x[:,columns],y[:,joint:joint+1],train,valid)["rmse"][0])
    return np.asarray(values)


def torque_report_fit(runs,delay_s,train,valid):
    rows=[];targets=[]
    for run_index,run in enumerate(runs):
        ff=delayed(run,"tau_ff",delay_s);qcmd=delayed(run,"q_command_rad",delay_s)
        dqcmd=delayed(run,"dq_command_rad_s",delay_s)
        requested=ff+run["kp_command"]*(qcmd-run["q_measured_rad"])+run["kd_command"]*(dqcmd-run["dq_measured_rad_s"])
        dummy=np.zeros((len(run["time"]),len(runs)));dummy[:,run_index]=1.
        rows.append(np.c_[requested,run["q_measured_rad"]-NOMINAL,run["dq_measured_rad_s"],dummy])
        targets.append(run["tau_est_at_feedback_nm"])
    return ridge_fit(np.vstack(rows),np.vstack(targets),train,valid)


def identify(paths):
    runs=[load_run(path) for path in paths];train,valid=_split_indices(runs)
    candidates=[]
    for delay_ms in range(0,31):
        x,y=acceleration_features(runs,delay_ms*.001)
        fit=ridge_fit(x,y,train,valid)
        candidates.append((delay_ms,float(np.mean(fit["rmse"])),fit))
    delay_ms,_,best=min(candidates,key=lambda item:item[1])
    x,y=acceleration_features(runs,delay_ms*.001)
    state_only=ridge_fit(x[:,5:],y,train,valid)
    diagonal=diagonal_acceleration_rmse(runs,delay_ms*.001,train,valid)
    input_matrix=best["coef"][:5].T
    rank=int(np.linalg.matrix_rank(input_matrix));condition=float(np.linalg.cond(input_matrix))
    effective_mass=(np.linalg.pinv(input_matrix) if rank==5 and condition<500 else None)
    torque_candidates=[]
    for candidate_ms in range(0,31):
        fit=torque_report_fit(runs,candidate_ms*.001,train,valid)
        torque_candidates.append((candidate_ms,float(np.mean(fit["rmse"])),fit))
    tau_delay,tau_error,tau_fit=min(torque_candidates,key=lambda item:item[1])
    # Apparent inverse dynamics against reported tau_est.  Useful for relative
    # friction/coupling screening only; it cannot establish an absolute scale.
    inverse_rows=[];inverse_targets=[]
    for run_index,run in enumerate(runs):
        dummy=np.zeros((len(run["time"]),len(runs)));dummy[:,run_index]=1.
        inverse_rows.append(np.c_[run["qdd"],run["q_measured_rad"]-NOMINAL,
                                  run["dq_measured_rad_s"],np.tanh(run["dq_measured_rad_s"]/.05),
                                  run["imu"],dummy])
        inverse_targets.append(run["tau_est_at_feedback_nm"])
    inverse=ridge_fit(np.vstack(inverse_rows),np.vstack(inverse_targets),train,valid)
    full=np.asarray(best["rmse"]);state=np.asarray(state_only["rmse"])
    result=dict(schema="g1_arm_closed_loop_identification_result_v1",
        inputs=[run["path"] for run in runs],run_count=len(runs),sample_period_s=DT,
        timing_alignment=dict(feedback="state_received_monotonic_ns",
            commands="write_begin_monotonic_ns with zero-order hold",
            imu="received_monotonic_ns"),
        training_rule="first 70% of every run",validation_rule="last 30% of every run",
        usable_samples=int(sum(len(run["time"]) for run in runs)),
        acceleration_response=dict(
            selected_common_delay_ms=delay_ms,
            delay_scan_mean_validation_rmse_rad_s2=[dict(delay_ms=d,rmse=e) for d,e,_ in candidates],
            full_5x5_validation_rmse_rad_s2=full.tolist(),
            diagonal_input_validation_rmse_rad_s2=diagonal.tolist(),
            state_only_validation_rmse_rad_s2=state.tolist(),
            improvement_vs_state_only_percent=(100*(state-full)/state).tolist(),
            improvement_vs_diagonal_percent=(100*(diagonal-full)/diagonal).tolist(),
            effective_acceleration_per_torque_rad_s2_per_nm=input_matrix.tolist(),
            input_matrix_rank=rank,input_matrix_condition=condition,
            provisional_effective_mass_matrix_kg_m2=None if effective_mass is None else effective_mass.tolist()),
        reported_torque_path=dict(selected_common_delay_ms=tau_delay,
            validation_rmse_nm=np.asarray(tau_fit["rmse"]).tolist(),
            apparent_same_joint_gain=np.diag(tau_fit["coef"][:5].T).tolist(),
            warning="tau_est is the robot's own estimate, not independent shaft torque"),
        apparent_inverse_dynamics_from_tau_est=dict(validation_rmse_nm=inverse["rmse"].tolist(),
            mass_matrix_kg_m2=inverse["coef"][:5].T.tolist(),
            diagonal_viscous_nm_per_rad_s=np.diag(inverse["coef"][10:15].T).tolist(),
            diagonal_coulomb_nm=np.diag(inverse["coef"][15:20].T).tolist()),
        acceptance=dict(automatic_controller_update=False,
            preliminary_candidate=bool(len(runs)>=3 and rank==5 and condition<500
                and np.mean(full)<np.mean(diagonal) and np.mean(full)<np.mean(state))),
        limitations=["closed-loop local identification around one arm pose and FSM 500",
            "reported tau_est cannot identify absolute motor torque scale by itself",
            "Savitzky-Golay dq differentiation filters impacts and high-frequency dynamics",
            "raw local-frame IMU is used only as a nuisance regressor",
            "repeatability across at least three captures is required before proposing model changes"])
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw",nargs="+",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    result=identify(args.raw)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(output=str(args.output),run_count=result["run_count"],
        selected_delay_ms=result["acceleration_response"]["selected_common_delay_ms"],
        validation_rmse=result["acceleration_response"]["full_5x5_validation_rmse_rad_s2"],
        preliminary_candidate=result["acceptance"]["preliminary_candidate"]),indent=2))


if __name__=="__main__": main()
