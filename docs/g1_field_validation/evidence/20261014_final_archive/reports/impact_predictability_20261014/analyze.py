"""Read-only input analysis; no hardware imports, no controller modifications."""
import json
from pathlib import Path
import sys
import numpy as np
import orjson
from scipy.spatial.transform import Rotation
from scipy.signal import lfilter, find_peaks, welch

ROOT=Path('/home/fjk/g1_ws/hold-my-beer-mpc')
sys.path.insert(0,str(ROOT/'tools/g1_commissioning'))
from hardware_mpc_predictor import GAIN

def stats(x):
    x=np.asarray(x)
    return dict(p50=float(np.percentile(x,50)),p90=float(np.percentile(x,90)),p95=float(np.percentile(x,95)),p99=float(np.percentile(x,99)),max=float(np.max(x)))

def analyze(path):
    commands=[];imus=[];session=None
    for line in path.open('rb'):
        r=orjson.loads(line)
        if r.get('schema')=='g1_mpc_session_v1':session=r
        elif r.get('schema')=='g1_mpc_predictor_imu_v1':imus.append(r)
        elif r.get('schema')=='g1_hardware_mpc_command_v1' and 'predictor' in r and 5<=r['task_elapsed_s']<15:commands.append(r)
    origin=commands[0]['predictor']['anchor_monotonic_ns']
    imus.sort(key=lambda r:r['received_monotonic_ns'])
    times=np.array([(r['received_monotonic_ns']-origin)*1e-9 for r in imus])
    ticks=np.arange(int(np.ceil(times[0]/.002)),int(np.floor(times[-1]/.002))+1)
    grid=ticks*.002
    indices=np.searchsorted(times,grid,side='right')-1
    yaw=commands[0]['yaw0_rad'];c,s=np.cos(yaw),np.sin(yaw);h0=np.array([[c,s,0],[-s,c,0],[0,0,1]])
    rot=Rotation.from_quat(np.array([r['quaternion_wxyz'] for r in imus])[:,[1,2,3,0]])
    raw=(rot.apply(np.array([r['accelerometer_raw_m_s2'] for r in imus]))+[0,0,-9.81])@h0.T
    raw=raw[indices]
    filt,_=lfilter([GAIN],[1,-(1-GAIN)],raw,axis=0,zi=((1-GAIN)*raw[0])[None,:])
    anchors=np.array([(r['predictor']['anchor_monotonic_ns']-origin)*1e-9 for r in commands])
    anchor_i=np.rint(anchors/.002).astype(int)-ticks[0]
    current=np.array([r['predictor']['current_y'][:3] for r in commands])
    forecasts=np.array([r['predictor']['forecast_y'] for r in commands])[:,:,:3]
    parity=np.sqrt(np.mean((filt[anchor_i]-current)**2,axis=0))
    if max(parity)>.05:raise ValueError(parity)
    sends=np.array([(r['write_end_monotonic_ns']-origin)*1e-9 for r in commands])
    queries=np.array([(r['predictor']['query_monotonic_ns']-origin)*1e-9 for r in commands])
    epoch=(commands[0]['write_begin_monotonic_ns']-origin)*1e-9-commands[0]['task_elapsed_s']
    # Task timestamps are only used for approximately 5..15 s selection; predictor anchors remain exact.
    mask=(grid>=0)&(grid<anchors[-1])
    rn=np.linalg.norm(raw,axis=1);fn=np.linalg.norm(filt,axis=1)
    threshold=float(np.percentile(rn[mask],90))
    peaks,_=find_peaks(rn,height=threshold,prominence=1.,distance=125)
    peaks=peaks[mask[peaks]]
    events=[]
    for i in peaks:
        near=np.flatnonzero((grid>=grid[i]-.020)&(grid<=grid[i]+.060))
        j=near[np.argmax(fn[near])]
        row=dict(approx_task_s=float(grid[i]-epoch),raw_norm_m_s2=float(rn[i]),filtered_peak_m_s2=float(fn[j]),filtered_peak_lag_ms=float((grid[j]-grid[i])*1000),predicted_before={})
        for lead in (.006,.012,.024):
            ci=np.searchsorted(sends,grid[i]-lead,side='right')-1
            if ci<0:continue
            k=int(np.floor((grid[i]-anchors[ci])/.006))
            if k<0 or k>=9:continue
            pk=int(np.argmax(np.linalg.norm(forecasts[ci],axis=1)))
            row['predicted_before'][f'{lead*1000:.0f}ms']=dict(actual_lead_from_write_ms=float((grid[i]-sends[ci])*1000),interval_index=k,norm_m_s2=float(np.linalg.norm(forecasts[ci,k])),current_filtered_norm_m_s2=float(np.linalg.norm(current[ci])),predicted_max_over_horizon=float(np.linalg.norm(forecasts[ci,pk])),predicted_peak_time_error_ms=float((anchors[ci]+.006*pk+.002-grid[i])*1000),predicted_peak_interval_index=pk)
        events.append(row)
    rmse={}
    for k in (0,1,3,5,8):
        ii=anchor_i+3*k
        tf=(filt[ii]+filt[ii+1]+filt[ii+2])/3
        tr=(raw[ii]+raw[ii+1]+raw[ii+2])/3
        rmse[str(k)]=dict(interval_ms=[6*k,6*(k+1)],learned_filtered_rmse=float(np.sqrt(np.mean(np.sum((forecasts[:,k]-tf)**2,axis=1)))),hold_filtered_rmse=float(np.sqrt(np.mean(np.sum((current-tf)**2,axis=1)))),learned_raw_rmse=float(np.sqrt(np.mean(np.sum((forecasts[:,k]-tr)**2,axis=1)))),hold_raw_rmse=float(np.sqrt(np.mean(np.sum((current-tr)**2,axis=1)))))
    f,psd=welch(raw[mask],fs=500,nperseg=1024,axis=0)
    power=psd.sum(axis=1)
    lags=np.arange(-30,31)*.002
    cors=[]
    for lag in lags:
        shifted=np.column_stack([np.interp(grid[mask]+lag,grid,filt[:,a]) for a in range(3)])
        cors.append(np.corrcoef(raw[mask].ravel(),shifted.ravel())[0,1])
    result=dict(source=str(path.relative_to(ROOT)),sample_count=len(commands),filter_current_reconstruction_rmse=parity.tolist(),window='command task elapsed [5,15), raw grid matching first to last command anchors',raw_acc_norm=stats(rn[mask]),filtered_acc_norm=stats(fn[mask]),raw_vector_rms=float(np.sqrt(np.mean(rn[mask]**2))),filtered_vector_rms=float(np.sqrt(np.mean(fn[mask]**2))),raw_power_above_15hz_fraction=float(power[f>15].sum()/power.sum()),cross_correlation_best_filtered_delay_ms=float(lags[np.argmax(cors)]*1000),cross_correlation_best=float(max(cors)),query_to_write_ms=stats((sends-queries)*1000),anchor_to_write_ms=stats((sends-anchors)*1000),imu_age_at_query_ms=stats([r['predictor']['imu_age_ms'] for r in commands]),peaks_selection=dict(raw_norm_height_m_s2=threshold,min_prominence_m_s2=1,min_spacing_ms=250),peaks=events,forecast_rmse=rmse)
    result['peak_summary']=dict(count=len(events),median_raw=float(np.median([r['raw_norm_m_s2'] for r in events])),median_filtered=float(np.median([r['filtered_peak_m_s2'] for r in events])),median_filtered_lag_ms=float(np.median([r['filtered_peak_lag_ms'] for r in events])))
    for lead in ('6ms','12ms','24ms'):
        pp=[r for r in events if lead in r['predicted_before']]
        errors=np.array([r['predicted_before'][lead]['predicted_peak_time_error_ms'] for r in pp])
        result['peak_summary'][lead]=dict(count=len(pp),median_predicted_norm=float(np.median([r['predicted_before'][lead]['norm_m_s2'] for r in pp])),median_predicted_over_raw=float(np.median([r['predicted_before'][lead]['norm_m_s2']/r['raw_norm_m_s2'] for r in pp])),peak_time_error_median_ms=float(np.median(errors)),peak_time_abs_error_p90_ms=float(np.percentile(abs(errors),90)),peak_time_within_6ms_fraction=float(np.mean(abs(errors)<=6+1e-6)))
    return result

out=ROOT/'evaluation/hardware_shadow/commissioning/impact_predictability_20261014'
out.mkdir(exist_ok=True)
paths=sorted((ROOT/'evaluation/hardware_shadow/commissioning').glob('morning_M01*2026101[34]*/raw.jsonl'))
results=[analyze(p) for p in paths]
output=dict(method='Diagnostic only. No training or hardware output; detected torso IMU raw acceleration maxima are impact proxies, not verified foot contact labels.',filter_gain=GAIN,low_frequency_filter_group_delay_ms=(1-GAIN)/GAIN*2,runs=results)
(out/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
for r in results:
    print(Path(r['source']).parent.name)
    print(json.dumps({k:r[k] for k in ('raw_acc_norm','filtered_acc_norm','raw_vector_rms','filtered_vector_rms','raw_power_above_15hz_fraction','cross_correlation_best_filtered_delay_ms','query_to_write_ms','peak_summary','forecast_rmse')},indent=2))
