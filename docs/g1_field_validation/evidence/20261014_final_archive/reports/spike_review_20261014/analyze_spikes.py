"""Read-only controller/capture audit; outputs are isolated in this directory."""
import os
os.environ['MPLCONFIGDIR']='/tmp/g1_spike_matplotlib'
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/g1_commissioning'))
import hashlib,json
import orjson
import numpy as np
from scipy.signal import savgol_filter,find_peaks,butter,sosfiltfilt
from scipy.spatial.transform import Rotation
from analyze_hardware_pid import _interp,_continuous_quaternions,_angular_velocity,_rotation_z
from endpoint_pose import EndpointModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=ROOT/'evaluation/hardware_shadow/commissioning'
OUT=Path(__file__).resolve().parent
IDX=[15,16,17,18,19,22,23,24,25,26,12]

def norm(x):return np.linalg.norm(x,axis=-1)
def scalar(x):
 return dict(rms=float(np.sqrt(np.mean(x*x))),p95=float(np.percentile(x,95)),
             p99=float(np.percentile(x,99)),max=float(np.max(x)))
def blocks(t,mask):
 change=np.diff(np.r_[False,mask,False].astype(int))
 return [(float(t[a]),float(t[b-1]+.005)) for a,b in zip(np.where(change==1)[0],np.where(change==-1)[0])]

def load(path):
 low=[];high=[];imu=[];commands=[];events=[];session=None;epoch=None;yaw=None;sha=hashlib.sha256();errors=0
 for line in path.open('rb'):
  sha.update(line)
  try:r=orjson.loads(line)
  except orjson.JSONDecodeError:errors+=1;continue
  ev=r.get('event');sc=r.get('schema')
  if ev=='session_start':session=r
  if ev=='task_epoch':epoch=r['task_epoch_monotonic_ns']
  if ev=='heading_reference_frozen':yaw=r['yaw0_rad']
  if ev in ('session_end','session_fault','capture_drained','space_key','task_stage','normal_weight_release_started'):
   events.append(r)
  if sc=='g1_lowstate_raw_v1' and r.get('crc_valid'):
   m={int(v['index']):v for v in r['motors']}
   low.append((r['received_monotonic_ns'],[float(m[i]['q_rad']) for i in IDX]))
  if sc=='g1_mpc_predictor_low_v1':
   high.append((r['received_monotonic_ns'],[r['q_rad'][i] for i in IDX],[r['dq_rad_s'][i] for i in IDX]))
  if sc=='g1_torso_imu_raw_v1':
   imu.append((r['received_monotonic_ns'],r['quaternion_wxyz'],r['accelerometer_raw_m_s2'],r['gyroscope_rad_s']))
  if ev=='dds_write' and r.get('task_elapsed_s') is not None:
   commands.append((r['task_elapsed_s'],r.get('stage'),r.get('mpc_active',False)))
 meta=dict(directory=path.parent.name,raw_sha256=sha.hexdigest(),invalid_json_lines=errors,
           events=events,commands=len(commands),has_epoch=epoch is not None)
 if not commands or epoch is None or yaw is None:return meta,None
 meta.update(predictor=session['predictor_mode'],left_gain_scale=session['left_pd_gain_experiment']['scale'],
             config_hashes={k:session[k] for k in ('mpc_config_sha256','torque_config_sha256','controller_config_sha256')})
 return meta,(session,epoch,yaw,low,high,imu,commands)

def derive(path,raw,source):
 session,epoch,yaw,low,high,imu,commands=raw
 data=high if source=='full_rate_feedback' else low
 lt=np.array([(r[0]-epoch)*1e-9 for r in data]);it=np.array([(r[0]-epoch)*1e-9 for r in imu]);ct=np.array([r[0] for r in commands])
 start=max(3.,lt[0],it[0]);end=min(ct[-1],lt[-1],it[-1])
 t=np.arange(np.ceil(start*200)/200,end,.005)
 slots=np.zeros((len(t),13));slots[:,:11]=_interp(lt,[r[1] for r in data],t)
 quats=_continuous_quaternions(_interp(it,_continuous_quaternions([r[1] for r in imu]),t))
 rb=np.einsum('ij,njk->nik',_rotation_z(-yaw),Rotation.from_quat(quats[:,[1,2,3,0]]).as_matrix())
 bodyacc=np.einsum('nij,nj->ni',rb,_interp(it,[r[2] for r in imu],t))+[0,0,-9.81]
 model=EndpointModel(path.parent/'controller_config.yaml')
 pos={s:[] for s in ('left','right')};rot={s:[] for s in pos}
 for q,R in zip(slots,rb):
  rel=model.relative(q)
  for s in pos:
   p,r=rel[s];pos[s].append(R@p);rot[s].append(R@r)
 ix=np.clip(np.searchsorted(ct,t,side='right')-1,0,len(ct)-1)
 active=np.array([commands[i][2] for i in ix],bool)
 stages=np.array([commands[i][1] for i in ix])
 walk=(stages=='forward_walk')&active
 masks={'walk':walk,'all_control':active&(t>=5)}
 masks['interior_walk']=np.zeros(len(t),bool)
 for a,b in blocks(t,walk):masks['interior_walk']|=(t>=a+.25)&(t<b-.25)
 ht=np.array([(r[0]-epoch)*1e-9 for r in high]);dq=_interp(ht,[r[2] for r in high],t)
 meta=dict(source=source,low_dt_ms=scalar(np.diff(lt[(lt>=5)&(lt<end)])*1000),
           forward_segments_s=blocks(t,walk),control_segments_s=blocks(t,masks['all_control']))
 return t,bodyacc,{s:np.array(v) for s,v in pos.items()},{s:np.array(v) for s,v in rot.items()},masks,dq,meta

def metrics(t,body,pos,rot,masks,window):
 n=round(window/.005);n+=not n%2
 acc={s:savgol_filter(body,n,3,axis=0)+savgol_filter(pos[s],n,3,deriv=2,delta=.005,axis=0) for s in pos}
 omega={s:savgol_filter(_angular_velocity(rot[s],.005),n,3,axis=0) for s in pos}
 alpha={s:savgol_filter(omega[s],n,3,deriv=1,delta=.005,axis=0) for s in pos}
 result={}
 for name,mask in masks.items():
  if mask.sum()<10:continue
  out=dict(duration_s=float(mask.sum()*.005),sides={})
  for s in pos:
   an=norm(acc[s]);wn=norm(omega[s]);aln=norm(alpha[s])
   peak_times={};counts={}
   for thr in (4.,6.,8.):
    pk,_=find_peaks(an,height=thr,prominence=1.,distance=30)
    pk=pk[mask[pk]];counts[str(thr)]=len(pk);peak_times[str(thr)]=t[pk].tolist()
   pp,_=find_peaks(aln,height=15.,prominence=4.,distance=30);pp=pp[mask[pp]]
   z=rot[s][mask,:,2];mean=z.mean(0);mean/=norm(mean)
   tilt=np.degrees(np.arccos(np.clip(z[:,2],-1,1)));swing=np.degrees(np.arccos(np.clip(z@mean,-1,1)))
   out['sides'][s]=dict(acc=scalar(an[mask]),alpha=scalar(aln[mask]),omega=scalar(wn[mask]),
           tilt_mean_deg=float(tilt.mean()),swing_rms_deg=float(np.sqrt(np.mean(swing*swing))),
           acc_peaks=counts,acc_peak_times_s=peak_times,alpha_peaks_over15=len(pp),
           fraction_acc_over4=float(np.mean(an[mask]>4.)))
  result[name]=out
 # Raw torso acceleration peaks: impact proxy only, no force/contact sensor.
 envelope=savgol_filter(norm(body),7,2)
 pk,_=find_peaks(envelope,height=4.,prominence=2.,distance=50)
 pk=pk[masks['interior_walk'][pk]]
 impacts=[]
 for p in pk:
  m=(t>=t[p]-.08)&(t<=t[p]+.08)
  impacts.append(dict(time_s=float(t[p]),body_peak_m_s2=float(envelope[p]),
     left_acc_peak=float(norm(acc['left'][m]).max()),right_acc_peak=float(norm(acc['right'][m]).max()),
     left_alpha_peak=float(norm(alpha['left'][m]).max()),right_alpha_peak=float(norm(alpha['right'][m]).max())))
 result['impact_proxies']=impacts
 return result,acc,alpha,omega

def main():
 paths=sorted(p for p in BASE.glob('morning_M*_*/raw.jsonl')
   if any(x in p.parent.name for x in ('20261013','20261014'))
   and any(x in p.parent.name for x in ('M01_','M06_','M08_','M10_')))
 results=[]
 for path in paths:
  print('Reading',path.parent.name,flush=True)
  meta,raw=load(path)
  if raw is None:
   meta['excluded_reason']='No completed command stream/epoch/heading';results.append(meta);continue
  if any(r.get('event')=='session_fault' for r in meta['events']):
   meta['excluded_reason']='faulted capture; not included in summary';results.append(meta);continue
  meta['sources']={}
  for source in ('decimated_recorded_lowstate','full_rate_feedback'):
   t,body,pos,rot,masks,dq,info=derive(path,raw,source)
   source_result=dict(**info,windows={})
   for window in (.075,.105,.135):
    result,acc,alpha,omega=metrics(t,body,pos,rot,masks,window)
    source_result['windows'][str(round(window*1000))]=result
    if source=='full_rate_feedback' and window==.105:
     fig,axs=plt.subplots(3,1,figsize=(13,8),sharex=True)
     axs[0].plot(t,norm(body),label='Torso raw IMU |a|',color='.6',alpha=.8)
     for s,c in [('left','tab:blue'),('right','tab:orange')]:
      axs[0].plot(t,norm(acc[s]),label=s,c=c,lw=1)
      axs[1].plot(t,norm(alpha[s]),label=s,c=c,lw=1)
      axs[2].plot(t,np.degrees(np.arccos(np.clip(rot[s][:,2,2],-1,1))),label=s,c=c,lw=1)
     for ax,y in zip(axs,('Linear acc (m/s^2)','Angular acc (rad/s^2)','Tilt (deg)')):
      ax.set_ylabel(y);ax.legend(loc='upper right');ax.grid(alpha=.3)
      for a,b in blocks(t,masks['walk']):ax.axvspan(a,b,color='green',alpha=.05)
     axs[-1].set_xlabel('Task time (s)');axs[0].set_title(path.parent.name.replace('morning_',''))
     fig.tight_layout();fig.savefig(OUT/(path.parent.name+'.png'),dpi=120);plt.close(fig)
     np.savez_compressed(OUT/(path.parent.name+'_arrays.npz'),t=t,body=body,walk=masks['walk'],
          left_acc=acc['left'],right_acc=acc['right'],left_alpha=alpha['left'],right_alpha=alpha['right'])
   sos=butter(3,[15,60],fs=200,btype='bandpass',output='sos')
   dqhp=sosfiltfilt(sos,dq,axis=0)
   m=masks['interior_walk']
   source_result['joint_velocity_band_15_60hz_rms_rad_s']={s:np.sqrt(np.mean(dqhp[m,j:j+5]**2,axis=0)).tolist()
          for s,j in [('left',0),('right',5)]}
   meta['sources'][source]=source_result
  results.append(meta)
  print('Finished',path.parent.name,meta['sources']['full_rate_feedback']['windows']['105']['walk']['sides'],flush=True)
 (OUT/'summary.json').write_text(json.dumps(dict(runs=results,notes=[
   'Model reconstruction, no endpoint accelerometer or rod-flexibility measurement.',
   'Peak count: norm height >=4/6/8 m/s2, prominence>=1 m/s2, distance>=150ms; same rule both arms.',
   'Full-rate callback q is cross-check; old comparable LowState is decimated.',
   '35ms smoothed raw torso acceleration peaks >=4 m/s2 with prominence>=2, spacing>=250ms are impact proxies, not verified foot contacts.',
   '75/105/135ms centered Savitzky-Golay filters are offline estimators; high-frequency impulses attenuated.'
 ]),indent=2)+'\n')

if __name__=='__main__':main()
