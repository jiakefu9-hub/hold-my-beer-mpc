"""Recompute all six published phase metrics from archived full-rate NPZs.

Needs only numpy/scipy. No original workstation paths, raw decompression,
MuJoCo, model imports, network or robot SDK. Writes nothing.
"""
from pathlib import Path
import json
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent


def vector_rms(x):return float(np.sqrt(np.mean(np.sum(x*x,axis=1))))


def main():
    report=json.loads((HERE/'reports/phase_comparison_20261014/summary.json').read_text())
    checks=0
    for name,run in report['runs'].items():
        with np.load(HERE/'runs'/run['directory']/'endpoint_analysis/metrics.npz') as z:
            t=z['task_elapsed_s'];assert np.allclose(np.diff(t),.005)
            for side in ('left','right'):
                R=Rotation.from_quat(z[side+'_orientation_h0_quaternion_wxyz'][:,[1,2,3,0]]).as_matrix()
                axis=R[:,:,2]
                vel=savgol_filter(z[side+'_position_from_torso_imu_h0_m'],21,3,deriv=1,delta=.005,axis=0)
                increments=Rotation.from_matrix(np.einsum('nij,njk->nik',R[:-1].transpose(0,2,1),R[1:])).as_rotvec()/.005
                omega=np.empty((len(t),3));omega[:-1]=np.einsum('nij,nj->ni',R[:-1],increments);omega[-1]=omega[-2]
                omega=savgol_filter(omega,21,3,axis=0)
                for phase,intervals in report['windows'].items():
                    m=np.logical_or.reduce([(t>=a)&(t<b) for a,b in intervals])
                    mean=axis[m].mean(axis=0);mean/=np.linalg.norm(mean)
                    dev=np.degrees(np.arccos(np.clip(axis[m]@mean,-1,1)))
                    values=dict(tilt_mean_deg=float(z[side+'_tilt_deg'][m].mean()),
                        swing_rms_deg=float(np.sqrt(np.mean(dev**2))),
                        acc_rms_m_s2=vector_rms(z[side+'_linear_acceleration_h0_m_s2'][m]),
                        alpha_rms_rad_s2=vector_rms(z[side+'_angular_acceleration_h0_rad_s2'][m]),
                        relative_speed_demeaned_rms_m_s=vector_rms(vel[m]-vel[m].mean(axis=0)),
                        omega_rms_rad_s=vector_rms(omega[m]))
                    for key,value in values.items():
                        np.testing.assert_allclose(value,run['windows'][phase][side][key],rtol=1e-9,atol=1e-9,
                            err_msg=f'{name}/{phase}/{side}/{key}');checks+=1
    for cohort,groups in report['cohorts'].items():
        for group,g in groups.items():
            for phase in report['windows']:
                for side in ('left','right'):
                    for metric in ('tilt_mean_deg','swing_rms_deg','acc_rms_m_s2','alpha_rms_rad_s2',
                                   'relative_speed_demeaned_rms_m_s','omega_rms_rad_s'):
                        value=np.mean([report['runs'][r]['windows'][phase][side][metric] for r in g['ids']])
                        np.testing.assert_allclose(value,g['windows'][phase][side][metric]['mean'],atol=1e-10);checks+=1
    print(json.dumps(dict(status='PASS',runs=10,metrics_checked=checks,
                         basis='archived full-rate NPZ; no 50 Hz previews or local evaluation used'),indent=2))


if __name__=='__main__':main()
