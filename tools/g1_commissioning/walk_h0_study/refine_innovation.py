#!/usr/bin/env python3
"""History forecast + current-IMU innovation with learned horizon decay.

Uses frozen k8 local absolute/delta predictors; no base refit or hardware I/O.
Weights are fitted ONLY on development OOF, not a new independent CV score.
09--12 comparison is exploratory (already inspected in earlier experiments).
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import benchmark as b
from refine_blend import blend

NEAR='legs_current_imu_delta_knn8'
FAR='legs_current_imu_knn8'


def decreasing_projection(target, weight):
    """Weighted pool-adjacent-violators projection, then [0,1] clipping."""
    blocks=[]
    for i,(v,w) in enumerate(zip(target,weight)):
        blocks.append([i,i+1,float(w),float(v*w)])
        while len(blocks)>1 and blocks[-2][3]/blocks[-2][2]<blocks[-1][3]/blocks[-1][2]:
            right=blocks.pop();left=blocks.pop()
            blocks.append([left[0],right[1],left[2]+right[2],left[3]+right[3]])
    result=np.empty(len(target))
    for a,z,w,total in blocks:
        result[a:z]=np.clip(total/w,0.,1.)
    return result


def weights(near,far,truth):
    out=np.zeros((9,3))
    for g in range(3):
        sl=slice(g*3,g*3+3)
        difference=near[:,:,sl]-far[:,:,sl]
        denominator=np.maximum(np.sum(difference**2,axis=(0,2)),1e-15)
        target=-np.sum(difference*(far[:,:,sl]-truth[:,:,sl]),axis=(0,2))/denominator
        out[:,g]=decreasing_projection(target,denominator)
    return out


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir',type=Path,required=True)
    p.add_argument('--local-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    paths=[args.local_dir/f'development_oof/trial{i:02d}.npz' for i in b.DEVELOPMENT]
    protocol=dict(development=b.DEVELOPMENT,comparison=b.HELDOUT,old_five_excluded=True,
        status='Exploratory, not new blind validation; decay fitting is meta-training on development OOF.',
        formula='absolute_future + decay[h,group]*(delta_future-absolute_future); bounded 1>=w6>=...>=w54>=0.',
        orientation='Use delta predictor unchanged; not part of decay fitting.',
        near=NEAR,far=FAR,script_sha256=sha(Path(__file__)),
        dependency_sha256={str(Path(x.__file__)):sha(Path(x.__file__)) for x in (b,__import__('refine_blend'))},
        input_sha256={str(path):sha(path) for path in paths})
    b.dump(args.output_dir/'protocol.json',protocol)
    ds=[dict(np.load(path)) for path in paths]
    near,far,truth=[np.concatenate([d[k] for d in ds]) for k in (NEAR,FAR,'truth')]
    w=weights(near,far,truth)
    assert np.all((w>=0)&(w<=1)) and np.all(np.diff(w,axis=0)<=1e-12)
    # With identical neighbor retrieval, the correction is constant across
    # horizons before learned decay: current minus neighbors' current states.
    delta=near[:,:,:9]-far[:,:,:9]
    np.testing.assert_allclose(delta,np.repeat(delta[:,:1,:],9,axis=1),atol=1e-11,rtol=0)
    np.savez_compressed(args.output_dir/'model.npz',decay=w,near=NEAR,far=FAR)
    b.dump(args.output_dir/'weights.json',dict(horizons_ms=list(range(6,55,6)),group_order=['acc','omega','alpha'],decay=w.tolist()))
    threshold=json.loads((args.baseline_dir/'benchmark/protocol.json').read_text())['large_acc_threshold_m_s2']
    rows=[]
    for i in b.HELDOUT:
        d=dict(np.load(args.local_dir/f'predictions/trial{i:02d}.npz'))
        pred=blend(d[NEAR],d[FAR],w)
        record=dict(np.load(args.baseline_dir/f'data/trial{i:02d}_prepared.npz'))
        rows+=b.metrics(pred,d['truth'],d['raw_acc_truth'],d['raw_acc_hold'],record,d['anchors'],'decayed_innovation',i,threshold)
        np.savez_compressed(args.output_dir/f'trial{i:02d}_predictions.npz',prediction=pred,truth=d['truth'],
            time=d['time'],anchors=d['anchors'],raw_acc_truth=d['raw_acc_truth'],raw_acc_hold=d['raw_acc_hold'])
    b.dump(args.output_dir/'metrics_per_trial.json',rows)
    aggregate=[]
    for key in sorted({(r['region'],r['horizon_ms'],r['group']) for r in rows}):
        rs=[r for r in rows if (r['region'],r['horizon_ms'],r['group'])==key]
        sse,count=sum(r['sse'] for r in rs),sum(r['scalar_count'] for r in rs)
        aggregate.append(dict(method='decayed_innovation',region=key[0],horizon_ms=key[1],group=key[2],
            rmse=float(np.sqrt(sse/count)),sse=sse,scalar_count=count))
    b.dump(args.output_dir/'metrics_aggregate.json',aggregate)
    b.dump(args.output_dir/'artifact_sha256.json',{p.name:sha(p) for p in args.output_dir.iterdir() if p.is_file()})
    print('acc full',[(r['horizon_ms'],r['rmse']) for r in aggregate if r['group']=='acc' and r['region']=='full'])
    print('decay',w[:,0].tolist())


if __name__=='__main__':
    main()
