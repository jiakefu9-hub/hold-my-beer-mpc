#!/usr/bin/env python3
"""Independently compare the exported runtime bank with saved research forecasts.

No retraining or robot access. Requires local ignored study artifacts; the
normal hardware runtime does not. Comparison runs are already-inspected data,
not fresh blind validation. Also check causal preprocessor physical features.
"""
import argparse
import json
import os
from pathlib import Path
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"
import numpy as np
from hardware_mpc_predictor import FrozenInnovationBank, HardwareMpcPredictor, ROOT


def verify(study, refinement):
    bank=FrozenInnovationBank()
    maximum=0.; count=0; per_trial=[]
    for trial in (9,10,11,12):
        d=dict(np.load(study/f"data/trial{trial:02d}_prepared.npz"))
        saved=dict(np.load(refinement/f"innovation/trial{trial:02d}_predictions.npz"))
        for i,index in enumerate(saved["anchors"]):
            current=d["y"][index]
            feature=np.r_[d["qf"][index],d["dqf"][index],current[:9]]
            actual,_=bank.predict(feature,current)
            expected=saved["prediction"][i]
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10)
            maximum=max(maximum,float(np.max(np.abs(actual-expected))));count+=actual.size
        per_trial.append(dict(trial=trial,anchors=len(saved["anchors"])))
    # At the same accepted receive-time samples/grid, online filtering in W
    # then rotating to H0 equals the research's H0-then-filter computation.
    raw=dict(np.load(study/"data/trial09.npz"));prepared=dict(np.load(study/"data/trial09_prepared.npz"))
    p=HardwareMpcPredictor("hold_current")
    origin=10_000_000_000;p.set_grid_origin(origin)
    li=max(0,np.searchsorted(raw["low_t"],-.5,side="right")-1)
    ii=max(0,np.searchsorted(raw["imu_t"],-.5,side="right")-1)
    feature_max=0.
    for index in range(0,len(prepared["t"]),3):
        t=float(prepared["t"][index]); stamp=origin+round(t*1e9)
        while li<len(raw["low_t"]) and raw["low_t"][li] <= t:
            p.observe_low(origin+round(float(raw["low_t"][li])*1e9),raw["q"][li],raw["dq"][li]);li+=1
        while ii<len(raw["imu_t"]) and raw["imu_t"][ii] <= t:
            m=raw["imu"][ii]
            p.observe_imu(origin+round(float(raw["imu_t"][ii])*1e9),m[2:6],m[9:12],m[12:15]);ii+=1
        result=p.query(stamp,float(prepared["yaw0_rad"]))
        if t>=5:
            expected=np.r_[prepared["qf"][index],prepared["dqf"][index],prepared["y"][index,:9]]
            error=float(np.max(np.abs(expected-result.diagnostics["features"])))
            feature_max=max(feature_max,error)
    if feature_max>1e-6:
        raise AssertionError(f"online preprocessor mismatch: {feature_max}")
    return dict(schema="g1_hardware_mpc_predictor_verification_v1", passed=True,
        bank_sha256=bank.manifest["bank_sha256"],comparison=per_trial,
        predicted_scalars_checked=count,max_prediction_absolute_error=maximum,
        max_causal_preprocessor_feature_absolute_error=feature_max,
        limitations=["same accepted samples for preprocessing parity; live callback downsampling may differ",
                     "filtered forecast parity, not SO3-rollout forecast accuracy or physical control validation",
                     "comparison episodes were previously inspected, not new blind evaluation"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=ROOT/"evaluation/hardware_shadow/commissioning/walk_h0_predictor_study_20260925")
    parser.add_argument("--refinement",type=Path,default=ROOT/"evaluation/hardware_shadow/commissioning/walk_h0_refinement_20260925")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args();result=verify(args.study,args.refinement)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x") as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
