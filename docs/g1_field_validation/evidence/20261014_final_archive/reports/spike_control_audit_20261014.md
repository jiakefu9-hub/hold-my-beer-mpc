# Learned MPC control-side spike audit (2026-10-14)

Read-only analysis of physical captures. No controller/configuration changes and no robot output.

## Captures and window

All statistics below use successful DDS writes in task [5,15) s (forward walking, excluding stop settling).

- 20261013_164838: morning_M01_mpc_learned_zero_payload75_walk_20261013_164838_063152_fd65/raw.jsonl, 1657 writes.
- 20261014_124321: morning_M01_mpc_learned_zero_payload75_walk_20261014_124321_974583_1119/raw.jsonl, 1666 writes.
- 20261014_125333: morning_M01_mpc_learned_zero_payload75_walk_20261014_125333_831147_ee9f/raw.jsonl, 1663 writes.

All use learned_filtered, alignment disabled, normal left PD 20/1. All ended normal_release_completed, weight zero, capture dropped zero. Logs explicitly do not independently certify physical stopping.

## Observed command discontinuities

| Quantity | 13th 164838 | 14th 124321 | 14th 125333 |
|---|---:|---:|---:|
| Shoulder pitch maximum absolute per-write total-torque change Nm | 6.265 | 5.231 | 6.721 |
| Shoulder roll same Nm | 2.309 | 2.828 | 3.725 |
| Shoulder yaw same Nm | 1.725 | 2.108 | 2.393 |
| Elbow same Nm | 3.081 | 3.177 | 3.326 |
| Pitch total-torque-change P99 Nm | 2.846 | 2.913 | 3.469 |
| Pitch acceleration-command maximum per-write change rad/s2 | 30 | 29.002 | 30 |
| Any total-torque saturation | none | none | none |
| Pitch predictive brake active samples percent | .121 | 0 | .361 |
| Yaw predictive brake active samples percent | .302 | .540 | .661 |
| Prewrite work P99 ms | 4.522 | 4.126 | 4.348 |
| Joint observation host-receive age at write P99 ms | 6.981 | 6.463 | 6.737 |
| Prediction grid anchor to write P99 ms | 4.799 | 5.699 | 4.979 |

Most large command changes occur without a latched brake. Braking can amplify individual episodes but is too sparse to explain the general phenomenon alone. Absolute torque limits do not constrain changes between successive commands. There is no active torque slew constraint or delta-acceleration objective in this configuration.

Examples in latest 125333 capture:

- At task 9.672463 s, shoulder pitch total command changes by +6.721 Nm, of which changing model bias contributes +4.001 Nm. Controller work 2.927 ms; no brake.
- At 14.742712 s, acceleration-command changes across pitch/roll/yaw/elbow/wrist are [30,-30,-25.774,30,-30] rad/s2. Pitch torque changes +5.533 Nm, elbow +3.184 Nm. Pitch brake active; controller work 3.374 ms.
- At 14.766457 s, pitch acceleration change +24.561 rad/s2 and torque change +5.779 Nm; no brake.

The largest events are not concentrated in computation overruns. Clock/actuator/filter delays remain relevant even when software completes within 6 ms.

## Joint movement evidence (not differentiated endpoint metrics)

Latest capture measured angular speed P99, left/right:

| Joint | Left rad/s | Right rad/s |
|---|---:|---:|
| shoulder pitch | .4844 | .5427 |
| shoulder roll | .4656 | .7529 |
| shoulder yaw | .2441 | .9210 |
| elbow pitch | .3018 | .4054 |
| wrist roll | .0261 | .0997 |

Additional frequency check: linearly resample DDS q/dq to 6 ms, t=5.02..14.98 s, scipy.signal.welch nperseg=256; integrate PSD over 15..70 Hz. These are descriptive host-observation spectra, not calibrated structural modal estimates.

| Signal/band RMS | 13th left | 13th right | 14th125333 left | 14th125333 right |
|---|---:|---:|---:|---:|
| Shoulder roll dq rad/s | .03584 | .09812 | .03384 | .10772 |
| Shoulder yaw dq rad/s | .01786 | .11475 | .01902 | .13093 |
| Shoulder roll q rad | .00029 | .00061 | .00023 | .00061 |
| Shoulder yaw q rad | .00012 | .00077 | .00011 | .00075 |

Right high-frequency joint activity is greater in both logs; the effect appears in joint position as well as velocity. It cannot be dismissed solely as differentiation of velocity noise. No endpoint acceleration or collision claims are made from this table.

## Code interpretation and hypotheses

- hardware_mpc_learned.py:136 computes tau_total=M*a+b, tau_ff=tau_total-[Kp(qref-q)+Kd(dqref-dq)]. Firmware still provides local feedback between host updates. It is incorrect to call this 'no PD', or to assume it has the same fixed-reference behavior as the left arm.
- hardware_mpc_torque_control.py:294 anchors shoulder yaw qref at nominal; the other four references are one-step predictions following measured state. Right packet Kp=[20,20,2,20,20], Kd=[1,1,.2,1,1], versus all-left 20/1. Yaw gains are not comparable between sides.
- hardware_mpc_torque_control.py:317 explicitly records absolute torque constraints true, torque-rate constraints false. hardware_mpc_learned.py:137 rejects an unplanned slew bound. Therefore adding an output clip alone is inconsistent with the existing final model envelope.
- hardware_mpc_predictor.py:219 filters body acceleration/gyro and leg features using a causal 15 Hz first-order filter on a 2 ms grid; angular acceleration is derived from filtered gyro then filtered again. MPC q/dq observations are raw latest joint feedback, not passed through that same body filter. Sensor acquisition and actuator-internal latency are unknown.
- The learned forecast targets filtered disturbance signals. It cannot be assumed to predict a sharp raw impact onset early enough to switch a controller before the impact. Replay detection timing must be assessed separately.

Evidence supports a possible impact -> noisy/fast dq feedback and changing moving-base bias -> rapidly reversing MPC correction -> additional transient loop. It does not identify the causal contribution of each component or prove model signs are wrong.

## Recommended investigation order

1. Preserve baseline. Compare peak/high-frequency metrics alongside existing 15 Hz metrics. Use independently defined body-impact events and count joint/control transients around them.
2. First controller candidate: a small in-QP penalty on consecutive desired accelerations, including first action versus the last actually sent/accepted action. This targets observed sign reversal while keeping q/dq/torque/braking constraints in the optimization. A penalty on total predicted torque change is more direct but must handle changing b; smoothing M*a alone does not remove bias jumps. Both require closed-loop validation; replay only establishes altered commands.
3. Audit high-frequency raw dq against q-derived velocity before selecting any velocity filter. Filtering feedback without accounting for delay may worsen damping.
4. Treat impact-triggered hold as an alternative experiment, not a guaranteed fix. Freeze a reference once, maintain support torque, blend total torque continuously in/out, keep MPC updated, and bound duration/hysteresis. Direct tau_ff=0 plus current-angle capture removes roughly 2-3 Nm support on pitch/elbow and can itself create a new disturbance. Fixed joint references also transmit torso angular motion and may trade upright attitude for relative stiffness. No duration/gain is validated by this audit.

One cannot infer that increasing right Kp/Kd alone will recreate left PD: at the measured feedback instant the current implementation subtracts the commanded PD contribution. Any modified feedback law must be represented consistently in prediction and total torque accounting.
