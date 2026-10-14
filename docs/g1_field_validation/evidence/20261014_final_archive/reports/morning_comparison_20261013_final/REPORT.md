# 2026-10-13 morning comparison

All six runs included. Primary [5,18), 200 Hz, 105 ms cubic SG.
Relative velocity is demeaned per window. Per-group means average TWO independent run metrics.

| Run | Side | Tilt mean deg | Swing RMS deg | Acc RMS m/s² | Alpha RMS rad/s² | Relative speed RMS m/s | Omega RMS rad/s |
|---|---|---:|---:|---:|---:|---:|---:|
| M01 | left | 12.97528 | 1.10847 | 1.89957 | 5.42830 | 0.11468 | 0.32897 |
| M01 | right | 0.63165 | 0.59095 | 1.98640 | 5.34694 | 0.10395 | 0.28523 |
| M02 | left | 12.94992 | 1.22198 | 1.97310 | 5.42917 | 0.11407 | 0.32606 |
| M02 | right | 0.61444 | 0.57711 | 2.15937 | 6.07252 | 0.10407 | 0.28926 |
| M03 | left | 12.92104 | 1.08424 | 1.87259 | 5.75032 | 0.10972 | 0.32231 |
| M03 | right | 2.48210 | 3.09144 | 1.91900 | 5.90088 | 0.12233 | 0.35183 |
| M04 | left | 13.03506 | 1.09133 | 1.85411 | 5.40472 | 0.10981 | 0.32599 |
| M04 | right | 2.22797 | 2.82381 | 1.81085 | 5.37231 | 0.12034 | 0.34274 |
| M05 | left | 12.67017 | 1.19067 | 1.98218 | 5.92127 | 0.11299 | 0.33414 |
| M05 | right | 0.63809 | 0.57121 | 2.07618 | 6.36457 | 0.09955 | 0.28705 |
| M06 | left | 12.82081 | 1.15695 | 2.02146 | 5.38457 | 0.11381 | 0.32529 |
| M06 | right | 0.62992 | 0.57294 | 2.20110 | 5.54212 | 0.10171 | 0.27992 |

## Group means: primary

| Controller | Side | Tilt | Swing | Acc | Alpha | Relative speed | Omega |
|---|---|---:|---:|---:|---:|---:|---:|
| learned_mpc | left | 12.89805 | 1.13271 | 1.96051 | 5.40644 | 0.11425 | 0.32713 |
| learned_mpc | right | 0.63078 | 0.58195 | 2.09375 | 5.44453 | 0.10283 | 0.28258 |
| hold_mpc | left | 12.81005 | 1.20633 | 1.97764 | 5.67522 | 0.11353 | 0.33010 |
| hold_mpc | right | 0.62626 | 0.57416 | 2.11777 | 6.21854 | 0.10181 | 0.28816 |
| position_pid | left | 12.97805 | 1.08778 | 1.86335 | 5.57752 | 0.10977 | 0.32415 |
| position_pid | right | 2.35503 | 2.95763 | 1.86492 | 5.63660 | 0.12134 | 0.34728 |

## Group means: walk

| Controller | Side | Tilt | Swing | Acc | Alpha | Relative speed | Omega |
|---|---|---:|---:|---:|---:|---:|---:|
| learned_mpc | left | 12.94016 | 0.78977 | 1.91361 | 5.25056 | 0.10180 | 0.30030 |
| learned_mpc | right | 0.62058 | 0.56499 | 2.02857 | 5.27901 | 0.09331 | 0.25951 |
| hold_mpc | left | 12.83584 | 0.94227 | 1.93908 | 5.29815 | 0.10576 | 0.31296 |
| hold_mpc | right | 0.60319 | 0.54548 | 2.07506 | 5.94634 | 0.09733 | 0.27661 |
| position_pid | left | 12.93731 | 0.91831 | 1.92912 | 5.83271 | 0.10920 | 0.32890 |
| position_pid | right | 2.50692 | 3.00192 | 1.91281 | 5.66793 | 0.11715 | 0.33871 |

## Group means: stop

| Controller | Side | Tilt | Swing | Acc | Alpha | Relative speed | Omega |
|---|---|---:|---:|---:|---:|---:|---:|
| learned_mpc | left | 12.75767 | 1.85189 | 2.10746 | 5.88792 | 0.14824 | 0.40375 |
| learned_mpc | right | 0.66479 | 0.62147 | 2.29745 | 5.93320 | 0.12956 | 0.34858 |
| hold_mpc | left | 12.72408 | 1.80187 | 2.10085 | 6.75841 | 0.13581 | 0.38108 |
| hold_mpc | right | 0.70317 | 0.65257 | 2.25433 | 6.94942 | 0.11492 | 0.32339 |
| position_pid | left | 13.11385 | 1.50384 | 1.62136 | 4.62552 | 0.11161 | 0.30776 |
| position_pid | right | 1.84874 | 2.25712 | 1.69322 | 5.52764 | 0.13430 | 0.37444 |

## Group means: later_walk

| Controller | Side | Tilt | Swing | Acc | Alpha | Relative speed | Omega |
|---|---|---:|---:|---:|---:|---:|---:|
| learned_mpc | left | 13.00889 | 0.63460 | 1.86405 | 4.87816 | 0.09746 | 0.28277 |
| learned_mpc | right | 0.57718 | 0.51781 | 1.90813 | 4.85813 | 0.09162 | 0.25110 |
| hold_mpc | left | 12.93592 | 0.66608 | 1.87201 | 4.82929 | 0.09908 | 0.28557 |
| hold_mpc | right | 0.57568 | 0.50468 | 1.98714 | 5.74584 | 0.09424 | 0.26414 |
| position_pid | left | 12.99561 | 0.70745 | 1.87521 | 5.42907 | 0.10264 | 0.30379 |
| position_pid | right | 1.60404 | 1.76189 | 1.80616 | 5.09502 | 0.11058 | 0.31008 |

## Group means: late_walk

| Controller | Side | Tilt | Swing | Acc | Alpha | Relative speed | Omega |
|---|---|---:|---:|---:|---:|---:|---:|
| learned_mpc | left | 13.08138 | 0.62669 | 1.83169 | 4.92929 | 0.09726 | 0.28370 |
| learned_mpc | right | 0.56758 | 0.50961 | 1.89282 | 4.77290 | 0.09123 | 0.24857 |
| hold_mpc | left | 12.98278 | 0.61089 | 1.89054 | 4.79084 | 0.10144 | 0.29144 |
| hold_mpc | right | 0.58392 | 0.50288 | 1.98315 | 5.78583 | 0.09701 | 0.27226 |
| position_pid | left | 13.06766 | 0.70421 | 1.89471 | 5.55166 | 0.10180 | 0.30062 |
| position_pid | right | 1.31154 | 1.39882 | 1.82293 | 5.21012 | 0.10986 | 0.30648 |

## Timing and excitation

| Run | Work P99 ms | Max ms | Deadline misses / samples | Torso acc RMS m/s² |
|---|---:|---:|---|---:|
| M01 | 5.358 | 7.193 | 13/2154 | 2.0350 |
| M02 | 5.089 | 6.567 | 6/2161 | 2.1017 |
| M03 | 5.036 | 8.394 | 16/2151 | 2.0138 |
| M04 | 3.894 | 6.302 | 5/2161 | 2.0335 |
| M05 | 4.323 | 5.949 | 1/2166 | 2.0884 |
| M06 | 5.520 | 9.153 | 16/2150 | 2.1473 |

## Semantics

- aggregation: Arithmetic mean of per-run metrics, n=2 per controller; no significance claim
- tilt: Angle between endpoint local +Z and frozen H0/world vertical; mean over samples
- swing: RMS angular distance from each axis to normalized mean axis of that run/window
- velocity: Derivative of endpoint-minus-torso IMU position in fixed H0, demeaned per window; not absolute translation
- acceleration: 3D vector norm RMS, gravity removed; angular metrics use full 3D rotation
- hold: Learning prediction ablation; measured disturbances, gravity and moving-base inverse dynamics retained
- measurement: Joint/IMU model reconstruction; not external optical or liquid-surface ground truth
- timing: MPC through timing enqueue; PID through command enqueue; both exclude device actuation delay

Detailed joint/reference/torque margins, recording hashes and 75/135 ms sensitivity: summary.json.
