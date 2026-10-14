# M01 impact predictability: read-only log audit

Scope: four M01 learned-MPC runs, 2026-10-13 15:04:26 and 16:48:38, and 2026-10-14 12:43:21 and 12:53:33. Command window is `[5,15)` s. No controller edits, model training, DDS/network access, or physical outputs. `analyze.py` reproduces `summary.json`.

## What the current predictor actually predicts

`hardware_mpc_predictor.py:30-32,213-222,242-250,270-289` uses a causal 2 ms grid, a first-order 15 Hz low-pass, and nine future 6 ms intervals. Linear acceleration targets are means of future **filtered** values at interval offsets 0, 2, and 4 ms. They are not raw foot-impact peaks. Angular acceleration additionally differentiates filtered gyro and low-pass-filters the derivative. The 33 input features are filtered leg q/dq and body acceleration/omega/alpha. No explicit foot-contact label or force sensor is consumed.

The captured G1 HG `LowState` and the SDK's `_LowState_.py` do not contain a named foot force/contact field. Logs contain torso IMU and joint states. Body acceleration peaks are therefore only **impact proxies**; this audit does not certify exact foot contact or left/right foot labels.

## Quantitative findings

Raw acceleration means gravity-removed, H0-rotated torso IMU acceleration; it is not an endpoint measurement. We detect local magnitude maxima with prominence >=1 m/s², height above each run's 90th percentile, and spacing >=250 ms. Each run produces 20 candidate events. Filtered peak means the largest norm within [-20,+60] ms of each raw event.

| Run | Raw peak median (m/s²) | Filtered neighboring peak median (m/s²) | Filtered peak lag median | Query-to-Write median |
| --- | ---: | ---: | ---: | ---: |
| 10/13 150426 | 15.975 | 8.148 | 4 ms | 3.639 ms |
| 10/13 164838 | 16.983 | 8.766 | 5 ms | 3.693 ms |
| 10/14 124321 | 16.692 | 8.895 | 4 ms | 3.526 ms |
| 10/14 125333 | 16.862 | 8.801 | 4 ms | 3.516 ms |

The raw torso acceleration has about 71–74% of its measured vector spectral power above 15 Hz in these runs. This includes measurement noise and structural vibration; it is not a direct measure of physically controllable arm disturbance energy.

To inspect timing, take the latest saved forecast already written at least 12 ms before each raw event, find its maximum predicted norm within the 54 ms horizon, and compare the center of that predicted interval with the event. This is an event-conditioned diagnostic, **not** a tested online classifier or a false-positive/false-negative evaluation.

| Run | Forecast norm for event interval, median | Predicted peak timing error median | Absolute timing error P90 | Peak within ±6 ms |
| --- | ---: | ---: | ---: | ---: |
| 10/13 150426 | 4.018 m/s² | +6 ms | 18.0 ms | 47% |
| 10/13 164838 | 4.558 m/s² | +8 ms | 19.2 ms | 32% |
| 10/14 124321 | 5.036 m/s² | +4 ms | 16.0 ms | 60% |
| 10/14 125333 | 4.861 m/s² | +6 ms | 12.6 ms | 60% |

Forecast amplitude should not equal raw peak amplitude: their target definitions differ. Nonetheless, simply interpreting the largest existing predicted value as a precisely timed raw impact is not justified. Even this retrospective diagnostic has a 12.6–19.2 ms P90 timing error, before unknown motor application delay.

The learning predictor still improves prediction of its intended filtered target. For `[18,24)` ms after the predictor anchor, learned/hold vector acceleration RMSE is 1.072/2.340, 1.208/2.387, 0.953/2.400, and 0.965/2.403 m/s² respectively. This is approximately 49–60% lower prediction error; it does not establish a corresponding endpoint reduction.

Reconstructed causal current acceleration matches logged `current_y` to floating-point precision in the first three runs. Last-run per-axis RMSE is [0.00977,0.00158,0.00465] m/s², consistent with occasional differences in callback availability when replaying by receive timestamp. Thus the comparison has a measured reconstruction bound.

## Implication for briefly switching to fixed-angle PD

The data support exploring an impact-specific controller, but not an immediate production switch based solely on these predicted filtered maxima. The existing predictor contains useful gait information, but is not a validated raw-impact event predictor. A 6–12 ms switching pulse would often be mistimed; widening the interval changes the controller for a material fraction of each step.

Priority should be diagnosing whether post-impact endpoint oscillation correlates with right-arm torque/reference changes. If a later controlled test is justified, first evaluate a default-off short hold/damping mode with continuous torque at entry/exit, hysteresis and a fixed maximum duration, then validate triggers offline on held-out episodes. Freezing a right-arm target at current q and abruptly clearing feedforward can itself introduce a torque step; it is not equivalent to leaving the existing controller unchanged for one frame. A stiffness increase is not automatically shock isolation.

Theoretical low-frequency group delay of the first-order filter is 9.64 ms; measured sharp-peak lag is 4–5 ms here. Neither is a universal sensor delay. Host receive stamps do not reveal sensor acquisition or motor application timestamps. `--align-accel-forecast` shifts filtered forecast sampling to a host-write estimate and does not solve those missing timestamps or restore high-frequency impact information.
