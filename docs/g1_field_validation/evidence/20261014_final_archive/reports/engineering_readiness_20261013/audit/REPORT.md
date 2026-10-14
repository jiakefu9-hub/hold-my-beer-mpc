# Field readiness audit, 2026-10-13

Offline only. Same 12 valid T-series runs and exclusion rules as final comparison.
T05 remains included with operator-observed torso contact; no collision-free claim.
Endpoint window [5,18); group mean of per-run metrics. No independent payload sensor.

## Filter window sensitivity

| SG ms | Group | N | Left acc | Right acc | Right alpha | Right relative speed | Right omega |
|---|---|---:|---:|---:|---:|---:|---:|
|75|mpc_off|3|2.0178|2.2660|6.8606|0.1124|0.3077|
|75|mpc_on|2|2.1543|2.4761|6.9290|0.1154|0.3106|
|75|c_on|3|2.0195|2.3373|6.3778|0.0972|0.2677|
|105|mpc_off|3|1.8149|1.8982|5.2474|0.1107|0.3012|
|105|mpc_on|2|1.9369|2.0560|5.1924|0.1134|0.3038|
|105|c_on|3|1.8122|1.9256|4.5703|0.0948|0.2606|
|135|mpc_off|3|1.6939|1.7287|4.3844|0.1094|0.2958|
|135|mpc_on|2|1.8006|1.8455|4.2818|0.1119|0.2984|
|135|c_on|3|1.6884|1.7465|3.7342|0.0935|0.2554|

## Execution and deadlines

| Run | Complete P99 ms | Deadlines missed | Samples | LowState age at write P99 ms | IMU age P99 ms |
|---|---:|---:|---:|---:|---:|
|T01_162329|5.003|2|2165|7.031|6.752|
|T02_163351|5.179|7|2160|7.138|6.833|
|T03_164534|4.628|0|2166|6.234|6.411|
|T04_165151|5.130|2|2165|6.644|6.739|
|T05_165900|4.596|2|2165|6.431|6.397|
|T07_172138|5.430|10|2157|7.539|7.293|
|T08_172328|4.635|1|2165|6.251|6.384|
|T10_173715|5.481|14|2153|7.373|7.017|
|T11_174034|4.633|0|2167|6.410|6.226|
|T12_174355|4.669|2|2165|6.199|6.390|
|T01_174738|5.509|16|2151|7.478|7.197|
|T03_175013|4.725|1|2166|6.413|6.523|

## Matched-band joint tracking (105 ms nominal window)

Steady walking [6.2,14.2). Both modeled integrated acceleration and received dq use the same SG derivative.
No fitted delay/gain. Physical disturbance, filtering and model error are not separated.

| Run | Joint | Model ddq RMS | Observed ddq RMS | Error RMS | Zero-lag correlation |
|---|---|---:|---:|---:|---:|
|T01_162329|shoulder_pitch|2.934|3.696|4.798|-0.032|
|T01_162329|shoulder_roll|4.182|4.717|5.742|0.176|
|T01_162329|shoulder_yaw|3.295|5.092|5.609|0.163|
|T01_162329|elbow_pitch|6.473|2.667|6.955|0.051|
|T01_162329|wrist_roll|4.771|0.727|4.871|-0.069|
|T03_164534|shoulder_pitch|5.799|5.438|7.432|0.150|
|T03_164534|shoulder_roll|5.483|5.907|8.015|0.012|
|T03_164534|shoulder_yaw|6.860|6.550|8.955|0.114|
|T03_164534|elbow_pitch|7.140|3.218|7.550|0.120|
|T03_164534|wrist_roll|7.436|1.102|7.599|-0.078|
|T05_165900|shoulder_pitch|5.280|4.157|6.139|0.185|
|T05_165900|shoulder_roll|5.780|4.168|7.107|0.011|
|T05_165900|shoulder_yaw|7.115|5.171|8.054|0.174|
|T05_165900|elbow_pitch|7.675|2.688|7.875|0.121|
|T05_165900|wrist_roll|7.458|0.806|7.548|-0.061|
|T07_172138|shoulder_pitch|3.055|3.488|4.709|-0.042|
|T07_172138|shoulder_roll|4.032|4.387|5.280|0.220|
|T07_172138|shoulder_yaw|3.644|4.814|5.562|0.192|
|T07_172138|elbow_pitch|6.651|2.521|7.010|0.148|
|T07_172138|wrist_roll|5.003|0.757|5.055|0.009|
|T01_174738|shoulder_pitch|3.716|3.481|5.106|-0.020|
|T01_174738|shoulder_roll|4.167|5.059|5.994|0.171|
|T01_174738|shoulder_yaw|3.836|4.818|5.739|0.166|
|T01_174738|elbow_pitch|5.361|2.303|5.791|0.060|
|T01_174738|wrist_roll|5.440|0.659|5.504|-0.043|
|T03_175013|shoulder_pitch|4.020|4.635|5.613|0.180|
|T03_175013|shoulder_roll|5.125|5.236|7.456|-0.035|
|T03_175013|shoulder_yaw|6.976|6.387|8.832|0.129|
|T03_175013|elbow_pitch|7.750|2.814|8.144|0.117|
|T03_175013|wrist_roll|7.421|0.895|7.502|-0.031|

## Forecast labels at host write time

| Run | Acc RMSE original / aligned | Alpha RMSE original / aligned |
|---|---:|---:|
|T01_162329|1.2049 / 0.8275|3.5463 / 2.3263|
|T07_172138|1.2367 / 0.9484|3.6129 / 2.6150|
|T01_174738|1.1778 / 0.8887|3.4984 / 2.4120|

## Limitations

- Offline inputs and host-receive times are not independent physical ground truth.
- Changing an offline derivative window does not alter the robot or the online 15 Hz filter.
- Model-executed acceleration is an estimate, not measured acceleration.
- LowState dq differentiated offline is a bandwidth-limited motion estimate; ddq_raw is not trusted.
- Tracking comparison uses no fitted time shift; it does not identify motor or transport latency.
- Model force limits do not certify physical torque or collision safety.
- Predictor input replay is not closed-loop performance or hardware real-time validation.
- Small historical nonrandomized sample; repeat-run variability and contact remain relevant.
