# Acceleration forecast timing: offline label check

Vector RMSE against the same filtered IMU interval [host Write end, +6 ms).
This table does not measure endpoint control improvement.

| Run | Acc baseline / aligned (m/s²) | Alpha baseline / aligned (rad/s²) |
| --- | ---: | ---: |
| mpc_learned_acc001_alpha0005_omega1_walk_20261008_164546 | 1.1933 / 0.8823 | 3.5032 / 2.2943 |
| mpc_zero_neutral_164546_walk_20261009_155844 | 1.0824 / 0.8006 | 3.2020 / 2.1509 |
| mpc_zero_leftpd100_nopulse_right164546_repeat_walk_20261009_174504 | 1.1496 / 0.8346 | 3.5333 / 2.4502 |
| mpc_zero_leftpd100_nopulse_right164546_repeat2_walk_20261009_180354 | 1.2180 / 0.8762 | 3.5655 / 2.3059 |
