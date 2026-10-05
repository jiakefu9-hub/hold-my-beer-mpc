# Nominal arm delay propagation

SDK-free native equivalent of the bounded 2 ms loop in
`tools/g1_commissioning/hardware_mpc_delay_preview.py`. This library has no
network, DDS, mode switching or command publisher. It predicts nominal arm
state from measured q/dq, torso forecasts and previously issued commands;
it does not identify real motor latency or torque gain.

Build against the **same MuJoCo installation** as the `g1_mpc` Python process:

```bash
cmake -S cpp/g1_arm_delay -B build/g1_arm_delay -DCMAKE_BUILD_TYPE=Release \
  -DMUJOCO_ROOT=/home/fjk/miniforge3/envs/g1_mpc/lib/python3.10/site-packages/mujoco
cmake --build build/g1_arm_delay -j2
```

`native_arm_delay.py` checks the ABI, compiled source SHA256 and MuJoCo
header/runtime versions, and records the library SHA256. Rebuild after changing
`delay.cpp`; a stale binary is rejected, not silently used. Generated libraries
stay under ignored `build/`; they are not committed.

`test_native_arm_delay.py` compares 80 deterministic randomized moving-base,
partial-weight and command-switch cases against the original Python/MuJoCo
loop, including startup before any command. Position tolerance is `2e-12 rad`,
velocity tolerance `2e-11 rad/s`. Numerical parity is not physical validation.
