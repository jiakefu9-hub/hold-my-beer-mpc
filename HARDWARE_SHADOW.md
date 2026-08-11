# Unitree G1 hardware shadow mode

This is a read-only integration stage. It is intentionally incapable of
publishing a robot command.

## Safety boundary

The shadow path has two independent output barriers:

1. `unitree_arm_state_bridge` subscribes only to `rt/lowstate`. Its source does
   not include a LowCmd type, command topic, or publisher.
2. `run_hardware_shadow.py` opens the POSIX object with a read-only file
   descriptor and private copy-on-write mapping. It has no command sink. The
   built `ShadowArmCommand` always has `arm_weight=0`, zero `tau_ff`,
   `request_output=false`, `publish_performed=false`, and
   `ready_for_output=false`.

The existing output-capable adapter is not used by either shadow command. Do
not run `unitree_arm_adapter_dds --enable-output` during this stage.

```text
rt/lowstate
    -> state-only C++ DDS bridge
    -> protocol-v2 state slot
    -> read-only Python state source
    -> verified units / indices / frames / timestamps
    -> template or hybrid predictor (hybrid can fall back to template)
    -> existing KinematicsHelper + MPC
    -> in-memory ShadowArmCommand
    -> JSON timing/diagnostics only (no sink, no DDS publish)
```

## Implemented hardware contract

The only supported model mapping is the repository's
`g1_23dof_rev_1_0` arm5 model:

- motors 0..11: six left-leg then six right-leg joints;
- motor 12: waist yaw;
- motors 15..19: left arm5;
- motors 22..26: right arm5;
- arm SDK command shape: left 15..19, right 22..26, waist 12..14.

Joint positions are radians, velocities are rad/s, gyroscope values are
rad/s, accelerometer values are m/s^2, and all state freshness checks use the
bridge host's monotonic clock. Every mapped joint is checked against the MJCF
range, and all 35 state vectors are checked for shape, finite values, declared
velocity/IMU bounds, motor temperatures, monotonic sample IDs/timestamps, and
20 ms freshness. The bridge timestamp is host arrival time; the independent
raw robot tick is also required to advance monotonically (with uint32 wrap).
An unexpected `mode_pr` or `mode_machine` is fatal.

The IMU conversion implements exactly the convention declared in
`configs/g1_hardware_shadow.yaml`: W-from-IMU quaternion in wxyz order,
gyro/specific force expressed in IMU, and an explicit torso-from-IMU rigid
rotation. The checked torso pose, acceleration, angular velocity, and causal
angular acceleration then enter the same H-frame predictor path used in
simulation. No convention is inferred from the raw numbers.

The checked-in configuration deliberately leaves the joint-map, robot-tick,
and IMU verification flags false and the allowed mode lists empty. Full shadow
control therefore fails closed until the physical robot contract is confirmed.

## Build only the state bridge

The local checkout currently expects Unitree SDK2 at
`/home/fjk/g1_ws/unitree_sdk2`. Configure the build and request only the
state-only target:

```bash
cd /home/fjk/g1_ws/disturbance-lab

cmake -S cpp/unitree_arm_adapter \
  -B /tmp/hold-my-beer-mpc-unitree-arm-adapter-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUNITREE_ARM_ADAPTER_BUILD_DDS=ON \
  -DUNITREE_SDK2_DIR=/home/fjk/g1_ws/unitree_sdk2

cmake --build /tmp/hold-my-beer-mpc-unitree-arm-adapter-build \
  --parallel --target unitree_arm_state_bridge
```

## First robot session: state inspection only

Choose the wired robot interface explicitly; do not guess it. In terminal 1:

```bash
taskset -c 5 \
  /tmp/hold-my-beer-mpc-unitree-arm-adapter-build/unitree_arm_state_bridge \
  YOUR_INTERFACE \
  --shm-name /g1_arm_mpc_shadow \
  --unlink-on-exit
```

In terminal 2:

```bash
cd /home/fjk/g1_ws/disturbance-lab
MPLCONFIGDIR=/tmp/disturbance-lab-matplotlib \
  /home/fjk/miniforge3/envs/g1_mpc/bin/python run_hardware_shadow.py \
  --inspect-state-only \
  --shared-memory /g1_arm_mpc_shadow \
  --inspect-samples 500 \
  --duration-s 10
```

This inspection mode does not require the verification flags. It reports
raw modes, quaternion norms, state age, IMU data, and right-arm q/dq. It does
not run MPC and cannot write the command slot. Stop the bridge with Ctrl-C;
`--unlink-on-exit` removes only its temporary shared-memory name.

Before changing the verification flags, confirm from the exact robot/firmware
documentation or a Unitree-supported interface:

- the robot is the 23-DOF arm5 variant and motor indices match;
- the target firmware's `tick` advances monotonically with uint32 wrap;
- LowState quaternion ordering and W/body direction;
- gyro frame and units;
- whether accelerometer is specific force or gravity-removed acceleration;
- the fixed IMU-to-model-torso rotation;
- the physical IMU origin matches the MJCF `imu_in_torso` site (otherwise a
  measured translation and lever-arm acceleration correction are required);
- the allowed `mode_pr` and `mode_machine` values in the intended read-only
  locomotion state.

Observed plausible values alone are not proof of the frame convention.

## Complete target-runtime shadow run

After those fields are verified in the YAML, the PREEMPT_RT environment is
checked and the complete state-to-command-build path can be run with one
command:

```bash
cd /home/fjk/g1_ws/disturbance-lab
./tools/realtime/run_hardware_shadow.sh YOUR_INTERFACE \
  --control-cpu 7 \
  --bridge-cpu 5 \
  --predictor template \
  --duration-s 30 \
  --group first_g1_readonly_shadow
```

The control process uses the existing target runtime gate, CPU 7 affinity,
SCHED_RR/10, and single-threaded numerical libraries. DDS receive threads stay
on housekeeping CPU 5. The summary includes complete path mean/p95/p99/max,
per-stage timing, state age, QP success, predictor diagnostics, command build
count, source-to-command age, and a command publish count that must remain
zero.

## Inputs still missing from LowState

LowState does not provide the lower-body policy target, runtime walking
command, or gait phase used to train the MLP. `LocomotionContext` defines the
required 12 + 3 + 2 values and applies its own monotonic timestamp/freshness
checks, but this repository does not yet know the real lower-body controller's
transport or schema. The current runner supplies no such context, so selecting
`hybrid_residual` explicitly falls back to template and records the reason.

The phase template also needs a verified gait epoch/phase relationship for a
meaningful walking comparison. Until that signal is connected, its internal
clock is only suitable for exercising the shadow computation path, not for
claiming phase-aligned hardware prediction quality.

Finally, LowState alone lacks a validated floating-base pose/twist/contact
estimator for hardware inverse dynamics. Shadow mode therefore does not invent
a feedforward torque: `tau_ff` stays zero and the command remains explicitly
not ready for output.
