# G1 Arm SDK commissioning tools

This directory contains isolated A1a/A1b/A2/A3 and raw walking-capture field tools, not the production hardware
adapter and not an extension of the publisher-absent HIL. The operator-facing
entry and latest evidence index are in
[`docs/g1_field_validation/README.md`](../../docs/g1_field_validation/README.md);
the detailed staged procedure is in [RUNBOOK.md](../../docs/g1_field_validation/RUNBOOK.md).

## Capability separation

| Target | Default build | Network | Command capability |
| --- | --- | --- | --- |
| `g1_arm_static_preview` | yes | none; no SDK linked | records offline would-write frames only |
| `g1_commissioning_query` | no | LowState + four getter RPCs | no LowCmd type or command publisher |
| `g1_imu_zero_observer` | no | torso IMU + getter-only FSM | continuously logs quaternion/RPY/FSM; no publisher or setter |
| `g1_commissioning_mode_step` | no | LowState + FSM get/set RPC | one request: 0 to 1, or 1 to 4; no joint publisher |
| `g1_arm_static_execute` | no | LowState + one publisher | only `rt/arm_sdk`; explicit A2 gates |
| `g1_arm_balance_hold_execute` | no | LowState + one publisher | only `rt/arm_sdk`; explicit grounded A3/FSM 500 gates |
| `g1_arm_stop_observe` | no | LowState + FSM getter | three-second stop-monitor observation; no joint publisher |
| `g1_phase_probe` | no | torso IMU + LowState + FSM/phase getters | 30-second raw observer; no motion/mode output |
| `g1_walk_capture` | no | raw subscribers + getters + arm publisher + velocity RPC | separate opt-in 19-second arm/forward-walk capture; no mode setter |

All networked targets are opt-in at CMake configure time. Merely running the
A2 executable without its complete arguments exits before DDS initialization.
Even with valid arguments, each output executable validates its matching field-reviewed profile and consecutive
fresh states, requires an interactive `EXECUTE <robot_id>` response, revalidates
new states, and only then constructs the publisher.

The query uses small `Client` subclasses which register only getter IDs. It does
not use the official `LocoClient::Init`, because that method registers setter IDs
as well. RPC discovery/request traffic is still network output; “read-only” here
means it requests data and never calls a motion/mode mutator.

`g1_imu_zero_observer` is built with the same device-query option. It subscribes
only to `rt/secondary_imu`, registers only the FSM getter, prints quaternion/RPY
at 5 Hz and writes a new JSONL log. Its field procedure is
[`IMU_ZERO_REFERENCE_TEST.md`](../../docs/g1_field_validation/IMU_ZERO_REFERENCE_TEST.md).

The new [raw phase/walking capture guide](../../docs/g1_field_validation/RAW_WALK_CAPTURE.md)
covers the two new tools, build flags, field commands and log schema. The phase
observer uses `G1_COMMISSIONING_BUILD_DEVICE_QUERY`; the walking collector requires
the separate `G1_COMMISSIONING_BUILD_WALK_CAPTURE` option (default OFF). Its fixed
heading target is **IMU navigation-world +X, yaw=0**, never the starting heading.
It retains raw data throughout the 19-second session; no disturbance template,
world-frame data conversion, angular-acceleration derivation or MPC runs online.
The velocity command and heading correction are active only during seconds 5–13.
Both tools are offline-tested only; no hardware execution has been performed.

The separate A1b mode tool is **not read-only**: a mode RPC changes motor behavior.
It registers only GetFsmId/SetFsmId, accepts only `damp` (1) or `locked-stand` (4),
requires the explicit `A1B_HOISTED_MODE_ONLY` permit, and verifies the source mode
before sending at most one setter. Already being in the requested mode is a no-op.
After a successful setter it observes state for six seconds and checks FSM readback.
CRC, tick, freshness, raw mode and finite-value checks are reused/checked; failure
stops further requests without retries or automatic mode rollback. It is not an
emergency stop and does not independently certify the physical pose or support.
The A1a query source and A2 profile/gates are unchanged by this addition.

## Build and offline test

Default offline build (no device target is created):

```bash
cmake -S tools/g1_commissioning -B /tmp/g1-commissioning-default \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build /tmp/g1-commissioning-default --parallel 4
ctest --test-dir /tmp/g1-commissioning-default --output-on-failure
```

Build the A1b mode tool without enabling the A2 joint publisher:

```bash
cmake -S tools/g1_commissioning -B /tmp/g1-commissioning-modes \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DG1_COMMISSIONING_BUILD_DEVICE_QUERY=ON \
  -DG1_COMMISSIONING_BUILD_MODE_STEP=ON \
  -DG1_COMMISSIONING_BUILD_REAL_OUTPUT=OFF \
  -DUNITREE_SDK2_DIR=/home/fjk/g1_ws/unitree_sdk2
cmake --build /tmp/g1-commissioning-modes --parallel 4
ctest --test-dir /tmp/g1-commissioning-modes --output-on-failure
```

Network execution requires the explicitly authorized hoisted A1b procedure in the
field guide; do not run a valid mode command as an offline build test.

Compile-check the query and A2/A3 opt-in device targets without running them
(the walking collector has its own build instructions in the guide above):

```bash
cmake -S tools/g1_commissioning -B /tmp/g1-commissioning-full \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DG1_COMMISSIONING_BUILD_DEVICE_QUERY=ON \
  -DG1_COMMISSIONING_BUILD_REAL_OUTPUT=ON \
  -DUNITREE_SDK2_DIR=/home/fjk/g1_ws/unitree_sdk2
cmake --build /tmp/g1-commissioning-full --parallel 4
ctest --test-dir /tmp/g1-commissioning-full --output-on-failure
```

Offline preview with test-only inputs:

```bash
/tmp/g1-commissioning-default/g1_arm_static_preview \
  --profile tools/g1_commissioning/tests/data/synthetic_profile.conf \
  --state tools/g1_commissioning/tests/data/synthetic_state.conf \
  --output /tmp/g1-arm-static-synthetic-preview.jsonl
```

The synthetic profile is permanently rejected by the real-output gate. The
field template also fails that gate until every identity, behavior and numeric
review item is explicitly filled and confirmed. Software hard caps (5 degrees,
0.1 rad/s, weight 0.5, weight rate 0.2/s, 30 seconds) are additional engineering
ceilings, not manufacturer limits or recommendations.

The separate [`a3_balance_hold.template`](profiles/a3_balance_hold.template) is
for the grounded balance test. The current static world-upright pose targets
shoulder pitch -4 degrees on both arms, shoulder roll left -1/right +1 degree,
and elbow pitch left -8.1/right -7.8 degrees; other arm axes and the single waist
yaw remain zero. During its 3-second entry, each target
interpolates from the freshly measured pose while global weight ramps
from 0 to 1; it holds for 20 seconds, then keeps the target while weight
returns to 0 over 3 seconds. The A3 executor does not switch modes or command
locomotion. It requires continuously observed `GetFsmId()==500`, rejects the A2
schema/permit, and retains the state, remote L2+B, FSM and manual
`EXECUTE <robot_id>` gates. The template is `DRAFT`; a fresh FSM-500 query is
still required before field use.

## Runtime behavior that needs field review

### A2 stop interlock (2026-09-15)

The executor now independently monitors actual `GetFsmId()==4`; it does not treat
`LowState.mode_machine==4` as locked standing. On this robot that raw byte was 4
in both damping and locked standing. A getter-only worker uses a 100 ms RPC
timeout, a 50 ms wait between calls, and a 200 ms maximum sample age measured
from request start. These are software refusal limits, **not guaranteed physical
stop latencies**. Failures, malformed/late responses, observation gaps, non-4 FSM,
or a stop key latch the interlock for the lifetime of the process.

CRC-valid LowState remote bytes 2/3 are decoded using the pinned SDK's
`example/g1/low_level/gamepad.hpp`: L2 is bit 5, B is bit 9. A short L2+B sample
remains latched after key release. This byte interpretation has offline coverage;
the earlier field button test established a mode transition, not this decoder's
live reaction time.

The gate is checked during startup, again before publisher creation, during
normal and SIGINT-release loops, and immediately before their writes. On an
interlock trip the process stops normal output and exits with code 4, without
sending a final frame or changing robot mode. It cannot retract an already
in-flight command. No automatic reset/reconnect/restart is provided. CRC/state/
tracking/deadline failures retain their existing best-effort fault-frame path.
Neither path certifies robot behavior after network loss or process failure.

`g1_arm_stop_observe NIC NEW_JSONL A2_READ_ONLY_STOP_OBSERVE` uses the same state
callback and FSM monitor but has no joint-command capability. It is built only
with the opt-in real-output build, and running it still requires authorization
for live read-only access. Tests never run it against a robot automatically.

The stop-interlock change did not modify the trajectory, gains, or state gates.
The subsequent user-authorized pretest review introduces profile v2: ambiguous
loss/release success confirmations are replaced by explicit **procedure reviews**,
not by unconditional approval. See [pretest review](profiles/PRETEST_REVIEW.md).
A working stop interlock does not turn a DRAFT template into a reviewed profile.

### Existing trajectory and failure behavior

- All valid Arm5 slots start at the same fresh LowState pose. The selected right
  arm slot follows a finite out/hold/back trajectory; every other valid slot
  holds that captured pose. Invalid waist roll/pitch slots remain all-zero.
- The global weight ramps from zero and returns to zero. An `active_mask` is not
  used to claim independent hardware ownership.
- Normal completion returns to the captured start pose, then ramps weight down.
  SIGINT is a separate graceful release: with fresh valid state it holds the
  current pose while reducing weight; it does not continue an old return path.
- CRC failure, stale/future state, tick rollback, raw-mode mismatch, limit,
  tracking, deadline, or write failure enters a distinct fault path. It attempts
  one all-zero, weight-zero `rt/arm_sdk` frame and exits. This attempt is not a
  proven hardware stop, and the log says whether the DDS write succeeded.
- A process crash cannot run cleanup code. Target firmware crash/cable-loss
  behavior remains unknown unless separately evidenced. Before the first short
  hoisted trial, review an independent operator response assuming prior output
  may persist, and prohibit automatic resume. A hardware watchdog/fault-injection
  campaign is not a prerequisite for that trial and is not claimed as passed.
- Profile v2 requires `loss_response_plan_reviewed` and
  `normal_release_plan_reviewed`, as well as the existing independent emergency,
  identity/mapping/parameter and no-competing-publisher requirements. Actual smooth
  hand-back is an A2 outcome, not required historical proof before the first A2.
  Old v1 profiles/flags are rejected rather than silently reinterpreted.

### A3 balance hold (implemented 2026-09-15; first hardware run 2026-09-16)

- `g1_arm_balance_hold_execute` is built only when real-output targets are
  explicitly enabled. The first FSM-500 hardware run completed its 3/5/3-second
  plan and normal weight-zero release; see the linked field session record below.
- The profile and executable are stage-bound: A3 accepts only
  `g1_arm_balance_hold_site_v1` plus permit
  `A3_GROUNDED_BALANCE_HOLD_ONLY`; A2 accepts only its A2 schema and permit.
- A3 requires the operator to establish grounded, stationary self-balance first.
  It only monitors FSM 500 and publishes the arm message; it never calls
  `Start()`, `SetFsmId`, `ReleaseMode`, `rt/lowcmd`, or a walking API.
- A3's weight-1 envelope is separate from A2's weight-0.5 hard cap. The current
  photo profile uses 3/20/3 seconds; the A3-only hold bound is 20 seconds, and a
  3/20/3-second field repeat completed on 2026-09-16. Initial gains remain the
  field-proven conservative `kp=20, kd=1`, not the larger simulation gains.
- A3 deliberately does not abort on measured joint velocity, tracking error,
  exact LowState `mode_pr/mode_machine`, a shared ±1.2 rad measured-angle
  envelope, or a single control-thread deadline miss. Those values remain in
  command/state logs where applicable; delayed A3 cycles resume from the current
  time without catch-up write bursts. A2 keeps its existing gates unchanged.
- A3 still requires continuous FSM 500 monitoring, valid finite CRC-checked
  state no more than 100 ms old, no tick rollback, no L2+B request, successful
  DDS writes, and a finite profile-bounded plan (11 seconds for the default
  profile, 26 seconds for the validated 20-second hold). These are the minimal
  runtime refusal conditions, not hardware safety certification.
- Hardware evidence and its physical-observation boundary are recorded in
  [`20260916_A3_HARDWARE.md`](../../docs/g1_field_validation/sessions/20260916_A3_HARDWARE.md).

### Bottle-center endpoint pose

[`endpoint_pose.py`](endpoint_pose.py) performs offline forward kinematics using
the XML selected by `configs/g1.yaml`, `imu_in_torso` as base, and the actual
`left_grasp_site` / `right_grasp_site` bottle-center sites. Run it with the existing
`g1_mpc` Python environment, an execute JSONL, and a concurrent torso observer log:

```bash
python tools/g1_commissioning/endpoint_pose.py \
  --execute SESSION/execute.jsonl --imu SESSION/torso_imu.jsonl \
  --output SESSION/endpoint_pose.jsonl
```

It reports measured-joint FK position relative to the torso IMU, orientation in
that base frame, and orientation in the IMU navigation frame. The body-frame
bottle-Z tilt and world-vertical bottle-Z tilt are separate metrics: current
pose tuning minimizes the former. Absolute world
translation is unavailable. IMU samples are matched by host receive time within
125 ms (the observer logs at about 5 Hz); missing matches produce null world
poses. This is static-pose evidence, not synchronized acceleration estimation.
XML and input hashes are stored. Tests:
`python tools/g1_commissioning/tests/test_endpoint_pose.py`.

The current target envelope allows shoulder pitch [-4,0] degrees, elbow pitch
[-10,0] degrees, left shoulder roll [-1,0] degree, and right shoulder roll
[0,+1] degree, with all other targets zero. Legacy zero profiles remain valid.
See [endpoint frame definition](../../docs/g1_field_validation/ENDPOINT_FRAME.md).

Do not run any networked target during offline preparation. Actual A1a/A1b/A2/A3
commands are intentionally kept in the field guide, next to their human gates.
