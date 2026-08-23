# Hardware offline preparation

状态：**offline code/test validated；H1 仍为 PARTIAL；hardware-unverified**。

本文记录机器人不在现场期间可以完成的 H2-prep 与 O0/O1。它不替代第一次真实 G1
只读 session，不修改 `configs/g1_hardware_shadow.yaml` 的 verification flags，也不
授予 `rt/arm_sdk`、`rt/lowcmd` 或任何其他真实输出能力。Latency 实验保持冻结。

## 状态 trace 离线审计（H2-prep）

[`right_arm_runtime/hardware_state_replay.py`](right_arm_runtime/hardware_state_replay.py)
读取 state-only inspection 保存的 `raw_state_trace.jsonl`。它检查：

- 35-slot `q/dq/ddq/tau_est`、温度和 IMU 数组的 shape/有限性；
- sample ID、host monotonic timestamp、uint32 robot tick 的单调性；
- persisted right-arm mapping 是否严格等于 slots 22..26；
- bridge summary 是否声明 output capability absent，CRC-valid/paired 数是否覆盖 trace；
- state age、sample dt、tick delta、quaternion norm、速度/力矩/温度的观察统计；
- trace 和 bridge summary 的 SHA256，保证后续人工审核绑定到准确输入。

审计结果把 `offline_trace_contract_passed` 与 `hardware_session_verified` 分开。即使真实
capture 的结构审计 PASS，后者仍固定为 `false`，并列出型号/固件、motor sign、tick/
mode、torso IMU frame/gravity/lever-arm 等现场 gate。工具不会写 YAML：

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
/home/fjk/miniforge3/bin/conda run -n g1_mpc \
  python tools/realtime/audit_hardware_state_trace.py \
  evaluation/hardware_shadow/state_inspection/SESSION/raw_state_trace.jsonl \
  --source-kind unverified_real_capture \
  --bridge-summary \
  evaluation/hardware_shadow/state_inspection/SESSION/state_bridge_summary.json \
  --output \
  evaluation/hardware_shadow/state_inspection/SESSION/offline_trace_audit.json
```

`synthetic_test_fixture` 只能用于单元测试；其报告不得列为硬件证据。2026-08-23 的
H1 连接尝试没有产生任何 state sample，不能运行该审计，也不能推进 H2 verification。

## Future output 离线合同（O0/O1）

[`right_arm_runtime/hardware_output_contract.py`](right_arm_runtime/hardware_output_contract.py)
定义不含发布能力的以下边界：

1. `ValidatedStateIdentity`：session nonce、source sample/timestamp 和 13-slot 当前 q；
2. `TaskClockEvent`：由未来真实 locomotion producer 明确提供 task epoch、绝对 task
   time、6 ms anchor、planned/runtime command 和 heading；不能从第一条 LowState 猜测；
3. `HardwareControlProposal`：绑定 source state、task epoch/绝对 6 ms anchor、生成/过期
   时间、mode、active mask、13-slot `q/dq/ddq/kp/kd/tau` 和 diagnostics；
4. `CertifiedHardwareCommand`：只表示通过 **offline transport contract**，字段固定为
   `hardware_safety_certified=false`、`hardware_output_authorized=false`；
5. `ExecutionReceipt`：明确 sink、接收原因、DDS write 和 hardware output 是否发生；
6. `FakeHardwareCommandSink`：纯内存 fake sink，拒绝 session/state replay、command
   replay、过期和 watchdog failure，receipt 永远记录 DDS/hardware write 为 false。

离线认证检查 source-state binding、13-slot shape/有限性、expiry、active/inactive slot
语义和两种互斥控制模式。robot-side PD 模式的 `tau` 只能表示 feedforward；direct
torque 模式必须令 robot-side `kp/kd=0`，避免 double PD。inactive slot 必须保持当前
q 且 `dq/ddq/kp/kd/tau=0`。

MuJoCo predicted qacc 可以进入 diagnostics，但不会在这里自动变成真机 hard-stop；
`max_abs_qacc=10 rad/s²` 没有被复制为硬件阈值。真实 hard-stop、soft guard 与
diagnostics 的阈值/持续时间/恢复条件仍需厂商资料、目标机器人确认和吊架试验。

该模块不导入 Unitree SDK、DDS、`unitree_shm` writer 或 publisher，且任何 live
launcher 都不导入它。它是 O0/O1 的接口与故障注入准备，不是 output adapter 已完成。

## 测试与能力隔离

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
PYTHONPYCACHEPREFIX=/tmp/hold-my-beer-mpc-pycache \
  /home/fjk/miniforge3/bin/conda run -n g1_mpc \
  python -m unittest -v \
  right_arm_runtime.tests.test_hardware_state_replay \
  right_arm_runtime.tests.test_hardware_output_contract
```

测试覆盖结构 round-trip、synthetic/real 标签、非有限值、时间/tick/mapping/bridge
不一致、verification flags 保持 false、session/state binding、过期、inactive slots、
double PD、duplicate/restart nonce、watchdog 和 fake receipt。C++ state bridge 的
publisher 隔离、protocol v2、safety release 仍由既有测试覆盖。

## 必须留到现场的 gates

- **H1**：在专用有线 NIC 收到足量 CRC-valid、paired LowState + torso IMU；人工确认
  数据来自目标 `g1_23dof_rev_1_0` Arm5。当前状态保持 PARTIAL。
- **H2 verification**：确认 35-slot index/motor sign、tick wrap、mode whitelist、IMU
  坐标/重力/lever arm 后，才允许人工修改 verification flags。
- **H3**：接入真实 locomotion producer 的 `TaskClockEvent`，才可运行 full-task v2
  read-only shadow；第一条 LowState 不能被猜作 task t=0。
- **O2**：吊架、无 payload、人工急停、低 arm weight 下验证 ownership、全 13 slots、
  release/crash/watchdog 真实行为；只做固定姿态 PD。
- **O3/O4**：完成硬件状态估计、inverse-dynamics/torque contract、执行 receipt 和
  独立授权后，才分别允许短时 MPC 与行走 full-task 主动闭环。

在新的现场授权前，不得构建/启动 output-capable launcher，不得传
`--enable-output`，不得发送任何真实 G1 控制命令。
