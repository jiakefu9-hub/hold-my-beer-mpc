# Hardware offline preparation

状态：**offline code/test validated；H1 仍为 PARTIAL；hardware-unverified**。

本文记录机器人不在现场期间完成的 H2-prep 与 Stage-2 O0/O1。它不替代第一次真实 G1
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

## Future output 离线合同（O0）

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

该模块不导入 Unitree SDK、DDS、`unitree_shm` writer 或 publisher。它是平台无关
O0 合同和故障注入层，不是 output adapter 已完成。

## Protocol-v3 与 publisher-absent C++ HIL（O1）

[`right_arm_runtime/unitree_shm.py`](right_arm_runtime/unitree_shm.py) 和
[`cpp/unitree_arm_adapter/include/unitree_arm_adapter/protocol.hpp`](cpp/unitree_arm_adapter/include/unitree_arm_adapter/protocol.hpp)
共用一个锁定的 protocol-v3 ABI：

| 项目 | 大小 / 偏移 |
| --- | ---: |
| 整个 shared memory | 3328 B |
| command payload / slot offset | 768 B / 64 |
| paired-state payload / slot offset | 1440 B / 896 |
| receipt payload / slot offset | 928 B / 2368 |

command 完整绑定 producer/command ID、source sample/timestamp、session nonce、task
epoch/6 ms anchor、expiry、active mask 和 safety-policy ID/SHA256。state bridge 启动时
要求非零 ingress session nonce，并在每个 CRC-valid LowState + torso IMU 配对样本中
写入该 nonce、两源时间、skew 和 validation flags。HIL 只核对这份 ingress
证据，不能用 CLI 为 state 伪造 session。Python state-only inspector 和 read-only
shadow 同样必须从 launcher 接收 expected nonce，核对三项 required flags、两源时间
关系与不超过 5 ms 的 skew，并把这些字段原样写入 evidence；不能用本地读取时刻
重造 bridge 的 validation timestamp。

Python 唯一的正式全绑定写入口是
`write_certified_hil_command(CertifiedHilCommandEnvelope)`。它只接受 O0 生成的
`CertifiedHardwareCommand`，验证其 offline-only scope 和两个 authorization 字段为
false，然后仍将 `REQUEST_OUTPUT` 保持为 0。普通 `write_*` API 传
`request_output=True` 会在 Python 端拒绝。

C++ [`unitree_arm_adapter_hil`](cpp/unitree_arm_adapter/src/hil_main.cpp) 固定每 2 ms 执行一次最后边界；
非 2000 us 的 `--period-us` 会在打开 shared memory 前拒绝：

1. 用 command seqlock sequence 区分新 proposal、未变 slot 的 hold 和重写/replay；
2. 从有界 cache 中精确匹配 `(source_sample_id, source_timestamp_ns)`，同时
   把最新 2 ms state 作为 actuation state；
3. 复核 finite/session/source/task/policy/expiry/mode/mask/double-PD/13-slot limits、
   ownership、deadline 和 supervisor 状态转换；
4. 一个新 6 ms proposal 只能在相对 `0/2/4 ms` 三拍保持，之后必须收到
   下一 anchor；
5. recording command sink 在 would-write 调用点用自己的时钟再检查 deadline/
   expiry，只有真实通过才记录 `sink_write_performed=true`；
6. receipt 完整回显 identity、observed state、guard/reason、请求/实际 mask 与
   weight、selected 13-slot command 和 fake-sink 结果。被拒绝拍仍保留 receipt，
   但 command sink 结果为 false。

supervisor 状态为 `disarmed -> arming-PD -> active -> soft-guard-releasing`，并有
`latched-fault`。production policy 中 site limits、ownership、startup-PD、active control、
release behavior 和 output authorization 全部默认未验证，因此不可 arming。
只有显式 `offline_fixture_policy` 能为测试放行 would-write，这些宽限值不得
记为真机 policy。

HIL 只链接本地 core 和 recording sink，不链接 Unitree SDK、`LowCmd`、
`ChannelPublisher` 或 `rt/arm_sdk`。`dds_main.cpp` 和 command publisher target 已移除；
`UNITREE_ARM_ADAPTER_BUILD_DDS=ON` 会在 CMake 配置阶段 fail closed。因此
`would_write_command_sink_count` 只是本地 formatter/supervisor 验收证据，
`device_command_transport_present=false`、`hardware_output_performed=false`。

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
double PD、duplicate/restart nonce、watchdog 和 fake receipt。C++ 测试另覆盖
protocol-v3 ABI parity、supervisor 状态机、精确 source-state cache、`0/2/4 ms`
hold、sink-time deadline/expiry、receipt/sink 完整一致和 HIL binary 输出能力隔离。

```bash
cpp/unitree_arm_adapter/build_and_test.sh
/home/fjk/miniforge3/bin/conda run -n g1_mpc \
  python -m unittest -v right_arm_runtime.tests.test_unitree_shm
```

## 必须留到现场的 gates

- **H1**：在专用有线 NIC 收到足量 CRC-valid、paired LowState + torso IMU；人工确认
  数据来自目标 `g1_23dof_rev_1_0` Arm5。当前状态保持 PARTIAL。
- **H2 verification**：确认 35-slot index/motor sign、tick wrap、mode whitelist、IMU
  坐标/重力/lever arm 后，才允许人工修改 verification flags。
- **H3**：接入真实 locomotion producer 的 `TaskClockEvent`，才可运行 full-task v2
  read-only shadow；第一条 LowState 不能被猜作 task t=0。
- **O2**：先以现场证据冻结 production safety policy，再在吊架、无 payload、
  人工急停、低 arm weight 下验证 ownership、全 13 slots、
  release/crash/watchdog 真实行为；只做固定姿态 PD。
- **O3/O4**：完成硬件状态估计、inverse-dynamics/torque contract、执行 receipt 和
  独立授权后，才分别允许短时 MPC 与行走 full-task 主动闭环。

在新的现场授权前，不得恢复或新增 output-capable target/launcher，
不得发送任何真实 G1 控制命令。
