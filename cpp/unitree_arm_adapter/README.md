# G1 右臂 C++ hardware adapter 边界

状态：**publisher-absent offline HIL 已实现；H1 仍为 PARTIAL；hardware output
未授权且当前无 command publisher target**。

这个目录把 Unitree 的 2 ms 状态/未来命令边界与 Python 控制核心隔开。
正式 MuJoCo 不经过这里，而是使用独立的
[`cpp/right_arm_sim_runtime`](../right_arm_sim_runtime/)。

## 当前只有三个可执行边界

| executable | 用途 | 真机输出能力 |
| --- | --- | --- |
| `unitree_arm_state_bridge` | 订阅 LowState + torso IMU，CRC/配对后写 state slot | 无 publisher |
| `unitree_arm_adapter_hil` | 2 ms supervisor + recording command sink + receipt | 无 Unitree SDK，只 would-write |
| `unitree_arm_adapter_dry_run` | 本地 ABI/周期/故障注入 | 无 DDS |

`dds_main.cpp` 和 `unitree_arm_adapter_dds` target 已在 Stage 2 移除。CMake 参数
`UNITREE_ARM_ADAPTER_BUILD_DDS=ON` 会直接 fail closed；当前没有
`--enable-output` 或其他可在 CLI 上启用 Unitree command publisher 的路径。

## Protocol v3

Python 镜像位于
[`right_arm_runtime/unitree_shm.py`](../../right_arm_runtime/unitree_shm.py)。一个 3328 B
POSIX shared-memory object 包含三个 64-byte-aligned seqlock slot：

| slot | payload | offset | writer |
| --- | ---: | ---: | --- |
| command | 768 B | 64 | Python/offline producer |
| paired state | 1440 B | 896 | C++ state bridge |
| receipt/status | 928 B | 2368 | C++ HIL/dry-run |

command 携带 producer/command ID、session nonce、source sample/timestamp、task epoch/
6 ms anchor、expiry、active mask、safety-policy ID/SHA256 和 13-slot
`q/dq/ddq/kp/kd/tau`。state 携带 35-slot 运动量/温度、LowState 与 torso IMU
各自时间、pair skew、validation flags 和 ingress session nonce。receipt 完整
回显 command/source/task/policy identity、observed state、deadline/expiry、requested/
executed mask 和 weight、guard reason 及最终 selected 13-slot command。

C++ `static_assert` 和 Python ctypes 测试同时锁定总大小、payload 大小、字段偏移和
64-byte alignment。改字段时必须升级 protocol version，不允许只维持总字节数。

## Paired state ingress

`unitree_arm_state_bridge` 只含 `ChannelSubscriber`，订阅 `rt/lowstate` 和
`rt/secondary_imu`。LowState 先按 SDK2 规则检查 CRC；两路都有新样本且
host-arrival skew 在限内时才写 paired state。

bridge 必须从 `--session-nonce` 接收非零 uint64。state-only launchers 为每次进程
生成新 nonce；HIL/下游只能核对 payload 内的 identity，不能为旧状态伪造
新 session。state bridge 源码和 binary 隔离测试会拒绝 `LowCmd`、command
topic 和 `ChannelPublisher`。

23-DOF Arm5 索引仍是：

```text
13-slot arm_sdk = 左臂[15..19] + 右臂[22..26] + 腰[12..14]
arm weight      = motor_cmd[29].q（仅作 future contract）
```

该索引、motor sign、mode whitelist 和 IMU frame 仍要由目标 G1 现场确认。

## Publisher-absent HIL

HIL 不读取 DDS，也不包含 device transport。它打开现有 protocol-v3 shared
memory，固定每 2 ms 执行；非 2000 us 的 `--period-us` 会在打开 shared memory 前
fail fast：

```text
paired state --> bounded exact-state cache ------+
                                                  v
protocol-v3 command --> dispatcher --> HardwareCommandSupervisor
                                         |
                                         v
                              13-slot formatter / guards
                                         |
                                         v
                              RecordingCommandSink (would-write)
                                         |
                                         v
                              protocol-v3 receipt + JSONL
```

`ValidatedStateCache` 以 seqlock sequence 去重，并按
`(source_sample_id, source_timestamp_ns)` 精确查找 proposal 绑定的历史状态。
最新 2 ms state 另作为 actuation state；二者不得混用。找不到 exact source
时 fail closed。

dispatcher 以 command seqlock sequence 区分“slot 未改变的 hold”与“重写/新
command”。一个新 6 ms proposal 只允许在相对 `0/2/4 ms` 三拍使用；
第四拍未到新 anchor、重写旧 ID、anchor 跳变或 replay 都会拒绝。hold 也会
逐拍重做 state freshness、ownership、deadline 和原 command expiry 检查。

`HardwareCommandSupervisor` 实现 disarmed、arming-PD、active、
soft-guard-releasing 和 latched-fault。production `SupervisorPolicy` 的 site limits、
ownership、startup/active/release verification 和 output authorization 全部默认 false；
因此仅填写一组数值 limit 不能 arming。宽松的 fixture policy 只能通过显式
HIL 测试开关启用，不是真机安全 policy。

`RecordingCommandSink` 与 receipt logger 是两个独立边界。command sink 在真正
would-write 调用点自行重读 `CLOCK_MONOTONIC`，再验证 deadline 和 expiry。
被拒绝的拍仍可以落盘 receipt，但 `sink_write_performed=false`。完整 HIL 的
`device_command_transport_present=false`、`hardware_output_performed=false`。

## 控制语义

只允许两种互斥模式：

1. `robot_pd_plus_feedforward`：`q/dq/kp/kd` 由 robot-side PD 消费，`tau`
   只能是纯 feedforward；
2. `direct_torque`：`tau` 是最终合成力矩，formatter 强制 robot-side
   `kp=kd=0`，避免 double PD。

inactive slot 必须使用当前 actuation `q`，且 `dq/ddq/kp/kd/tau=0`。
MuJoCo mapper 的 `max_abs_qacc=10 rad/s²` 没有被复制成硬件 hard-stop；真机
hard-stop/soft-guard/diagnostics 仍需厂商资料和吊架试验。

## 构建与测试

不依赖 Unitree SDK 的 core + HIL：

```bash
cpp/unitree_arm_adapter/build_and_test.sh
```

只构建 state bridge（仍不包含 command publisher）：

```bash
UNITREE_ARM_BUILD_DDS=OFF \
UNITREE_ARM_BUILD_STATE_BRIDGE=ON \
UNITREE_ARM_ADAPTER_BUILD_DIR=/tmp/hold-my-beer-mpc-unitree-state-only-build \
UNITREE_SDK2_DIR=/home/fjk/g1_ws/unitree_sdk2 \
cpp/unitree_arm_adapter/build_and_test.sh
```

CTest 覆盖 protocol ABI、supervisor state machine、exact source cache、2/6 ms hold、
sink-time deadline/expiry、receipt/sink parity、Python-to-C++ HIL end-to-end 和 HIL 源码/
binary capability scan。这些都是离线验证，不能提升 H1 或任何 hardware verification flag。

## 现场前仍不可越过的 gate

- H1 要先收到真实 CRC-valid、paired G1 state，当前仍是 PARTIAL；
- 人工冻结 model/index/sign/mode/IMU 合同，不得自动修改 verification flags；
- 用厂商资料和吊架测试确立 production limits、ownership、arm-weight 与 release/
  crash/watchdog 行为；
- 只有独立授权的后续阶段才能新增 Unitree command sink。它应复用本阶段的
  supervisor/formatter，但不得把 publisher-absent HIL 写成真机输出证据。

更完整的现场阶段见
[`HARDWARE_INTEGRATION_PLAN.md`](../../HARDWARE_INTEGRATION_PLAN.md)。
