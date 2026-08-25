# G1 Hardware Shadow 与 Future Hardware Output 实施计划

状态：**H1 仍为 PARTIAL；publisher-absent HIL 已离线实现；
hardware output 未授权、无 publisher target、未实机执行**。

本计划冻结 latency 实验之后的真机工程边界。当前实施终点是第一次真实 G1
state-only inspection：真实 `LowState` 和 torso IMU 可以进入本仓库的只读状态入口，
但不运行 MPC、不改变机器人模式、不取得 arm ownership，也不创建任何命令
publisher。只有人工审阅该证据并冻结型号、索引、时钟和 IMU 契约后，才进入完整
read-only control shadow。

## 1. 当前代码事实

当前仓库有三个不同边界，不能混称为一条已经完成的真机控制链：

```text
Shared control core
  full-task protocol / continuous-H / template v2 / ArmMPCPolicy

Simulation adapter
  MuJoCo state -> RightArmSimProcess -> RNEA + MuJoCo mapper
  -> certified final_tau -> d.ctrl -> mj_step

Hardware state adapter (read-only)
  rt/lowstate + rt/secondary_imu -> state-only bridge
  -> protocol-v3 paired-state slot -> read-only Python mapping / inspection

Publisher-absent HIL (offline only)
  control proposal -> offline certification -> protocol-v3 command
  -> C++ supervisor -> recording command sink -> full receipt

Future hardware output adapter (absent)
  site-certified command -> same supervisor/formatter
  -> future Unitree command sink (not implemented)
```

`RightArmSimProcess`、MuJoCo DDQ-to-torque mapper 和 `d.ctrl` 属于 simulation
adapter。它们依赖 MuJoCo 的求解器和接触状态，不能直接搬到真机并声称完成了
hardware certification。

当前完整 `HardwareShadowController` 仍只兼容 legacy phase template；它没有
final full-task v2 的真实 task epoch、continuous-H 任务绑定和 24 ms handoff。
因此本轮第一次真实 session 只做状态契约发现，不运行该 controller，也不宣称
final control stack 已在真实状态上验证。

## 2. Unitree 官方实现核对

以下官方实现只提供型号、消息和平台语义参考；本仓库自己的 MJCF/URDF、23-DOF
集合和 motor-index 合同始终优先：

- Unitree `unitree_rl_gym` 将 `g1_23dof_rev_1_0` 列为
  `mode_machine=4`，结构为双腿各 6 DOF、waist yaw 1 DOF、双侧 Arm5：
  <https://github.com/unitreerobotics/unitree_rl_gym/blob/main/resources/robots/g1_description/README.md>
- Unitree SDK2 的 G1 Arm5 示例使用左臂 15..19、右臂 22..26、腰 12..14，
  `rt/arm_sdk` 的全局 arm weight 位于 slot 29：
  <https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/g1/high_level/g1_arm5_sdk_dds_example.cpp>
- SDK2 官方 G1 low-level 示例先校验 `LowState` CRC，再消费电机和 IMU 数据；
  同时明确区分 `LowState.imu_state` 的 pelvis IMU 与 `rt/secondary_imu` 的 torso IMU：
  <https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/g1/low_level/g1_ankle_swing_example.cpp>
- `xr_teleoperate` 的 G1_23 motion-mode 实现也使用 35-slot LowState、左臂
  15..19 和右臂 22..26，并通过 `rt/arm_sdk` 与已有 locomotion controller 协作：
  <https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/robot_arm.py>
- `unitree_rl_gym` 的实机 deploy 可借鉴显式网卡、等待 LowState、遥控状态机和
  20 ms lower-body policy loop；它的通用 G1 arm/waist index 列覆盖另一种 DOF
  布局，不能复制为本项目 Arm5 映射：
  <https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/deploy_real.py>

当前本地 SDK2 pin 为 `fa925bf6bb3fff439000266d70bde32eb5cd3597`。
不同 Unitree 官方仓库中的“23DOF”还存在紧凑索引和不同 `mode_machine` 代际；
因此 `mode_machine=4` 在第一次 session 只是要核对的官方参考，不是自动通过门。
任何不匹配都必须 fail closed，并由目标机器人型号、固件和已知关节运动重新确认。

## 3. 两部分共用的窄接口

接口按“数据可信度”和“输出授权”分层，避免 shadow 在未来被一个布尔参数静默升级
为主动输出。

### 3.1 `RawHardwareStateFrame`

唯一 hardware ingress 应包含：session/sample ID、LowState tick/version/mode、
35-slot `q/dq/ddq/tau_est/temperature/motorstate`、pelvis IMU、独立 torso IMU、
两个本机接收时间、pair skew、CRC/freshness/reject reason、机器人/固件/映射 ID。

当前 protocol v3 已携带 35-slot 运动量、温度、mode、tick、配对后的
torso IMU，以及 LowState/torso 两个来源的 host timestamp、pair skew、
validated timestamp、ingress flags 和显式 session nonce。CRC 在 bridge 写 shared
memory 之前验证；启动脚本每次生成非零 nonce 并交给 bridge，下游只能核对而
不能改写该证据。ABI 总长 3328 B，由 command 768 B、state 1440 B 和
receipt/status 928 B 三个 64-byte-aligned seqlock slot 组成；Python/C++ 使用字段
offset 回归锁定，不是只对比总字节数。

### 3.2 `ValidatedHardwareObservation`

只有 model/index、单位、frame、时间单调、freshness、有限性、温度和 IMU 合同全部
通过后才生成。它向共享控制核心提供右臂 `q/dq`、torso `R/omega/acc/alpha` 及
每一项 validity；shadow 与 future output 必须共用这一转换，不能各维护一套 pelvis/
torso 语义。

### 3.3 `TaskClockEvent`

final full-task template 必须从真实 locomotion command producer 得到显式 task epoch、
planned/runtime command、heading reference 和 6.4 s stop event。第一条 LowState 或
shadow 进程启动时刻不能被猜作 task t=0。该接口是后续完整 full-task read-only
shadow 的前置条件，本轮不实现。

### 3.4 `HardwareControlProposal`

共享 MPC 只产生 proposal：`source_sample_id/timestamp`、task epoch/anchor、
`q_ref/dq_ref/ddq_des`、predictor/QP/timing diagnostics。proposal 不携带 publisher
capability，也不能直接表示“安全可执行”。

### 3.5 `CertifiedHardwareCommand` 与 `ExecutionReceipt`

future output 的独立认证层才生成完整 13-slot `q/dq/kp/kd/tau`、command mode、
active mask、arm-weight transition、expiry、hard-stop/soft-guard/diagnostic 结果。
output adapter 必须回报实际写出的 command ID、绑定的 source state、mode、weight、
clamp/guard/watchdog reason 和 DDS write time。shadow 不链接或实例化这一层。

Stage 2 中 Python 唯一的正式全绑定 writer 是
`write_certified_hil_command(CertifiedHilCommandEnvelope)`。它只接受真实
`CertifiedHardwareCommand` 类型，且要求 scope 固定为
`offline_transport_contract_only`、`hardware_safety_certified=false`、
`hardware_output_authorized=false`。它始终清除 `REQUEST_OUTPUT`；普通 `write_*`
API 传 `request_output=True` 会立即拒绝。这是 HIL transport 边界，不是真机认证。

若使用 robot-side PD，则 `tau` 必须是纯 feedforward；若使用 direct torque，则
robot-side `kp/kd` 必须为零。绝不能在两侧重复计算 PD。

## 4. 本轮实施：H0/H1 state-only inspection

### H0：发布隔离与数据完整性

已冻结的实现要求：

1. CMake 用 `UNITREE_ARM_ADAPTER_BUILD_STATE_BRIDGE=ON` 单独构建 bridge，同时
   `UNITREE_ARM_ADAPTER_BUILD_DDS=OFF`；state-only build directory 中出现
   `unitree_arm_adapter_dds` 即拒绝运行。
2. bridge 编译单元只包含 `ChannelSubscriber`，只订阅 `rt/lowstate` 和
   `rt/secondary_imu`，没有 `LowCmd`、`ChannelPublisher`、`rt/arm_sdk` 或
   `MotionSwitcherClient`。
3. 每条 LowState 先按 SDK2 官方算法验证 CRC；错误包计数并拒绝写入 shared memory。
4. 两个状态流各有新样本且本机 callback skew 不超过 5 ms 才形成 paired state。
5. Python 以 `O_RDONLY + MAP_PRIVATE` 打开 shared memory，只收集状态；不实例化
   predictor、MPC 或 command builder。
6. 必须收到请求数量的 fresh、unique、finite、timestamp/tick 单调 paired samples；
   incomplete、stale、NaN/Inf、重复或倒退一律 fail closed。
7. 保存完整 35-slot JSONL trace、inspection summary、bridge log/summary、repo/SDK/
   config/binary hash 和显式 NIC 信息。

单用途入口：

第一次上机应先阅读详细
[H1 现场操作手册](G1_H1_FIELD_RUNBOOK.md)，现场使用
[一页速查表](G1_H1_FIELD_CHECKLIST.md)。

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
env -u UNITREE_INGRESS_SESSION_NONCE \
  ./tools/realtime/run_hardware_state_inspection.sh YOUR_VERIFIED_G1_INTERFACE \
  --duration-s 10 \
  --inspect-samples 500 \
  --group first_real_g1_readonly
```

该命令不要求把 YAML verification flags 改为 `true`，不要求 PREEMPT_RT 或
`SCHED_RR`，因为它不运行控制计算。它仍要求人工确认专用有线 NIC 和机器人安全
状态，不能自动选择接口。

### H1：第一次真实 G1 session 验收

第一次 H1 由合格吊架把机器人完全吊起，并按目标版本官方说明上电后停留在 factory
零力矩状态；不按准备/运动/debug/诊断姿态组合键。项目不调用 `ReleaseMode`、
`SelectMode`，不进入 debug/develop，不取得 arm ownership。session 后人工审查：

- bridge output capability absent，command publish/write 为 0；
- LowState received/CRC-valid/CRC-rejected 数量及 secondary-IMU/pair/skew 统计；
- bridge 最后收到的 version 已记录；`mode_pr`、`mode_machine` 的整段观察值是否稳定并
  与目标型号资料一致（当前 trace 不能证明 version 在整段稳定）；
- sample ID、host timestamp、robot tick 和 rate/gap 是否单调合理；
- 35-slot 状态有限，右臂 22..26 与已知姿态的符号/幅值一致；
- quaternion norm、静止重力方向/模长、gyro bias 和 torso/pelvis 来源没有混用；
- 电机温度和 state age 合理。

观察值不会自动回写 `configs/g1_hardware_shadow.yaml`。仅“数值看起来合理”也不能
证明 IMU frame 或最后一轴物理语义。第一次 session 在这里停止。

### 2026-08-23 第一次连接尝试

在当时唯一有 carrier 的有线接口 `enx6c1ff701509c`（`192.168.31.159/24`）上执行了
旧版 state-only 连接尝试。它早于当前 protocol-v3 HEAD，且该接口没有收到任何
`rt/lowstate` 或
`rt/secondary_imu`：LowState received/CRC-valid/CRC-rejected 为 `0/0/0`，torso
IMU 和 paired state 均为 0。collector 因 `0/500` fresh paired samples 按合同
fail closed；没有运行 predictor/MPC，没有 publisher/command write，也没有残留
bridge 或 shared-memory 对象。旧目录缺少当前 H1 所需的 nonce、inspection log、raw
trace 和 Python summary，因此不能作为当前 protocol-v3 launcher 的完整执行证据。

证据位于
`evaluation/hardware_shadow/state_inspection/first_real_g1_readonly_20260823/`。
这是一条 **连接失败证据**，不是第一次有效真实 G1 shadow 数据。当前网络邻居只有
普通 `192.168.31.1` 网关，另一张候选 USB Ethernet 无 carrier，所以 H1 状态保持
`PARTIAL`。下一次必须先由现场人员确认目标 G1 专用网线、接口和机器人安全状态，
然后原样重跑；不得通过换 topic、关闭 CRC、减少完整性门或预填 verification flags
绕过失败。

## 5. 后续 hardware shadow 阶段

### H2：人工冻结状态契约

基于 H1 证据和目标 G1/固件资料确认 model、35-slot mapping、motor sign、tick wrap、
mode whitelist、torso IMU 坐标/重力语义和固定外参。完成 replay/unit/frame 测试后，
才把相应 verification flags 改为 true。

机器人不在现场期间已完成 **H2-prep**：

- `hardware_state_replay.py` 可对未来 H1 `raw_state_trace.jsonl` 做 shape、finite、
  sample/timestamp/tick monotonic、slots 22..26 mapping 和 bridge CRC/pair counter 审计；
- 报告严格区分 `offline_trace_contract_passed` 与
  `hardware_session_verified=false`，synthetic fixture 必须显式标记；
- 审计器只报告 observed mode candidates 和现场 gates，不写配置，也不允许自动修改
  verification flags。

当前 H1 没有收到样本，所以这里的实现/单测通过不能推进 H2 verification。命令与
详细边界见 [HARDWARE_OFFLINE_PREPARATION.md](HARDWARE_OFFLINE_PREPARATION.md)。

### H3：完整 read-only control shadow

接入权威 `TaskClockEvent`，再让真实状态进入 final full-task template v2、continuous-H
和共享 `ArmMPCPolicy`。虚拟 24 ms handoff 必须保留绝对 task anchor 和真实时间；
missed 6 ms anchor 不能通过压缩 control index 隐藏。输出仍只是
`HardwareControlProposal`，command write/publish 仍为 0。

这一阶段才能回答“真实状态是否进入当前最终控制链”，但仍不能回答“命令在真机上
是否安全有效”。

## 6. Future hardware output：仅设计的 O0-O4

### O0：输出合同与 raw replay

在 H1/H2 raw trace 上离线定义 `CertifiedHardwareCommand`、session nonce、
source-state binding、activation deadline、active mask 和 `ExecutionReceipt`。
把安全规则分为：

- hard-stop：非有限命令、状态失信/超时、明确的硬位置/力矩/温度限制、ownership 或
  watchdog 失败；
- soft guard：可以经限幅、降权、保持、退出主动控制处理的风险，并明确持续时间、
  回滞和人工复位；
- diagnostics：qacc、dq、位置余量、torque/torque-rate、温度、通信延迟、接触与
  estimator confidence。

MuJoCo `max_abs_qacc=10 rad/s²` 不是 Unitree 的已确认物理极限，不得默认复制为
真机 hard-stop。尤其是单拍轻微 qacc 超限是否终止主动控制，必须结合传感器质量、
执行器/传动限制和吊架试验单独判定。

### O1：无机器人输出的 fake DDS/HIL

future publisher 只能读取与 shadow 相同的 validated state ingress，不得再从
`LowState.imu_state` 另建 pelvis-IMU 状态链。使用 fake sink 验证 CRC、command-state
binding、过期拒绝、watchdog、process crash、重启旧命令、mode/weight 状态机和
receipt；输出 capability 默认不存在。

当前已完成的 **O0/O1 offline preparation** 有两层：

1. 平台无关 Python contract 和内存 fake sink，验证 source binding、expiry、
   13-slot mask、inactive zero-action hold、double-PD、replay/restart nonce 和 watchdog；
2. protocol-v3 + publisher-absent C++ HIL，在 2 ms final-sink 边界运行共享
   `HardwareCommandSupervisor`。状态机为 disarmed、arming-PD、active、
   soft-guard-releasing 和 latched-fault；production policy 的 site/ownership/startup/
   active/release/output 验证项默认全为 false，因此不可 arming。只有
   显式 test-fixture policy 能在无 publisher 的 HIL 中走到 would-write sink。

2 ms dispatcher 用 command seqlock sequence 判定新 proposal 还是同一 slot 的合法
hold。一个 6 ms proposal 只可在相对 `0/2/4 ms` 三拍使用；超出三拍、
重写旧 ID 或错过下一 anchor 都 fail closed。有界 cache 用精确
`(sample_id, source_timestamp_ns)` 查找生成 proposal 的历史 source state，最新
2 ms state 只用于当前 actuation/hold 组装，不能替代 source identity。

HIL 的 recording command sink 在真正 would-write 调用点再读一次时钟，重新检查
deadline 和 expiry。receipt 完整回显 session/producer/command/source/task/policy、
请求与实际 active mask/weight、guard/clamp、13-slot selected command 和 sink 结果。
receipt 落盘与 command sink 是两个不同事实；被拒绝拍仍可写 JSONL，但
`sink_write_performed=false`。无论是否 would-write，DDS/hardware write 始终为 false。

HIL executable 只链接本地 core/recording sink，隔离测试会扫描源码和 binary，
拒绝 `LowCmd`、`ChannelPublisher`、`rt/arm_sdk` 或 Unitree SDK 输出依赖。
`dds_main.cpp` 和真实 command target 已从 Stage 2 移除；把
`UNITREE_ARM_ADAPTER_BUILD_DDS=ON` 交给 CMake 会直接失败。

arm-weight 的真实 ownership/transition、真实 publisher crash/release 以及硬件阈值
仍是现场 gate；没有用自定阈值伪造这些验证。

### O2：吊架固定姿态 PD commissioning

独立授权后，在无 payload、低 weight、人工急停和物理限位条件下，先验证
`rt/arm_sdk` ownership。所有 13 slots 从 live q 无跳变初始化；确认 weight 是否同时
影响双臂/腰、13/14 无效槽如何处理，以及 release/crash 的真实行为。此时不运行 MPC。

### O3：吊架限时 MPC

在完整状态估计、硬件 inverse-dynamics/torque contract 和 previous-executed-command
反馈具备后，才允许单臂、短时、低风险 MPC。先验证静止和小扰动，再逐级增加 payload。

### O4：行走 full-task 主动闭环

最后才把真实 locomotion task epoch、24 ms handoff、6.4 s direct stop 和
full-task template v2 接到主动输出。模板提前知道停车时刻，仍不是任意任务预测器。

每一级都需要新的显式授权；H0/H1 的通过不会自动授予 O0-O4 输出权限。

## 7. 本轮明确未做

- 仓库中已无 `unitree_arm_adapter_dds` target/launcher，开启 DDS build 会 fail closed；
- 未创建 `rt/arm_sdk` 或 `rt/lowcmd` publisher；
- 未调用 motion switcher 或改变机器人控制模式；
- 未把 synthetic replay/fake sink 当作真实硬件证据；
- 未修改任何依赖目标机器人确认的 verification flag 或 mode whitelist；
- 未运行完整 MPC hardware shadow；
- 未接入 full-task v2 的真实 task epoch；
- 未修改 MPC、template、continuous-H、mapper 或安全阈值；
- 未重新开启 latency、L1-D 或 async/free-running；
- publisher-absent HIL 仅记录 would-write 和 receipt，未启用任何 hardware output。
