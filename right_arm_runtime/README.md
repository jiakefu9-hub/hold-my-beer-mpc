# 右臂运行时边界

该目录提供共享 Python 控制核心与两种平台适配器之间的窄接口。两条进程/IPC
路径用途不同，不能互相冒充：

- `sim_process.py` 对接 `cpp/right_arm_sim_runtime`，只用于 MuJoCo；
- `unitree_shm.py` 对接 `cpp/unitree_arm_adapter`，用于硬件状态/命令协议。

最终仿真使用第一条路径。hardware shadow 只读使用第二条路径的 state slot，
尚未支持 full-task template v2 + 24 ms handoff，也没有真机命令输出。

## 正式 MuJoCo process 链

正式入口和数据流为：

```text
run.sh
  -> main_sim.py
  -> FullTaskTemplatePredictor v2 + MPC
  -> RightArmSimProcess
  -> C++ RNEA
  -> MuJoCo DDQ-to-torque mapper / forward-dynamics certification
  -> right-arm executor
  -> mapper-certified feedforward + latest-state PD/guards -> final_tau
  -> main_sim writes MuJoCo d.ctrl before mj_step
```

`RightArmSimProcess` 每个虚拟 2 ms 物理拍原子发送完整 MuJoCo 状态、
`q_ref/dq_ref/ddq_des`、当前 right-arm PD torque 和虚拟时间戳。当前完整 6 ms
interval 内有两次 DDQ-to-torque 更新；其余物理拍只允许复用上一份已经认证的
feedforward，再用当前 q/dq 运行 executor。大数组通过 seqlock shared memory
传输，两根 pipe 只发请求/完成通知。

正式 full-task 的 `[0,24 ms)` 右臂由 `main_sim.py` 直接执行当前 fixed-posture
PD；predictor/H/task clock 同时从 task t=0 推进，但 MPC 不接入右臂。24 ms、
absolute template anchor 4 接管时，切换前最后一拍真实 PD torque 作为
`previous_executed_tau` 送入 mapper。task clock、gait phase 和 template 都不
reset、不重播。

运行时的历史配置值有三种：

- `process`：正式 full-task，执行独立 C++ worker 返回的 `final_tau`；
- `sync`：同步 C ABI 数值回归基线；
- `shadow`：同拍比较 sync/process，比较通过后执行 process 结果。

正式 `--full-task-smoke` 只允许 `process`。这里的“shadow”是仿真进程 parity
模式，不是 Unitree hardware shadow。

### fail-closed 执行合同

C++ mapper 的 normal candidate、second pass、rescue、hold-last、PD safe-hold
和最多四次 line search 都必须经过当前 MuJoCo 状态的 forward-dynamics 验收。
worker 只在 `final_output_certified=1`、`NO_SAFE_TORQUE=0` 且最终力矩有限时返回
成功。Python 还检查 session/request/command/state id；以下任一事件都会终止
当前运行，不会把旧力矩或未认证 candidate 写入 `d.ctrl`：

- worker 启动失败、超时、EOF 或非零 status；
- 错帧或 session/request/state id 不一致；
- mapper 返回 `NO_SAFE_TORQUE` 或未认证 final output；
- 最终力矩包含 NaN/Inf。

这是 mapper 更新拍的 MuJoCo 当前状态验收与每拍 executor guard 组成的
执行合同。中间 2 ms 拍没有重做 forward-dynamics 验收，这也不是实体电机、
接触或机器人固件已经认证的证据。

默认关闭的 `sim_mpc_latency.py` 仅属于 MuJoCo adapter：它延后 MPC result packet
的激活，并在 activation state 上调用同一 mapper，不改变 process ABI 的
fail-closed 语义。`heldout_pair_02_minus` 2 ms 短 smoke 在 44 ms 因最低真实
candidate `10.293 rad/s^2` 超过门限 10 返回 `NO_SAFE_TORQUE`。该实验已在
L1-C PARTIAL 后冻结，不进入 L1-D 或 async/free-running。

正常使用采用上下文管理器：

```python
with RightArmSimProcess(...) as runtime:
    result = runtime.execute(...)
```

command/state 使用 MuJoCo 虚拟时间；`publish_monotonic_ns` 只测 IPC 墙钟，二者
不能相减。协议和 worker 测试见
[cpp/right_arm_sim_runtime/README.md](../cpp/right_arm_sim_runtime/README.md)。

## 正式环境 preflight

full-task 必须经根目录 `run.sh` 启动。第一个 `mj_step` 前会核对 parent/worker
affinity 都严格为 `[7]`、六个数值库线程变量均为 1、Torch intra/inter-op 为
1、control-loop GC 已关闭、dynamic arming 为 false、startup duration 为
24 ms 且 handoff anchor 为 4。详见
[REALTIME_RUNTIME.md](../REALTIME_RUNTIME.md)。

## Unitree protocol v3

`unitree_shm.py` 是 `cpp/unitree_arm_adapter` 的 Python client。它镜像一个锁定的
protocol-v3 ABI，可读 35 电机 paired state 和完整 C++ receipt；client 本身
不创建 DDS。hardware shadow 使用 `read_only=True` 打开 C++ state-only bridge
创建的对象，没有 command sink。publisher-absent HIL 的离线 producer 则只能经
下文的受限全绑定 writer 写 command slot。

| 项目 | protocol v3 |
| --- | ---: |
| magic | `0x473141524d504331` |
| 总大小 | 3328 B |
| command 槽偏移 / payload | 64 / 768 B |
| state 槽偏移 / payload | 896 / 1440 B |
| receipt/status 槽偏移 / payload | 2368 / 928 B |

Python 打开时检查 magic、version、总大小、字段偏移和 64-byte alignment。
跨进程原子读写使用 `libatomic` acquire/release + seqlock。三个槽保持单写者：
上层写 command、paired state bridge 写 state、C++ 2 ms HIL 写 receipt。

command 不再只是数值向量：它完整绑定 producer/command ID、session nonce、
source sample/timestamp、task epoch/6 ms anchor、expiry、active mask 和 safety-policy
ID/SHA256。state 保存 LowState/torso IMU 两源时间、pair skew、validated timestamp、
ingress flags 和 state-bridge session nonce。receipt 回显完整 identity、observed state、
deadline/expiry、guard、requested/executed mask 与 weight 和最终 selected 13-slot command。
Python read-only inspector/shadow 与 C++ HIL 都必须核对 launcher 指定的同一 nonce、
三项 required flags、两源时间关系和不超过 5 ms 的 skew。

### 离线 writer 与两种命令语义

- robot-PD + feedforward：底层执行 q/dq PD，`tau` 只能是纯 feedforward；
- direct torque：`tau` 已包含反馈，C++ formatter 将 robot-side `kp/kd` 强制置零，
  避免 double PD。

普通 `write_*` API 传 `request_output=True` 会直接抛出 `PermissionError`。唯一的正式
全绑定写口 `write_certified_hil_command(CertifiedHilCommandEnvelope)` 只接受真实
`CertifiedHardwareCommand`，要求 certification scope 为 offline-only，并要求
`hardware_safety_certified` 和 `hardware_output_authorized` 都是 false。它仍然清除
`REQUEST_OUTPUT`。跨进程字符串 identity 使用 UTF-8 SHA256 前 8 字节的大端
uint64，禁止使用进程随机化的 Python `hash()`。

## 本地协议测试

```bash
cpp/unitree_arm_adapter/build_and_test.sh
/home/fjk/miniforge3/bin/conda run -n g1_mpc \
  python -m unittest right_arm_runtime.tests.test_unitree_shm -v
```

测试检查 C++/Python 全字段 layout、seqlock、错误 magic/version/layout、普通 writer
的输出拒绝、formal HIL writer 的 offline-only 约束及两种 PD 语义。它们不
访问 DDS，也不证明真机行为。


## Hardware offline preparation

`hardware_state_replay.py` 审计 state-only JSONL evidence，但永远不会修改 hardware
verification flags；`hardware_output_contract.py` 定义 source-state binding、expiry、
13-slot active mask、互斥 PD/torque 语义和纯内存 fake sink。

Stage 2 的 C++ `unitree_arm_adapter_hil` 固定在 2 ms final boundary 重做绑定、安全状态机、
deadline/expiry 和 13-slot formatter；非 2000 us 的 `--period-us` 会 fail fast。它用
有界 cache 精确找回 proposal 绑定的
source state，并用当前 2 ms state 组装 actuation plan。每个 6 ms proposal 只在
相对 `0/2/4 ms` 三拍合法；同一 seqlock slot 第四拍、重写旧 ID、source 丢失或
新 anchor 不连续都 fail closed。recording command sink 在 would-write 调用点再用实际
时钟检查 deadline/expiry。receipt logger 独立记录每拍：被拒绝 receipt 可以落盘，
但 `sink_write_performed=false`。

HIL binary 不链接 Unitree SDK，不包含 `LowCmd`、`ChannelPublisher` 或
`rt/arm_sdk`。`dds_main.cpp`/output target 已移除，CMake 对
`UNITREE_ARM_ADAPTER_BUILD_DDS=ON` 直接 fail closed。production supervisor policy 的现场
验证和授权字段默认全为 false，因而不可 arming。

详见 [HARDWARE_OFFLINE_PREPARATION.md](../HARDWARE_OFFLINE_PREPARATION.md)。H1
仍为 PARTIAL；真实 model/index/IMU、arm-weight ownership、release/watchdog 和安全
阈值必须留到现场验证。
