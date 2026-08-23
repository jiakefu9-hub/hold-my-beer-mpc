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

## Unitree protocol v2

`unitree_shm.py` 是 `cpp/unitree_arm_adapter` 的 Python client。它可写 13 维
Arm SDK command，读取 35 电机状态和 C++ 2 ms loop status；client 本身不创建
DDS。hardware shadow 使用 `read_only=True` 打开 C++ state-only bridge 创建的
对象，没有 command sink。

| 项目 | protocol v2 |
| --- | ---: |
| magic | `0x473141524d504331` |
| 总大小 | 2304 B |
| command 槽偏移 / payload | 64 / 656 B |
| state 槽偏移 / payload | 768 / 1392 B |
| status 槽偏移 / payload | 2176 / 96 B |

Python 打开时检查 magic、version、总大小、字段偏移和 64-byte alignment。
跨进程原子读写使用 `libatomic` acquire/release + seqlock。三个槽保持单写者：
上层写 command、DDS callback 写 state、C++ 2 ms loop 写 status。

### 两种命令语义

- `write_robot_pd_plus_feedforward(...)`：底层执行 q/dq PD；`tau_ff` 只能是纯
  feedforward；
- `write_direct_torque(...)`：`tau_cmd` 已包含反馈；C++ 将下发 kp/kd 强制置零，
  避免 double PD。

两种 Python API 的 `request_output` 默认都是 `False`。即使上层设为 `True`，
output-capable C++ 进程还必须显式带 `--enable-output`；本地 dry-run 和
`unitree_arm_state_bridge` 根本没有输出能力。当前项目没有授权任何真机输出
命令。

## 本地协议测试

```bash
cpp/unitree_arm_adapter/build_and_test.sh
/home/fjk/miniforge3/bin/conda run -n g1_mpc \
  python -m unittest right_arm_runtime.tests.test_unitree_shm -v
```

测试检查 C++/Python layout、seqlock、错误 magic/version/layout、默认输出关闭
及两种 PD 语义；它使用 dry-run，不访问 DDS，也不证明真机行为。完整硬件边界
见 [HARDWARE_SHADOW.md](../HARDWARE_SHADOW.md)。
