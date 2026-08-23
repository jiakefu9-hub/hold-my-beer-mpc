# 运行时架构：从 MuJoCo 到真实 G1

这是一张“运行时地图”，用于在几分钟内看懂程序实际怎样启动、哪些模块属于
独立进程，以及上真机时哪些部分会替换。算法、协议和安全门的详细设计不在这里
重复，见文末链接。

当前状态只有三句话：

- **Simulation** 是当前正式、已验证的控制路径；
- **Shadow** 只允许读取真实 G1 状态，当前仍缺真实机器人证据；
- **Future / Hardware output** 只有接口和离线准备，未授权、未启用。

图中实线表示当前已有的调用或数据路径，虚线表示现场 gate 之后或未来才允许接通
的路径。

## 一分钟总览

```mermaid
flowchart LR
  Sim["Simulation<br/>MuJoCo 正式路径"]
  Core["共享控制核心<br/>full-task v2 · continuous-H<br/>运动学 · ArmMPCPolicy"]
  Shadow["Shadow<br/>真实状态只读路径"]
  Hardware["Future / Hardware output<br/>未启用"]

  Sim <--> Core
  Shadow -. "完整 shadow 仍需现场 gate" .-> Core
  Core -. "proposal，尚未授权输出" .-> Hardware
```

“共享控制核心”是逻辑边界，不代表一个单独进程。full-task predictor、continuous-H
和 MPC 的逐 anchor 编排由
[`FullTaskRightArmControlCore`](right_arm_runtime/full_task_control_core.py) 保留一份；
Simulation 与 H3-offline shadow 只提供各自已验证的状态、权威 `TaskClockEvent` 和
平台运动学 helper，不各维护一套近似时钟或 MPC。

## Simulation：当前正式运行结构

```mermaid
flowchart LR
  Run["run.sh<br/>环境检查 · 构建 · taskset"]

  subgraph CPU7["逻辑 CPU 7（正式受控运行）"]
    subgraph Py["Python 主进程 · main_sim.py"]
      Physics["权威 MuJoCo 世界<br/>d.ctrl · mj_step · 2 ms"]
      Legs["TorchScript 下肢策略<br/>20 ms"]
      Predict["task clock · continuous-H<br/>full-task template predictor"]
      MPC["运动学 + ArmMPCPolicy / OSQP<br/>6 ms"]
      Client["RightArmSimProcess client"]

      Legs -->|"腿部控制"| Physics
      Physics -->|"torso state"| Predict
      Physics -->|"right-arm q / dq"| MPC
      Physics -->|"完整仿真状态"| Client
      Predict --> MPC --> Client
    end

    subgraph Worker["C++ 子进程 · right_arm_sim_runtime_worker"]
      RNEA["Pinocchio RNEA<br/>0 / 4 ms 更新"]
      Mapper["MuJoCo scratch mapper<br/>0 / 4 ms 候选验收"]
      Executor["Executor<br/>24 ms 接管后每 2 ms<br/>使用最新 q / dq"]
      RNEA --> Mapper -->|"validated / cached feedforward"| Executor
    end
  end

  Run --> Py
  Client -->|"request：refs + 当前状态<br/>shared memory + pipe"| RNEA
  Executor -->|"response：guarded final_tau[5]<br/>shared memory + pipe"| Client
  Client -->|"写右臂 d.ctrl"| Physics
```

这里有两个独立 OS 进程：Python 主进程和它启动的 C++ worker。Python 经
`taskset` 固定到 CPU 7，worker 继承同一 affinity。Python 拥有唯一的仿真时间和
物理世界，必须等 worker 返回通过当前执行链 guard 的 `final_tau`，写入 `d.ctrl`
后才调用 `mj_step()`。因此当前正式 MuJoCo 是 blocking lockstep，不是并行流水线
或 free-running simulator。

还要注意：

- MPC QP 在 Python 主进程；C++ worker 在 0/4 ms 更新 RNEA 与 MuJoCo 候选验收，
  接管后 executor 每 2 ms 用已验收/缓存的 feedforward 和最新 `q/dq` 运行；
- worker 内的 MuJoCo 是验收用 scratch model，不推进权威机器人状态；
- task/template/H 从 `t=0` 推进；右臂在 `[0, 24 ms)` 使用固定姿态 PD，24 ms 的
  anchor 4 才交给 MPC；
- `RightArmSimProcess`、MuJoCo mapper、`d.ctrl` 和 `mj_step()` 都属于
  **simulation adapter**，不是可直接搬到真机的控制核心。

入口与边界代码：[`run.sh`](run.sh)、[`main_sim.py`](main_sim.py)、
[`right_arm_runtime/sim_process.py`](right_arm_runtime/sim_process.py)、
[`cpp/right_arm_sim_runtime/`](cpp/right_arm_sim_runtime/)。

## Shadow 与 publisher-absent HIL

```mermaid
flowchart LR
  G1["目标真实 G1 DDS<br/>尚无有效现场样本<br/>rt/lowstate + rt/secondary_imu"]
  Bridge["C++ 独立进程<br/>unitree_arm_state_bridge<br/>CRC + 两路配对 + session nonce"]
  SHM["protocol-v3 shared memory<br/>state slot"]
  Inspect["Python 独立进程<br/>inspect-state-only<br/>read-only / private mapping"]
  Evidence["JSONL + summary<br/>只读证据"]
  Fixture["synthetic / replay state<br/>+ explicit TaskClockEvent"]
  Core["共享 full-task core<br/>HardwareControlProposal"]
  Cert["Python offline certification<br/>CertifiedHardwareCommand"]
  Cmd["protocol-v3 command slot"]
  Hil["C++ 2 ms HIL<br/>state cache + supervisor"]
  Fake["recording command sink<br/>would-write only"]
  Receipt["protocol-v3 receipt + JSONL<br/>DDS / hardware write = 0"]

  G1 --> Bridge --> SHM --> Inspect --> Evidence
  Fixture --> Core --> Cert --> Cmd --> Hil --> Fake --> Receipt
  SHM -. "现场契约通过后可作 HIL ingress" .-> Hil
```

当前批准入口是
[`tools/realtime/run_hardware_state_inspection.sh`](tools/realtime/run_hardware_state_inspection.sh)：
C++ bridge 只有 subscriber，没有 `LowCmd`、command topic 或 publisher。launcher 每次
生成非零 ingress session nonce，bridge 把它与 CRC-valid 的 LowState 和配对 torso
IMU 一起写入 state slot；Python 只读检查也必须核对同一个 nonce、三项 ingress
flags、两路时间与不超过 5 ms 的 skew。机器人目前不在现场，所以 H1
仍是 **PARTIAL**，离线测试不是真实 G1 证据。

Stage 2 另增了一条 **publisher-absent C++ HIL**，专用于把 Python 产生的
offline-certified command 经 protocol-v3 送到最后一道 C++ 安全边界。HIL 的
supervisor 复核 session/source/task/policy 绑定、过期、deadline、13-slot mask、
double-PD、ownership 和状态机；通过时只写入独立 recording command sink
和完整 receipt。该 binary 不链接 Unitree SDK，不含 `LowCmd`、
`ChannelPublisher` 或 `rt/arm_sdk`，因此它的“would write”不是 DDS write。

HIL 固定运行在 2 ms 周期，其他 `--period-us` 会 fail fast。它用 seqlock sequence
区分“原 slot 未更改”与“重写/新命令”。每个新 6 ms
proposal 只可在对应 2 ms sink 拍的 `0/2/4 ms` 三次使用；此后必须有下一个
anchor。新 proposal 必须在有界 cache 中精确命中
`(source_sample_id, source_timestamp_ns)`，不允许用“最新状态”冒充源状态；
实际准备执行时另用当前 2 ms actuation state。

入口与边界代码：[`tools/realtime/run_hardware_shadow.sh`](tools/realtime/run_hardware_shadow.sh)、
[`run_hardware_shadow.py`](run_hardware_shadow.py)、
[`right_arm_runtime/hardware_shadow.py`](right_arm_runtime/hardware_shadow.py)、
[`right_arm_runtime/unitree_shm.py`](right_arm_runtime/unitree_shm.py)、
[`cpp/unitree_arm_adapter/src/state_bridge_main.cpp`](cpp/unitree_arm_adapter/src/state_bridge_main.cpp)。

## Future / Hardware：仅保留边界，当前无 output target

```mermaid
flowchart LR
  G1State["G1 DDS state"]
  Ingress["C++ state ingress process<br/>CRC + paired state"]
  StateSHM["state shared memory"]
  Clock["真实 locomotion producer<br/>TaskClockEvent"]

  subgraph PyFuture["Python future hardware runtime · 未授权"]
    Observe["ValidatedHardwareObservation<br/>mapping · frame · freshness"]
    Estimator["future full-state / contact estimator"]
    Core["共享控制核心<br/>full-task v2 + continuous-H<br/>运动学 + MPC"]
    Proposal["HardwareControlProposal"]
    Cert["future site certification<br/>现场 policy · ownership · 13-slot command"]
    Receipt["ExecutionReceipt<br/>仅代表 write / status 证据"]

    Observe -.-> Estimator -.-> Core
    Clock -.-> Core -.-> Proposal -.-> Cert
  end

  CommandSHM["protocol-v3 command / receipt"]
  subgraph CppFuture["C++ future output process · 尚不存在"]
    Adapter["复用 Stage-2 supervisor / formatter<br/>+现场验证 policy"]
    DDS["future Unitree command sink<br/>rt/arm_sdk"]
    Adapter -. "site gate 后才能新增" .-> DDS
  end
  Robot["真实 G1"]

  G1State -.-> Ingress -.-> StateSHM -.-> Observe
  Cert -. "command" .-> CommandSHM -.-> Adapter
  DDS -.-> Robot
  Adapter -. "status" .-> CommandSHM -.-> Receipt
```

仓库已有 protocol-v3、C++ supervisor/13-slot formatter、publisher-absent HIL
和 receipt，但**没有真实 Unitree command publisher target 或 launcher**。原
`dds_main.cpp` 已移除，`UNITREE_ARM_ADAPTER_BUILD_DDS=ON` 会在 CMake 阶段 fail
closed。这保证 Stage 2 只验证“如果到达 sink 边界会写什么”，而不是在仓库中
预留一个可被 CLI 误开的真机输出。

未来真机输出应复用已测 C++ supervisor/formatter，另外加入经现场验证的
policy/ownership 和独立 Unitree sink。当前 production policy 的验证/授权字段默认全为
false，数值 limit 未经 site verification 时 supervisor 必然 unarmable。
`ExecutionReceipt` 只能证明 command-sink 路径，不能证明实体电机执行了力矩。

上真机时，边界应保持如下：

| 继续保留的一份共享控制核心 | 由 Hardware adapter 替换或新增 |
| --- | --- |
| full-task protocol、绝对 task time、continuous-H | MuJoCo state、`d.ctrl`、`mj_step()` |
| full-task template v2 与 predictor | `RightArmSimProcess`、simulation C++ worker |
| 运动学/模型后端接口、`ArmMPCPolicy` | MuJoCo DDQ-to-torque mapper 与仿真认证 |
| 右臂关节定义、`HardwareControlProposal` 接口 | G1 motor index/sign、状态/接触估计、真实 task epoch |
| 控制意图和 fail-closed 原则 | 13-slot command、hardware torque 合同、ownership、watchdog、急停和 receipt |

MuJoCo 的 `max_abs_qacc=10 rad/s²` 不能默认照搬成真机 hard-stop；真机必须根据硬件
证据分别定义 hard-stop、soft guard 和 diagnostics。

## 当前验证边界

| 路径 | 当前状态 |
| --- | --- |
| Simulation：full-task v2 + continuous-H + 24 ms handoff + process | **simulation-validated**；仅限冻结模型、任务和受控运行环境 |
| H1 state inspection | 代码与离线合同已就绪；无真实 G1 样本，**PARTIAL** |
| 完整只读 shadow | H3-offline full-task proposal replay 已通过；真实 G1 launcher 仍受现场配置 gate 阻止且保持 legacy 兼容，**hardware-unverified** |
| Publisher-absent HIL | protocol-v3、C++ supervisor、2/6 ms hold、fake command sink 和 receipt 已离线实现；DDS/hardware write 固定为 0 |
| Future hardware output | 真实 publisher target 已移除/禁止构建；**未集成、未授权、hardware-unverified** |

## 架构与代码同步约定

`ARCHITECTURE.md` 是本仓库的运行时架构基准。以下任一变化都必须在**同一次代码
变更**中同步更新这里的图和状态说明：

- 启动入口、独立进程/worker 数量或 CPU/生命周期关系；
- IPC、状态来源、控制命令方向或谁拥有权威时间/物理状态；
- shared control core 与 Simulation/Shadow/Hardware adapter 的边界；
- 某条路径从 disabled、read-only 或 site-gated 变为可执行。

反过来，文档中的目标架构只有在对应代码、测试和验证 gate 落地后，才能从虚线
改成实线。评审运行结构相关改动时，至少同时核对实际 launcher、进程创建代码、
IPC 协议和输出许可；仅修改图或仅修改代码都视为未完成。

## 详细设计从哪里继续读

- 固定任务、模板与 H：[`FULL_TASK_TEMPLATE.md`](FULL_TASK_TEMPLATE.md)
- MPC 数学：[`MPC_DESIGN.md`](MPC_DESIGN.md)
- Simulation process 与计时：[`REALTIME_RUNTIME.md`](REALTIME_RUNTIME.md)、
  [`right_arm_runtime/README.md`](right_arm_runtime/README.md)
- Shadow 操作边界：[`HARDWARE_SHADOW.md`](HARDWARE_SHADOW.md)
- 真机接口和阶段 gate：[`HARDWARE_INTEGRATION_PLAN.md`](HARDWARE_INTEGRATION_PLAN.md)
- 离线准备和现场待验项：[`HARDWARE_OFFLINE_PREPARATION.md`](HARDWARE_OFFLINE_PREPARATION.md)
