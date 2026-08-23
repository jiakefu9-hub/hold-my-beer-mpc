# G1 右臂 C++ 高频适配器

状态：**hardware-unverified；当前项目只授权 state-only shadow，不授权主动输出**。

这个目录是独立进程、默认不发命令的 C++ 硬件适配器，用于把共享 Python
控制核心与 Unitree 2 ms 状态/命令循环隔开。Python 端由
`right_arm_runtime/unitree_shm.py` 按 protocol v2 对接；正式 MuJoCo 仿真不走
DDS，而是使用独立的 `cpp/right_arm_sim_runtime`。

最终仿真方案的 full-task template v2、continuous-H 和 24 ms startup-PD handoff
**尚未接入这里**。不要把本目录的 dry-run、DDS 订阅测试或只读 shadow 写成
最终控制器已完成真机迁移。

## 文件职责

- **核心**：`protocol.hpp`、`seqlock.hpp`、`safety.cpp` 和 `periodic_loop.cpp`，分别定义跨进程协议、无锁一致快照、唯一发布前安全入口，以及基于 `CLOCK_MONOTONIC` 和绝对时间的 2 ms 周期。
- **半核心**：`shared_memory.cpp` 和 `dds_main.cpp`，负责 POSIX 共享内存以及 Unitree SDK2 的 `rt/lowstate`/`rt/arm_sdk` 适配。`state_bridge_main.cpp` 是独立的只读硬件入口，只订阅 LowState 与 secondary torso IMU，先验证 LowState CRC，编译单元中没有命令消息或 publisher。
- **非核心**：`dry_run_main.cpp`、`core_test.cpp` 和 `build_and_test.sh`，只用于布局检查、故障注入、计时和构建验证。

## 控制链和双重 PD 约束

共享命令同时携带 `q_ref`、`dq_ref`、`ddq_des`、`kp`、`kd` 和 `tau`，但当前有且只有两种互斥的执行语义：

1. `kRobotPdPlusFeedforward`：机器人底层根据 `q_ref/dq_ref/kp/kd` 计算 PD，`tau` **只能是纯前馈力矩**。C++ 适配器不再计算一次 PD。
2. `kDirectTorque`：上游给出的 `tau` 已经是最终合成力矩。适配器会把发往机器人的 `kp/kd` 强制清零，并把 `q` 设为当前关节角，避免双重 PD。

`ddq_des` 现在只作为明确的协议字段保留。适配器**不会**在缺少完整 floating-base 状态时擅自从 `ddq_des` 计算 RNEA 力矩；当前真机路径应由上层提供经过验证的 `tau_ff` 或 `tau_cmd`。

## Unitree 索引和 arm weight

顺序严格来自相邻 Unitree SDK2 的 G1 arm5 官方示例：

```text
13维arm_sdk顺序 = 左臂[15..19] + 右臂[22..26] + 腰[12..14]
右臂局部位置    = 5..9
arm weight      = motor_cmd[29].q
```

本项目把右臂 22..26 解释为 shoulder pitch、shoulder roll、shoulder yaw、elbow pitch、wrist roll；官方示例把索引 26 命名为 `RightElbowRoll`。这处命名差异必须在真机启用输出前根据具体 G1 型号和 URDF 再核对，不能只凭项目名称假设。

`arm weight` 是官方 arm SDK 的全局混合量，不是只属于右臂的五维开关。因此协议保留了完整 13 维命令；上游必须同时给出左臂和腰部的安全参考，不能把未控制的八维留成默认零后直接把 weight 拉高。weight 的切换规律和失效后的机器人行为也必须先在吊架/急停条件下验证。

## 共享内存协议

默认 POSIX 名称是 `/g1_arm_mpc`。一个固定布局内含三个 seqlock 槽：

- `command`：Python/上层写，C++ 2 ms 循环读；
- `state`：Unitree DDS 回调写，Python/上层读；除运动状态和 IMU 外，还原样保存 35 个电机的机壳/绕组温度；
- `status`：C++ 写，包括数据年龄、唤醒迟到、执行耗时、deadline 次数、过温拍数和安全模式。

时间戳必须来自同一台电脑的 `CLOCK_MONOTONIC`；Python 对应 `time.monotonic_ns()`。seqlock 保证读者只接受同一偶数版本前后的完整 POD 快照，不依赖 Python 锁。每个槽按单写者设计，不允许两个进程同时写同一个槽。

Python 对接前先运行：

```bash
/tmp/hold-my-beer-mpc-unitree-arm-adapter-build/unitree_arm_adapter_dry_run --print-layout
```

Python 端必须逐字段复现版本、大小、偏移和 64 字节对齐；版本或布局不一致时 C++ 会拒绝覆盖已有共享内存。

## 安全闸门

默认配置包括：命令超过 30 ms、状态超过 20 ms、NaN/Inf、异常四元数、未来时间戳、无逐拍授权、无效控制模式或错过完整 2 ms deadline 时，进入 `weight=0, kp=kd=tau=0` 的安全释放计划。正常命令还会经过关节角、速度、增益、力矩和 weight 限幅。

过热保护单独检查 arm SDK 接管的 13 个关节：Unitree `temperature[0]` 是机壳温度，默认硬上限 85 °C；`temperature[1]` 是绕组温度，默认硬上限 120 °C。这两个默认值与 SDK2 的 `unitree/robot/g1/common/terminations.hpp` 一致。任一受控关节超过上限即进入 `kSafeReleaseOvertemperature`，状态保存全部原始温度，`overtemperature_count` 按 2 ms 过温控制拍累加。腿部温度不会由右臂适配器处理，仍应由机器人全身安全层负责。

当前实现采用简单硬上限，没有恢复回滞：温度回到上限以内后才重新允许新鲜且有效的上游命令。真机前应根据具体电机型号确认阈值，并评估是否增加低于硬上限的降额区和带回滞的人工复位锁存；不要把这里的软件释放替代机器人固件自身的过温保护。

真正的输出有两层许可：

1. 进程必须显式带 `--enable-output`，否则甚至不会创建 `rt/arm_sdk` publisher；
2. 活跃控制命令必须逐拍带 `kCommandRequestOutput`。若第一层已打开而第二层失效，进程只发布 `weight=0` 的释放帧，不会发布活跃控制。

安全释放不是已经通过真机认证的急停。weight 突然归零可能把控制权交回机器人内部控制器，其物理过渡必须在吊架、硬件急停和低 weight 条件下验证。普通 Linux 内核上的 2 ms 绝对周期也不等于硬实时保证；后续仍需评估 CPU 绑核、优先级、内存锁定和最坏时延。

## LowState 与 C++ RNEA 的边界

当前 `rt/lowstate` 能直接提供 35 个电机的 `q/dq/ddq/tau_est`，以及 IMU 四元数、角速度、线加速度和 RPY；它**没有直接提供** walking floating base 的世界系位置和线速度，也没有完整接触集合、足底外力或状态估计协方差。

因此，真机上的 floating-base Pinocchio RNEA 还需要一个经过验证的 full-state estimator/接触估计接口，至少明确 base pose、twist、acceleration、接触和外力的坐标系与时间戳。在这些输入齐全前，本目录不声称已经安全实现“2 ms RNEA”；当前 2 ms 部分只负责最新状态快照、已计算命令的限幅/超时/NaN/deadline 保护和 DDS 发送。

同理，MuJoCo DDQ-to-torque mapper 的 forward-dynamics certification 不能复制到
这里：它依赖仿真接触求解状态。未来 hardware output path 必须先定义并验证真实
状态估计、inverse dynamics、torque contract 和 fail-closed 接口；本任务没有
增加这些能力。

### 真机 qacc 门限不是 MuJoCo 常量的复制

未来主动输出不得默认把 MuJoCo 的 `max_abs_qacc=10 rad/s^2` 直接设为真机
hard-stop。仿真值来自当前模型和 forward-dynamics 验收，不是 Unitree 硬件物理
极限。硬件合同必须分别定义 hard-stop、soft guard 和 diagnostics，并联合检查
`q/dq`、位置余量、torque/torque-rate、温度、通信/watchdog、接触与状态估计可信度。
单拍轻微 qacc 超限是否应立即终止主动控制，需要厂商限制和吊架分级试验支持；
本 adapter 当前没有实现或验证这项策略，也没有因此获得输出许可。

## 构建与验证

不依赖 Unitree SDK 的核心构建和测试：

```bash
chmod +x cpp/unitree_arm_adapter/build_and_test.sh
cpp/unitree_arm_adapter/build_and_test.sh
```

只构建 state-only bridge（第一次真实 session 的批准方式）：

```bash
UNITREE_ARM_BUILD_DDS=OFF \
UNITREE_ARM_BUILD_STATE_BRIDGE=ON \
UNITREE_ARM_ADAPTER_BUILD_DIR=/tmp/hold-my-beer-mpc-unitree-state-only-build \
UNITREE_SDK2_DIR=/home/fjk/g1_ws/unitree_sdk2 \
cpp/unitree_arm_adapter/build_and_test.sh
```

output-capable DDS 只允许为离线设计审查单独构建，不能与第一次 state-only session
共用 build directory，也不属于本轮批准的运行路径。

纯本地 2 ms 干运行（永远不访问 DDS）：

```bash
/tmp/hold-my-beer-mpc-unitree-arm-adapter-build/unitree_arm_adapter_dry_run \
  --iterations 2000 --synthetic-input
```

需要观察 wall-clock 长尾时，可先预热，再把逐拍原始计时写到 CSV：

```bash
/tmp/hold-my-beer-mpc-unitree-arm-adapter-build/unitree_arm_adapter_dry_run \
  --warmup-iterations 1000 --iterations 30000 --synthetic-input \
  --csv /tmp/unitree_arm_2ms_timing.csv
```

`iterations` 只表示正式统计拍数，预热拍不进入汇总和 CSV。CSV 在循环
结束后统一写盘，不把文件 I/O 混进 2 ms 热路径。`work_time` 从线程被唤醒
后开始，到一次 status seqlock 写完为止；`completion_lateness` 是完成时刻
减去本拍 deadline，负值表示仍有余量；`period_jitter` 是相邻实际启动间隔
相对 2 ms 的绝对偏差。`deadline_miss_event` 表示该拍是否完成超时，
`skipped_periods` 则记录唤醒时已经跨过的完整周期，二者不能混为同一统计。
这个干运行不含 DDS 传输，也尚不含缺少 floating-base 输入的 C++ RNEA，
因此只能评价当前共享内存、安全计划和周期调度路径，不能直接等同于真机
端到端延迟。

DDS 只读干运行（订阅状态，但不发布命令）：

```bash
/tmp/hold-my-beer-mpc-unitree-arm-adapter-build/unitree_arm_adapter_dds enp3s0
```

硬件 shadow 阶段优先使用更强隔离的 state-only binary；它没有任何可开启
输出的参数：

```bash
/tmp/hold-my-beer-mpc-unitree-state-only-build/unitree_arm_state_bridge \
  enp3s0 --shm-name /g1_arm_mpc_shadow --unlink-on-exit
```

完整只读检查和 PREEMPT_RT shadow 运行步骤见仓库根目录的
[HARDWARE_SHADOW.md](../../HARDWARE_SHADOW.md)。

只有完成索引、13 维参考、weight 过渡、超时回退和急停验证后，才允许显式执行：

```bash
/tmp/hold-my-beer-mpc-unitree-arm-adapter-build/unitree_arm_adapter_dds \
  enp3s0 --enable-output
```

不要把最后一条命令当作当前可直接上真机的启动说明。

当前正式边界是：

- `unitree_arm_state_bridge`：可用于只读状态检查，没有 publisher；
- `run_hardware_shadow.py`：只读 state-to-MPC-to-command-build，publish count
  必须为零，且只兼容 legacy phase template；
- `unitree_arm_adapter_dds --enable-output`：未来输出路径，当前未获授权、未完成
  full-task v2 + 24 ms 集成、未经过真机验收。

完整说明见 [HARDWARE_SHADOW.md](../../HARDWARE_SHADOW.md)，统一接口和分阶段
output 设计见 [HARDWARE_INTEGRATION_PLAN.md](../../HARDWARE_INTEGRATION_PLAN.md)。
