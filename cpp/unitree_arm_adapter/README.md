# G1 右臂 C++ 高频适配器

这个目录是**独立进程、默认不发命令**的 C++ 真机运行时适配器，用于把较低频的 Python MPC 与 2 ms 状态/命令循环隔开。Python 端已由 `right_arm_runtime/unitree_shm.py` 按 protocol v2 对接；主仿真不走 DDS，而是通过 `libright_arm_executor.so` 验证相同的 PD、限幅和超时语义。

## 文件职责

- **核心**：`protocol.hpp`、`seqlock.hpp`、`safety.cpp` 和 `periodic_loop.cpp`，分别定义跨进程协议、无锁一致快照、唯一发布前安全入口，以及基于 `CLOCK_MONOTONIC` 和绝对时间的 2 ms 周期。
- **半核心**：`shared_memory.cpp` 和 `dds_main.cpp`，负责 POSIX 共享内存以及 Unitree SDK2 的 `rt/lowstate`/`rt/arm_sdk` 适配。
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

## 构建与验证

不依赖 Unitree SDK 的核心构建和测试：

```bash
chmod +x cpp/unitree_arm_adapter/build_and_test.sh
cpp/unitree_arm_adapter/build_and_test.sh
```

可选构建 DDS 适配器：

```bash
UNITREE_ARM_BUILD_DDS=ON \
UNITREE_SDK2_DIR=/home/fjk/g1_ws/unitree_sdk2 \
cpp/unitree_arm_adapter/build_and_test.sh
```

纯本地 2 ms 干运行（永远不访问 DDS）：

```bash
/tmp/hold-my-beer-mpc-unitree-arm-adapter-build/unitree_arm_adapter_dry_run \
  --iterations 2000 --synthetic-input
```

DDS 只读干运行（订阅状态，但不发布命令）：

```bash
/tmp/hold-my-beer-mpc-unitree-arm-adapter-build/unitree_arm_adapter_dds enp3s0
```

只有完成索引、13 维参考、weight 过渡、超时回退和急停验证后，才允许显式执行：

```bash
/tmp/hold-my-beer-mpc-unitree-arm-adapter-build/unitree_arm_adapter_dds \
  enp3s0 --enable-output
```

不要把最后一条命令当作当前可直接上真机的启动说明。
