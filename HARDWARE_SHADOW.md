# Unitree G1 硬件 Shadow 模式

状态：**硬件未验证（hardware-unverified）**。这是只读集成阶段，被明确设计为
无法发布机器人控制命令。话题选择、消息配对、关节映射、单位和坐标转换已经
实现并通过本地测试，但尚未在本仓库的目标 G1 真机及其固件上确认。

当前系统事实、仿真结果和分阶段真机 checklist 见
[PRE_HARDWARE_FREEZE.md](PRE_HARDWARE_FREEZE.md)；运行完整 shadow 前还必须满足
[REALTIME_RUNTIME.md](REALTIME_RUNTIME.md) 的目标环境要求。

## 安全边界

Shadow 路径具有两层相互独立的输出屏障：

1. `unitree_arm_state_bridge` 只订阅状态话题 `rt/lowstate` 和
   `rt/secondary_imu`。其源码不包含 LowCmd 类型、command topic 或 publisher。
2. `run_hardware_shadow.py` 用只读文件描述符和 private copy-on-write mapping
   打开 POSIX shared memory，没有 command sink。生成的 `ShadowArmCommand`
   始终满足 `arm_weight=0`、`tau_ff=0`、`request_output=false`、
   `publish_performed=false` 和 `ready_for_output=false`。

已有的 output-capable adapter 不属于任何 shadow 命令的调用链。本阶段不要运行
`unitree_arm_adapter_dds --enable-output`。

```text
rt/lowstate + rt/secondary_imu（torso）
    -> 仅状态 C++ DDS bridge，主机到达时间配对偏差 <= 5 ms
    -> protocol-v2 state slot
    -> 只读 Python state source
    -> 单位 / 索引 / 坐标系 / 时间戳检查
    -> template 或 hybrid predictor（hybrid 可回退 template）
    -> 现有 KinematicsHelper + MPC
    -> 内存中的 ShadowArmCommand
    -> 仅输出 JSON timing/diagnostics（无 sink、无 DDS publish）
```

## 已实现的硬件契约

当前只支持仓库中的 `g1_23dof_rev_1_0` arm5 模型映射：

- motors 0..11：左腿 6 个关节，然后右腿 6 个关节；
- motor 12：waist yaw；
- motors 15..19：左臂 arm5；
- motors 22..26：右臂 arm5；
- Arm SDK command 顺序：左臂 15..19、右臂 22..26、waist 12..14。

关节位置单位为 rad，速度为 rad/s，gyroscope 为 rad/s，accelerometer 为
m/s^2。所有状态 freshness 检查都使用 bridge 主机的 monotonic clock。每个已
映射关节都要检查 MJCF range；全部 35 维状态还要检查 shape、有限性、声明的
速度/IMU 范围、电机温度、单调 sample ID/时间戳和 20 ms freshness。bridge
时间戳使用主机到达时间；独立的原始 robot tick 还必须单调前进，并允许 uint32
wrap。任何未列入允许集合的 `mode_pr` 或 `mode_machine` 都会立即报错。

本地固定版本的 Unitree SDK2 G1 示例把 `rt/lowstate` 内的 IMU 视为 pelvis
IMU，并另以 `rt/secondary_imu` 提供 torso IMU。这只是选择话题的软件依据，
不是目标机器人硬件契约已经确认的证据。state-only bridge 使用后者，并且只有
两个话题都提供了新样本，且主机到达时间偏差不超过 5 ms 时，才发布一个配对
状态。配对时间戳取两个到达时刻中较早者，因此 freshness 同时覆盖两个来源。

之后的 IMU 转换严格实现 `configs/g1_hardware_shadow.yaml` 中声明的唯一约定：
quaternion 为 wxyz 顺序、含义为 W-from-IMU，gyro/specific force 表达在 IMU
坐标系，并显式应用 torso-from-IMU 固定旋转。检查后的 torso 姿态、线加速度、
角速度和因果角加速度进入与仿真相同的 H-frame predictor 路径。代码绝不会根据
原始数值“猜测”坐标约定。

仓库内配置有意把 joint-map、robot-tick 和 IMU verification flag 保持为
`false`，允许的 mode 列表也保持为空。因此在目标真机契约确认前，完整 shadow
control 必然 fail closed。

## 只构建 state bridge

当前本地 checkout 预期 Unitree SDK2 位于 `/home/fjk/g1_ws/unitree_sdk2`。
只配置并构建 state-only target：

```bash
cd /home/fjk/g1_ws/disturbance-lab

cmake -S cpp/unitree_arm_adapter \
  -B /tmp/hold-my-beer-mpc-unitree-arm-adapter-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUNITREE_ARM_ADAPTER_BUILD_DDS=ON \
  -DUNITREE_SDK2_DIR=/home/fjk/g1_ws/unitree_sdk2

cmake --build /tmp/hold-my-beer-mpc-unitree-arm-adapter-build \
  --parallel --target unitree_arm_state_bridge
```

## 第一次真机 session：只检查状态

必须明确选择连接机器人的有线网卡，不能猜测。在 terminal 1 运行：

```bash
taskset -c 5 \
  /tmp/hold-my-beer-mpc-unitree-arm-adapter-build/unitree_arm_state_bridge \
  YOUR_INTERFACE \
  --shm-name /g1_arm_mpc_shadow \
  --max-source-skew-us 5000 \
  --unlink-on-exit
```

在 terminal 2 运行：

```bash
cd /home/fjk/g1_ws/disturbance-lab
MPLCONFIGDIR=/tmp/disturbance-lab-matplotlib \
  /home/fjk/miniforge3/envs/g1_mpc/bin/python run_hardware_shadow.py \
  --inspect-state-only \
  --shared-memory /g1_arm_mpc_shadow \
  --inspect-samples 500 \
  --duration-s 10
```

inspection 模式不要求 verification flag 为 true。它报告原始 mode、quaternion
norm、state age、IMU 数据以及右臂 q/dq；它不运行 MPC，也不能写 command slot。
用 Ctrl-C 停止 bridge；`--unlink-on-exit` 只删除该进程创建的临时 shared-memory
名称。

在修改任何 verification flag 前，必须根据精确机器人/固件文档或 Unitree
支持的接口确认：

- 机器人确为 23-DOF arm5 版本，且 motor indices 一致；
- 目标固件的 `tick` 单调增加，并按 uint32 wrap；
- LowState quaternion 顺序以及 W/body 旋转方向；
- gyro 坐标系和单位；
- accelerometer 表示 specific force 还是已经移除重力的 linear acceleration；
- IMU 到模型 torso 的固定旋转；
- 物理 IMU 原点是否与 MJCF `imu_in_torso` site 一致；若不一致，需要实测平移
  和杆臂加速度修正；
- 预期只读 locomotion 状态下允许的 `mode_pr` 和 `mode_machine`。

观测值看起来合理，不能证明坐标系契约正确。

## 完整 target-runtime shadow 运行

只有在上述字段已经写入 YAML 并得到验证后，才能先检查 PREEMPT_RT 环境，再用
一条命令运行完整 state-to-command-build 路径：

```bash
cd /home/fjk/g1_ws/disturbance-lab
./tools/realtime/run_hardware_shadow.sh YOUR_INTERFACE \
  --control-cpu 7 \
  --bridge-cpu 5 \
  --predictor template \
  --duration-s 30 \
  --group first_g1_readonly_shadow
```

控制进程使用现有 target runtime gate、CPU 7 affinity、`SCHED_RR/10` 和
单线程数值库。DDS receive threads 留在 housekeeping CPU 5。summary 包含完整
路径 mean/p95/p99/max、各阶段 timing、state age、QP success、predictor
diagnostics、command build count、source-to-command age，以及必须始终为零的
command publish count。

## LowState 尚未提供的输入

LowState 不包含 MLP 训练时使用的 lower-body policy target、runtime walking
command 或 gait phase。`LocomotionContext` 已定义所需的 12 + 3 + 2 个值，并
执行独立的 monotonic timestamp/freshness 检查，但本仓库目前不知道真实下肢
控制器的 transport 或 schema。当前 runner 不提供这些上下文，因此显式选择
`hybrid_residual` 时会回退到 template，并记录原因。

相位模板还需要经过验证的 gait epoch/phase 关系，才能进行有意义的行走比较。
在该信号接入前，内部时钟只适合验证 shadow 计算路径，不能作为真机相位对齐
预测质量的证据。

最后，仅靠 LowState 还缺少经过验证的 floating-base pose/twist/contact estimator，
不足以支持硬件 inverse dynamics。Shadow mode 不会虚构 feedforward torque：
`tau_ff` 始终为零，command 也明确保持 not ready for output。
