# Unitree G1 硬件 Shadow

状态：**只读、hardware-unverified**。

该路径用于核对 Unitree LowState 接入、坐标/索引契约、共享 MPC 计算和 timing；
它被结构性地禁止发布控制命令。当前 shadow 仍只兼容旧 legacy phase template，
**不支持最终 full-task template v2，也没有 24 ms startup-PD handoff**。因此它
不是最终仿真方案的真机迁移，更不是已经完成的真机闭环。

当前冻结仿真、实时环境和平台边界分别见
[PRE_HARDWARE_FREEZE.md](PRE_HARDWARE_FREEZE.md)、
[REALTIME_RUNTIME.md](REALTIME_RUNTIME.md) 和
[ARCHITECTURE.md](ARCHITECTURE.md)。两部分共用接口、Unitree 官方实现核对和
H0-H3/O0-O4 阶段门见
[HARDWARE_INTEGRATION_PLAN.md](HARDWARE_INTEGRATION_PLAN.md)。

## 安全边界

只读 shadow 有两层相互独立的输出屏障：

1. `unitree_arm_state_bridge` 只订阅 `rt/lowstate` 和
   `rt/secondary_imu`。该 binary 的编译单元没有 LowCmd、command topic 或
   publisher；LowState 先按 Unitree 官方 SDK2 算法校验 CRC，坏包计数并拒绝
   写入 shared memory；
2. `run_hardware_shadow.py` 以只读文件描述符和 private copy-on-write mapping
   打开 POSIX shared memory。第一次 inspection 在原始状态证据处停止；以后若启用
   完整 controller，生成的 `ShadowArmCommand` 仍固定满足 `arm_weight=0`、
   `tau_ff=0`、`request_output=false`、`publish_performed=false` 和
   `ready_for_output=false`。

output-capable 的 `unitree_arm_adapter_dds` 是未来硬件输出适配器，不属于 shadow
调用链。本阶段不要运行 `unitree_arm_adapter_dds --enable-output`。

```text
rt/lowstate + rt/secondary_imu
    -> C++ state-only bridge（DDS receive，无 publisher）
    -> protocol-v2 state slot
    -> read-only Python state source
       |-> H0/H1: strict raw inspection + JSONL evidence（本轮在这里停止）
       `-> H2/H3 after contract verification:
           legacy phase template -> shared MPC -> in-memory proposal only
```

这条链不使用仿真的 `RightArmSimProcess`，也不使用 MuJoCo
DDQ-to-torque mapper：后者依赖 MuJoCo 的接触求解状态，不能冒充真机 forward
dynamics certification。shadow 中 `ddq_des` 只进入 command proposal，
`tau_ff` 保持零。

## 当前硬件状态契约

当前只支持仓库中的 `g1_23dof_rev_1_0` arm5 映射：

- motors 0..11：左腿 6 个关节、右腿 6 个关节；
- motor 12：waist yaw；
- motors 15..19：左臂 arm5；
- motors 22..26：右臂 arm5；
- 13 维 Arm SDK 顺序：左臂 15..19、右臂 22..26、腰 12..14。

关节位置是 rad，速度是 rad/s，gyroscope 是 rad/s，accelerometer 是 m/s²。
freshness 使用 bridge 主机 `CLOCK_MONOTONIC`。映射关节检查 MJCF range；35 维
状态还检查 shape、有限性、速度/IMU 范围、电机温度、单调 sample ID、robot
tick、时间戳和 20 ms freshness。未列入白名单的 `mode_pr` 或
`mode_machine` 会 fail closed。

本地固定 SDK2 示例把 `rt/lowstate` 中的 IMU 视为 pelvis IMU，并从
`rt/secondary_imu` 取得 torso IMU。这只是软件选题依据，不是目标机器人契约
已经确认。state-only bridge 只在两个话题都有新样本且主机到达时间偏差不超过
5 ms 时发布配对状态；配对时间取较早到达时刻，让 freshness 覆盖两个来源。

之后的转换严格采用 `configs/g1_hardware_shadow.yaml`：quaternion 顺序 wxyz、
含义 W-from-IMU，gyro/specific force 表达在 IMU frame，并应用明确的
torso-from-IMU 固定旋转。代码不根据数值外观猜测坐标约定。

仓库配置有意把 joint map、robot tick 和 IMU verification flags 保持为
`false`，允许的 mode 列表也为空。所以在目标 G1/固件契约确认前，完整 shadow
controller 必然 fail closed；这不是需要绕开的报错。

## 构建 state-only bridge

当前 checkout 默认从 `/home/fjk/g1_ws/unitree_sdk2` 找 SDK2。state bridge 有独立
CMake 开关；构建时必须显式关闭 output-capable DDS：

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
cmake -S cpp/unitree_arm_adapter \
  -B /tmp/hold-my-beer-mpc-unitree-state-only-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUNITREE_ARM_ADAPTER_BUILD_DDS=OFF \
  -DUNITREE_ARM_ADAPTER_BUILD_STATE_BRIDGE=ON \
  -DUNITREE_SDK2_DIR=/home/fjk/g1_ws/unitree_sdk2
cmake --build /tmp/hold-my-beer-mpc-unitree-state-only-build \
  --parallel --target unitree_arm_state_bridge
```

该 build directory 中若出现 `unitree_arm_adapter_dds`，inspection launcher 会拒绝
运行。它还检查 unresolved libraries、command topic 字符串和
`ChannelPublisher` 符号。

## 第一次 session：只检查状态

必须人工选择连接机器人且已核对的有线网卡，不能猜测。机器人保持厂商现有安全
stand/locomotion mode；本项目不调用 motion switcher，不取得 arm ownership。使用
单用途入口：

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
./tools/realtime/run_hardware_state_inspection.sh YOUR_VERIFIED_G1_INTERFACE \
  --duration-s 10 \
  --inspect-samples 500 \
  --group first_real_g1_readonly
```

inspection 不要求 verification flags 为真，也不运行 predictor/MPC。它要求完整收到
指定数量的 fresh、unique、finite、timestamp/tick 单调 paired samples；否则 fail
closed。证据目录保存完整 35-slot `raw_state_trace.jsonl`、Python summary、bridge
log/summary、CRC/skew 计数和 repo/SDK/config/binary/NIC metadata。

2026-08-23 的首次连接尝试在 `enx6c1ff701509c` 上运行 10 s，结果为 LowState 0、
secondary IMU 0、paired state 0，collector 因 `0/500` fail closed。证据在
`evaluation/hardware_shadow/state_inspection/first_real_g1_readonly_20260823/`。
该接口当时位于普通 `192.168.31.0/24` LAN，没有证据证明已连接目标 G1；所以这不是
有效真实状态 session，verification flags 保持不变，完整 shadow 不得继续。

在改变任何 verification flag 前，必须用目标型号/固件的正式资料或受控测量
确认：

- 23-DOF arm5 型号和 motor indices；
- uint32 robot tick 的单调与 wrap 语义；
- quaternion 顺序和旋转方向；
- gyro/accelerometer 的坐标、单位和重力语义；
- IMU 到 torso 的旋转，以及 IMU 原点导致的杆臂项；
- 只读 locomotion 状态下合法的 `mode_pr` / `mode_machine`。

“数值看起来合理”不能证明坐标契约正确。`mode_machine=4` 是
`g1_23dof_rev_1_0` 的官方参考观察值，但在目标固件实测和人工确认前也不会自动
写入白名单。

### 无机器人时完成的 H2-prep

仓库现已提供 `right_arm_runtime/hardware_state_replay.py` 和
`tools/realtime/audit_hardware_state_trace.py`。它们对未来有效 H1 trace 做纯离线
schema、有限性、单调性、slots 22..26 mapping 及 bridge counter 审计。报告中的
`offline_trace_contract_passed=true` **不等于** hardware session verified；
`hardware_session_verified` 和 `verification_flags_modified` 固定为 false，并列出仍需
现场确认的 model/firmware、motor sign、tick/mode 和 torso IMU gates。

当前 H1 没有 state sample，无法用 synthetic fixture 替代。详细命令、O0/O1 fake
sink 和现场 gate 见
[HARDWARE_OFFLINE_PREPARATION.md](HARDWARE_OFFLINE_PREPARATION.md)。

## 完整只读 shadow 计算

只有契约字段已经核对并写入 YAML，且 PREEMPT_RT checker 通过后，才能运行：

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
./tools/realtime/run_hardware_shadow.sh YOUR_INTERFACE \
  --control-cpu 7 \
  --bridge-cpu 5 \
  --predictor template \
  --duration-s 30 \
  --group first_g1_readonly_shadow
```

`template` 是 CLI 唯一允许的 shadow predictor。它指旧 phase template，仅用于
只读计算路径兼容；它不是正式仿真的默认方案。launcher 把 state bridge 放在
housekeeping CPU 5，把 Python shadow controller 放在 CPU 7、`SCHED_RR/10`，
并设置六个数值库线程变量为 1。summary 报告完整 shadow path timing、state
age、QP diagnostics、command build count 和必须始终为零的 publish count。

## 与最终 full-task 方案的缺口

最终仿真使用 reset 后绝对 task time、continuous causal-H、24 ms startup-PD 和
full-task template v2；hardware shadow 目前没有：

- 经过验证的 task epoch、6.4 s direct-stop 时序和 gait clock 对齐；
- full-task v2 资产/manifest/checksum 的硬件侧加载合同；
- `[0,24 ms)` 固定 PD 到 anchor 4 MPC 的真机 handoff；
- startup 最后一拍真实执行力矩到 mapper 的 previous-torque 连续性；
- 真机 floating-base pose/twist/contact estimator；
- 可认证的真机 inverse-dynamics / torque mapping；
- 输出许可、watchdog、arm weight 过渡和急停的实机验证。

这些缺口必须作为后续硬件阶段单独设计和验收，不能通过在 shadow 内重放 template
时钟来宣称完成。直到那时，hardware shadow 始终是只读且
hardware-unverified。

## Future hardware output 的安全分类原则

该原则只记录后续设计边界，本阶段没有增加或授权任何 hardware output：

- 不默认把 MuJoCo mapper 的 `max_abs_qacc=10 rad/s^2` 原样继承为真机
  hard-stop。该值依赖仿真 forward dynamics、接触状态和当前冻结模型，不是已经由
  Unitree 电机、传动或整机试验确认的物理极限。
- **hard-stop** 应保留给经硬件依据确认、继续输出会造成不可接受风险的条件，例如
  非有限命令、通信/watchdog 失效、明确越过关节/力矩/温度硬限或失去状态可信度。
- **soft guard** 应覆盖可以受控降级、限幅、保持或退出主动控制的风险；阈值、持续
  时间和恢复条件必须通过吊架及逐级真机试验冻结。
- **diagnostics** 应保留单拍 qacc 峰值、dq、位置余量、torque/torque-rate、温度、
  接触与估计器置信度，支持事后关联，而不能把单一模型量当作全部安全证据。

尤其是“单拍轻微 qacc 超限是否立即终止主动控制”必须作为独立硬件安全问题评估；
在取得厂商限制、传感器质量、执行器/传动动态及重复吊架实验之前，既不能默认沿用
`10 rad/s^2` hard-stop，也不能据此放宽当前 MuJoCo 门限。
