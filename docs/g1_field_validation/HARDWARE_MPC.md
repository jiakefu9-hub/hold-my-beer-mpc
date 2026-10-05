# G1 真机力矩 MPC：当前程序与首次运行

对应 [`g1_walk_mpc.py`](../../tools/g1_commissioning/g1_walk_mpc.py)，更新于 2026-10-05。
已实现显式开启的受控首次力矩试验入口；**还没有 MPC 真机效果或硬实时验收结果**。
本轮开发未向机器人发命令。6 ms PID 已通过现场功能复验并冻结，见
[PID 基线](sessions/20261005_PID_6MS_FIELD.md)。

## 1. 控制路线

腿部用内置运控，腰 yaw=0。左臂固定 PD，名义五轴 `[-4,-1,0,-8.1,0]°`；
右臂 MPC，名义五轴 `[-4,+1,0,-7.8,0]°`，左右瓶各 250 g。
只发 `rt/arm_sdk` 和已有的 Loco 速度请求，不使用 `rt/lowcmd`，不自动切模式或进入 debug。

右臂：实测 q/dq → 结合已发命令估计假定执行时刻状态 → 九段、每段 **6 ms** 的加速度 MPC →
逆动力学名义力矩 → 局部修正、多候选比较及正动力学检查 → 力矩前馈＋PD。
发送前从选中总力矩减去已计入的 PD，同时发一拍 q/dq 参考和 `kp=20,kd=1`，固件只加一次 PD。
这不是旧版纯位置参考控制，也不是从持续积累的命令参考而不看实测状态出发。

局部模型对力矩是精确仿射关系，因此使用质量矩阵逆替代重复数值扰动、批量检查候选；
没有删除候选、缩短预测窗口或取消最终约束检查。QP 与仿真的代价、运动学主线一致。
延迟补偿中的 2 ms 小步名义动力学循环移至 `cpp/g1_arm_delay`，保持原模型、步长和已发命令历史；
80 组变化身体运动／部分权重／命令切换与 Python 原实现逐项对照通过。库加载检查源码哈希和
MuJoCo 头文件／运行库版本；缺库或旧库在真实输出前拒绝，不静默切回较慢实现。
静止任务使用当前身体运动估计，行走任务才查询冻结的状态索引扰动库。
H0 在每轮走前 3–5 秒平均 yaw 后固定，不跟随身体转动。库、滤波和预测时间轴未缩放或重新拟合。

真实力矩增益、摩擦、惯量误差、腿部反作用及电机内部延迟未标定。
默认 **6 ms 执行延迟是假设**（含计算与后续应用），不是测出了 DDS 往返时间。
主机接收时间也不是传感器内部采样时间；所有假设和原始测量分别入日志。

## 2. PID 结果与首次力矩边界

PID 平顺、停车和退权正常；完整 `[5,18)` 有 0.934% 超时，不称为硬实时通过。
复用其通信／记录链路，不改 PID 增益或 governor。实测角有静差，抬臂结束时右肩接近 +5°；
MPC 入口必须检查实测姿态，不用目标角冒充实测角，也不为避免拒绝而放宽旧模型的关节外层边界。

[`hardware_mpc_torque_field.yaml`](../../configs/hardware_mpc_torque_field.yaml) 将右臂五轴总力矩估计／
前馈绝对上限设为 `[5,3,2,5,1.5] N·m`，不用旧仿真的统一 ±25 N·m。
这是工程试验限值，**不是厂家额定值或由 tau_est 完成的标定**。
保留 q/dq/加速度约束，加入已离线研究的 q+dq/rate 制动边界；
没有新增“手腕实测速度瞬时达到某值便停”的独立规则。

## 3. 停止与退权

正常 18 秒、Ctrl-C 和可处理计算故障时，先请求零行走速度，再至少三秒退权。
冻结最后成功发出的 q/kp/kd；把 `kd*dq_ref` 转入前馈后令 dq_ref=0，保持完整 PD＋前馈关系连续。
退权全程保留支撑前馈，weight=0 最后一帧才清零 tau；故障路径不再求解 MPC。
正常、计算故障、主动停止和遥控打断已用无网络测试检查，不等于实机物理交还已验证。

CRC、FSM、遥控 L2+B、状态失效和 DDS 写失败保护保留。发现模式切换／失联／写失败后不继续
发渐退序列，避免与阻尼控制冲突。断网、强杀进程不能保证交还；异常仍用本机已验证的现场停止方式。
力矩写前要求所用状态不超过 25 ms、计算不超过 10 ms；这是异常拒绝边界，**不是改成 10 ms 周期**。

## 4. CPU 与实时调度

本机已有 PREEMPT_RT 和隔离核 6–7；旧 PID/早期 MPC 在 CPU 2、SCHED_OTHER 运行。
新版可把控制线程放 CPU 7，DDS、RPC、日志线程避开 SMT 同核 6–7，仅控制线程可选 FIFO 20。
隔离不等于没有所有 IRQ。最终两轮离线整链平均 **5.079／4.963 ms**、P99 **5.940／5.438 ms**，
仍有 **1.168%／0.324%** 超时；暂保留 6 ms，不称为硬实时通过，也未包含真实 DDS 负载。
完整计时证据见 [本轮记录](sessions/20261005_MPC_FIELD_PREP.md)。
10-05 操作者已完成 governor/FIFO 设置并跑过两轮；旧版本仍有 13.6%／22.6% 超时，不能据此放行。
随后又减少计算开销，并将整机电源档由 balanced 切到 performance。最终结果以本轮记录为准。
在**准备运行的同一个本机终端**执行（密码只在你自己的终端输入）：

```bash
sudo prlimit --pid $$ --rtprio=40:40
sudo cpupower -c 6-7 frequency-set -g performance
powerprofilesctl set performance
```

第一条只给该终端及其子进程 RT 权限，不改永久 PAM 配置；程序结束恢复自己的原调度。
第二条改变 governor，本轮原值 powersave；结束后可用
`sudo cpupower -c 6-7 frequency-set -g powersave` 恢复。其他电脑先核对 CPU 拓扑。
整机电源档可用 `powerprofilesctl set balanced` 恢复；性能档可能增加功耗、温度和风扇转速。
不要对整个 Python 进程套 `chrt`，以免工作线程也继承实时优先级。

## 5. 首次运行

机器人已双脚着地、FSM 500 自主平衡且静止，没有其他用户程序接管手臂。先原地，不自动行走。
复用已确认 PID profile 的机器人／网络事实，新的显式选项表示此次力矩试验选择，不伪造力矩验收。

本机依赖已安装、本地库已构建。其他克隆或更新本地 C++ 后，先用同一个 `g1_mpc` 环境构建
（不连接机器人，不构建任何 command publisher）：

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
cmake -S cpp/g1_arm_delay -B build/g1_arm_delay -DCMAKE_BUILD_TYPE=Release \
  -DMUJOCO_ROOT=/home/fjk/miniforge3/envs/g1_mpc/lib/python3.10/site-packages/mujoco
cmake --build build/g1_arm_delay -j2
```

不连接机器人的依赖／构包预检（会检查本地库，但不代表通过 6 ms 现场计时）：

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/g1_commissioning/g1_walk_mpc.py \
  --preflight --cpu 2 --torque-config configs/hardware_mpc_torque_field.yaml
```

完成上节主机设置后，以下是真实输出命令；程序仍先只读检查，再要求输入 `EXECUTE <robot_id>`：

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
MPC_OUT="evaluation/hardware_shadow/commissioning/mpc_torque_stationary_$(date +%Y%m%d_%H%M%S)"
taskset -c 0-17 /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/g1_walk_mpc.py enx6c1ff701509c \
  --execute --task stationary --cpu 7 --rt-priority 20 \
  --profile evaluation/hardware_shadow/commissioning/g1_pid_6ms_20261005_132052/arm_profile.conf \
  --output-dir "$MPC_OUT" --pid-6ms-validated \
  --permit-real-output MPC_WALK_H0_CAPTURE --allow-first-torque-field-trial \
  --assumed-command-delay-ms 6
```

0–3 s 渐接管抬臂，3–18 s 右臂 MPC／左臂固定，18 s 后停车回复等待和至少三秒退权。
原地任务前进／转向请求始终为零。现场平顺、日志及交还正常后，另一次才改 `--task walk`、
使用新目录并加 `--torque-stationary-validated`。行走任务 5–15 s 请求 0.5 m/s，15–18 s 停车及航向保持；
速度×时间不是物理距离限位。原地试验未通过时，不通过删保护、改位置参考版或直接行走绕行。

## 6. 运行后看什么

保存 raw.jsonl、实际 float32 命令、q/dq、IMU、tau_est、原始 ddq、预测、候选、时间及配置／源码哈希。
主指标是完整 `[5,18)`；原地任务同样取此窗口，但不称为行走效果。

```bash
python tools/g1_commissioning/analyze_mpc_execution.py "$MPC_OUT/raw.jsonl" \
  --output-dir "$MPC_OUT/execution_analysis"
python tools/g1_commissioning/analyze_hardware_mpc.py "$MPC_OUT/raw.jsonl" \
  --output-dir "$MPC_OUT/endpoint_analysis"
```

先看完整接管／退权、故障、日志覆盖、实际周期和状态年龄；再看总力矩目标与之后的 tau_est、
实测 dq 差分加速度与期望／模型，以及瓶子姿态与加速度。不能把发送前状态当本条命令响应，
或把 tau_est 当独立测力计。对齐规则见 [执行效果分析](HARDWARE_MPC_EXECUTION.md)。

## 7. 真要增加周期时必须同步修改

当前仍为 6 ms。先完成 FIFO／性能模式与实际 DDS 负载计时，没有把普通调度未优化当成必须降频。
若最终必须调整，要同时检查：积分矩阵／一拍 q/dq、QP horizon 与物理预测长度、制动约束、
力矩 slew、weight 步长、命令历史／执行时刻、扰动节点和**区间均值**重采样、创新衰减时间、
滤波物理时间常数、时钟槽、profile、日志及分析器。不能仅改 sleep，或把九段 6 ms 数字当九段 8 ms。
原始 2 ms 训练数据可能复用，但新预测时距与区间目标必须重建／验证。
PID 的历史现场计时来自 CPU 2／普通调度；与新 MPC 的 CPU 7／FIFO 时间不能直接当成算法公平速度对比。
比较控制效果时也需保留瓶重、路径和完整 `[5,18)` 窗口，不为突出 MPC 而削弱 PID。

旧位置参考实现只作 [历史对照](HARDWARE_MPC_REFERENCE_SERVO.md)。
