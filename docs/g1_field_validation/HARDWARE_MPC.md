# G1 真机力矩 MPC：当前程序与首次运行

对应 [`g1_walk_mpc.py`](../../tools/g1_commissioning/g1_walk_mpc.py)，更新于 2026-10-06。
**当前状态：两次原地力矩 MPC 已完整执行，行走尚无完整成功记录。**
原地操作者确认平顺、保持时微调，首次右瓶模型重建倾角从约 5.34° 降到 0.49°；
仍有加速度跟踪偏差及截止时间错过，不是力矩精度、行走效果或硬实时验收。

最新 [10-06 全日复盘与明日版本](sessions/20261006_MPC_CLOSEOUT.md)：起步多次失败主要是
规划／执行约束冲突，最后一次还涉及未经标定的状态外推。当前现场配置已取消活跃段的
总力矩／偏置相对力矩变化率硬限制和附加 `q+dq/6` 包络，**默认直接用实测 q/dq**。
保留原 MPC 关节范围、速度／加速度限值、绝对力矩上限及停止交还。
今天保存的 4,869 个活跃／故障状态离线重算通过，但这些是旧命令下的反馈，
**不是新版闭环行走成功**。明日先用 `hold_current` 做一条动态功能复验，不同时切学习预测器。
后续 [提前制动与力矩上限对照](sessions/20261006_MPC_BRAKING_PREPARATION.md)：保留仿真原停止距离处理，
新增接近边界时的软减速代价，未增加退出条件；同一批状态改成 ±25 N·m，输出力矩完全不变，
但该批主要是原地状态，不能证明旧上限足够行走。进一步检查完整 H0 走停数据后，
上限适度提高为 `[10,6,4,7,1.5] N·m`。新增控制逻辑只有离线证据，尚无实机闭环复验。
此前四次失败仍保留于 [启动记录](sessions/20261005_MPC_STARTUP_FAULTS.md)，
修复过程见 [离线复盘](sessions/20261005_MPC_OFFLINE_POSTMORTEM.md)，不覆盖历史失败。
6 ms PID 已通过现场功能复验并冻结，见 [PID 基线](sessions/20261005_PID_6MS_FIELD.md)。

## 1. 控制路线

腿部用内置运控，腰 yaw=0。左臂固定 PD，名义五轴 `[-4,-1,0,-8.1,0]°`；
右臂 MPC，名义五轴 `[-4,+1,0,-7.8,0]°`，左右瓶各 250 g。
只发 `rt/arm_sdk` 和已有的 Loco 速度请求，不使用 `rt/lowcmd`，不自动切模式或进入 debug。

右臂：当拍实测 q/dq → 九段、每段 **6 ms** 的加速度 MPC →
逆动力学名义力矩 → 局部修正、多候选比较及正动力学检查 → 力矩前馈＋PD。
发送前从选中总力矩减去已计入的 PD，同时发一拍 q/dq 参考和 `kp=20,kd=1`，固件只加一次 PD。
这不是旧版纯位置参考控制，也不是从持续积累的命令参考而不看实测状态出发。

局部模型对力矩是精确仿射关系，因此使用质量矩阵逆替代重复数值扰动、批量检查候选；
没有删除候选、缩短预测窗口或取消最终约束检查。QP 保留仿真的七项代价和运动学主线；
现场另加下述软减速项，这是明确记录的差异，不改仿真默认代码。
若原候选及保持候选均不满足模型加速度上限，再尝试有界最小二乘候选，仍受相同力矩／
前馈绝对范围和最终正动力学检查约束。候选通过范围检查不代表准确达到期望加速度；
跟踪误差单独记录，不能据此宣布 MPC 控制效果通过。
当前主 QP 内用局部关系 `tau=M*ddq+b` 提前考虑绝对总力矩及第一拍前馈范围，
避免先规划一个明显无法执行的加速度；没有加入任何力矩变化率行。
可选延迟补偿中的 2 ms 小步名义动力学循环位于 `cpp/g1_arm_delay`，保持原模型、步长和已发命令历史；
80 组变化身体运动／部分权重／命令切换与 Python 原实现逐项对照通过。库加载检查源码哈希和
MuJoCo 头文件／运行库版本；缺库或旧库在真实输出前拒绝，不静默切回较慢实现。
当前本地库 ABI=2，亦批量计算相同的条件质量矩阵／偏置。凝聚 QP 直接使用原代价块，
不再重复构造不用的 145×145 矩阵；随机数值对照检查代价与解，仿真默认路径保持。
静止任务使用当前身体运动估计，行走任务才查询冻结的状态索引扰动库。
H0 在每轮走前 3–5 秒平均 yaw 后固定，不跟随身体转动。库、滤波和预测时间轴未缩放或重新拟合。

真实力矩增益、摩擦、惯量误差、腿部反作用及电机内部延迟未标定。
**默认不做状态外推**。以前的 6 ms 执行延迟是假设，不是测出了 DDS 往返时间。
只有显式给 `--assumed-command-delay-ms` 才启用模型外推；指定 `0` 仍会外推反馈年龄，
不等于禁用。当前现场命令应省略此参数，日志值为 `null`，`state_input` 明确记为实测输入。
主机接收时间也不是传感器内部采样时间；所有假设和原始测量分别入日志。

## 2. PID 结果与当前力矩边界

PID 平顺、停车和退权正常；完整 `[5,18)` 有 0.934% 超时，不称为硬实时通过。
复用其通信／记录链路，不改 PID 增益或 governor。实测角有静差，抬臂结束时右肩接近 +5°；
MPC 入口必须检查实测姿态，不用目标角冒充实测角，也不为避免拒绝而放宽旧模型的关节外层边界。

[`hardware_mpc_torque_field.yaml`](../../configs/hardware_mpc_torque_field.yaml) 将右臂五轴总力矩估计／
前馈绝对上限由首次试验的 `[5,3,2,5,1.5]` 提高为 `[10,6,4,7,1.5] N·m`，不用统一 ±25 N·m。
依据是旧 H0 完整走停记录中，实测姿态／名义姿态下，模型覆盖 `|ddq|≤8` 的总力矩与前馈需求
约 `[7.640,4.317,2.681,5.071,0.321] N·m`；乘 1.25 后向上取整，腕部不低于原上限。
此模型估算不是实际 MPC 全程命令，更不是电机输出标定；所有细节和限制见上述对照记录。
这是工程试验限值，**不是厂家额定值或由 tau_est 完成的标定**。
10-05 第一次现场失败后，写前总力矩估计检查改为仅在 MPC 活跃段应用；抬臂／等待稳定／退权保留
前馈限值与有限值检查。该次修复没有提高数值（10-06 后续才提高），但**过渡阶段不再受该总力矩检查约束**，不能称为保护完全未变，
也不能由此前 PID 成功推导加入前馈后的过渡已通过实机验证。
保留原 q/dq/加速度约束与原求解器的停止距离处理，不启用后加的 `q+dq/6` 制动边界；
没有新增“手腕实测速度瞬时达到某值便停”的独立规则。

`predictive_braking_enabled: true`：用实测速度估算 `|dq|*0.012 + dq²/(2*4)` 的停止行程。
当该落点进入外层角度边界前 1°，平滑增加该关节未来速度趋零的代价（最大权重 50），
让 MPC 提前减速。这是软目标，**不增 QP 约束行、不增加拒绝条件**；无风险时该项为零。
4 rad/s²、12 ms 是减速能力／响应时间的设计假设，不是已经测出的执行器性能。
该项及激活程度写入 `predictive_braking` 日志；不能据此保证真实关节绝不越界。

活跃段 `active_slew_reference: none`：既不限制总力矩的逐拍变化，也不限制减去模型支撑项后的变化。
例如旧 50 N·m/s 相当于每 6 ms 只许变化 0.3 N·m，这条活跃限制现在已取消。
`transition_rate_nm_s` 仅用于进入／退出的渐变，不再传入活跃 MPC 或候选筛选。
`recovery_envelope_enabled`、`recovery_reentry_enabled`、`enforce_mapper_state_envelope` 均为 false；
不保留一个默认仍打开的隐藏恢复门槛。历史分支仅用于显式离线对照。
取消变化率限制意味着允许更快的力矩变化，不等于已经证明所有新命令都平顺。

## 3. 停止与退权

正常 18 秒、Ctrl-C 和可处理计算故障时，先请求零行走速度，再至少三秒退权。
冻结最后成功发出的 q/kp/kd；把 `kd*dq_ref` 转入前馈后令 dq_ref=0，保持完整 PD＋前馈关系连续。
退权全程保留支撑前馈，weight=0 最后一帧才清零 tau；故障路径不再求解 MPC。
正常、计算故障、主动停止和遥控打断已用无网络测试检查；10-06 已有一次平顺正常退权的
实机与人工观察记录，不等于其他异常路径也已完成物理验证。

CRC、FSM、遥控 L2+B、状态失效和 DDS 写失败保护保留。发现模式切换／失联／写失败后不继续
发渐退序列，避免与阻尼控制冲突。断网、强杀进程不能保证交还；异常仍用本机已验证的现场停止方式。
力矩写前要求所用状态不超过 25 ms、计算不超过 10 ms；这是异常拒绝边界，**不是改成 10 ms 周期**。
启动时先完成垃圾回收、RPC 构造和控制线程调度设置，再建立任务／预测器起点；运行中仍保留
50 ms 积压拒绝条件。输入确认本身发生在这些步骤之前，不计入控制实验时间。
`--compute-process` 只隔离数值计算，不持有 DDS。单次请求最多等待 9 ms；超时、进程退出、
序号不匹配即终止本次计算，不自动重启或使用旧结果。主进程仍单独检查 FSM、遥控、状态年龄、
力矩边界及 10 ms 写前时限，并保留最后成功发送的包用于退权。计算进程退出不影响这份快照；
这不等于主进程崩溃、网络失效或机器人内部异常时都能安全交还。

## 4. CPU 与实时调度

本机已有 PREEMPT_RT 和隔离核 6–7；旧 PID/早期 MPC 在 CPU 2、SCHED_OTHER 运行。
使用下方 `--compute-process` 时，独立计算主线程放 CPU 7，传输控制主线程放 CPU 2，
DDS、RPC、日志线程避开 SMT 同核 6–7；只有两个主线程可选 FIFO 20。
工作线程保持普通调度，计算与传输分别拥有 Python 执行锁；不把实时优先级套在整个 Python 进程上。
隔离不等于没有所有 IRQ。较早两轮离线平均 **5.079／4.963 ms** 没有模拟持续 SDK 回调争用，
不足以代表现场；历史证据见 [准备记录](sessions/20261005_MPC_FIELD_PREP.md)。
现已增加两路 500 Hz SDK 消息反序列化／回调／CRC／日志的离线并发测试，
计时和重复结果以 [最新复盘](sessions/20261005_MPC_OFFLINE_POSTMORTEM.md) 为准。
不包含真实 DDS Write、网络及完整 RPC 查询，仍不是硬实时验收。
10-05 操作者已完成 governor/FIFO 设置并跑过两轮；旧版本仍有 13.6%／22.6% 超时，不能据此放行。
随后又减少计算开销，并将整机电源档由 balanced 切到 performance；这不是只改电源档的对照实验。
在**准备运行的同一个本机终端**执行（密码只在你自己的终端输入）：

```bash
sudo prlimit --pid $$ --rtprio=40:40
sudo cpupower -c 2,6-7 frequency-set -g performance
powerprofilesctl set performance
```

第一条只给该终端及其子进程 RT 权限，不改永久 PAM 配置；程序结束恢复自己的原调度。
第二条改变 governor；10-06 原地日志显示 CPU 7 已为 performance，而传输线程 CPU 2 仍为 powersave，
故下一次一并设置 CPU 2。它与 CPU 1 共享物理核，设置不意味着独占，也不保证消除所有超时。
按自己的运行前记录恢复 governor；本项目最初的 powersave 可用
`sudo cpupower -c 2,6-7 frequency-set -g powersave` 恢复。其他电脑先核对 CPU 拓扑。
整机电源档可用 `powerprofilesctl set balanced` 恢复；性能档可能增加功耗、温度和风扇转速。
不要对整个 Python 进程套 `chrt`，以免工作线程也继承实时优先级。

## 5. 首次运行

机器人已双脚着地、FSM 500 自主平衡且静止，没有其他用户程序接管手臂。
下方先列原地命令；本机已有原地成功记录，下一轮动态复验用后面的行走命令，不必机械重复原地试验。
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
  --execute --task stationary --cpu 7 --rt-priority 20 --compute-process \
  --profile evaluation/hardware_shadow/commissioning/g1_pid_6ms_20261005_132052/arm_profile.conf \
  --output-dir "$MPC_OUT" --pid-6ms-validated \
  --permit-real-output MPC_WALK_H0_CAPTURE --allow-first-torque-field-trial
```

0–3 s 渐接管抬臂，**3–4 s 固定姿态＋支撑前馈，4–5 s 原地右臂 MPC 调姿，5–18 s 继续 MPC**，
左臂固定。18 s 后停车回复等待和至少三秒退权。4 s 切入是为了避免调姿与起步同时发生，
不是保证 5 s 前真机一定已经完全静止；
当拍实测状态和原有约束仍须满足，否则拒绝。普通 `--preflight` 只预检数学／构包，
独立进程的并发与故障验证由复盘中的专用离线测试完成。
原地任务前进／转向请求始终为零。10-06 已完成旧版及新规则的原地复验；另一次才改 `--task walk`、
使用新目录并加 `--torque-stationary-validated`。行走任务 5–15 s 请求 0.5 m/s，15–18 s 停车及航向保持；
速度×时间不是物理距离限位。原地试验未通过时，不通过删保护、改位置参考版或直接行走绕行。

### 下一次行走基线（新版已离线检查，真机待验）

当前命令使用“实测状态、无活跃力矩变化率限制、软制动、适度提高后的力矩上限”的现场配置。
依据与剩余问题见 [最新准备结论](sessions/20261006_MPC_BRAKING_PREPARATION.md)。先做一条并检查完整停车／退权，
再判断是否采重复轨迹；保存实际配置／源码哈希，不把旧版原地成功当作新版行走已经通过。

先显式选 `hold_current`，保持原地已经使用的扰动预测方式，不同时切到学习前馈。
当前真实加速度与模型仍有差距，这一轮是动态功能／响应检查，不是预先宣布性能通过。
采用当前 `[10,6,4,7,1.5] N·m` 上限和软制动；6 ms、多候选方法不变，
之后再对照 `learned_filtered`，不同时更改模型补偿。
机器人处于已验证的 FSM 500 自主平衡、静止起步和既有可用行走场地；程序不会替人调模式。
下面命令**会真正行走及控制双臂**；不是离线检查指令：

```bash
MPC_OUT="evaluation/hardware_shadow/commissioning/mpc_torque_walk_hold_$(date +%Y%m%d_%H%M%S)"
taskset -c 0-17 /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/g1_walk_mpc.py enx6c1ff701509c \
  --execute --task walk --predictor hold_current --cpu 7 --rt-priority 20 --compute-process \
  --profile evaluation/hardware_shadow/commissioning/g1_pid_6ms_20261005_132052/arm_profile.conf \
  --output-dir "$MPC_OUT" --pid-6ms-validated --torque-stationary-validated \
  --permit-real-output MPC_WALK_H0_CAPTURE --allow-first-torque-field-trial
```

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
综合本轮时序、姿态变化与延迟假设可用新离线脚本（先完成上面的末端分析）：

```bash
python tools/g1_commissioning/analyze_mpc_field_trial.py "$MPC_OUT/raw.jsonl" \
  --endpoint-dir "$MPC_OUT/endpoint_analysis" --output-dir "$MPC_OUT/field_review"
```

## 7. 真要增加周期时必须同步修改

当前仍为 6 ms。先完成 FIFO／性能模式与实际 DDS 负载计时，没有把普通调度未优化当成必须降频。
若最终必须调整，要同时检查：积分矩阵／一拍 q/dq、QP horizon 与物理预测长度、制动约束、
进入／退出的力矩渐变、weight 步长、可选命令历史／执行时刻、扰动节点和**区间均值**重采样、创新衰减时间、
滤波物理时间常数、时钟槽、profile、日志及分析器。不能仅改 sleep，或把九段 6 ms 数字当九段 8 ms。
原始 2 ms 训练数据可能复用，但新预测时距与区间目标必须重建／验证。
PID 的历史现场计时来自 CPU 2／普通调度；与新 MPC 的 CPU 7／FIFO 时间不能直接当成算法公平速度对比。
比较控制效果时也需保留瓶重、路径和完整 `[5,18)` 窗口，不为突出 MPC 而削弱 PID。

旧位置参考实现只作 [历史对照](HARDWARE_MPC_REFERENCE_SERVO.md)。
