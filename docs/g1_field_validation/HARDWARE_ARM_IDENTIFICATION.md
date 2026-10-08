# G1 右臂小幅系统辨识

对应采集程序：[g1_arm_system_identification.py](../../tools/g1_commissioning/g1_arm_system_identification.py)；
初步离线分析：[analyze_arm_system_identification.py](../../tools/g1_commissioning/analyze_arm_system_identification.py)；
三轮联合复核：[review_arm_identification_batch.py](../../tools/g1_commissioning/review_arm_identification_batch.py)。

**2026-10-08 结果：正式三轮已完成，重复性较好；尚不能可靠标定惯量、摩擦和绝对力矩增益，因此没有修改
MPC 物理参数。** 下一步是新版不加学习前馈 MPC，不再重复同配置辨识。见[联合分析记录](sessions/20261008_ARM_IDENTIFICATION.md)。

目标不是重新测瓶子，而是初步估计三类量：

1. Arm SDK 力矩通道到关节响应的有效延迟和相对增益；
2. 当前姿态附近的等效粘性摩擦、库仑摩擦和惯量；
3. 五个右臂关节之间的局部耦合，即一个实测候选 `5×5` 动力学关系。

结果不会自动写回 MPC。`tau_est` 是机器人自身的估计，不是独立测力计，因此不能仅凭这次实验宣布
“请求 1 N·m 就精确输出 1 N·m”。瓶子质量、质心和固定方式沿用现有模型，不在本轮重新辨识。

## 为什么腿部力矩和脚底力不是本轮必需量

当前逆动力学问的是：**已知身体这一刻怎样运动，右臂五个关节要产生多少力矩。**
如果身体 IMU 能无误差地给出刚性躯干的姿态、线加速度、角速度和角加速度，而且手臂没有接触外物，
那么腿部通过何种力矩／脚底力造成了这段身体运动，并不是计算当前右臂条件力矩的必需输入。
身体匀速向前本身也不产生额外惯性力；起步、刹车、落脚晃动和转动才会影响手臂。

腿部和接触信息在另外两件事上有价值：预测未来 54 ms 身体会怎样运动，以及计算手臂动作反过来对整机平衡／
脚底接触的影响。当前学习预测器用腿部 q/dq 和近期 IMU 处理第一件事；当前五关节模型没有解决第二件事。
现实中 IMU 还有噪声、偏置、时间戳和角加速度差分误差，所以“IMU 绝对准确”只是用于划清模型边界的理想假设。

## 当前模型到底有没有摩擦

对**真机当前使用的加速度到力矩映射**来说，可以近似理解成“理想刚性、没有摩擦补偿”，但不等于只剩质量和
关节限位。它仍包含完整五轴惯量耦合、重力、科氏／离心项、`0.01 kg·m²` 的模型 armature，以及 IMU
给出的移动基座运动。关节角限制属于 MPC 的运动约束；没有撞到限位时，它不是逆动力学里的一个额外力。

仿真 MJCF 的默认关节确实写有 `damping=0.001`、`frictionloss=0.1` 和 `armature=0.01`。但当前真机
`RightArmInverseDynamics.compute()` 明确把 passive torque 和 friction loss 以零传给 RNEA；原因是这两个
仿真默认值没有被证明等于真实减速器／电机摩擦，不能直接拿来补偿真机。armature 则已进入质量矩阵。

当前没有建模的主要是：静摩擦、随速度变化的粘性摩擦、方向相关摩擦、齿隙、传动柔性、迟滞，以及固件内部
PD／力矩通道的真实增益和延迟。因此“关节完全光滑”是一个便于理解但稍显过头的说法：更准确地说，是控制器
目前只算理想刚体力矩，把这些非理想执行器效应都留给真实系统和反馈修正承担。

## 程序实际做什么

- 完成进入渐变后，右臂参考角是当前 MPC 的持瓶中立位：肩 pitch `-4°`、肩 roll `+1°`、肩 yaw `0°`、
  肘 pitch `-7.8°`、腕 roll `0°`；这也是此前调到瓶子接近竖直的名义姿态。
- 辨识是在这组角度附近叠加小幅力矩，不是让五个关节从一个大范围逐点扫描。
- 配置中的“相对中立位不超过 `±10°`”是激励阶段的停止边界，不是目标运动幅度；实际摆幅由小力矩、
  固定参考角的 PD 和真实执行通道共同决定，并由日志记录。
- 机器人必须已由操作者进入 Regular Motion Mode / FSM 500，原地自主平衡；程序不切模式。
- 全程请求零行走速度，不进入走路阶段。
- 0～3 s：沿用已验证的渐进接管并到达当前持瓶姿态。
- 3～5 s：静止保持。
- 5～15 s：右臂五轴同时叠加不同频率的平滑小力矩；参考角仍固定在名义姿态。
- 15～18 s：停止激励并保持，用于观察余振。
- 随后至少 3 s 渐变退权到 weight=0。

首次筛查使用 `[0.8,0.5,0.2,0.6,0.15] N·m`：肩 pitch 响应清楚，但肩 roll、肩 yaw、肘和腕的
同频响应与残差接近。后续正式采集将五轴最大前馈调整为 `[0.8,0.8,0.5,0.9,0.35] N·m`；肩 pitch
保持不变，另外四轴仍低于各自既有力矩外层的 25%。每轴由三个互不重复的 0.4～2.4 Hz
正弦分量组成，进入／退出有 0.75 s 平滑包络。放大后的筛查中 yaw 峰峰运动约 9.7°；
耦合和静摩擦是可能原因，尚未单独证实。因此正式辨识使用辨识专用 yaw 稳定：
`kp=[20,20,6,20,20]`、`kd=[1,1,0.5,1,1]`。命令和分析都会保存这组实际增益，动力学结果不会
假称是在 MPC 的 `2/0.2` yaw PD 下直接测得。这是一轮筛查幅度，不是电机额定值。

采集保存实际发包、q/dq、`tau_est`、身体 IMU、时间戳、周期和退出结果。激励阶段另外检查既有总力矩外层、
相对名义角不超过 10°、速度不超过 2 rad/s；模式、遥控、CRC、反馈和三秒退权沿用共享现场程序。

## 实验顺序

6 ms PID 已完成真机验证。当前主线见[实验进度简表](EXPERIMENT_STATUS.md)：先完成本辨识和离线分析，
只把有充分依据的延迟、摩擦或局部耦合修正接入新版模型；然后先跑不加学习前馈的微调 MPC，最后跑学习前馈 MPC。
辨识数据本身不会自动让模型变准，必须先检查重复性、拟合误差和参数是否合理，再明确修改模型。
本批已完成此检查，决定保留现有模型；“必须改出一组参数”不是进行下一轮 MPC 的条件。

在机器人处于 Regular Motion Mode 原地稳定后：

1. 先做一轮，现场确认只是小幅平顺动作、没有持续漂移；
2. 第一轮日志完整才重复两轮，每轮使用新目录；
3. 离线分析后再决定哪些参数可以接入，不把未经检查的拟合值自动写进 MPC；
4. 模型修正通过离线回放和测试后，再依次运行不加学习前馈、加学习前馈的两轮 MPC。

辨识本身不需要行走。水瓶继续采用当前两侧各 250 g 的配置。

### 电脑离线预检

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
PYTHONPATH=tools/g1_commissioning \
  /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/g1_arm_system_identification.py --preflight --cpu 7
```

输出必须明确为 `dds_initialized=false`、`publisher_created=false`。

### 一轮真实采集

下面的命令会实际接管并轻微驱动右臂，只能在上述现场状态下运行：

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
sudo -v
ID_OUT="evaluation/hardware_shadow/commissioning/arm_identification_$(date +%Y%m%d_%H%M%S)"
taskset -c 0-17 /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/g1_arm_system_identification.py enx6c1ff701509c \
  --execute --cpu 7 --rt-priority 20 \
  --profile evaluation/hardware_shadow/commissioning/g1_pid_6ms_20261005_132052/arm_profile.conf \
  --output-dir "$ID_OUT" --pid-6ms-validated \
  --permit-real-output ARM_ID_STATIONARY
```

每次重新执行整段命令，时间戳会产生新目录。第一轮尚未检查前，不连续无观察地批量执行。

### 三轮完成后的分析

正式联合分析采用整轮分离：第 1 轮拟合、第 2 轮选延迟，然后用前两轮拟合最终系数，第 3 轮只检验。
没有给第三轮单独拟合偏置，也没有使用第三轮 IMU 均值调整输入。

```bash
PYTHONPATH=tools/g1_commissioning OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/g1_mpl_cache \
  /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/review_arm_identification_batch.py \
  evaluation/hardware_shadow/commissioning/arm_identification_20261008_121118/raw.jsonl \
  evaluation/hardware_shadow/commissioning/arm_identification_20261008_121256/raw.jsonl \
  evaluation/hardware_shadow/commissioning/arm_identification_20261008_121407/raw.jsonl \
  --output-dir evaluation/hardware_shadow/commissioning/arm_identification_review_REPLAY
```

输出目录必须尚不存在。结果含 `review.json`、第三轮预测数组及对比图。另在现有刚体模型上拟合
“偏置＋非负粘性／库仑摩擦”候选，检查第三轮能否改善、参数是否可以解释；即使误差减小，也不会自动写回。
246／366 ms 居中差分仅用于离线敏感性检查，不修改控制器的实时滤波或延迟。

旧初步分析器保留每轮前 70%／后 30% 的筛查结果，但同一保留段同时参与选延迟，不能称作独立检验。
其中 `preliminary_candidate`、满秩或条件数合适，都不证明矩阵是物理惯量；最终采用上面的整轮复核。

反馈按日志中的 `state_received_monotonic_ns` 对齐，命令按实际 `write_begin_monotonic_ns` 作零阶保持，IMU 使用
`received_monotonic_ns`。机器人消息没有源端采样时间戳，因此这里估计的是主机所见的有效延迟，不能当作电机内部
纯延迟的精确测量。

## 会记录什么，怎样让模型更接近真机

每个 6 ms 控制拍保存：五轴前馈力矩、q/dq 参考、PD 增益、weight、实际 q/dq、机器人报告的 `tau_est`、
命令与反馈时间戳。另行保留 2 ms 采样／20 ms 落盘的完整 LowState、约 5 ms 的身体 IMU、CRC／遥控／模式、
实际控制周期、写入耗时、退出原因和最终 weight。`tau_est` 对应发包前的反馈，分析时会与更早的命令按延迟
重新对齐，不会把同一行误当成命令的即时响应。

离线程序从这些记录得到：

- 0～30 ms 范围内，使离线 qdd 与历史激励力矩最符合的有效延迟候选；这不是实时未来预测；
- `qdd ≈ A·tau_ff + 状态项 + 身体 IMU 项` 中完整的 `5×5 A`，以及条件允许时的 `M_eff≈A⁻¹`；
- 完整耦合、只看同轴力矩、完全不使用力矩三种模型在保留数据上的误差；
- 与 `tau_est` 相对一致的延迟／增益，以及粘性摩擦、库仑摩擦的候选量级。

通过三轮重复后，可靠的结果可以分层使用：先把辨识延迟写入命令执行时间模型；再加入小幅、连续的摩擦补偿；
最后把局部有效输入矩阵作为 MPC 的执行／跟踪模型，或与现有刚体 `M` 作受约束的校正。不会用一次拟合直接覆盖
全部物理惯量，也不会根据 `tau_est` 自动乘一个力矩比例。任何写回都应一次只改一类参数，并用未参与拟合的一轮
数据以及原 PID／MPC 基线复核。

本轮之后仍难准确获得的内容主要是：

- 没有独立测力或可信电流标定时的**绝对实际关节力矩**；`tau_est` 与控制器来自同一机器人，不能自证准确；
- 固件内部 weight 混合、力矩／PD 的限幅、滤波、增益调度和精确应用时间；
- 齿隙、柔性、迟滞和随温度／负载变化的摩擦，这些通常需要更多方向、幅值和温度条件的实验；
- 整机平衡控制、脚底接触和手臂反作用的完整耦合；内置腿部策略不是主机可见的白盒模型；
- 不同手臂姿态下的全局模型。本轮只辨识当前持瓶姿态附近，扩大工作空间需要在多个姿态重复，而不是外推一次结果。

前三项是小实验可以尝试改善的方向，不保证本批数据足以分离参数，更不会产生覆盖所有动作的真机数字孪生。
