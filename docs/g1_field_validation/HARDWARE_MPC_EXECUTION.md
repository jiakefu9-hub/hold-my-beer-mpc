# 力矩 MPC：怎样判断真实执行是否跟上

更新：2026-10-05。对应 `arm_execution_record.py`、`analyze_mpc_execution.py`。
本轮 MPC 只有离线开发和检查，没有发送 MPC 控制或获得 MPC 真机效果；
同日 PID 已完成实机复验，见 [PID 记录](sessions/20261005_PID_6MS_FIELD.md)。

## 当前控制路线与未完成项

首次真实 MPC 使用：实测关节状态 → MPC 期望加速度 → 逆动力学名义力矩 →
局部修正和多候选正动力学筛选 → 力矩前馈＋PD。
主算法已经实现；旧 `reference_servo` 现只留作离线对照，CLI 禁止它真实输出。
默认 preflight、运行时构造及离线计时均选择 `measured_torque_preview`，不再默认位置参考版。
数据包构造统一为 `MpcRuntime.make_message`：本地预检、回放以及仍受锁保护的共享传输入口
都按该分支构造非零 tau，不能再误用 PID 的零前馈构包器。正常／故障交接现已接入：
冻结最后成功包的完整 PD＋前馈关系、至少三秒 weight 退权；模式／失联打断后不继续发包。

先前阻塞项的处理和剩余边界（当前操作以 [MPC 程序文档](HARDWARE_MPC.md) 为准）：

- 09-29 完整链路平均 8.5–8.7 ms 是历史结果；本轮保留模型修正和 6 ms，优化记录见
  [2026-10-05](sessions/20261005_MPC_FIELD_PREP.md)。普通调度的长尾仍不等于硬实时通过。
- 共享现场停止／退权路径已加入非零前馈连续性，不在该路径继续依赖 QP。
  无网络故障注入已检查，真实物理交还仍需首轮验证。
- 首次试验配置不再使用统一 ±25 N·m；逐轴限值见 `hardware_mpc_torque_field.yaml`，
  它们仍是工程边界，不是标定或厂家额定值。

无需为此先采十几条轨迹；实际力矩响应要通过首次受控原地试验取得证据。
入口显式开启且先原地再行走，不把 6 ms PID 正常当作力矩 MPC 成功。

## 要比较的三层数据

| 比较 | 数据来自哪里 | 能说明什么 |
| --- | --- | --- |
| 总力矩目标与 `tau_est` | 实际命令包的 `tau/kp/kd/q/dq`，加上之后收到的关节 q/dq/力矩估计 | 执行器报告的力矩与请求是否大致一致 |
| 实际关节加速度与 MPC 期望、模型预测 | 实测 dq 的时间差分；`raw_mpc_ddq_rad_s2`；最终 `post_transition_ddq_rad_s2` | 机器人是否产生了想要的加速度，而非仅在同一模型内自洽 |
| 瓶子是否更稳 | 原有 `analyze_hardware_mpc.py`，实测关节＋身体 IMU 的 H0 末端分析 | 姿态、线加速度等最终任务效果；不能单由关节力矩误差推断 |

SDK 包中 `tau` 是前馈项，不能直接拿它和总力矩估计比较。这里按下式重建预期总量：

`tau_expected = tau_ff + kp * (q_ref - q_measured) + kd * (dq_ref - dq_measured)`

候选筛选针对总力矩；发送前减掉已计入的 PD，避免重复加 PD。保留 q/dq、kp/kd 不代表
退回原先的“只积分位置参考”路线，也不应为了看起来像力矩控制而直接把所有 PD 去掉。

Unitree 官方 [Arm5 例程](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/g1/high_level/g1_arm5_sdk_dds_example.cpp)
展示同一个 Arm SDK 消息内的 q/dq/kp/kd/tau；[HG MotorState 定义](https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/idl/hg/MotorState_.hpp)
包含 q/dq/ddq/tau_est。接口存在不等于已证明这台机器的力矩精度。
没有独立测力仪时，**不能仅凭 `tau_est` 给出绝对力矩标定结论**，也不自动拟合“乘 1.1”这样的补偿。

## 记录和时间对齐

共享运行器新增字段（PID 也记录，控制增益／周期／保护未改）：

- `packet_q_rad/packet_dq_rad_s/packet_kp/packet_kd/tau_ff/packet_weight`：实际消息对象按 IDL
  float32 精度保存，不把未发送的候选值当成命令；索引也显式保存。
- `tau_est_at_feedback_nm`、`ddq_raw_at_feedback_rad_s2`：控制时所用 LowState 的反馈，保留原值。
  缺失／非有限值不伪造为零；原完整 LowState 记录继续保留。
- 原有状态接收时刻、Write 起止、实际周期、重复状态和控制诊断继续记录。
  **同一命令记录里的反馈来自发送前，不是这条新命令的响应。**

分析器按主机时间戳排序，使用“此前成功写出的命令”对照“后来收到的状态”。
控制记录携带的快照与原 LowState 按接收时刻去重；不会把重复用到的同一帧算成新测量。
缺少明确 `tau_ff` 的命令不能从候选值补造，也不能假设为零。

只在完整接管 weight≈1 时做定量比较；接管／退出阶段的内置运控混合未辨识，不算进主要误差。
主要窗口仍为 `[5,18)`，即开始走到停车保持结束；静止任务也使用该窗口，但明确不称为行走结果。
另列 `[3,5)` 站立基线。异常提前终止的数据只报告实际覆盖范围，不冒充完整成功实验。

真实角加速度不用曾经全零的 `ddq_raw` 替代：默认用相隔约 24 ms 的实测 dq 求区间平均加速度，
并对**同一个区间**内实际发出的 MPC 期望／最终正动力学预测做按时间加权平均。
保留真实区间长度，遇到数据间隔过大或不满接管权重就不计算。该方法仍受速度噪声、
接收抖动影响；24 ms 平均会淡化更短尖峰，重叠窗口也不是独立统计样本。

默认不平移命令，即 `--assumed-delay-ms 0`，不代表测得延迟为零。
可对预先指定的延迟假设作敏感性比较，但不能在同一条轨迹上找最小误差后宣布“测出了 DDS 延迟”。
Write 返回与主机收包时刻并不提供电机内部应用时刻。

## 获得真实数据后怎么查看

从仓库根目录、`g1_mpc` Python 环境运行，输出目录必须是新目录：

```bash
python tools/g1_commissioning/analyze_mpc_execution.py /path/to/run/raw.jsonl \
  --output-dir /path/to/run/execution_analysis
python tools/g1_commissioning/analyze_hardware_mpc.py /path/to/run/raw.jsonl \
  --output-dir /path/to/run/endpoint_analysis
```

执行分析产生 `execution.png`（五关节各三栏：力矩、加速度、角度）、`summary.json`、
`torque.csv`、`acceleration.csv`、`execution.npz`，均保留输入哈希／计算参数可复算。
检查顺序：先看周期与反馈质量，再看期望总力矩和 `tau_est`，接着看关节加速度跟踪，最后看瓶子指标。
输出是逐关节偏差、RMSE 等证据，**不自动颁发成功或安全判定**；估计值不变、量化明显、
动作激励很小时，都不能据此断言力矩响应准确。

离线回放消息会被排除，不会生成“真实力矩跟踪成功”的报告。
本轮测试用已知 0.125 N·m 偏差和 0.8 rad/s² 加速度的合成记录核对计算，合成数值不是机器人结果。

## 优化前的执行记录链检查（历史，2026-10-05）

以下是加入遥测后、整链优化前的数字。当前准备和计时以
[后续优化记录](sessions/20261005_MPC_FIELD_PREP.md)为准，不用旧版本数字冒充当前结果。

- 62 项相关测试通过：PID、6 ms 周期、实测状态 MPC、力矩包／单次 PD、延迟任务、
  新执行分析、历史力矩分析和禁止真实输出检查。未初始化 DDS。
- 最终默认 preflight 通过，选中力矩分支，本地 SDK 包的右臂 tau 非零；
  `publisher_created=false`、`field_output_supported=false`。不表示发送过命令。
- 增加记录后的首轮离线诊断平均 8.344 ms，主窗口 1083/1083 超过 6 ms。
  统一构包入口后的最终版本重新跑完整任务，平均 **8.645 ms**、P99 **11.738 ms**，
  主窗口 **1078/1078** 超时，周期中位数约 **12.000 ms**。没有把较快的首轮当作最终版本成绩。
- 最终回放独立审计 2246 条命令，其中 1245 条 MPC 活跃命令，状态起点、仅一次 PD、力矩、
  变化率及最终 weight=0 检查通过；日志无丢弃。**这些都不抵消时序失败，也不证明物理力矩准确。**

最终原始结果保存在本机
`evaluation/hardware_shadow/commissioning/torque_execution_telemetry_final_20261005/`。
`run1/summary.json` 保存源码／模型哈希和运行设置，`raw.jsonl` 保留完整诊断，`timing.json` 保留逐周期时间。
原始记录哈希：`0420140984a33c5ea3ae484c8b2588e407675350b014e0bc5b911c97dc9dd1aa`。
这仍是合成静止输入、CPU 2、普通调度、假定观测延迟 4 ms／命令时间 6 ms 的离线计时，
不含真实 DDS 通信和机器人响应；原始大文件按原约定不提交 Git。
