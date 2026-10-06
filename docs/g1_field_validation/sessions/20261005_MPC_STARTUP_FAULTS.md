# 2026-10-05：四次原地 MPC 启动失败与修复

前三次均没有成功发出 MPC 活跃段命令，不是有效 MPC 性能实验。
第三次完成抬臂，在约 3 秒第一次计算 MPC 时拒绝候选，随后完整退权到零。
第四次成功 Write **一条** MPC 活跃命令，下一拍超时；随后完整退权。
不能将这一条命令称为完成真机 MPC 验证。离开实验室后的改动与离线结果另见
[整链性能与接管复盘](20261005_MPC_OFFLINE_POSTMORTEM.md)。

## 16:53：抬臂力矩检查及退权中止

日志：`evaluation/hardware_shadow/commissioning/mpc_torque_stationary_20261005_165321/raw.jsonl`。
最后正常抬臂命令时间 0.870751 s，weight=0.290250，右肘估算总力矩约 −4.992 N·m；
随后抛出 `field total/feedforward torque envelope exceeded; hand back`。
故障退权也被同一检查打断，最后成功发包 weight=0.270513，`fault_release.completed=false`。
**不能解释成已安全退权到零**。操作者随后报告通过坐姿→运动模式恢复内置运控接管。

当时修改 `hardware_mpc_field.py`：MPC 活跃段检查总力矩；过渡段检查有限值及前馈限值。
这改变了保护的适用范围，虽然配置限值数字未增大。过渡段真实行为仍待验证；
也不保证其他前馈／状态／通信错误下渐退一定完成。测试通过只是软件证据。

## 17:05：预测器启动积压，手臂零发包

日志：`evaluation/hardware_shadow/commissioning/mpc_torque_stationary_20261005_170513/raw.jsonl`。
记录了 publisher 创建，**没有成功的手臂 Write**；有零行走速度请求。
故障为 `predictor backlog exceeded; refusing unbounded catch-up`。

- task epoch：13729899599407 ns。
- task_epoch 日志：13729938630479 ns，已过去 39.031 ms。
- control_runtime：13729957176277 ns，距 epoch 57.577 ms。
- fault：13729957303034 ns，距 epoch 57.704 ms。

原代码先设 epoch 并推进预测器，再做 `gc.collect()`，随后创建速度线程、设置调度并读取主机信息。
第一拍到来时超过实际代码中的 50 ms backlog 上限。用户输入确认在 epoch 之前；
不能把此次失败归因于操作者输入慢。此前口头提及的 40 ms 也不是这里的 backlog 阈值。

修复：把 GC 放到 publisher 创建前；速度 RPC 在线程所属的工作核完成构造后等待启动信号；
主线程完成 affinity/FIFO 及主机信息记录，重新检查状态，再设 task epoch、初始化预测器并启动循环。
PID 原有 epoch 顺序保持。运行中的 backlog／陈旧状态保护未放宽，也没有运行中重置滤波。

验证使用禁止 socket 的假 SDK runner 和真实只读预测器：GC、RPC 构造、调度阶段各注入
120 ms 停顿，第一包仍从低权重开始，受控停止正常退到零；启动失败路径正常清理线程。
相关预测器测试继续确认真实运行积压会被拒绝。随后检查原地／行走假通信正常结束和力矩字段测试。
随后 17:19 真机运行完成抬臂，未再出现这次启动积压；不宣称已获得成功 MPC 实验。

## 17:19：抬臂完成，首次 MPC 候选搜索失败

日志：`evaluation/hardware_shadow/commissioning/mpc_torque_stationary_20261005_171958/raw.jsonl`。
SHA256：`07d880fa9cb43db31dc8455ce16a3cfdd701f440ea3582e85b04ffd384451500`。
操作者看到手抬起后放下。首包时间 0.014865 s；最后正常抬臂命令为
2.996917 s、weight=0.998972。首次 MPC 的 QP 求解成功，但力矩候选检查抛出
`no candidate satisfies the current forward-model envelope`，没有发出 MPC 活跃命令。
故障退权完成，耗时 3.242176 s，最后成功命令 weight=0；全程成功手臂命令 917 条。
这次日志证明软件退权序列完整，不代表所有故障情况下均能安全交还。

### 原因与窄范围修复

原搜索先算无约束局部修正，再逐关节裁剪。耦合关节与窄的单拍力矩变化范围导致
多轮搜索停在同一边界：候选右肘**模型预测**加速度约 −11.768 rad/s²，超过 10 的上限。
这是模型数值，不是测到了真实关节加速度。此时手臂仍在运动，实测右肘速度约 −0.706 rad/s。

同一个失败快照、相同约束下，有界最小二乘可找到另一组力矩，正动力学复查的最大
绝对加速度约 7.609 rad/s²。因此本次是候选搜索漏解，不能据原错误宣布无可行力矩。
`hardware_torque_mapper.py` 仅在原候选、上一拍和保持候选均失败时，针对调用方提供的
精确仿射模型增加有界最小二乘候选（最多 20 次迭代），之后仍调用正动力学复查。
本次不改力矩绝对上限、变化率、模型加速度上限或运行时保护；真正无可行候选仍停止。

**允许的候选不等于准确跟踪**：该快照期望右肘 +8 rad/s²，新候选模型结果约 −7.609 rad/s²，
`tracking_within_limit=false`。原有保持／退回候选也不保证跟踪精度；本次没有把这一结果称为
成功执行 MPC 加速度。下一次原地试验还需验证连续运行与真实响应，不能直接转为行走试验。

回归测试固化了该快照数值，不依赖未纳入 Git 的原始日志；另外检查真正不可行时仍拒绝。
共 20 项相关测试通过，复现命令：

```bash
PYTHONPATH=tools/g1_commissioning:tools/g1_commissioning/tests \
MPLCONFIGDIR=/tmp/g1-mpl OPENBLAS_NUM_THREADS=1 \
/home/fjk/miniforge3/envs/g1_mpc/bin/python -m unittest test_measured_torque_mpc test_mpc_field
```

该快照的 mapper 单独运行 100 次，普通调度下平均约 0.594 ms、P99 约 0.765 ms；
不包含完整控制链、DDS 或真实机器人，不是 6 ms 实时验收。随后第四次运行使用了该修复。

## 17:27：成功一拍，第二拍计算过期

日志：`evaluation/hardware_shadow/commissioning/mpc_torque_stationary_20261005_172729/raw.jsonl`。
SHA256：`fca6f0df0ac55aef121df433b35cc36088ece0bd435ac60aada4a334217deb69`。
3.000676 s 成功发出第一条 MPC 活跃命令；该拍控制计算约 7.063 ms，写前约 7.864 ms。
下一拍核心计算 **13.801596 ms**，触发 `torque computation older than 10 ms; hand back`，未发出该拍候选。
928 条成功手臂命令包含后续退权；退权耗时 3.253785 s，最终 weight=0，日志丢失数为零。

失败拍内部耗时：QP 装配约 5.119 ms（代价 3.876 ms、运动学项 1.134 ms）；
求解接口外层约 5.000 ms，但 DAQP 调用约 0.549 ms、求解器内部约 0.030 ms；
逆动力学约 1.223 ms（其中 C++ RNEA 约 0.0076 ms），映射约 1.345 ms。
这些是不同层级、可能嵌套的 wall time，不能把所有数字重复相加。
**不能把 13.8 ms 全归因于求解器或电源档**；日志本身也没有独立测出每段等待 GIL 的时间。
当时 CPU 7/FIFO 20、governor=performance，整机 platform_profile=balanced。
过去切换整机档位的比较还同时改过代码，不能当成单独电源档位的因果实验。

四次日志的统一提取程序是 `tools/g1_commissioning/analyze_mpc_startup.py`，只读分析，不连接机器人。
本机结果位于 `evaluation/hardware_shadow/commissioning/mpc_postmortem_20261005/field_audit.json`。
实测 dq 与模型 ddq 分开；一条命令不足以标定力矩增益、实际加速度跟踪或 DDS 执行延迟。
