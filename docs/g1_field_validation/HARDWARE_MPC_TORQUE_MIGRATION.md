# MPC 按仿真迁移：实测状态、逆动力学与多候选力矩

2026-09-28。对应 `g1_walk_mpc.py --actuation measured_torque_preview`，也是当前 CLI 默认选择。
**可离线运行和复验，禁止真实输出；还不能直接上机。** 本轮没有连接机器人、切模式或创建 DDS 发布器。
旧 PID、`reference_servo` 基线和仿真控制器均保留；没有把新力矩限值偷偷套到旧现场程序上。

本轮落实了“先把仿真的主要控制链迁移过来，再检验差异”的建议。不是宣称整套仿真执行环境
已经等价搬上真机：闭环压力测试暴露了延迟／模型偏差下的不可行问题，完整 6 ms 时间检查也未通过。

## 1. 现在一拍究竟怎样算

1. **读实测 q、dq，作为 MPC 当前状态。** 上一拍求解结果只用于优化初值，不再拿持续保存的命令参考
   假装真实手臂已经到位。保持九段 × 6 ms、54 ms 预测窗和仿真七项代价。
2. 由同一个加速度 MPC 求出第一拍 `ddq_des`，生成一拍 PD 参考：
   `q_ref=q+dq·0.006+0.5·ddq_des·0.006²`，`dq_ref=dq+ddq_des·0.006`。
   和仿真一样受外层位置／速度边界约束；失败解不会进入执行层。
3. 用实测 q/dq、身体当前 H0 运动及 250 g 瓶子模型计算逆动力学名义力矩。
   加上当前 PD 贡献，得到待检查的**总力矩**。
4. 像仿真一样，小幅扰动力矩，估计“改变一点力矩，会改变多少加速度”的局部关系。
   产生 `1、0.5、0.25、0.125` 四种修正幅度的候选；按局部预测排序，再至少试算两个候选，
   必要时试算更多、第二次修正或最多两次救援。以前的力矩也必须在当前状态重新检查才能复用。
5. 选择通过模型检查的总力矩，消息中写入
   `tau_ff=选定总力矩−当前PD贡献`。固件若按预期再加 PD，才是选定总量，**没有叠加两次 PD**。
   这只是消息与模型算术一致；真实固件混合、执行延迟和力矩精度还没验证。

这里的“多个候选”是**当前一拍的多组关节力矩**，不是同时求解多个完整 MPC、也不是发给机器人逐个试。
所有候选都只在本地模型中试算。没有用 `tau_est` 自动改变输出增益。

DAQP 求解器对原仿真 QP 做代数等价消元：10 个十维状态＋9 个五维加速度，共 145 个变量，
消元后是 45 个加速度变量。它不改变七项代价或初始实测状态；原仿真 OSQP 仍可用于独立对照。

## 2. 哪些与仿真一致，哪些不能假装一致

| 项目 | 本轮情况 |
| --- | --- |
| 实测 q/dq 起点、一拍参考、七项代价 | 已迁移；与原 `ArmMPCPolicy` 独立对照 |
| 逆动力学＋PD 后检查总力矩 | 已实现；C++ RNEA 与独立 MuJoCo 质量矩阵／偏置力交叉验证 |
| 局部修正、多个候选、二次／救援、当前状态重检回退 | 已实现；同一 MuJoCo 正动力学下与原仿真 mapper 对照 |
| 仿真完整接触／摩擦／全身状态 | **未迁移成真机事实**；采用给定实测身体运动的右臂条件模型 |
| 实际执行器响应、延迟、负载惯量误差 | 未辨识；压力测试明确加入偏差，不称为实机标定 |
| 执行节拍 | 候选每 6 ms 更新；仿真内部还有 2 ms PD、4 ms mapper 子步，并未谎称完全相同 |
| 进入、正常退出 | 新增离线总力矩变化率设计、最终输出复查、weight=0 后清零前馈；尚未现场验收 |
| 故障后的实机停车／力矩交还 | 尚未为新路径验收；新路径不能进入现有 `run_device` |

条件正动力学是 `ddq = M_arm(q)^−1 · (tau_total − h_arm)`。
身体姿态、角速度、角加速度以及 IMU 点去重力后的线加速度来自现有 H0 管线；
通过 IMU 杠杆项重构虚拟基座，不虚构腿部加速度、接触力或机器人全局位置。
在固定状态下，这个条件模型对力矩是线性的；多候选检验增加的是执行约束／耦合检查，
**不是凭空增加了关于实机摩擦、力矩误差的新信息**。

因此，同模型逆动力学再正动力学算回期望加速度，本身只是数学一致性检查。
下面加入 PD、积分、负载偏差和延迟的闭环试验，才进一步检查执行效果；仍不是实机证据。

## 3. 配置与边界

参数集中在 [hardware_mpc_torque_preview.yaml](../../configs/hardware_mpc_torque_preview.yaml)。
右臂五轴顺序为肩 pitch、肩 roll、肩 yaw、肘 pitch、腕 roll。

- QP 外层角度范围沿用仿真：`[-5,5]、[-5,3]、[-20,5]、[-40,40]、[-40,40]` 度；
  两个肩关节有 1° 内层运行裕量。速度／加速度模型约束为 1 rad/s、8 rad/s²。
- 总力矩 ±25 N·m、模型加速度检查 10 rad/s²、变化率 `[100,100,100,100,40] N·m/s`
  都是**离线设计参数，不是本机已批准的力矩安全限值**。
- 右臂 PD 保持 kp=20、kd=1，不复制仿真更高的 PD 增益。
  名义右臂仍为 `[-4,+1,0,-7.8,0]°`；左臂仍为固定 PD `[-4,-1,0,-8.1,0]°`，唯一腰 yaw 为 0。
- 用户已确认左右水瓶各 250 g，与 XML 质量一致；绑法、质心、惯量、关节 armature／摩擦仍未标定。
  旧持瓶目标带有明显 PD 静差，新增重力前馈后不能保证实物仍保持同一姿态。
- 主动阶段把力矩变化范围先纳入候选搜索，再检查最终输出；不能先验证候选、后逐轴裁剪而不复查。
  18 s 后不突然去掉重力前馈；正常退权保留支撑计算，直到 weight=0 才清零前馈。
- weight 不满 1 时内置运控与 Arm SDK 怎样混合，未包含在条件模型中。
  正常过渡代码的存在不等于已证明实际接管／交还平顺，尤其不能保证断网后的交还。

## 4. 本轮实测的离线结果

这里“实测”仅指实际运行程序得到的计算结果和主机耗时，**不是机器人实验**。

### 数学、代码与消息检查

本轮针对 MPC、共享运行器、日志、力矩链及分析器的 54 项测试通过，另有 18 项 PID／6 ms 回归通过，
**合计 72 项**。新增迁移测试 11 项，
覆盖实测状态不会被旧参考替代、原 QP 对照、原 mapper 对照、耦合／非线性候选检查、失败回退、
仅一次 PD、正常过渡、独立日志审计，以及命令行和底层运行器均拒绝真实力矩输出。
动力学测试包含 50 组随机状态；精简质量矩阵／偏置计算与完整 MuJoCo 流程一致，没有通过删检查省时间。

### 三秒闭环模型对照

给身体指定同一段解析扰动，右臂闭环积分；控制每 6 ms 更新，试验中的 PD／物理积分每 2 ms 更新。
使用理想已知扰动预测，隔离执行层问题。起始已经处于名义姿态、weight=1；不是全身行走，
不含完整接管／退权，也没有把冻结旧关节轨迹当成新控制器的响应。

模型匹配时，四组均完成三秒：

| 路径 | 关节加速度跟踪 RMSE（rad/s²） | 瓶轴倾斜 RMS（度） | 末端线加速度向量 RMS（m/s²） |
| --- | ---: | ---: | ---: |
| 旧持续参考＋PD | 2.36290 | 3.15228 | 2.58485 |
| 实测状态 MPC＋名义逆动力学＋PD | 0.39282 | 1.95739 | 1.26363 |
| 实测状态 MPC＋局部多候选修正 | 0.00833 | 1.98609 | 1.20955 |
| 原仿真未消元 QP＋同一条件 mapper | 0.00833 | 1.98609 | 1.20954 |

结论：新 MPC 与原仿真 QP 基本一致；修正总力矩有效消除了这套匹配模型里的大部分加速度误差。
但姿态 RMS 相比名义逆动力学略差，不能声称所有指标都改善。旧参考路径约束不同，
也不能把整张表当成仅改变一个因素的严格消融实验。

压力测试只改被控模型：瓶子质量／惯量 +20%、少量摩擦、观测延迟 4 ms、执行延迟 6 ms。
修正版在 **2.484 s** 因 QP 不可行而停止；当时实测输入右肩 pitch 为约 −5.00088°，
已经越过 −5° 外层边界且速度仍向外。独立 HiGHS 可行性检查也确认约束无解。
名义逆动力学版在 2.574 s 发生类似失败；原 OSQP 对照在 2.478 s 先出现收敛失败，
该时刻独立可行性检查仍有解，**不把 OSQP 收敛失败误写成物理约束无解**。

失败轨迹的指标只覆盖停止前，不与三秒完整指标混成一次“通过”统计。
这不证明真实机器人一定会这样失败，因为所加偏差是假定压力条件；但它足以反驳“照搬仿真就已经稳健”。

图和可复算数据见[闭环对照图](../../evaluation_summary/hardware_mpc_torque_migration_20260928/closed_loop/comparison.png)、
[完整结果及失败状态](../../evaluation_summary/hardware_mpc_torque_migration_20260928/closed_loop/summary.json)。
每个 NPZ 保留时间、q/dq、期望／实际加速度、力矩、末端指标和计算耗时，曲线可直接复画。

### 完整周期计时与独立消息审计

CPU 2、BLAS 单线程、普通 SCHED_OTHER；主机 PREEMPT_RT 内核已开启，未改系统调度／电源设置。
开启现有 learned_filtered 预测器，用合成静止反馈跑两轮完整接管／控制／退权。
它主要测固定姿态下计算路径，不是行走反馈或最坏状态测试，也不证明物理跟踪性能。

| `[5,18)` 主窗口 | 第 1 轮 | 第 2 轮 |
| --- | ---: | ---: |
| 控制器耗时 p99 | 3.017 ms | 3.111 ms |
| 预测全流程 p99 | 1.992 ms | 2.078 ms |
| 完整离线工作 p99 | 6.122 ms | 6.577 ms |
| 完整离线工作最大值 | 9.838 ms | 9.783 ms |
| 实际循环间隔最大值 | 12.067 ms | 12.300 ms |
| 6 ms 截止时间超限 | 29 / 2137 | 67 / 2099 |

合计 **96 / 4236，约 2.27%** 超时。两轮流程完整、日志无丢弃／失败、weight 最终为零，
但**6 ms 时间要求没有通过**。耗时已包含缓存推进、模型查询、QP、逆／正动力学、候选筛选、
消息 CRC／序列化和日志入队；尚未包含真实 DDS、网络和 RPC 竞争，不能据此乐观扣除余量。

已消除重复状态准备、无关接触组装以及候选包重复 CRC；未修改约束或把检查移出计时窗口。
两轮独立日志审计核对了 3472／3427 条命令，其中主动 MPC 2471／2426 条；
实测初值、一拍参考、PD 算术、最终力矩／变化率和最终零前馈均通过。
旧 `audit_hardware_mpc_replays.py` 假定持续参考积分，**不用于新路径**。

## 5. 程序入口与复验

| 文件 | 职责 |
| --- | --- |
| [hardware_mpc_torque_control.py](../../tools/g1_commissioning/hardware_mpc_torque_control.py) | 实测状态 MPC、执行层、正常过渡、离线消息 |
| [hardware_arm_inverse_dynamics.py](../../tools/g1_commissioning/hardware_arm_inverse_dynamics.py) | 移动基座 RNEA、独立条件质量矩阵／偏置力 |
| [hardware_torque_mapper.py](../../tools/g1_commissioning/hardware_torque_mapper.py) | 局部扰动、候选排序与试算、重检回退及完整候选诊断 |
| [validate_measured_torque_mpc.py](../../tools/g1_commissioning/validate_measured_torque_mpc.py) | 移动基座闭环对照、偏差／延迟试验、失败可行性诊断、画图 |
| [benchmark_hardware_mpc.py](../../tools/g1_commissioning/benchmark_hardware_mpc.py) | 完整离线周期计时；反馈回放不响应新控制 |
| [audit_measured_torque_replay.py](../../tools/g1_commissioning/audit_measured_torque_replay.py) | 从日志重新计算消息／参考关系，不相信单独的 passed 标记 |
| [test_measured_torque_mpc.py](../../tools/g1_commissioning/tests/test_measured_torque_mpc.py) | 与原仿真和消息语义的独立回归检查 |

以下均不会连接机器人。输出目录必须尚不存在：

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
conda activate g1_mpc
python tools/g1_commissioning/g1_walk_mpc.py --preflight --cpu 2

python tools/g1_commissioning/validate_measured_torque_mpc.py \
  --duration 3 --cpu 2 \
  --output-dir evaluation/hardware_shadow/commissioning/torque_closed_loop_recheck

python tools/g1_commissioning/benchmark_hardware_mpc.py \
  --actuation measured_torque_preview --predictor learned_filtered --runs 2 --cpu 2 \
  --output-dir evaluation/hardware_shadow/commissioning/torque_timing_recheck

python tools/g1_commissioning/audit_measured_torque_replay.py \
  evaluation/hardware_shadow/commissioning/torque_timing_recheck/run1
python tools/g1_commissioning/audit_measured_torque_replay.py \
  evaluation/hardware_shadow/commissioning/torque_timing_recheck/run2
```

闭环比较器会保存失败案例而不是停止全部比较；进程退出不等于所有案例通过，要看 summary 的各项 `status`。
计时器的 `complete` 同样仅代表流程完整，要单独看 `deadline_misses`。

相关测试命令保存在[本轮证据入口](../../evaluation_summary/hardware_mpc_torque_migration_20260928/README.md)。
源码、模型和库哈希随结果保存；旧名义逆动力学阶段的证据另见
[历史逆动力学说明](HARDWARE_MPC_INVERSE_DYNAMICS.md)，不能把它的更短时间用于本版。

## 6. 接下来应做什么

主线继续使用**实测状态 MPC＋逆动力学＋局部候选修正**，不退回“命令参考就是实际状态”的假设。
下一步先在离线处理两件实际暴露的问题：

1. 把观测年龄、已发送命令及执行延迟纳入短时状态估计／跟踪模型，检查负载偏差下的恢复能力；
   维持外层边界，不以扩大活动范围、忽略失败或删约束来消除不可行。
2. 降低完整周期的尾部耗时，为真实 DDS 留出余量；用相同记录范围重测，不能只报 QP 求解时间。

这两项通过后，再准备新力矩路径的现场正常／异常交还，并从静止小幅响应、重力补偿渐入／渐出开始。
用 q/dq、tau_est 和命令日志评估真实延迟与加速度跟踪；tau_est 不是独立绝对力矩标定仪。
目前没有发布新非零力矩的现场指令，也没有声称这一步可以被一次离线测试替代。
