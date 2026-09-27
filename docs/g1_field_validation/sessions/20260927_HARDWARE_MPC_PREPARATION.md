# 2026-09-27：6 ms 真机 MPC 离线准备

## 结论范围

新增独立 `g1_walk_mpc.py`：左臂固定已验证的非零 PD 姿态、右臂参考轨迹 MPC、单腰 yaw 固定，
腿部保留内置运控。对应说明是 [HARDWARE_MPC.md](../HARDWARE_MPC.md)。
这是**可以进入受控首次上机的候选版本**，不是已经证明有效或零超时的硬实时控制器。
今天没有连接机器人、初始化 DDS、创建真实 publisher、修改模式或发送控制命令。

本轮同时补齐 6 ms PID 的显式首次复验开关，默认输出锁仍保留；不需要现场改源码解锁。
已验证的 20 ms PID 保留于 `a1d0197`，此前的 6 ms 离线候选为 `5ddf7cf`。
下一次顺序：**6 ms PID 复验 → 静止 MPC → 行走 MPC**。本记录不能代替这些现场结果。

## 实现及与仿真的差别

- 复用仿真九段 × 6 ms 的 MPC 目标函数；代数消元为 45 个加速度变量，完整原约束还原检查。
  仿真源码／力矩执行器未改。硬件通过 `rt/arm_sdk` 的 q/dq + kp/kd 控制，不是 `rt/lowcmd`。
- 优化并保持连续的命令参考，用实测 q/dq 跟踪误差修正预测几何；固件 PD 动态尚未辨识。
  名义角 ±5°、参考速度 0.07 rad/s、参考加速度 0.20 rad/s² 保留。没有新增实测腕速停止门。
- 本地 C++ 运动学库放在可重建的 `build/right_arm_rnea/`；模型、QP 在任何 DDS 初始化前加载和预热。
- 固定 H0 预测库随 Git 发布，约 6.7 MB、15,099 行、33 个输入量；七条开发轨迹训练，旧五条不使用。
  每拍一次最近邻检索，得到未来九段滤波扰动，不在现场重新训练。
- 保留完整 `[5,18)` 主评价窗口，以及原始 IMU、关节、预测输入／输出、参考、求解和时序记录。
  分析器沿用 PID 的瓶子中心 FK/H0 指标，并检查主窗口状态和命令覆盖，明确标记 MPC 身份。

## 为什么换求解器

原 OSQP 及早期消元版本在部分实测轨迹、尤其参考接近位置边界时出现未收敛或不可行。
排查后修正了参考 governor 的离散制动余量，并进行了数值缩放与等价消元。
部分 OSQP 版本在第 09 条完成，但第 10 条仍失败，因此没有把单条通过宣称成完成。

最终采用 DAQP 0.9.1：同一凸 QP、无软化约束、无无效解输出。原生求解预算 3.5 ms，
只接受最优状态，返回后检查墙钟时间及完整约束残差 ≤1e-6。
本机只补装这一项依赖，没有重装、升级或移动 SDK2／既有 Python 环境。
依据：[DAQP 官方实现](https://github.com/darnstrom/daqp)、
[参数及返回码](https://darnstrom.github.io/daqp/parameters/)。

失败尝试没有删除。本地 `evaluation/hardware_shadow/commissioning/` 下
`mpc_offline_20260927_a` 至后续试验目录、`..._final10`、`..._row10` 等保留旧参数与失败日志；
不要将它们的耗时与最终 DAQP 版本混写。最终目录以 `mpc_offline_20260927_final_daqp` 为前缀。

## 数值与行为验证

1. 30 项 Python 测试通过：等价 QP 消元、与原代价逐项对照、DAQP 与独立高精度求解器解对照、
   600 周期命令范围／连续性、失效解拒绝、因果预测、SO(3)、模型溯源、输出显式许可，
   PID 既有控制／退权、时钟、MPC/PID 指标一致及记录缺口拒绝。
2. C++ 与 Python Pinocchio 一致性：30 个随机状态，以及 13 个批量运动学节点，
   本次输出的力矩／雅可比／姿态等最大差异均为 0（打印精度内）。这不是实物模型标定。
3. 冻结模型与原研究保存的预测逐项比较：09–12 各 2,157 个锚点，合计 931,824 个预测数值；
   最大绝对差 `7.105427357601002e-15`。相同接收样本的因果预处理差异最大 `1.7763568394002505e-14`。
   证明导出／在线实现一致，不是新的盲测；现场 2 ms 接收筛选与原始高频记录仍可能不同。
4. 默认离线 preflight 通过：真实 QP 成功、SDK LowCmd 构造／CRC／1004 字节序列化成功，
   `dds_initialized=false`、`publisher_created=false`、`robot_connected=false`。

模型 SHA256：`0e7b25f9a8ddb6dcb893891cee21cfbc216f9174f6efe2e7c1d3f400ce1705c7`。
对应的冻结清单、源数据／方法哈希随 `assets/g1_hardware_mpc_predictor/manifest.json` 保存。

## 当前电脑与时间口径

实查内核 `6.8.1-1057-realtime`，`/sys/kernel/realtime=1`，PREEMPT_RT 已开启。
当前进程仍为 `SCHED_OTHER`、优先级 0；没有调整实时调度权限、IRQ、CPU governor 或系统配置。
平台 profile 为 performance，intel_pstate 为 powersave／balance_performance；该名称不代表 CPU 固定低频。
控制线程 CPU 2、BLAS 单线程；CPU 2 与 CPU 1 共享 SMT 核，不是独占 CPU。

以下回放运行真实预测、几何、QP、SDK 包构造／CRC／序列化及异步日志，按 6 ms 墙钟时间槽运行。
包含原始样本送入缓存和日志入队；不包含真实 DDS 解包、网络 Write、Loco RPC 并发或电机响应。
回放反馈来自旧轨迹，不会响应新 MPC 命令，所以不能用于声称物理控制改善。

### 最终 DAQP 版本：六轮完整回放

下面只统计 `[5,18)`；耗时单位 ms。P99 表示 99% 的样本不超过该值，不代表最大值。

| 输入轨迹／重复 | 完整工作 P50 | 完整工作 P99 | 完整工作最大 | 实际周期 P99 | deadline miss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 09／1 | 3.617 | 4.212 | 5.584 | 6.012 | 1 / 2165 |
| 09／2 | 3.644 | 4.256 | 6.707 | 6.013 | 3 / 2163 |
| 09／3 | 3.650 | 4.236 | 6.307 | 6.014 | 1 / 2165 |
| 10／1 | 3.649 | 4.180 | 4.711 | 6.016 | 0 / 2166 |
| 11／1 | 3.596 | 4.239 | 6.109 | 6.013 | 1 / 2165 |
| 12／1 | 3.671 | 4.394 | 7.067 | 6.015 | 3 / 2163 |

合计 12,987 个主窗口周期，9 次错过 deadline，约 **0.0693%**；超时跳过旧槽、不追赶发送。
最长实际循环间隔 12.014 ms。晚唤醒也可能导致 deadline miss，即使本拍工作耗时不足 6 ms。
这六轮全部正常退权到 0，日志丢失／写盘失败均为 0；不能把这项结果写成“硬实时已通过”。

预测全流程（接收缓存筛选／滤波／检索／预测域构造，不只是最近邻搜索）P99 为 **1.114–1.222 ms**，
控制计算 P99 **2.396–2.589 ms**。不同阶段的 P99 不能简单相加当成完整循环 P99。
最终版本之前的重复测量有更长尾部，说明负载／调度会改变尾延迟；未以无限重跑筛选零超时。

另逐条审计所有 active MPC 命令：所有 QP 解均成功，无 fallback，完整约束最大残差约 `9.14e-12`；
参考角最大偏移 5°、速度 0.07 rad/s、加速度 0.20 rad/s²（浮点尾差约 `4e-13`）。
回放确实到过参考边界；反馈不会响应新命令，因此它既不能证明真机会饱和，也不能证明控制有效。
现场要看实际跟踪与触边频率，不在此次离线准备中放宽范围。

可分享的[机器可读摘要](../../../evaluation_summary/hardware_mpc_20260927/validation.json)
包括每轮完整统计、主机状态、参数、模型／源码／输入／日志哈希和预测一致性结果。
原始完整日志、逐帧 timing 和失败轮保留在本地上述目录。

## 如何复验

从仓库根目录、`g1_mpc` 环境运行：

```bash
python tools/g1_commissioning/g1_walk_mpc.py --preflight --cpu 2

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest \
  tools.g1_commissioning.tests.test_hardware_mpc \
  tools.g1_commissioning.tests.test_hardware_pid_control \
  tools.g1_commissioning.tests.test_pid_6ms \
  tools.g1_commissioning.tests.test_analyze_hardware_pid \
  tools.g1_commissioning.tests.test_mpc_host

python tools/g1_commissioning/benchmark_hardware_mpc.py \
  --source-npz evaluation/hardware_shadow/commissioning/walk_h0_predictor_study_20260925/data/trial09.npz \
  --output-dir /tmp/g1-mpc-new-replay --runs 3 --cpu 2

python tools/g1_commissioning/verify_hardware_mpc_predictor.py

# 从六轮已保存的原始日志重新计算参考边界审计和时间统计总数：
python tools/g1_commissioning/audit_hardware_mpc_replays.py \
  evaluation/hardware_shadow/commissioning/mpc_offline_20260927_final_daqp{09,10,11,12}
```

输出目录必须是新目录。最后两条使用本地研究数据；不提供 `--source-npz` 时，基准程序可以用
合成静止数据做新克隆的基本检查，但它不能代替实测行走数据回放。
原始日志仍在被忽略的 `evaluation/`，不提交；可分享的模型、程序、验证摘要和文档提交到 Git。

## 明天仍需验证

- 6 ms PID 的实际 DDS 周期、状态年龄和重复包、平顺性、停车及三秒退权。
- MPC 静止接管／反馈方向，再单独验证走动；不能用离线 QP 成功推断稳瓶有效。
- 实际 DDS／日志／RPC 并发下的完整循环，超时比例、最坏间隔与状态年龄。
- MPC 闭环下的新身体扰动是否仍在冻结模型适用范围内，是否频繁碰到参考边界；
  用本轮完整输入与预测日志和后续真实测量对照，再比较 PID 与 MPC 的 `[5,18)` 瓶子指标。

预测器含 15 Hz 因果滤波；单层低频延迟约 9.64 ms，角加速度包含两层。
这不是无延迟冲击预测。纯主机耗时、DDS Write 返回和电机闭环延迟仍是不同的量。
