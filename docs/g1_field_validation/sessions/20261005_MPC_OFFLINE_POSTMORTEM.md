# 2026-10-05：离开实验室后的 MPC 整链与接管复盘

范围：只分析保存的数据、修改本地程序、做禁止网络的离线验证；没有连接机器人或发送命令。
**不是只调整电源档，也没有改成位置参考、提高力矩／加速度上限或放宽 10 ms 写前保护。**
6 ms 控制周期、9 段预测、2 ms 原始滤波时间轴、250 g 瓶重与扰动库保持。
本次改动尚待真机原地复验，不应直接进入行走试验。

## 1. 今天失败不是同一个原因

四次现场证据见 [故障记录](20261005_MPC_STARTUP_FAULTS.md)。主要问题分别是：

1. 抬臂／退权套用了活跃 MPC 总力矩检查，第一次退权也中止。此前已改检查适用阶段；
   这确实改变保护覆盖范围，不能用“数值限值没改”把它说成保护完全没变。
2. GC、速度服务构造、主机调度检查发生在任务计时开始后，启动即积压。改为先准备，再设 epoch。
3. 耦合关节的逐轴裁剪搜索漏掉可行力矩。增加有界仿射候选后仍做同样正动力学复查；
   模型可行不等于准确跟踪，故障快照的新候选仍有较大加速度误差。
4. 第四次成功一拍，下一拍核心 wall time 13.802 ms，触发原 10 ms 写前保护。
   当时已经是 CPU 7/FIFO 20，不能把问题全部解释为没开实时模式。

另外，两次现场进入 MPC 时抬臂尚未稳定。把时间压到 6 ms 以下并不能解决接管瞬间的动力学约束。

## 2. 计算之外，还有状态接收的争用

`benchmark_mpc_ingress.py` 不创建 DDS participant／publisher，不调用 Write 或 Loco RPC；
两条普通线程各按 2 ms 周期反序列化官方 SDK 消息，再进入实际的 LowState／IMU 回调、
CRC、预测器入口和日志。反馈是固定合成姿态，**不是响应命令的闭环机器人**。

早期诊断结果（同批探索过程中逐步改代码，非全部严格单因素实验）：

| 配置 | 活跃段平均 wall time | P99 | 超过计划 6 ms 截止时刻 |
|---|---:|---:|---:|
| 同进程、持续两路状态回调，原默认 Python 切换间隔 | 6.974 ms | 9.879 ms | 85.68% |
| 同进程、缩短切换间隔至 0.5 ms | 6.805 ms | 9.287 ms | 70.36% |
| 不并发，只在每拍顺序调用回调（较低接收负载，对照线索） | 4.809 ms | 5.763 ms | 0.93% |
| 同进程、数值开销优化后 | 6.068 ms | 8.730 ms | 50.91% |

第一行本线程 CPU time 平均约 4.631 ms，明显小于 wall time。结合并发／串行和后面的
进程隔离结果，证据支持 **Python 执行锁与线程调度竞争是重要原因**；没有把差额全部伪称为
精确测出的 GIL 等待时间。0.05 ms 的过密线程切换反而更慢，未采用。
旧 5.079／4.963 ms 报告没有这个持续 SDK 回调负载，不能直接外推为现场整链。

## 3. 已落地的修改

- 凝聚 QP 直接消费相同的代价块，不再重复构造／稠密化不用的 145×145 大矩阵。
  状态消元、约束、完整解重构和目标值仍保持；仿真默认路径不变。
- 小型 C++ RNEA／名义动力学调用不反复让出 Python 锁；ABI 2 批量计算原条件质量矩阵和偏置。
  原 Python/MuJoCo 路径保留作对照。库源码哈希／ABI 检查继续拒绝旧二进制。
- MPC 局部 CRC 适配器取消“每个字转 Python 整数再拷回 C 数组”的往返，保留 SDK 打包与
  原生 CRC 算法；120 个随机命令／状态结果与 SDK 逐位相同，不修改 SDK 安装或 PID 单例。
- 显式 `--compute-process`：计算在 CPU 7 的独立进程，主进程控制循环在 CPU 2；
  SDK 回调／日志／RPC 使用工作核。两边不共享 Python 执行锁。主进程仍唯一持有输出权限。
  固定大小共享缓冲区、单请求、序号、9 ms 等待上限；无自动重启、无旧结果补发。
- 计算进程只收到数值、状态时间戳和上一条**成功 Write 的实际 float32 包**。
  主进程先本地保存成功包，异常退权不依赖计算进程继续存活。
  故障测试还发现进程被杀时 `multiprocessing.Event` 通知可能卡死，因此改用有界等待的
  单次信号量协议，并验证子进程退出、超时、错序后拒绝且本地退权仍可完成。
- 首次预测器预热按两路已收到状态的共同时间，而非超前的 wall time，避免没有新回调时首拍回退。
  没有重置运行中的滤波，也没有提高 50 ms backlog 上限。
- 现场时序改为 0–3 s 抬臂，3–5 s 固定姿态＋支撑前馈，5–18 s MPC，再退权。
  3–5 s 仍属于过渡段；不会因“等了两秒”就绕过随后当拍的实际约束。

## 4. 接管时序的模型对照

`compare_mpc_entry.py` 从第四次现场第一条命令读取真实初始 q/dq，其余条件明确假设：
竖直静止身体、250 g 模型瓶、4 ms 观测延迟、6 ms 执行延迟、名义 weight 混合。
它不是完整真机复现，也不把模型混合规律说成已确认的固件实现。

- 3 s 立即切入：3 s 右肘速度约 −0.405 rad/s，3.12 s QP 拒绝。
- 5 s 切入：4.998 s 右肘速度约 +0.00018 rad/s，连续运行到 6.996 s 未拒绝。
- 同一模型、相同力矩／速度／加速度边界，没有为通过扩大数值范围；仅此数值实验把求解器
  wall budget 设为 0.1 s，以免离线算力影响动力学对照，**现场预算未改**。

这支持等待稳定的修改，不保证真实摩擦、固件、身体运动与模型完全一致。
MPC 活跃时右臂本来就不是固定姿态；模型结果不是已经取得瓶子稳定的实机效果。

## 5. 重复计时与验收边界

计时产物位于本机 `evaluation/hardware_shadow/commissioning/mpc_postmortem_20261005/`。
`final_isolated_1..3` 为同一套源码、配置及本地库哈希、相同 CPU 放置与 0.5 ms 切换间隔的重复测试。
普通 SCHED_OTHER 调度、CPU 7 计算／CPU 2 传输、performance 整机档；本轮未在 FIFO 下复测。

| 同版本轮次 | 活跃拍数 | 平均完整工作 | P99 | 最大工作耗时 | 6 ms 截止超时 |
|---|---:|---:|---:|---:|---:|
| final_isolated_1 | 2152 | 4.419 ms | 5.821 ms | 10.137 ms | 15 次／0.697% |
| final_isolated_2 | 2147 | 4.488 ms | 5.508 ms | 8.781 ms | 20 次／0.932% |
| final_isolated_3 | 2159 | 4.431 ms | 5.167 ms | 7.393 ms | 8 次／0.371% |

三轮均完成最终 weight=0、日志丢失=0、写前保护拒绝=0，并核对源码／配置／库哈希一致。
第一轮最大完整工作超过 10 ms，但发生在写前检查之后的日志／记账部分；该轮写前最大
9.849 ms，因此没有触发 10 ms **写前**拒绝。完整截止超时仍如实统计，不能隐藏尾部抖动。
结果显著改善，但依旧不是零超时，不宣称 6 ms 硬实时或真实 DDS 链路通过。

完整结果取每轮 `summary.json` 的 `active`，时间窗 `[5,18)`；`timing.json` 为逐拍计时，
`raw.jsonl` 为合成状态与候选审计。最大值／超时比例同样保留，不只看平均。
不同线程各自的 CPU time 不可误当进程总 CPU；隔离后 `thread_cpu_ms` 只统计主进程控制线程。

这个程序仍未覆盖真实网络接收、DDS Write、全部状态查询／速度 RPC、设备响应和传感器时间戳。
离线遇到写前拒绝时会记下并以假想命令历史继续统计尾延迟；**真机仍拒绝发送并退权**。
独立计算进程超时本身则直接终止离线测试，不掩盖失败。
未完成或异常的探索目录保留，不能算完整通过轮次。

## 6. 复现

先按 [主文档](../HARDWARE_MPC.md) 构建 ABI 2 库。全部以下命令均离线，不连接机器人。

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
PYTHONPATH=tools/g1_commissioning:tools/g1_commissioning/tests \
MPLCONFIGDIR=/tmp/g1-mpl OPENBLAS_NUM_THREADS=1 \
/home/fjk/miniforge3/envs/g1_mpc/bin/python -m unittest \
  test_mpc_compute_process test_mpc_field test_mpc_delay_lifecycle test_hardware_mpc \
  test_mpc_crc test_hardware_runner test_measured_torque_mpc test_native_arm_delay \
  test_hardware_arm_inverse_dynamics

# 每轮使用新的目录，顺序运行；不要同时跑其他 CPU 密集验证。
taskset -c 0-17 env PYTHONPATH=tools/g1_commissioning MPLCONFIGDIR=/tmp/g1-mpl \
  OPENBLAS_NUM_THREADS=1 /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/benchmark_mpc_ingress.py --cpu 7 --isolated --switch-ms 0.5 \
  --output-dir /tmp/g1_mpc_ingress_new_run

PYTHONPATH=tools/g1_commissioning MPLCONFIGDIR=/tmp/g1-mpl OPENBLAS_NUM_THREADS=1 \
/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/g1_commissioning/compare_mpc_entry.py \
  evaluation/hardware_shadow/commissioning/mpc_torque_stationary_20261005_172729/raw.jsonl \
  --output /tmp/g1_mpc_entry_new_result.json
```

55 项相关测试通过，包括数学对照、现场故障快照、原启动顺序修复、父进程退权、CRC、ABI、
有界进程退出与报文提交语义。不是 55 次实机试验。
另运行 `test_mpc_delay_preview test_mpc_recovery test_mpc_host test_mpc_serialization test_mpc_execution`
共 42 项通过，合计 **97 项**；禁止 socket 的本地 preflight 同样通过，QP solved、1004 字节构包、
ABI 2／源码哈希匹配、无 DDS 初始化及 publisher。这里的 preflight 没有实机权限。
原始现场数据和大量基准数据在忽略目录，本 MD 和分析／测试程序在仓库；跨机器复现现场种子
需要另外提供原始日志，算法单测及合成并发基准不依赖这些私有日志。

## 7. 下一次现场只验证尚缺的事实

用主文档显式 `--compute-process` 的**原地力矩**命令；先实际确认 6 ms PID 基线、FSM 500 静止、
既有停止／恢复方式，然后做一轮。不要增加旁支试验，也不要因失败就删掉时限或提高力矩限值。
重点是：能否完整跨过 5 s 进入连续 MPC，真实 DDS 下的逐拍时间／反馈年龄是否合适，
抬臂及退权是否平顺，q/dq/tau_est 是否给出合理响应。通过后再谈行走与 MPC 效果。

`tau_est` 是驱动估计，不是独立扭矩传感器；新加入的计算优化不能证明真实力矩增益准确。
若现场仍反复超时，应依据新分段日志处理实际瓶颈，再决定是否必须修改控制周期和整套预测时间轴。

## 8. 加速度怎样变成力矩（简版）

实测状态 → MPC 算期望关节加速度 → 逆动力学给名义力矩 → 局部多候选修正并用正动力学复查 →
选中总力矩减去同拍 PD 项后，以前馈 `tau` 发出；固件再加一次 PD。
所以不是“只积分成目标角度”，也不是只算一组逆动力学就直接发。候选的加速度是模型预测，
是否在真机达到仍要把后续实测响应与目标对齐检查。
