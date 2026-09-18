# 真机通信与主机计时：先用只读 phase probe

2026-09-17：已增加代码并完成离线测试和本机定时冒烟，**尚未取得本轮真机通信数字**。
这一步放在 `g1_phase_probe`，不为测延迟新增运动命令，也不修改 `g1_walk_capture` 的动作参数。

## 先说清楚“来回延迟”

电脑发一个查询，机器人服务回一条回复，可以用电脑自己的单调时钟量出
**应用层 RPC 往返时间**：调用前到调用返回。它包括 SDK/DDS、网络、双方调度和机器人服务处理。
不需要把两台电脑时钟对齐，但它不是纯网线／单向 DDS 时间，不能直接除以二。

`rt/lowstate` 和 `rt/secondary_imu` 是机器人持续发布的状态，**不是上一条手臂命令的确认包**。
发命令后收到下一帧状态，并不能证明那帧已经反映该命令。
当前接口没有能用于这件事的命令序号回显；torso IMU 还没有源时间戳。
LowState.tick 不是已经与本机单调时钟同步的时间，不能拿两个值直接相减。

## 这次实际测什么

| 指标 | 怎样测 | 不代表什么 |
| --- | --- | --- |
| FSM / phase 成功查询 RTT | 同一主机 request→reply；启动发现期剔除 | 不是电机执行回执；GetPhase 失败／超时另列 |
| IMU / LowState 收包间隔 | 相邻回调开始时间；频率、p95/p99/max | 包含发送周期与接收抖动，不是单向传输时延 |
| 回调开始→日志入队时间戳 | 复制、LowState CRC／inbox 锁、队列锁等 | 不含消息到达网卡之前、DDS 解码到回调之前的耗时；时间戳在队列插入前 |
| 日志排队／序列化 | 入队时间戳→出队→JSON 序列化完成 | 不是控制链必须同步等待的时间，也不是磁盘 fsync 完成时间 |
| CPU 7 的 6 ms 唤醒迟到 | 实际唤醒−固定绝对目标时刻 | 不是控制器计算时间 |
| 6 ms 状态读取循环 | 从醒来到读取、健康检查、上次记录入队和控制台打印完成 | 只是轻量只读 probe，没有 MPC、PID、运动学或力矩认证 |
| 消费时的主机侧状态年龄 | 读取快照完成−该状态本机接收时间 | 不包含传感器采样、机器人估计器和传输前半段 |

输出 mean、p50、p95、p99、max 和 `>6 ms` 数量；对 probe 的
`release_to_finish`，6 ms 才是对应周期期限。对于 RPC 和状态年龄，`>6 ms` 只是参考计数。
循环迟到时跳过错失时隙，不突发补跑；缺拍、错核、CRC 错误、tick 真回退、状态不新鲜分别记录。
日志保留原始样本；统计由程序结束后的离线脚本完成，不在线构造扰动模板。

## CPU 和“普通模式”怎么处理

本次默认 `--timing-cpu 7`，对应仿真用过的逻辑 CPU；可显式改编号，但报告必须标明不可直接视为同条件。
程序先验证该 CPU 可绑定，在创建 DDS/RPC/日志线程前，把支持线程的继承亲和性排除该物理核及其 SMT 同胞。
READY 后只把主线程的 6 ms probe 绑定到 CPU 7；程序结束后不影响其他进程。
因此**不要外面再 `taskset -c 7` 包住整个程序**，否则支持线程没有可用核，会在联网前拒绝。

要求调用线程为普通 `SCHED_OTHER`；不自动提权到 RR/FIFO，不改 governor、IRQ、GRUB 或内核。
日志在启动、READY、结束时保存内核、RT 标志、CPU governor、瞬时频率、SMT、
各线程实际 affinity／scheduler，以及能读取到的网口 carrier、速率、MTU、错误和丢包计数。
快照频率不是运行期间持续实测频率，也不声称自动实现了 IRQ 隔离。

2026-09-17 本机只读检查：

- 内核已经是 `6.8.1-1057-realtime`，`/sys/kernel/realtime=1`，已有 CPU 6–7 隔离启动参数。
- 当前普通进程 `SCHED_OTHER`；CPU 7 governor 为 `powersave`。
- 因此本次应标为“**现有 RT 内核＋普通调度＋现有 powersave**”，不是普通 generic 内核，
  也不是已经调到最快。保留现状，不为测试重启或修改系统。

仿真 parent/worker 当时同绑 CPU 7；这里是只读 probe 固定 CPU 7、支持线程分核。
可对照 6 ms 预算和统计口径，但工作负载／线程布局不同，不能拿空载 probe 冒充 MPC 整段计时。

## 现场运行与报告

仍按 [原始采集指南](RAW_WALK_CAPTURE.md) 先核对连接，机器人由你调至 FSM 500。
只读程序不替你切换模式。按该指南重新编译后，从仓库根目录运行：

```bash
NIC=enx6c1ff701509c  # 按本次连接检查确认
CAPTURE_DIR="evaluation/hardware_shadow/commissioning/phase_timing_$(date +%Y%m%d_%H%M%S)"
mkdir -p evaluation/hardware_shadow/commissioning
/tmp/g1-capture-device/g1_phase_probe "$NIC" \
  --output-dir "$CAPTURE_DIR" --permit-read-only PHASE_30S_READ_ONLY \
  --timing-cpu 7
```

看到 READY 后进行 30 秒观察。先静止观察一轮；另一轮按原计划用遥控器走几步、再停住，
保留现场／视频时间记录。若希望得到稳定的基线，同一种条件重复三轮，勿把不同条件直接混平均。
不并行跑仿真、编译或第二个采集程序，也不同时启动多个命令源。
保持原有 FSM getter 和 phase getter 的频率，不新增 500 Hz 查询压力。

结束后（原始数据不改写）：

```bash
python3 tools/g1_commissioning/analyze_capture_timing.py \
  "$CAPTURE_DIR/raw.jsonl" --output-dir "$CAPTURE_DIR/timing"
```

生成 `timing_summary.json` 和便于阅读的 `timing_report.md`。输出目录必须是新的。
未取得的指标显示 null／“未取得”，不会填成零延迟；残缺 JSON 拒绝分析，
未完成／状态不健康的采集会生成带问题的报告并返回非零，不标为通过。
分析器严格使用 READY 到结束的观察窗口，不把预热／发现期和退出清理揉进平均。

`raw.jsonl` 新增 `g1_host_probe_tick_v1`、`g1_host_timing_environment_v1` 和
`journal_*_ns` 字段；原始四元数、角速度、加速度等字段原样保留。
日志额外记录回调／序列化时间会带来少量开销；这些数据反映“采集打开时”的系统，不是无观测开销极限。

## 对照以前的仿真

[冻结计时记录](../simulation/REALTIME_RUNTIME.md)的 r1 完整 6 ms 区间：

| 条件 | mean | p99 | max | >6 ms |
| --- | ---: | ---: | ---: | ---: |
| nominal | 3.302370 ms | 3.630513 ms | 4.340295 ms | 0 |
| held-out | 3.299516 ms | 3.532013 ms | 4.228574 ms | 0 |

此处不把仿真预算解释成真机余量。
以前测的是状态／预测、控制算法和两次 DDQ-to-torque 等完整计算链，不含真实 DDS／电机。
真机还会有传感器与估计器更新、发包周期、网络／DDS 队列、线程调度、状态配对／滤波、
命令缓冲、固件控制周期及机械响应。这里只补到能够直接观测的主机／查询层。

**不要算“仿真 p99 ＋ RPC p99 = 真机总 p99”**：路径并不相同、部分过程并行，而且分位数不这样相加。
查询消息与整帧关节命令的格式、负载也不同，RPC 往返不是 Arm SDK 延迟的替代测量。

## 以后在 PID / MPC 里仍要加什么

1. 每个真正控制周期绑定本次输入的状态序号、原始接收时间和目标唤醒时间。
2. 分段记录取状态、配对／坐标转换／滤波、预测、PID/MPC、输出检查、编码以及本地 DDS Write 开始／结束。
3. 记录整个周期的 source-age-at-send、release-to-write-complete、丢拍、结果过期／hold／fallback。
   单测 solver 或几个微基准不代替带真实数据与负载的完整 6 ms 区间。
4. `g1_walk_capture` 已有本地 Arm SDK write 和速度 RPC 时间戳，可在获准运行时分析；
   它的控臂周期为 20 ms、速度刷新约 50 ms，仍不是未来 6 ms MPC 周期验证。
5. 若要量“命令→真实关节动作”，另做有界、获准的响应实验并保存命令、编码器及现场观察；
   那是包含固件和机械环节的系统响应，不是纯 DDS。精确单向传输需对端时间戳／时钟同步或专门回显支持，
   当前不在 PC1/PC2 部署新服务，也不假设低层固件提供这种接口。

本轮离线验证：SDK-free CTest 9/9、设备构建 CTest 12/12；报告脚本另覆盖成功／失败分组、
预热排除、缺拍、时间倒置、错核和截断日志。CPU 7 的无 SDK 本机定时冒烟（200 拍，约 1.2 秒）
正常完成：唤醒迟到 mean 0.145053 ms、nearest-rank p99 0.185779 ms、max 0.191507 ms，
跳过 0 拍。它只证明该环境下短时本机定时可运行，**不是 DDS／真机测量或长期稳定性结论**。

接口依据：已安装 SDK2 commit `fa925bf6bb3fff439000266d70bde32eb5cd3597` 的
`client/client_base.hpp`、`g1/loco/g1_loco_client.hpp`、`channel/channel_publisher.hpp` 和 `hg/IMUState_.hpp`；
在线交叉查看 [Unitree LocoClient](https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/g1/loco/g1_loco_client.hpp)、
[publisher 封装](https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/channel/channel_publisher.hpp)。
线程亲和性依据 [Linux sched_setaffinity](https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html)，
内核实时性与应用策略区别见 [Linux 实时抢占文档](https://docs.kernel.org/core-api/real-time/index.html)。
