# 2026-09-28：MPC 首次上机前的软件可靠性补充

上一版可回退基线：`113fc7bb47555cc80d06ade60b33af44a0a06385`。
本轮不更改 MPC／PID 增益、名义姿态、参考角／速度／加速度范围、预测模型或行走时间表。
没有连接机器人。这里的软件测试不能替代首次 6 ms PID 与 MPC 实机验证。

## 发现并修复了什么

| 问题 | 本轮处理 | 不代表什么 |
| --- | --- | --- |
| 行走 RPC 客户端初始化在异常保护外，线程可能直接退出而主循环未收到停止信号 | 初始化纳入保护，失败时设置停止标志，不进入正常行走流程 | 不能保证机器人内部服务可用 |
| arm DDS Write 失败后，旧故障退权路径仍可能再次尝试 arm Write | 锁存写入失败；返回 false 或抛异常后禁止后续 arm 输出，继续记录失败并尝试零速度请求 | 通信已失效时不能保证三秒退权，更不把拔网线当急停 |
| 部分订阅初始化失败发生在清理范围外 | 初始化也进入现有 try/finally，清理已建立的端点／线程，恢复信号与亲和设置 | 尚无真实 SDK 初始化故障实验 |
| 检查时先取时间、后读更新后的快照，可能将刚收到的数据误判成“未来” | 状态快照先取、校验时钟后取；FSM 时钟在快照锁内读取 | 不放宽 100 ms 状态／600 ms FSM 超时，不接受真正未来时间戳 |
| 写盘线程失败且队列已满时，关闭流程可能卡在投递结束标记；磁盘卡顿也可能无限等待 | 写线程自行排空并关闭，关闭等待有界，失败／超时保留 failed 状态；CLI 明确提示 capture incomplete | 五秒关闭等待不是磁盘永久写入保证；超时数据必须视为不完整 |
| 新版日志关闭的最终审查发现：队列判空后恰好入队最后一行，再发生关闭，可能漏写却报成功 | 关闭判定与入队共用状态锁，再检查队列为空；增加确定性并发测试 | 不把重复运行未出现竞争当成并发正确性证明 |
| 分析器将 stationary 记录写成 walk；只看主窗口覆盖，没有显式区分后续退出故障 | 静止／行走使用不同标签；单列 session_status、capture_quality 与警告 | 指标能算出来不等于整次实验成功 |
| 提前停止后，退权命令仍带 task time，可能被误用来补齐 `[5,18)` | 主窗口内 inactive MPC／提前停止命令直接拒绝完整控制评价 | 不丢弃原始失败日志 |
| 参考加速度审计只读程序自身的 ddq 字段 | 从连续发送的 q/dq、实际积分间隔与序号独立重算，并与日志字段交叉比较 | 这是发送参考审计，不是真实关节加速度测量 |

如果 `[5,18)` 控制完整，但 18 秒以后退出失败，分析器仍保留这段可用指标，
同时报告 `fault_recorded / review_required`。缺失退出、记录结束或 SDK 关闭证据也要求复核。
`capture_drained` 是关闭前写入的事件标记，不单独证明全部数据持久落盘；CLI 的关闭错误也必须看。

## 离线验证方法

`test_hardware_runner.py` 执行真正的 `run_device` 和线程／停止／退权流程，
只替换 SDK 通信为内存对象，禁止创建网络 socket。控制器用静止桩，真实 MPC 数学计算另由回放验证。
测试时使用 2 倍时钟，沿用实际三秒退权语义；不把这种测试耗时当作实时性能数据。

覆盖十二种场景：静止正常结束、行走正常结束、计算异常、操作者停止、Write 返回 false、
Write 抛异常、L2+B、FSM 离开 500、LowState 过期、订阅初始化失败、拒绝确认、速度 RPC 初始化失败。
检查了无模式 API／无 `rt/lowcmd`、静止任务零速度、停车、端点清理、信号恢复，
以及适用情况下的至少三秒渐退、每拍 weight 降幅不超过 0.002。

独立测试还覆盖了快照／时钟顺序、真实未来／过期状态拒绝、日志队列满／写线程失败／阻塞，
以及七种结果分析场景。部分初版假通信测试因虚构“未来快照”、未同步 heading 缓存及十倍时钟
放大测试负载而失败；测试桩已修正，没有通过放宽生产阈值让测试通过。

六轮旧回放重新独立审计：q 积分残差最大 `5.421010862427522e-20 rad`，
重算加速度与记录值差异为 0；最大参考加速度 `0.20000000000038515 rad/s²`（浮点尾差），
未发现隐藏的参考跳变。旧回放的控制、模型和时间结果仍属于 09-27 版本，不覆盖为新版测量。

## 最终版本验证结果

完整相关测试 **52 项通过**（54.972 秒），包含上述十二种主循环场景及关闭竞争测试。
离线 preflight 通过：实际 QP 返回 solved，LowCmd 构造／CRC／序列化为 1004 字节；
未建立 DDS participant 或 publisher。源码哈希、原始回放哈希及完整统计见
[可分享的验证摘要](../../../evaluation_summary/hardware_mpc_20260928/validation.json)。

最终代码使用同一份 trial10 数据、CPU 2、BLAS 单线程，做两轮完整 21 秒定时回放。
统计区间均为 `[5,18)`，不是仅选稳态：

| 指标 | 第 1 轮 | 第 2 轮 |
| --- | ---: | ---: |
| 控制器计算 p99 | 2.563 ms | 2.511 ms |
| 预测器计算 p99 | 1.234 ms | 1.198 ms |
| 完整单拍工作 p99 | 4.501 ms | 4.479 ms |
| 完整单拍工作最大值 | 7.923 ms | 6.543 ms |
| 实际周期 p99 | 6.014 ms | 6.014 ms |
| 实际周期最大值 | 12.067 ms | 12.009 ms |
| 超过 6 ms 截止时间／样本数 | 6 / 2160 | 2 / 2164 |

合计 **8 / 4324 拍超时，约 0.185%**，不是零超时或硬实时保证。
两轮均正常完成、最终 weight 为零、日志无丢弃／失败，独立 q/dq/dt 审计通过。
完整工作包含离线数据输入、预测、实际 QP、消息构造／CRC／序列化及日志入队，
**不包含真实 DDS 收发、网络延迟、现场 RPC 线程竞争或电机执行延迟**。
回放状态不受计算出的命令影响，不能据此宣称瓶子控制效果得到改善。

主机仍为 PREEMPT_RT 内核，但进程是普通 SCHED_OTHER、优先级 0；CPU 2 绑定不等于独占核心。
本轮没有改系统实时调度或电源设置。正式真机运行仍须看实际完整周期、状态新鲜度和正常退出。

最终关闭竞争修复前还做过两轮回放（5 / 4327 拍超时，p99 完整工作约 4.34–4.35 ms）；
它们连同源码哈希均保留在摘要，未混入最终版本统计，也未挑选更快结果替代最终结果。

## 复验命令

以下仅离线，不要给测试添加真实网卡或输出许可参数：

```bash
conda activate g1_mpc
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest \
  tools.g1_commissioning.tests.test_hardware_mpc \
  tools.g1_commissioning.tests.test_hardware_pid_control \
  tools.g1_commissioning.tests.test_pid_6ms \
  tools.g1_commissioning.tests.test_analyze_hardware_pid \
  tools.g1_commissioning.tests.test_mpc_host \
  tools.g1_commissioning.tests.test_hardware_runner \
  tools.g1_commissioning.tests.test_hardware_health \
  tools.g1_commissioning.tests.test_hardware_journal \
  tools.g1_commissioning.tests.test_hardware_mpc_analysis \
  tools.g1_commissioning.tests.test_mpc_replay_audit

python tools/g1_commissioning/g1_walk_mpc.py --preflight --cpu 2

# 输出目录必须是新目录；raw trial10 数据保存在本机，不随小型摘要上传。
python tools/g1_commissioning/benchmark_hardware_mpc.py \
  --source-npz evaluation/hardware_shadow/commissioning/walk_h0_predictor_study_20260925/data/trial10.npz \
  --output-dir evaluation/hardware_shadow/commissioning/mpc_robustness_recheck \
  --runs 2 --cpu 2
python tools/g1_commissioning/audit_hardware_mpc_replays.py \
  evaluation/hardware_shadow/commissioning/mpc_robustness_recheck
```

现场顺序不变：先 6 ms PID，再静止 MPC，正常后才单独运行行走 MPC。
本轮没有新增现场勾选项，也没有新增实测关节速度停止条件。
