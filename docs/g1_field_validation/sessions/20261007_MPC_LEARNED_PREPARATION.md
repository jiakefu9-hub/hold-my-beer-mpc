# 2026-10-07：独立学习前馈 MPC 离线准备

操作范围：读仓库／旧实验数据，新增学习版入口、yaw 预测、直接力矩计算，离线验证。
**未连接机器人、未初始化 DDS、未发任何控制命令，未更改主机电源档位或实时权限。**
现场指南：[HARDWARE_MPC_LEARNED.md](../HARDWARE_MPC_LEARNED.md)。
可分享的机器可读结果：[validation.json](../evidence/20261007_learned_mpc_offline/validation.json)。
该文件是首次准备时的结果；10-08 时间容错变更另存
[timing_grace.json](../evidence/20261007_learned_mpc_offline/timing_grace.json)，不覆盖早期源码哈希和结果。

## 基线与修改边界

- 原成功完整行走冻结于 `ab38f9374cdb3572dd17e2dda471e95cf460219c`，成功后的基础修复为 `46df505`。
- 原入口／配置／仿真不切换数学路线；只增加显式新版分派和可覆盖的 policy 类型。
- 新入口和新配置采用旧成功的限值、增益、目标姿态、6 ms、9 步、行走／停车／退权时间。
- yaw 回拉纳入预测：用净加速度变量表达含 PD 的状态方程，名义输入代价相应作等价变换。
  加速度限值现在覆盖含 yaw PD 的净加速度；不再末端额外加力。
- 删除新版同一仿射模型的重复候选搜索，保留最终检查与原有异常退权。
- 学习库原样使用，没有重训、没有增加 pitch PD、没有恢复三个动态末端代价，也未假设已辨识电机延迟。

## 检查结果

1. **56 项相关测试通过。** 包括 yaw 回拉变量变换的状态／代价等价，移动基座下直接力矩与独立 RNEA
   一致，发包 PD 只加一次，最终力矩越界拒绝，未来姿态改变 MPC 动作，独立进程计算一致，
   子进程失效后父进程保留退权能力，以及原基线的现场、制动、序列化测试。
2. 新入口 `--preflight` 通过；库、QP、IDL、CRC 构包正常，DDS／publisher 均未创建。
3. 预测库复算 09–12 四条已看过的对照轨迹：931,824 个预测数值，最大差异 `1.07e-14`；
   因果预处理最大差异 `1.78e-14`。这是导出／运行时一致性，不是新盲测，也不是预测误差小到这个量级。
4. 首次完整 MPC 的保存状态因果回放，三种方案均完成 2,322 个活跃命令，含完整 5–18 s 的 2,155 拍，
   正常退权最终 weight=0；没有使用未来反馈。学习版实际查库 2,155 拍。
5. 新版净加速度与最终力矩回算的最大差异约 `4.77e-7 rad/s²`，为 QP 容差／裁剪量级；
   同模型代数一致不等于真机实现了这个加速度。

### 真实日志回放时间

包含因果入队／预测／控制计算／本地构包序列化，不包含真实 DDS、RPC、独立进程通信与日志 IO。
固定 CPU 2，普通调度；QP 保留现场 `3.5 ms` 预算，没有打开 `--math-only`。

| 路径 | 平均 ms | P99 ms | 最大 ms |
| --- | ---: | ---: | ---: |
| 原成功基线逻辑、hold_current | 2.668 | 3.027 | 5.250 |
| 新版 yaw 预测／直接力矩、hold_current | 2.407 | 2.679 | 3.050 |
| 新版 yaw 预测／直接力矩、学习预测 | 2.869 | 3.269 | 3.644 |

新版学习预测部分平均 `0.732 ms`，控制／计划平均 `1.735 ms`。
三组在这个窗口计算均没有超过 6 ms；这不是整体现场实时证明。

同一保存状态下，打开学习预测后的五关节总力矩平均绝对变化约
`[0.149,0.088,0.041,0.083,0.012] N·m`，最大约 `[1.470,1.017,0.495,0.832,0.155] N·m`。
这证明学习预测确实进入计算；**保存状态不会响应新命令，不能从回放推出瓶子实际更稳。**

### 独立进程＋双路回调并发检查

合成静止反馈，两路 500 Hz IDL 反序列化／回调／CRC／日志，独立计算进程，6 ms 墙钟节拍。
包含学习预测，虽然合成腿部不动；测试计算负载，不是学习精度或闭环实验。

| 放置方式 | 5–18 s 平均 ms | P99 ms | 最大 ms | 截止时间错过 | 发送前门拒绝 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 计算 CPU 2／传输 CPU 0 | 4.119 | 5.205 | 11.651 | 15/2153 = 0.697% | 1 |
| 计算 CPU 7／传输 CPU 2 | 3.767 | 5.158 | 14.248 | 9/2157 = 0.417% | 1 |

内核为 PREEMPT_RT，但两轮线程均是 `SCHED_OTHER`，governor 为 powersave，平台为 performance；
不能把“装了实时内核”说成控制线程已经 FIFO。本轮未申请 sudo 修改设置。
CPU 7 那轮在 t≈6.939 s 发送前检查耗时 `14.059 ms`，触发当时已有的 10 ms 门。
测试为记录尾部而继续计算，最终 weight=0，**按当时的真实执行规则会退权，不会继续发送这拍**。
两轮均无日志丢弃／失败；保留超时证据，不增加周期，也不宣称已无中途退出风险。
这些是时间容错修改前的历史结果；新版处理见下面 10-08 追加记录，原入口仍保留旧规则。
明天按新指南使用已有 performance＋FIFO 入口；现场仍需记录真实截止时间／收包新鲜度。

## 原始证据与复算

本地目录：`evaluation/hardware_shadow/commissioning/mpc_learned_preparation_20261007/`。
最终回放 `replay_v3/{summary.json,frozen_baseline.jsonl,yaw_aware_hold.jsonl,learned.jsonl}`；
预测复算 `predictor_verification.json`；并发两轮 `ingress/`、`ingress_cpu7/`。
来源日志为首次成功的 `mpc_torque_walk_reboot_performance_20261007_172732/raw.jsonl`。
源码／配置／库／来源／输出摘要哈希在上述 JSON 中，原始大日志不入 Git。

初版验证脚本的 `replay/` 因第一拍 period=null 没有进入实验；`replay_v2/` 的控制窗口全部完成，
但错误地调用普通 plan 退权，在约18.294 s 停止。最终脚本已改用与现场相同的
`TorqueHandback`，不是放宽机器人门限。失败的验证记录原样保留，不当通过证据。

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
PYTHONPATH=tools/g1_commissioning:tools/g1_commissioning/tests \
  /home/fjk/miniforge3/envs/g1_mpc/bin/python -m unittest \
  test_mpc_learned test_mpc_actuation_constraints test_measured_torque_mpc \
  test_mpc_field test_mpc_serialization test_mpc_braking test_mpc_compute_process -q

# 输出目录必须是新的，避免覆盖本轮证据。
/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/g1_commissioning/validate_mpc_learned.py \
  --cpu 2 --output-dir evaluation/hardware_shadow/commissioning/mpc_learned_replay_repeat

taskset -c 0-17 /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/benchmark_mpc_ingress.py --learned --isolated \
  --cpu 7 --duration 24 --switch-ms 0.05 \
  --output-dir evaluation/hardware_shadow/commissioning/mpc_learned_ingress_repeat
```

## 下一步

软件入口已具备受控现场试运行条件；不需要重做所有早期阶段。新数学与学习预测尚无真机结果，
先做一轮，检查完整停车退权、右瓶全程倾角和停车 pitch 余量。若要将改善归因给学习本身，
使用同一新入口的 `hold_current` 对照。pitch 若缓慢漂移，再评估小幅名义姿态 PD；不在本轮提前追加。

## 10-08 追加：只对学习版加入有界时间容错

按用户要求减少偶发调度长尾造成的中止，不修改角度、力矩、速度、加速度限值和控制周期。
原成功入口不变。具体数字与行为见 [指南第 6 节](../HARDWARE_MPC_LEARNED.md#6-偶发慢一拍怎么处理)。

- 超过 10 ms 先复查最新 q/dq、命令总力矩及关节余量；满足条件的单拍最多容忍到 20 ms。
  原始反馈 25 ms 上限不变，复查反馈需在 10 ms 内；连续第三次慢拍仍退出。
- 子进程等待上限仅新版从 9 ms 改为 18 ms，之后仍需父进程检查；不接受失效或错序回复。
- QP 返回有效解但其局部墙钟计时超 3.5 ms，只记录；若整拍正常，不计为连续慢拍。
  求解器原生失败、NaN、约束失败不放行。没有扩展为“没看到超速就无限等待”。
- pitch 暂不加回拉，不减少右臂自由度，不增加全身接触仿真；先测试学习版这组明确变更。

**最新 70 项测试通过**（56.704 s），包含先前控制数学、原基线及新增容错测试，另测试共享主循环
正常放臂、非零力矩故障／人工退出及遥控交还。最新离线预检 `passed=true`、
`dds_initialized=false`、`publisher_created=false`。

单独运行的最终并发测试：合成静止反馈、计算 CPU 7／传输 CPU 2、普通 SCHED_OTHER／powersave，
未改性能档位。主动在 t≈6.003 s 插入一次 **11 ms 父进程暂停**：

| 项目 | 最终结果 |
| --- | ---: |
| 被延迟一拍的检查时耗时 | 15.723 ms |
| 用于计算的原始反馈年龄 | 18.516 ms |
| 复查的最新反馈年龄 | 1.197 ms |
| 该拍结果 | 检查通过，记录后继续 |
| 全程发送前检查拒绝 | 0 |
| 完成与最终 weight | terminal=true，0 |
| `[5,18)` 工作时间 平均／P99／最大 | 4.066／5.629／16.098 ms |
| `[5,18)` 截止时间错过，含故意暂停 | 15/2151 = 0.697% |
| 日志失败／丢弃 | 无／0 |

暂停期间收包回调继续，反馈合成值不响应命令；socket 被测试脚本禁止，没有真实 DDS／RPC／电机输出。
该结果验证“超过旧 10 ms 门但仍满足新检查时可继续”的软件路径，**不验证真机在 20 ms 延迟下安全或稳定**。
过期反馈、第三次连续超时、超过 20 ms、模式变化、最新总力矩超限、边界外向速度等拒绝分支由单元测试覆盖。

首版额外延迟测试 `timing_grace_injected/` 同样保留；最终证据为 `timing_grace_final/`。
复算命令、逐拍记录位置、源码／测试／输出哈希都在上述 `timing_grace.json`。
故意暂停只存在于离线 benchmark，现场入口没有这个参数。
