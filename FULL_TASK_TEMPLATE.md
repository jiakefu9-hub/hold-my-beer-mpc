# Full-task absolute-time disturbance template v2

状态：**受控 MuJoCo 仿真已验证 / hardware-unverified**。

当前正式 predictor 是 `FullTaskTemplatePredictor` v2。它针对一条固定的 8 s
direct-step 任务，按 reset 后的绝对 task time 查询完整扰动 horizon，再将 H 系结果
旋回当前世界系供右臂 MPC 使用。它提前知道 6.4 s 的直接停车，不是能处理任意命令
或未知停车时刻的通用预测器。

## 1. 为什么不再使用周期 phase 模板

旧 phase 模板适合重复的稳态步态，但任务启动和直接停车不是稳态周期的一部分。
若只按 gait phase 循环查询，启动瞬态会被稳态片段替代，6.4 s 的停车窗口也无法
表达“即将停车”。v2 因而直接保存从任务开始到结束的每一个 6 ms anchor：

```text
template row = f(absolute task time)
```

v2 不拼接旧周期模板、不在在线端循环、不按“最新资产”自动选择，也没有 neural
residual。旧 phase predictor 仅为只读 hardware shadow 的兼容项，不是正式仿真方案。

## 2. 冻结任务与事件时间

任务协议为 `full_task_direct_step_v1`：

| task time | 事件 |
|---:|---|
| 0 | planned 正式前进命令已经可见；heading control 开启；右臂执行固定姿态 PD |
| 20 ms | 下肢 20 ms policy 周期产生第一份新的 action |
| 24 ms | 合法 6 ms anchor 4；右臂 MPC/process 接管 |
| `[0,6.4 s)` | `planned vx=0.5 m/s, vy=0`，runtime `wz` 可由 heading controller 修正 |
| 6.4 s | planned `vx/vy` 在边界直接切零，不做 ramp |
| `[6.4,8.0 s)` | 停车响应仍属于唯一 headline |
| 7.998 s | headline 内最后一个 6 ms anchor，索引 1333 |
| 8.052 s | 该 anchor 的第 9 个未来节点 |
| 8.06 s | 离线 raw 记录尾端，保证最后 54 ms 标签完整 |

headline 始终是半开区间 `[0,8.0 s)`，共 1,334 个 anchors。8 s 以后的数据只是
hidden label tail，不是第二个评价阶段。

### 2.1 2/6/20 ms 顺序

MuJoCo physics step 是 2 ms，MPC anchor 是 6 ms，下肢 policy update 是 20 ms。
raw 样本采用 strict pre-step：时间戳、状态和 command 描述即将执行的
`[t,t+2 ms)`，在 `mj_step` 前写入。20 ms policy update 在上一物理区间结束后
提交，因此边界处下一条 pre-step 样本已经看到新 command。

task time、gait clock、template clock 和 continuous-H 都从 simulation time 0 启动。
24 ms 只切换右臂控制来源，不移动 task epoch。

## 3. Continuous causal-H v2

H 系是重力对齐、只绕世界 z 轴旋转的 heading frame：

```text
W_R_H = Rz(yaw_H)
v_H   = W_R_H^T v_W
v_W   = W_R_H v_H
```

`FullTaskContinuousHeadingFrame` 只消费当前及历史 6 ms anchor 的实测 torso yaw：

- `t=0`：`yaw_H` 就是第一帧实测 yaw；
- `0<t<0.8 s`：对 `0..t` 的全部 anchor yaw 做 sin/cos 圆周平均；
- `0.8<=t<6.4 s`：对可用的闭区间 `[t-0.8 s,t]` 做滑动圆周平均；
- `t>=6.4 s`：冻结最后一个停车前 anchor（6.396 s）已经确定的 H，不让停车晃动
  改变坐标系。

每个 anchor 的整个 54 ms window 固定使用该 anchor 已确定的同一个 H。实现不会
读取未来 yaw、不会直接线性平均角度或旋转矩阵，也不额外叠加 H 低通滤波。模板
构建、offline replay 和在线 predictor 共用
[`FullTaskContinuousHeadingFrame`](disturbance_template/full_task_protocol.py)。

## 4. 模板怎样建立

v2 源数据在 MuJoCo 中按同一 direct-step 命令采集：下肢 policy 和 heading control
正常运行，右臂全程使用配置中的 `fixed_posture_pd`。采集路径不调用右臂 MPC、
旧 phase predictor、process 或 DDQ-to-torque mapper，因此离线模板描述的是固定臂
采集分布；在线 MPC 闭环会产生需要单独量化的 distribution shift。

数据设计固定为 11 条 build（nominal + 5 对正/负小幅下肢初始 `q/dq` 扰动）和
4 条 held-out（2 对），所有 episode 的命令、heading、右臂 PD、payload、physics、
policy 和 task-time origin 相同。每条 episode 保存 2 ms strict pre-step raw，模板
只在 6 ms anchor 上构建。

跨 build episode 的向量在 H 系平均；姿态使用合法 SO(3) 平均并保存单位四元数，
不直接平均旋转矩阵元素。第一版 v2 没有额外 smoothing，也没有 raw/half/full 多套
候选。

正式资产固定为：

| 资产 | 仓库路径 | SHA256 |
|---|---|---|
| NPZ | [`disturbance_template/data/full_task_template_v2/20260815_162850/full_task_template.npz`](disturbance_template/data/full_task_template_v2/20260815_162850/full_task_template.npz) | `d4a0109adcff696936ef96160976161833ff9a7a7531e2e5d7ad9e50c10e17d4` |
| manifest | [`disturbance_template/data/full_task_template_v2/20260815_162850/full_task_template_manifest.json`](disturbance_template/data/full_task_template_v2/20260815_162850/full_task_template_manifest.json) | `6b48ee196d1f7d923dde057d3c0fb0e182f08512a65402c4c39c5e070a3243c6` |

manifest 记录 schema/protocol、build/held-out 设计、固定右臂模式、输入资产、轨迹和
H 诊断。运行时必须同时验证显式路径、两个 checksum、schema、protocol、shape、
anchor grid 和 SO(3)；禁止扫描目录选择“最新模板”。

## 5. Horizon schema 与在线查询

控制间隔 `dt=6 ms`、horizon `N=9`：

```mermaid
flowchart LR
  N0((node 0<br/>t measured)) -->|interval 0<br/>[t,t+6 ms)| N1((node 1<br/>t+6 ms))
  N1 -->|...| N8((node 8<br/>t+48 ms))
  N8 -->|interval 8<br/>[t+48,t+54 ms)| N9((node 9<br/>t+54 ms))
```

每一行包含：

- 10 个 nodes 的 `acc`、`alpha`、`omega` 和 rotation；
- 9 个 intervals 的 `acc`、`alpha`、`omega` 和 rotation；
- 向量量的跨 build 均值和标准差；rotation 使用 SO(3) 均值与
  `orientation_dispersion_rad`，而不是对旋转矩阵元素求标准差；
- anchor 的绝对 task time。

在线 predictor 的顺序是：

1. `reset()` 清除旧 task epoch 和 H 历史；reset 后第一条有效 observation 绑定
   `task t=0`。
2. 每个 6 ms anchor 必须且只能按顺序调用一次；时间倒退、重复、缺 anchor、
   off-grid 查询或协议/校验和错误都 fail closed 并给 reason code。
3. 用当前及历史实测 yaw 更新 shared continuous-H。
4. 精确读取当前绝对 anchor 的模板行，将 H 系的向量和 rotation 变回世界系。
5. `node 0` 始终用当前实测 `DisturbanceInput` 覆盖；nodes 1--9 和全部 intervals
   来自该模板行。

`[0,8.0 s)` 的正常查询不插值、不循环，predictor fallback 必须为 0。8 s 以后到
raw tail 的调用使用明确标记的 terminal measurement ZOH；它不计入 headline
fallback，也不会返回模板开头。

## 6. 24 ms startup-PD handoff

正式 task 在 simulation time 0 就开始，右臂仅在 `[0,24 ms)` 使用真实固定姿态
PD：target 是配置中的 `arm_waist_target`，target `dq=0`，不使用 MPC 生成的
`q_ref/dq_ref`。这段仍进入 `[0,8)` headline。

predictor、continuous-H 和 runtime preflight 在 PD 阶段正常推进，但 MPC 不向右臂
输出。24 ms 是不早于下肢 20 ms 第一次新 action 的第一个合法 6 ms anchor；接管时：

- 查询 absolute task time `0.024 s`，即 anchor 4，而不是 phase 0 或模板 row 0；
- 使用此刻真实 `q/dq` 和 torso state；
- 把 `[22,24) ms` 物理步真正执行的右臂 PD torque 作为
  `previous_executed_tau`；
- 第一份 MPC interval 仍按完整 6 ms 计时；不增加 torque ramp 或 off-grid solve。

`FixedStartupPdHandoff` 只按时间决定接管。足部接触、torso acceleration 和竖直速度
仅作诊断，不能推迟或提前 24 ms handoff；动态 arming 不属于正式路径。

## 7. Fallback 与安全边界

`FullTaskTemplatePredictor` 在合法 headline anchor 正常运行时不使用 fallback。
输入无效、资产/协议不匹配或 task-time 合同破坏会抛出带 reason code 的
`FullTaskPredictorError`，不能静默循环或猜测近似 anchor。headline 后的 terminal
measurement hold 是显式 tail 语义，不是 headline fallback。

predictor 之外，MPC 的 QP fallback 和 DDQ-to-torque 的 candidate、second pass、
rescue、hold-last、safe-hold、认证线搜索与 `NO_SAFE_TORQUE` 仍由独立执行合同负责。
0/4 ms mapper 更新拍只接受经当前状态 forward-dynamics 验收的 candidate。
2 ms 中间拍复用其已认证 feedforward，executor 用当前 `q/dq` 重算 PD 并做
限幅/超时/NaN guard，但不再做当前状态 forward-dynamics 验收。这套
MuJoCo 认证边界也不能被称为真机力矩安全证明。

## 8. 已有验证证据

### 8.1 离线 held-out 和 online parity

4 条 held-out 的 v2 离线平均 RMSE：

| 量 | nodes | intervals |
|---|---:|---:|
| acceleration | 0.08397 m/s^2 | 0.08080 m/s^2 |
| angular acceleration | 0.44422 rad/s^2 | 0.42240 rad/s^2 |
| angular velocity | 0.005426 rad/s | 0.005380 rad/s |
| SO(3) orientation geodesic | 0.001033 rad | 0.001032 rad |

4 条 held-out 逐 anchor offline-online replay 为 PASS：H yaw、H rotation、nodes 1--9
和 intervals 的普通数值最大绝对误差均为 0；task time 最大差
`6.59e-13 s`，rotation geodesic 的浮点上限为 `2.98e-8 rad`，predictor fallback
为 0，未使用隐式插值。

### 8.2 CPU 7 受控闭环历史证据

以下只引用清理前已经完成的 3 次 nominal + 3 次 `heldout_pair_02_minus` 历史受控
MuJoCo 证据，不预填后续冻结复跑结果：

| 场景 | tilt RMS / p95 / max | position RMS / p95 / max | 完整 6 ms mean / p99 / max | overrun |
|---|---|---|---|---:|
| nominal x3 | 0.002617 / 0.005713 / 0.007574 rad | 0.013735 / 0.024593 / 0.027716 m | 3.437 / 3.803 / 4.500 ms | 0/3,987 |
| held-out x3 | 0.002573 / 0.004393 / 0.006996 rad | 0.013680 / 0.021626 / 0.024426 m | 3.400 / 3.721 / 4.511 ms | 0/3,987 |

六条运行均通过 CPU 7/single-thread/GC preflight；合计 7,974 个完整区间的 mean/
p95/p99/max 是 `3.419/3.630/3.775/4.511 ms`，overrun 为 0。16,074 次 mapper
执行调用中 `final_output_uncertified=0`；同时 `predictor fallback=0`、
`QP fallback=0`、`final_unsafe=0`、`NO_SAFE_TORQUE=0`，无跌倒和 NaN/Inf。

证据入口是
[`controlled_runs_aggregate.json`](evaluation_summary/full_task_template_v2_final_freeze/controlled_runs_aggregate.json)
和
[`offline_online_parity.json`](evaluation_summary/full_task_template_v2_final_freeze/offline_online_parity/offline_online_parity.json)。

### 8.3 r1 最终冻结复跑

汇总语义修正后，正式入口完成 nominal 与 `heldout_pair_02_minus` 各一条完整运行；
轻量证据写入 `evaluation_summary/full_task_template_v2_final_freeze/final_runs_r1/`。

| 场景 | 完整区间 | mean / p95 / p99 / max | overrun | tilt RMS | position RMS | XY displacement |
|---|---:|---|---:|---:|---:|---:|
| nominal | 1,329 | 3.302370 / 3.464462 / 3.630513 / 4.340295 ms | 0 | 0.002617404 rad | 0.013735420 m | 3.222744 m |
| held-out | 1,329 | 3.299516 / 3.456565 / 3.532013 / 4.228574 ms | 0 | 0.002573218 rad | 0.013679703 m | 3.212114 m |

两条的 parent/worker affinity 都是 `[7]`，六个数值库线程变量都是 `1`，Torch
intra/inter-op 为 `1/1`，GC 在控制循环关闭，dynamic arming 为 `false`，24 ms
handoff 对应 anchor 4。每条有 2,679 次 mapper 调用；nominal 的 rescue/hold-last
为 `2/0`，held-out 为 `3/1`，但所有最终输出都通过认证。predictor/QP fallback、
final unsafe、`NO_SAFE_TORQUE`、未认证输出、跌倒及 NaN/Inf 均为 0，显式冻结门
全部 PASS。

r1 summary 把 task/protocol gate 与 nominal mapping path 分开：两条均为
`status=PASS`、`smoke_passed=true`、`nominal_mapping_path_passed=false`；fallback
计数 `3/5` 原样保留，并产生 `MAPPING_SAFETY_FALLBACK_USED` warning。因而 PASS
不等于没有 fallback；最终安全仍由认证输出、`final_unsafe`、`NO_SAFE_TORQUE`、
跌倒、NaN/Inf 和完整 6 ms deadline 单独验收。旧 tag 下 `final_runs/` 的原始 JSON
保持 legacy summary semantics，没有被改写。

## 9. 局限与硬件状态

- 模板仅覆盖冻结的命令、heading controller、payload、physics、policy 和 task-time
  origin；不能处理任意速度、方向或未知停车时刻。
- 模板由固定右臂 PD episode 建立；在线右臂 MPC 的 distribution shift 只能由闭环
  prediction-error diagnostics 衡量，不能假定消失。
- 受控 CPU 7 timing 是本机 MuJoCo 证据，不证明目标机 DDS、调度或最坏执行时间。
- hardware shadow 仍是只读 legacy phase-template 兼容路径，没有最终 v2 task epoch、
  continuous-H 和 24 ms handoff；主动真机闭环是 hardware-unverified。
- 神经扰动预测器已从当前源码移除，不再属于开发计划；清理前代码可从 archive
  checkpoint 恢复。下肢 RL policy 不受此清理影响。

真机适配边界见 [ARCHITECTURE.md](ARCHITECTURE.md)，正式运行环境和完整计时口径见
[REALTIME_RUNTIME.md](REALTIME_RUNTIME.md)。

## 10. 关键实现

| 文件 | 职责 |
|---|---|
| [`disturbance_template/full_task_protocol.py`](disturbance_template/full_task_protocol.py) | direct-step task clock、2/6/20 ms 网格、continuous-H |
| [`disturbance_template/full_task_template_asset.py`](disturbance_template/full_task_template_asset.py) | 运行时 NPZ/hash/schema/SO(3) 校验，不依赖 offline builder |
| [`disturbance_predictor.py`](disturbance_predictor.py) | `FullTaskTemplatePredictor`、严格 anchor 查询与 world-frame horizon |
| [`disturbance_template/full_task_startup_pd.py`](disturbance_template/full_task_startup_pd.py) | 24 ms fixed-PD handoff 和诊断 |
| [`disturbance_template/full_task_recording.py`](disturbance_template/full_task_recording.py) | 2 ms strict pre-step raw/schema/manifest |
| [`disturbance_template/full_task_fixed_pd_collector.py`](disturbance_template/full_task_fixed_pd_collector.py) | 固定右臂 PD 的 full-task episode 采集 |
| [`disturbance_template/full_task_template_builder.py`](disturbance_template/full_task_template_builder.py) | 6 ms window、SO(3) 均值、template/held-out 离线构建 |
| [`disturbance_template/full_task_online_parity.py`](disturbance_template/full_task_online_parity.py) | held-out offline-online replay parity |
| [`main_sim.py`](main_sim.py) | 正式任务、process runtime、headline 与控制质量记录 |
