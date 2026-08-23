# Pre-hardware freeze：full-task template v2

- 冻结日期：2026-08-15
- 最终仿真候选：**MPC + FullTaskTemplatePredictor v2**
- 状态：**受控 MuJoCo 仿真已验证 / 主动真机闭环未验证**
- 硬件输出：**未授权；hardware shadow 保持只读**

本文只记录当前可恢复、可复现的冻结方案。旧 phase-template、PID/LQR、
neural/hybrid 和 readiness 调参记录属于历史探索，不是当前正式控制方案。

## 1. 冻结结论

正式仿真链为：

```text
run.sh
  -> main_sim.py
  -> FullTaskTemplatePredictor v2
  -> ArmMPCPolicy
  -> RightArmSimProcess
  -> cpp/right_arm_sim_runtime
  -> Pinocchio RNEA
  -> MuJoCo DDQ-to-torque candidate validation at 0/4 ms
  -> certified feedforward + C++ latest-state PD/guards
  -> final_tau
  -> MuJoCo d.ctrl -> mj_step
```

这条链只有一份 MPC 和一份 full-task predictor。平台边界详见
[ARCHITECTURE.md](ARCHITECTURE.md)。

冻结行为如下：

- `task t=0` 立即发布 `vx=0.5, vy=0`；不使用 ramp。
- 下肢 RL policy 在 20 ms 边界产生第一份新动作。
- 右臂在 `[0, 0.024 s)` 使用配置中的固定姿态 PD。
- 右臂 MPC 在 `task t=0.024 s`、即 6 ms anchor 4 接管。
- 接管时模板直接查询绝对任务时间 24 ms，不 reset、不从模板零点重播。
- mapper 的 `previous_executed_tau` 是切换前最后一个 2 ms 物理拍真正执行的
  右臂 PD 力矩。
- `task t=6.4 s` planned `vx/vy` 直接切为零；heading control 全程开启。
- 唯一 headline 为 `[0, 8.0 s)`；PD 启动段也包含在 headline 内。
- MuJoCo 物理周期 2 ms，右臂 MPC 周期 6 ms，下肢策略周期 20 ms。

唯一正式运行命令见 [README.md](README.md)。正式入口必须通过仓库根
[`run.sh`](run.sh)，不能直接执行 `python main_sim.py`。

## 2. 冻结资产

| 资产 | 仓库相对路径 | SHA256 |
| --- | --- | --- |
| full-task v2 NPZ | `disturbance_template/data/full_task_template_v2/20260815_162850/full_task_template.npz` | `d4a0109adcff696936ef96160976161833ff9a7a7531e2e5d7ad9e50c10e17d4` |
| template manifest | `disturbance_template/data/full_task_template_v2/20260815_162850/full_task_template_manifest.json` | `6b48ee196d1f7d923dde057d3c0fb0e182f08512a65402c4c39c5e070a3243c6` |
| held-out 初态 manifest | `disturbance_template/data/full_task_template_v2/20260815_162850/episodes/heldout_pair_02_minus/episode_manifest.json` | `a40cd4180e46e62aa2f489a8bce233350317f807a1395e8bf6d2007774f40aa5` |

运行时只加载上述显式路径；禁止扫描目录或选择“最新模板”。加载器验证
template/manifest checksum、schema、protocol、shape、anchor 网格和 SO(3)
合法性。在线运行不依赖离线 builder、raw episodes 或 neural artifacts。

模板是由 11 条 build episode 构建的固定任务 baseline，4 条 held-out episode
用于离线验收。采集时下肢策略与 heading controller 正常运行，右臂使用固定姿态
PD；因此 T2 闭环中右臂改用 MPC 后的预测误差已被单独量化，不能把离线模板误差
当成闭环中的唯一误差。

## 3. Full-task 时间与 H 系

协议版本是 `full_task_direct_step_v1`：

| task time | planned command | 说明 |
| ---: | --- | --- |
| `0.000 <= t < 6.400` | `vx=0.5, vy=0` | direct start；heading controller 可生成小幅 runtime `wz` |
| `t >= 6.400` | `vx=0, vy=0` | direct stop；没有 ramp |
| `[0, 8.000)` | headline | 唯一正式评价区间 |
| `8.000..8.060+` | hidden tail | 仅覆盖最后 54 ms 标签，不属于 headline |

template anchor 使用 6 ms 网格，headline 内共有 1,334 个 anchor，最后一个为
7.998 s；它的第 9 个未来 node 到达 8.052 s。每次查询返回 10 个 nodes 和
9 个 intervals；node 0 始终由当前实测扰动覆盖。

连续 causal-H 版本是 `full_task_continuous_heading_v2`：

1. `t=0` 使用第一帧实测 torso yaw。
2. `0<t<0.8 s` 使用从 task epoch 到当前 6 ms anchor 的 yaw 圆周均值。
3. `0.8<=t<6.4 s` 使用截至当前 anchor 的过去 0.8 s 滑动圆周均值。
4. `t>=6.4 s` 冻结最后一个停车前有效 H。
5. 每个 54 ms horizon 内使用查询 anchor 已确定的同一个 H。
6. 只使用当前及历史 anchor；不插值、不读取未来 yaw、不额外低通。

模板中的 H 系 `acc/alpha/omega/rotation` 转回当前世界系后进入 MPC。姿态通过
合法 SO(3) 运算处理，不平均旋转矩阵元素。完整定义见
[FULL_TASK_TEMPLATE.md](FULL_TASK_TEMPLATE.md)。

## 4. 安全执行合同

MPC 输出 `ddq_des`。在 `twice_per_interval` 的 0/4 ms mapper 更新拍，
正常 candidate、second pass、rescue、hold-last 和 safe-hold/有界线搜索保持
既有顺序，被选的当拍总力矩 candidate 必须经过当前 MuJoCo 状态的 forward
dynamics 验收。2 ms 中间拍不重跑 mapper；worker 复用上一份已认证
feedforward，再用当前 `q/dq` 重算 PD 并执行限幅、超时与 NaN guard。
因此不声称每个 2 ms 的总 `final_tau` 都在该拍重做了 forward-dynamics 验收。

冻结安全条件包括：

- `ddq_execution_max_abs_qacc = 10 rad/s^2`，未放宽；
- 电机力矩上下限保持原配置；
- `NO_SAFE_TORQUE` 时终止该次仿真，不输出未认证 candidate；
- process 响应超时、session/request/state 不匹配或 nonfinite 时 fail closed；
- 右臂 startup-PD 的最后真实执行力矩跨 handoff 保留为 previous torque；
- `[0,24 ms)` 不豁免评价，首个 MPC 6 ms 区间也完整计时。

MuJoCo forward-dynamics certification 是仿真执行证据，不是实机物理验收。真机
不能直接复用 MuJoCo 接触求解状态来声称力矩安全。

### 4.1 Experimental latency 停止门

默认关闭的 MPC-result latency replay 在不改控制器或安全阈值的前提下完成了短
smoke。nominal 0/2/4 ms 与 `heldout_pair_02_minus` 0 ms 通过；held-out 2 ms
在 task/simulation time 44 ms fail closed。该拍 42 ms source packet 在 44 ms
activation state 上重新验收后，最低真实 candidate 的 `max|qacc|` 为
`10.293 rad/s^2`，高于当前 MuJoCo `10 rad/s^2` 门限，因而返回
`NO_SAFE_TORQUE`，没有写入未认证力矩或推进该拍物理。

结论限定为：`heldout_pair_02_minus` 对 2 ms MPC-result age 存在边缘敏感性。
实验在 L1-C 收束为 PARTIAL；L1-D、held-out 4 ms、完整任务 latency 比较和
async/free-running 均未执行，也不属于当前 freeze 的待办。

## 5. 受控 CPU7 证据

轻量、可提交的证据包位于
[`evaluation_summary/full_task_template_v2_final_freeze/`](evaluation_summary/full_task_template_v2_final_freeze/)。
它保存 3 条 nominal 和 3 条 `heldout_pair_02_minus` 受控运行、六份完整
`perf_intervals.csv`、环境 preflight、安全诊断、聚合结果和代表图。

六条运行都满足：

- parent affinity 和 C++ worker affinity 都严格为 `[7]`；
- `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB/BLIS` 线程数都为 1；
- Torch intra-op/inter-op 都为 1；
- 控制循环内 Python GC 关闭；
- dynamic arming 未启用；startup-PD 为 24 ms，handoff anchor 为 4；
- predictor fallback、QP fallback、final unsafe、`NO_SAFE_TORQUE`、未认证输出、
  跌倒和 NaN/Inf 均为 0。

### 5.1 完整 6 ms 时间

| 场景 | 区间数 | mean | p95 | p99 | max | overrun |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal，3 runs | 3,987 | 3.437 ms | 3.659 ms | 3.803 ms | 4.500 ms | 0 |
| held-out，3 runs | 3,987 | 3.400 ms | 3.591 ms | 3.721 ms | 4.511 ms | 0 |
| pooled | 7,974 | 3.419 ms | 3.630 ms | 3.775 ms | 4.511 ms | 0 |

pooled mean 的组成是 MPC policy 1.943 ms、两次 DDQ-to-torque 合计
1.037 ms、C++ executor bridge 0.037 ms、其余右臂路径 0.402 ms。
predictor 加权 mean 为 0.403 ms。该口径覆盖一个完整 6 ms 右臂区间内实际发生的
两次 mapper 调用，不是 solver-only 时间。

### 5.2 安全与控制质量

六条共记录 16,074 次 mapper 调用：未认证输出 0，final unsafe 0，
`NO_SAFE_TORQUE` 0。rescue 共触发 15 次；held-out 三条各有一次经重新验收成功的
hold-last。它们仍是认证输出，不是被隐藏的 unsafe candidate。

| 场景 | tilt RMS / p95 / max | position RMS / p95 / max | XY displacement / arc length |
| --- | --- | --- | --- |
| nominal | 0.002617 / 0.005713 / 0.007574 rad | 0.013735 / 0.024593 / 0.027716 m | 3.223 / 3.349 m |
| held-out | 0.002573 / 0.004393 / 0.006996 rad | 0.013680 / 0.021626 / 0.024426 m | 3.212 / 3.340 m |

offline-online replay 对 4 条 held-out episode 均通过。旧 tag 下早期综合
`full_task_smoke_summary.status` 保留 legacy 门控值；r1 语义将 task/protocol PASS
与 nominal mapping path 分开，且不删除或归零 fallback 诊断。

### 5.3 r1 最终冻结复跑

清理后的正式分支又完成 nominal 与 `heldout_pair_02_minus` 各一条完整 `[0,8)`
复跑。轻量证据位于
`evaluation_summary/full_task_template_v2_final_freeze/final_runs_r1/`。

| 场景 | 完整区间 | mean / p95 / p99 / max | overrun | tilt RMS | position RMS | XY displacement |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| nominal | 1,329 | 3.302370 / 3.464462 / 3.630513 / 4.340295 ms | 0 | 0.002617404 rad | 0.013735420 m | 3.222744 m |
| held-out | 1,329 | 3.299516 / 3.456565 / 3.532013 / 4.228574 ms | 0 | 0.002573218 rad | 0.013679703 m | 3.212114 m |

两条运行的 parent/worker affinity 均为 `[7]`，六个数值库线程变量均为 `1`，Torch
intra/inter-op 为 `1/1`，控制循环 GC 关闭，dynamic arming 为 `false`，MPC 在
24 ms 的 absolute anchor 4 接管。两条各记录 2,679 次 mapper 调用；nominal 的
rescue/hold-last 为 `2/0`，held-out 为 `3/1`。全部输出均已认证，且 predictor/QP
fallback、final unsafe、`NO_SAFE_TORQUE`、未认证输出、跌倒和 NaN/Inf 均为 0；
显式冻结验收门全部 PASS。

两条 r1 summary 都是 `status=PASS`、`smoke_passed=true`、
`nominal_mapping_path_passed=false`。fallback 计数 `3/5` 及
`MAPPING_SAFETY_FALLBACK_USED` warning 均保留。PASS 只表示 task/protocol 门通过，
不表示没有 fallback；最终安全仍以 `final_output_certified`、`final_unsafe`、
`NO_SAFE_TORQUE`、跌倒、NaN/Inf 和完整 6 ms deadline 为准。旧 tag 下的 JSON
保持 legacy summary semantics。

## 6. 已验证与未验证边界

| 模块 | 当前状态 |
| --- | --- |
| full-task v2 schema/hash、continuous-H、offline-online parity | simulation-validated |
| 24 ms PD→MPC handoff、previous torque 连续性 | simulation-validated |
| MPC + process + RNEA + MuJoCo mapper + certified executor | simulation-validated |
| CPU7 上完整 6 ms 时间 | 当前主机受控 MuJoCo 证据；不是硬实时证明 |
| experimental MPC-result age | 短 smoke：nominal 0/2/4 ms 与 held-out 0 ms 通过；held-out 2 ms 因最低真实 candidate `10.293 > 10 rad/s^2` fail closed；L1-C PARTIAL 后冻结 |
| hardware state bridge / shared-memory layout / read-only shadow | code/test validated，hardware-unverified |
| hardware shadow 的最终 full-task v2 + 24 ms 时钟 | 未实现 |
| 真实 floating-base/contact state estimation | hardware-unverified |
| Unitree 主动输出、ownership transition、watchdog 与急停 | hardware-unverified，未授权 |
| 真机 DDS/总线/驱动端到端 deadline | 未测量 |

当前 hardware shadow 只兼容旧 phase template，输出能力固定为 absent，不能作为
最终 full-task 方案已经迁移到真机的证据。`cpp/unitree_arm_adapter` 的 dry-run 和
协议测试只证明代码合同，不证明机器人上的主动闭环安全。

## 7. 冻结项与禁止外推

以下内容在进入独立硬件阶段前保持冻结：

- MPC 模型、权重、约束、OSQP 设置；
- DDQ-to-torque mapper、rescue/safe-hold/fail-closed 顺序与安全阈值；
- full-task 命令、heading controller、continuous-H、24 ms handoff；
- v2 模板内容、manifest 和 checksum；
- headline `[0,8)` 与 6.4 s direct stop。

Full-task template 是固定绝对任务时间 baseline，**提前知道 6.4 s 停车时刻**。
它不泛化到任意速度、方向、地形、下肢策略或未知停车时刻。neural disturbance
predictor 路线已冻结并从当前正式源码移除；下肢 TorchScript walking policy
`policy/motion.pt` 仍是必需运行资产，二者不能混淆。

## 8. 进入真机主动闭环前的独立门

下一阶段只能是 hardware-specific 的只读/吊架安全准备，不应继续修改冻结的
仿真算法。至少需要：

1. 为硬件适配器实现并验证 full-task v2 task epoch、continuous-H 和 24 ms
   PD→MPC handoff；不能沿用 legacy phase shadow 后直接发力矩。
2. 提供经验证的 floating-base pose/twist、接触和外力接口，明确估计延迟。
3. 定义不依赖 MuJoCo forward dynamics 的真机安全合同。不得默认把仿真的
   `max_abs_qacc=10 rad/s^2` 直接继承为真机 hard-stop；应分别定义：必须立即禁止
   输出的 hard-stop、允许受控降级/恢复的 soft guard，以及只记录趋势的 diagnostics。
   是否因单拍轻微 qacc 超限终止主动控制，必须结合真实 `q/dq`、位置余量、力矩与
   torque-rate、温度、接触/状态估计可信度和超限持续性，经吊架与分级试验确定。
4. 验证 13-DOF arm SDK 索引、左右臂/腰固定参考和 ownership transition。
5. 在吊架、急停和独立监督下分级验证 watchdog、超时释放、温度与通信故障。
6. 在目标计算机实测 DDS、驱动、总线和完整控制区间的 worst-case deadline。

这些门全部关闭前，项目结论必须保持为“受控 MuJoCo 仿真已验证，主动真机闭环
hardware-unverified”。
