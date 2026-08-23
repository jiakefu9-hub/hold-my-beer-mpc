# Experimental MPC-result latency replay

状态：**L1-A/L1-B 已完成；L1-C 在 held-out 2 ms 停止门收束为 PARTIAL；
latency 实验已冻结，L1-D 及以后未执行**
审查基线：`main@4376b1f43088b18eaba72ca865daad052f0ef4d8`
适用对象：full-task template v2、continuous-H、24 ms startup-PD、
`RightArmSimProcess` 正式 MuJoCo 链

## 1. 目的与结论

当前正式 MuJoCo 使用同步 external-step/lockstep：每个物理拍先读取状态，
等待 Python/C++ 控制链返回最终力矩，写入 `d.ctrl` 后才执行 `mj_step`。
因此几毫秒墙钟计算时间不会使 MuJoCo 状态在控制结果产生前继续演化。

这个缺口值得做一次受控的延迟敏感性实验，但第一版不直接改成
fully asynchronous/free-running simulator。推荐保留正式 zero-delay baseline，
新增一个显式、默认关闭的 experimental mode：

> 在 source 6 ms anchor 上生成完整 MPC 结果包；按照固定或预录的延迟，
> 到后续 2 ms physics tick 才激活；激活时使用最新 MuJoCo 状态运行现有
> RNEA/mapper/executor，并继续保持当前 fail-closed 安全合同。

这个模式模拟的是 **MPC result/reference availability latency with
fresh-state execution**，不是完整的真实 sensor-to-actuation latency。它可以回答
冻结控制器对旧 MPC 结果是否敏感，但不能证明真实 DDS、状态估计、总线、驱动、
固件或操作系统最坏时延合格。

第一版不排队执行 source state 上产生的旧 `final_tau`。该力矩的
forward-dynamics 认证只属于 source state；若在几毫秒后的 activation state
直接施加，原 `final_output_certified=true` 不再等价于当前状态安全。若未来硬件
输出合同正式冻结为 host-side direct torque，再另立阶段评估 stale final-torque
保守包络和 activation-state safety contract。

### 1.1 L1-A/L1-B/L1-C 实施结果与冻结决定

2026-08-21 完成了 capture、固定 0/2/4 ms result-age 调度、回归测试与受停止门
约束的短 smoke。正式 zero-delay 路径保持默认；实验调度仍是
simulation-only、默认关闭。

| 场景 | requested age | 结果 | 首次 source / activation |
| --- | ---: | --- | --- |
| nominal | 0 ms | PASS | 24 / 24 ms |
| nominal | 2 ms | PASS | 24 / 26 ms |
| nominal | 4 ms | PASS | 24 / 28 ms |
| `heldout_pair_02_minus` | 0 ms | PASS | 24 / 24 ms |
| `heldout_pair_02_minus` | 2 ms | **FAIL-CLOSED** | 42 / 44 ms 失败拍 |
| `heldout_pair_02_minus` | 4 ms | **NOT RUN** | 由停止门阻止 |

held-out 2 ms 的失败不是旧 source-state 力矩被直接输出：42 ms source packet
在 44 ms activation state 上使用最新 `q/dq` 与现有 mapper 重新验收。该拍各主要
candidate 的真实 forward-dynamics 结果均超过冻结门限；最低者为 R2/scale-1
candidate，`max|qacc|=10.293 rad/s^2`（mapper prediction 为
`8.68968 rad/s^2`）。其余最接近安全的摘要为 best-progress
`10.5312 rad/s^2`、hold-last `14.5751 rad/s^2`、safe-hold
`78.5875 rad/s^2`。mapper 因当前 MuJoCo
`max_abs_qacc=10 rad/s^2` 返回 `NO_SAFE_TORQUE`；该拍
`right_arm_d_ctrl_written=false`、`mj_step_performed=false`，没有输出未认证力矩。

逐拍证据保存在：

- `evaluation/experimental_mpc_latency_l1c/`
  `20260821_235116_heldout_pair_02_minus_fixed2_candidate_trace_v2/`
  `mpc_latency_trace.json`；
- 同目录 `heldout_0ms_vs_2ms_42_44_handoff_comparison.json`，用于对照 held-out
  0 ms 与 2 ms 在 42/44 ms 附近的 `q/dq/torque` 和 handoff transient。

因此当前结论严格限定为：`heldout_pair_02_minus` 对 2 ms MPC-result age 存在
**边缘敏感性**。这不是 latency replay 实现错误的证据，也不能外推为硬件物理
限制或真机 qacc hard-stop 设计依据。当前仿真控制器、mapper 和
`max_abs_qacc=10` 均不调整。

latency 工作在这里冻结：L1-D/L1-E 的完整任务与 trace replay 不执行，
held-out 4 ms 不补跑，不继续 async/free-running simulator。若未来重新开启，
必须另立明确阶段和验收合同；本文件以下 L1-D/L1-E 只保留为历史计划，不是当前
待执行任务。

## 2. 当前代码事实

### 2.1 正式 lockstep 链

- `configs/g1.yaml:5-10`：MuJoCo physics 为 2 ms，右臂 MPC 为 6 ms，
  下肢 policy 为 20 ms。
- `main_sim.py:1238-1459`：每拍先读取当前状态；6 ms anchor 上更新
  predictor、helper 和 MPC；zero-delay 下新 MPC 包同拍发布并激活。
- `right_arm_runtime/sim_process.py:804-957`：`RightArmSimProcess.execute()`
  发布 request 后同步等待同 ID response；超时不推进物理，也不复用旧响应。
- `main_sim.py:1783-1810`：只把 certified、非 `NO_SAFE_TORQUE`、有限的
  `final_tau` 写入 `d.ctrl`。
- `main_sim.py:1889-1942`：strict pre-step raw 在 `mj_step` 前写入，随后物理
  才推进 `[t,t+2 ms)`。

### 2.2 当前 0/4 ms mapper 语义

`configs/g1.yaml:185-192` 冻结为 `twice_per_interval`：

| 6 ms 区间相位 | 当前行为 |
| ---: | --- |
| 0 ms | 新 MPC 结果；用当前状态运行 RNEA/mapper并认证 feedforward |
| 2 ms | 复用已认证 feedforward；executor 用最新 `q/dq` 重算 PD |
| 4 ms | 用当前状态再次运行 RNEA/mapper并认证 feedforward |

对应选择逻辑在 `main_sim.py:1515-1548`；C++ 更新/缓存复用和最新状态
executor 在 `cpp/right_arm_sim_runtime/src/runtime.cpp:215-283`。缓存的是认证后的
feedforward，不是包含旧状态 PD 的完整力矩。

### 2.3 已有 command-delay 路径的边界

`sim_support.py:92-231` 已有 `ArmCommandDelayLine`：

- 原子延迟 `q_ref/dq_ref/ddq_raw/ddq_des`；
- 非网格延迟向上量化到下一 2 ms tick；
- 激活后用最新状态重新执行 torque mapping；
- 同拍多份 ready packet 只激活最新一份。

它提供了可复用的 packet、量化、ID 和激活思路，但不能只解除
`main_sim.py:168-175` 的 full-task 禁令：

1. `[0,24 ms)` startup-PD 阶段当前不会发布真实 MPC packet；启用非零延迟后，
   24 ms 可能先看到 synthetic zero-ddq 初始包；
2. packet 没有携带 source anchor 和完整 predictor/MPC diagnostics，当前记录器
   可能把最新 generated diagnostics 与旧 active command 错配；
3. 当前只支持一个固定 delay，没有版本化、checksum 固定的 per-anchor trace；
4. 现有 full-task 正式结果全部仍是 zero-delay。

### 2.4 现有 timing 不能直接作为注入值

六条 CPU7 冻结证据见
`evaluation_summary/full_task_template_v2_final_freeze/controlled_runs_aggregate.json`：

- 完整 6 ms interval mean/p99/max：约 `3.419/3.775/4.511 ms`；
- MPC policy mean/p99/max：约 `1.943/2.088/2.338 ms`；
- 两次 DDQ-to-torque 合计 mean/p99：约 `1.037/1.259 ms`。

完整 6 ms 数值包含三个 physics tick 和后续 4 ms mapper，不能直接作为
source state 到首份 actuator output 的 latency。实施前必须分别测量：

1. `source state -> MPC packet ready`；
2. `source state -> first certified torque ready`；
3. anchor-0、cached-2、mapper-4 三类 2 ms tick 的单拍 service time。

## 3. 冻结实验合同

### 3.1 不变项

- 正式 zero-delay `run.sh` 命令和默认行为不变；
- full-task template v2 资产、checksum 和查询值不变；
- continuous-H、MPC数学/权重/约束、mapper数学和安全阈值不变；
- 24 ms startup-PD source handoff、20 ms下肢首次新动作不变；
- task t=6.4 s direct stop、heading control和 `[0,8)` headline不变；
- 不修改 hardware adapter，也不把实验结论写成 hardware validation。

### 3.2 两条时间轴

每个 packet 必须同时保留：

- **source time**：状态、continuous-H、absolute template anchor 和 MPC 求解
  所属的 task/simulation time；
- **activation time**：该 packet 在后续 2 ms tick 成为 active command 的时间。

activation 时不得重新查询模板、重置 H/task/gait clock或从模板零点重播。
旧计划本身的 source anchor 必须保持不变，状态老化才是被测变量。

### 3.3 startup-PD

- `[0,24 ms)` 始终执行真正的 fixed-posture PD；
- 0--18 ms 可以推进 predictor/H并做既有preflight，但不得发布可执行MPC包；
- 24 ms、absolute template anchor 4生成第一份允许发布的MPC packet；
- 2/4 ms延迟下，首次实际MPC接管分别发生在26/28 ms；
- 首包ready以前继续执行fixed PD，不执行synthetic包；
- `previous_executed_tau` 必须是activation前最后一拍实际施加的PD力矩。

因此报告必须区分 logical/source handoff 24 ms 与 physical actuation handoff。

### 3.4 activation和mapper

1. `ready_tick = source_tick + ceil(latency / 2 ms)`；
2. ready前继续执行上一份active command；尚无真实active MPC包时继续fixed PD；
3. 激活新packet的当拍使用activation state立即运行mapper；
4. 第二次mapper认证相对activation后4 ms发生；
5. 没有新packet时也不得让mapper认证年龄无限增长；
6. 中间2 ms拍继续使用认证feedforward与latest-state executor PD；
7. `previous_executed_tau`始终来自上一物理拍真正写入的力矩；
8. `NO_SAFE_TORQUE`、未认证输出、错帧、timeout或nonfinite继续在
   `d.ctrl`和`mj_step`前fail closed。

### 3.5 任务边界

- 6.396 s source packet保持停车前horizon语义；
- 6.402 s source packet保持停车后horizon语义；
- 延迟只改变激活时间，不改变planned command或模板查询时间；
- 7.998 s source packet可能在8.0 s后激活；它只影响hidden tail，不改变
  `[0,8)` headline口径；raw tail仍记录到至少8.06 s。

## 4. 分阶段实施（历史计划；L1-C 后冻结）

### L1-A：精确计时与版本化trace，控制行为不变

目标是先得到可以解释、可以重放的延迟输入，不注入控制延迟。

计划改动：

- `main_sim.py`
  - 增加显式capture-only入口；
  - 在source anchor状态读取前记录 `source_sample_wall_ns`；
  - MPC结果包完成后记录 `mpc_packet_ready_wall_ns`；
  - 第一次process结果完成认证、但尚未写 `d.ctrl` 时记录
    `first_certified_tau_ready_wall_ns`；
  - 记录当前tick是anchor-0、cached-2还是mapper-4。
- 新增一个小型simulation-only模块，例如
  `right_arm_runtime/sim_mpc_latency.py`：
  - 定义 `mpc_latency_trace_v1` schema；
  - 负责有限值、单调anchor、shape、protocol和checksum校验；
  - 保存显式JSON/CSV或NPZ及SHA256；
  - 不包含控制调度逻辑以外的大型runtime重构。
- 新增对应单元测试，覆盖schema round-trip、checksum、时间单调性、非有限值和
  source/ready字段定义。

L1-A必须证明capture关闭和开启时的控制输出/轨迹数值一致；现有完整6 ms timing
定义保持不变。该阶段结束后先审查trace，再进入延迟注入。

### L1-B：固定2/4 ms MPC-result激活

- 新增默认关闭、与旧 `--mpc-command-delay-ms` 明确分开的experimental CLI；
- 扩展或小范围抽取现有delay packet，加入source anchor、diagnostics、真实包标志、
  ready/activation ID和严格reset合同；
- 分离 `mpc_result_generated`、`mpc_packet_active` 和
  `mpc_actuation_enabled`，避免24 ms synthetic包；
- 激活时使用当前状态运行现有process/mapper/executor；
- generated、active、applied数据分别记录，杜绝diagnostics错配；
- 第一阶段只允许固定0/2/4 ms；`>=6 ms`明确拒绝，避免在未定义多job调度时
  静默跨越一个完整MPC周期。

建议入口形态：

```text
--experimental-mpc-latency-mode off|fixed|trace
--experimental-mpc-latency-ms 2|4
--experimental-mpc-latency-trace /explicit/path
--capture-mpc-latency-trace /explicit/path
```

正式配置继续为zero-delay；禁止目录扫描或自动选择latest trace。

### L1-C：单元/集成测试与短smoke

至少覆盖：

- formal off和experimental 0 ms逐拍parity；
- `<2 ms`、`=2 ms`、`2 ms+epsilon`、4 ms量化；
- packet不可变、ID严格单调、同拍多ready和迟到旧packet处理；
- 24 ms source anchor 4，26/28 ms实际接管；
- 首包ready前固定PD、previous torque连续性；
- template/H不在activation时重查；
- 6.396/6.402 s和7.998 s/tail边界；
- activation与activation+4 ms mapper、2 ms cached语义；
- generated/active/applied diagnostics正确配对；
- process timeout、错ID、未认证输出和 `NO_SAFE_TORQUE` fail closed；
- full-task predictor、startup-PD、process、mapper及Python/C++ parity回归。

测试通过后，nominal和`heldout_pair_02_minus`分别运行experimental 0/2/4 ms
短smoke，至少覆盖首次接管后0.3 s。任一安全失败即停止，不进入完整运行。

### L1-D：配对完整任务和受控trace replay

- 先运行CPU7 zero-delay nominal和heldout baseline并捕获trace；
- 固定2/4 ms短smoke安全后，才运行配对完整 `[0,8)`；
- 只有固定4 ms结果表明值得继续时，才加载显式、checksum固定的baseline trace
  做确定性replay；
- 不用同一条delayed run自身的实时墙钟抖动驱动其物理轨迹；
- 不自动扩大到6 ms、多in-flight job或fully asynchronous模式。

### L1-E：证据和文档冻结

实验产物进入独立目录，例如
`evaluation/experimental_mpc_latency/<explicit-run-id>/`，不得覆盖正式freeze证据。

同步更新：

- `README.md`：正式命令不变，experimental入口单列；
- `REALTIME_RUNTIME.md`：区分compute budget、virtual result age和真实硬件延迟；
- `MPC_DESIGN.md`：source/ready/activation与0/4 ms时序；
- `FULL_TASK_TEMPLATE.md`：模板绑定source absolute task time；
- `PRE_HARDWARE_FREEZE.md`：只记为simulation latency sensitivity；
- `ARCHITECTURE.md`：scheduler属于MuJoCo adapter；
- `right_arm_runtime/README.md`和
  `cpp/right_arm_sim_runtime/README.md`：worker仍为blocking external-step。

## 5. 报告指标与停止门

### 5.1 新增延迟指标

- source-to-packet-ready与source-to-first-certified-torque-ready的
  mean/p95/p99/max；
- requested、quantized和effective activation delay；
- active command age、queue depth、held tick和drop/supersede count；
- source-to-activation的right-arm `q/dq`和torso状态变化；
- logical handoff与首次实际MPC actuation；
- mapper实际认证相位和认证年龄；
- produced/active/applied torque及handoff jump。

### 5.2 原有指标继续报告

- 完整6 ms mean/p95/p99/max/overrun及MPC、两次mapper、executor分量；
- tilt、position、EE acc/alpha、torso acc/alpha和XY距离；
- `ddq_des`与interval-average `ddq_real`；
- predictor/QP fallback；
- mapper rescue、hold-last、未认证输出、`NO_SAFE_TORQUE`；
- 跌倒和NaN/Inf。

成功且认证的rescue/hold-last继续作为warning和真实计数，不单独等同task failure。

### 5.3 立即停止条件

- 未认证输出、`NO_SAFE_TORQUE`、process hard failure或nonfinite；
- 跌倒或明显失控；
- predictor/QP出现新的系统性fallback；
- source/activation ID、task clock、template anchor或mapper相位不符合合同；
- trace缺anchor、倒退、schema/checksum不匹配或出现未支持的 `>=6 ms` latency；
- 控制质量明显退化。

出现上述情况时只定位和报告；不调整MPC权重、mapper阈值、24 ms handoff、
continuous-H、模板、heading、任务命令或评价窗口。

## 6. 后续升级条件

只有满足以下任一条件，才考虑下一阶段single-flight asynchronous、HIL或
stale-final-torque实验：

- CPU7或真机trace持续出现跨越完整6 ms周期的长尾；
- 2--4 ms结果显示显著控制退化，需要研究状态预测或deadline策略；
- 真机主动输出合同明确选择direct torque，并已冻结activation-state安全处理；
- 获得可用的sensor/DDS/driver/firmware端到端时间戳。

即使本实验PASS，结论也只能是“冻结MuJoCo任务对指定MPC结果延迟不敏感或可接受”，
不能写成真机硬实时或主动闭环已经验证。
