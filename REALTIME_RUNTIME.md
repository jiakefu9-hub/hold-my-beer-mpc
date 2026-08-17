# 实时运行边界与 CPU 7 冻结环境

本项目有两套容易混淆、但证据等级不同的运行环境：

1. 当前正式 MuJoCo 冻结仿真：CPU 7、单线程数值库、Python 主进程和
   `RightArmSimProcess` worker 同核运行；
2. 面向未来真机 shadow 的 PREEMPT_RT 环境：额外要求内核、CPU/IRQ 隔离和
   低优先级 `SCHED_RR`。

第一套已经用于 full-task template v2 + 24 ms startup-PD 的受控仿真；第二套只
准备主机运行环境。两者都不是 Unitree 真机硬实时或闭环控制证据。

## 正式 MuJoCo 运行链

唯一正式入口是仓库根目录的 `run.sh`：

```text
run.sh
  -> taskset CPU 7 + six numeric-library thread limits
  -> main_sim.py
  -> FullTaskTemplatePredictor v2 + right-arm MPC
  -> RightArmSimProcess
  -> cpp/right_arm_sim_runtime worker
  -> RNEA -> MuJoCo DDQ-to-torque certification at 0/4 ms
  -> cached certified feedforward + latest-state executor PD/guards
  -> final_tau -> MuJoCo d.ctrl -> mj_step
```

2 ms 中间拍不重跑 mapper/forward dynamics；它复用已认证 feedforward，再用
当前 `q/dq` 重算并限幅 PD。因此 16,074 是 mapper 调用次数，不是
“每个 2 ms 总力矩都重做 FD 认证”的证据。

正式 nominal 命令为：

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
MPC_CONTROL_CPU=7 MPC_CONTROL_NUM_THREADS=1 ./run.sh \
  --full-task-smoke \
  --disturbance-predictor full_task_template \
  --right-arm-runtime-mode process \
  --startup-pd-duration 0.024 \
  --headless \
  --no-video \
  --run-label final_freeze
```

`--full-task-smoke` 会使 `run.sh` 拒绝隐式/自动 CPU 和多线程数值库。禁止直接用
`python main_sim.py` 运行正式 full-task，因为这会绕过 launcher 标记、构建和
外层 affinity 检查。

## 第一个 `mj_step` 前的 fail-fast preflight

正式入口在两处检查运行环境：父进程在模型构建前做第一层检查；C++ worker 已
启动、Python GC 已关闭后，在第一个 `mj_step` 前做第二层检查并写出
`formal_full_task_runtime_preflight.json`。必须同时满足：

- parent affinity 严格等于 `[7]`；
- C++ worker affinity 严格等于 `[7]`；
- `OMP_NUM_THREADS=1`；
- `OPENBLAS_NUM_THREADS=1`；
- `MKL_NUM_THREADS=1`；
- `NUMEXPR_NUM_THREADS=1`；
- `VECLIB_MAXIMUM_THREADS=1`；
- `BLIS_NUM_THREADS=1`；
- Torch intra-op / inter-op 都等于 `1`；
- control loop 中 Python GC 已关闭；
- dynamic arming 为 `false`；
- startup-PD 为 `0.024 s`，MPC handoff 是 absolute anchor `4`。

任何条件不满足都会在物理仿真开始前终止，而不是在宽 affinity 或空线程环境下
继续产生一条不可比的结果。`run_metadata.json` 保存父/worker affinity、调度
策略、线程环境和 GC 状态；preflight JSON 保存正式门槛的逐项结果。

## 6 ms 指标口径与当前证据

完整 6 ms interval 从一个右臂 MPC anchor 的状态/预测开始，覆盖 MPC policy、
两次 DDQ-to-torque 调用及剩余执行路径；不能用 solver-only timing 代替。六条
CPU 7 受控运行的 compact evidence 位于
[`evaluation_summary/full_task_template_v2_final_freeze/`](evaluation_summary/full_task_template_v2_final_freeze/)：

- 7974 个完整 interval，聚合 mean `3.419 ms`，overrun `0`；
- 16074 次 mapper 调用，未认证输出 `0`；
- parent/worker 都固定在 CPU 7，六个数值库线程变量和 Torch 都为 1，GC 在
  control loop 中关闭。

这些数字证明的是当前主机、当前 MuJoCo process 链在受控启动环境下的重复仿真
结果。正式六条运行记录的 scheduler 是 `SCHED_OTHER`；因此不能由
`3.419 ms < 6 ms` 或 `overrun=0` 推导 Linux 硬实时，更不能推导 DDS、总线、
驱动和真实传感器延迟。

### r1 最终冻结复跑

清理后的正式入口又运行了 nominal 与 `heldout_pair_02_minus` 各一条；轻量证据
写入 `evaluation_summary/full_task_template_v2_final_freeze/final_runs_r1/`。

| 场景 | 完整区间 | mean | p95 | p99 | max | overrun |
|---|---:|---:|---:|---:|---:|---:|
| nominal | 1,329 | 3.302370 ms | 3.464462 ms | 3.630513 ms | 4.340295 ms | 0 |
| held-out | 1,329 | 3.299516 ms | 3.456565 ms | 3.532013 ms | 4.228574 ms | 0 |

两条都满足 parent/worker affinity `[7]`、六个数值库线程变量 `1`、Torch `1/1`、
控制循环 GC 关闭、dynamic arming `false` 以及 24 ms/anchor 4 handoff。每条各有
2,679 次 mapper 调用；nominal rescue/hold-last 为 `2/0`，held-out 为 `3/1`。
predictor/QP fallback、final unsafe、`NO_SAFE_TORQUE`、未认证输出、跌倒与 NaN/Inf
均为 0，所有显式冻结验收门 PASS。

两条 r1 `full_task_smoke_summary.status` 均为 `PASS`，同时保留
`nominal_mapping_path_passed=false`、fallback 计数 `3/5` 和 warning。PASS 不等于
没有 fallback；它只表示 task/protocol smoke 通过。认证输出、`final_unsafe`、
`NO_SAFE_TORQUE`、跌倒、NaN/Inf 和完整 6 ms deadline 仍独立决定安全验收。旧 tag
下 `final_runs/` 的原始 JSON 保持 legacy summary semantics。
控制质量同时保持在冻结水平：nominal/held-out 的 tilt RMS 分别为
`0.002617404/0.002573218 rad`，position RMS 为
`0.013735420/0.013679703 m`，XY displacement 为 `3.222744/3.212114 m`。

## 可选 PREEMPT_RT 目标环境

面向未来硬件 shadow，可先运行只读 checker：

```bash
/home/fjk/miniforge3/envs/g1_mpc/bin/python realtime_environment.py \
  --control-cpu 7
```

目标条件包括：

- `CONFIG_PREEMPT_RT=y`；`PREEMPT_DYNAMIC` 不算通过；
- 当前 XPro-16 上，control CPU 7 所在物理核的 SMT siblings 6--7 一起隔离；
- `isolcpus=domain,managed_irq,6-7`、`nohz_full=6-7`、`rcu_nocbs=6-7`；
- 普通 IRQ 留在 housekeeping CPU，control core 使用 `performance` governor；
- control Python 和同一控制进程链使用低优先级 `SCHED_RR/10`，保留有界 Linux
  RT throttling。

当前机器的一次性 boot 参数建议是：

```text
isolcpus=domain,managed_irq,6-7 nohz_full=6-7 rcu_nocbs=6-7 irqaffinity=0-5,8-17
```

这组编号只适用于 checker 已确认 topology 的当前机器，不能复制到其他 CPU
拓扑。仓库不会自动安装内核、修改 GRUB、改 IRQ affinity、停用
`irqbalance`、改变 governor 或永久授予 capability。

如果目标机使用 Ubuntu 22.04，可按供应商支持方式手动安装 realtime kernel；
安装和恢复前都应保留原 generic kernel，并在每次 boot 后实际检查：

```bash
uname -a
grep '^CONFIG_PREEMPT_RT=y$' "/boot/config-$(uname -r)"
cat /proc/cmdline
/home/fjk/miniforge3/envs/g1_mpc/bin/python realtime_environment.py \
  --control-cpu 7
```

`run.sh` 仍保留 `MPC_REQUIRE_REALTIME=1` 的低优先级 `SCHED_RR` guard，但它不是
当前 frozen simulation 命令的一部分。硬件 shadow launcher 会独立检查目标
环境；详见 [HARDWARE_SHADOW.md](HARDWARE_SHADOW.md)。

## 证据边界

- MuJoCo `RightArmSimProcess` 是 external-step 锁步仿真，不是 DDS 线程。
- mapper 的 forward dynamics certification 使用 MuJoCo 当前接触求解状态，不能
  直接迁移成真机安全证明。
- `unitree_arm_adapter` dry-run 不包含 DDS 端到端延迟，也不包含经验证的真机
  floating-base inverse dynamics。
- hardware shadow 只读、只构建内存 command，不发布；它当前仍是
  hardware-unverified。

因此，进入真机主动输出前仍需单独验证状态估计、DDS/驱动/总线延迟、watchdog、
13 维 arm reference、arm weight 过渡、急停和最坏时延。
