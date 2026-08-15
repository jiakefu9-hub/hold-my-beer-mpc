# Right-arm simulation runtime

这是只用于 MuJoCo 仿真的独立 C++ external-step worker。它不访问 Unitree
DDS，也不复用真机 `unitree_arm_adapter` 的业务 payload。一次请求原子携带
完整仿真状态和同一份 `q_ref/dq_ref/ddq_des`，worker 严格执行：

```text
C++ Pinocchio RNEA
→ C++ MuJoCo DDQ-to-torque mapper / candidate validation
→ C++ right-arm executor (PD, clamps, timeout and NaN fallback)
→ final_tau[5]
```

正式 full-task 调用链是根目录 `run.sh -> main_sim.py -> RightArmSimProcess ->`
本 worker。full-task template、continuous-H、24 ms handoff 和 MPC QP 都留在
共享 Python 控制核心；worker 不复制 predictor 或 MPC，也不拥有 task clock。

## 为什么单独建目录

MuJoCo mapper 需要 `qacc_warmstart`、`qfrc_applied` 和 `xfrc_applied`；这些
是仿真约束求解状态，不是真机 LowState。把它们塞进真机协议会模糊“仿真
候选验收”和“真实物理反馈”的边界。

## external-step 同步

父进程创建两根 pipe，并以 `pass_fds`/继承方式把读取请求的 fd 和写入完成
通知的 fd 交给 worker：

```bash
right_arm_sim_runtime_worker \
  --scene resources/g1_description/scene.xml \
  --shm-name /unique_run_name \
  --request-fd REQUEST_READ_FD \
  --response-fd RESPONSE_WRITE_FD
```

每个虚拟2 ms物理步：

1. Python作为唯一写者写完整 `request` seqlock；
2. Python向request pipe写1字节；
3. C++读取同一快照并计算一次；
4. C++作为唯一写者发布 `response`，再向response pipe写1字节；
5. Python只接受session/request/state id全部匹配的结果，然后执行 `mj_step`。

pipe只通知，不承载大数组。`publish_monotonic_ns` 只用于IPC墙钟耗时；
Executor的 `now/command/state` 全部使用 `simulation_time` 量化出的虚拟纳秒，
不能与 `CLOCK_MONOTONIC` 相减。仿真第0拍的command/state时间戳0合法。

`mapping_update_due=false` 时不会伪造一轮验收，而是复用上一轮已经验收的
前馈力矩，并用当前 `q/dq` 重新运行Executor。session变化或Executor配置
变化都会清空缓存；没有缓存时明确返回 `no_cached_feedforward`。

## 认证输出与 fail-closed

mapper 请求包含调用方当前计算出的 right-arm PD torque，供 normal candidate、
second pass、rescue 和 hold-last 均不能通过时作为 safe-hold 基准。safe-hold 与最多四个
line-search candidate 都必须重新运行真实 MuJoCo forward dynamics；插值本身
不是安全证据。

worker 只在 mapper 返回 `final_output_certified=1` 且
`no_safe_torque=0` 时缓存/执行 feedforward。`NO_SAFE_TORQUE`、mapper error、
非有限输出、错帧或 executor failure 都以非零 status 返回。Python client 再次
检查 status、所有 request/state ids、certification flag 和最终力矩有限性；失败
时 `main_sim.py` 会在写 `d.ctrl` 和 `mj_step` 前终止。

正式 full-task 的 `[0,24 ms)` fixed-posture PD 由 `main_sim.py` 执行，不由 worker
秘密运行 MPC。t=0 可以在不推进 MuJoCo 时间、也不改真实 `d.ctrl` 的条件下做
一次 dry preflight；24 ms handoff 时，上一物理拍真实执行的 PD torque 被显式
送入第一次 MPC mapping。

关闭时写带 `kRequestShutdown` 的请求；worker发布同request id的shutdown
响应后退出。EOF也会让worker退出。

## 构建和测试

```bash
chmod +x cpp/right_arm_sim_runtime/build_and_test.sh
cpp/right_arm_sim_runtime/build_and_test.sh
```

测试覆盖固定 ABI、seqlock、模型维度、RNEA/mapper/Executor 正常全链、缓存
复用、配置变更、NaN、维度错误、真实 fork/exec 共享内存与 pipe 握手，以及
shutdown。mapper 的 safe-hold、line search 与 `NO_SAFE_TORQUE` 分支由相邻
[`ddq_torque_mapper`](../ddq_torque_mapper/) 回归测试覆盖。

布局查询不需要scene和pipe：

```bash
/tmp/hold-my-beer-mpc-right-arm-sim-runtime-build/\
right_arm_sim_runtime_worker --print-layout
```

当前协议数组上限是 `nq/nv/nu/nbody=64/64/64/64`。worker启动时从scene
读取真实维度，每份请求必须严格相等；当前scene为 `30/29/23/27`。

CPU 7 affinity、六个单线程数值库环境变量、Torch 和 GC 并非本 worker 自己
静默修正，而是由正式 launcher 设置、由 `main_sim.py` 在第一个 `mj_step` 前
同时核对 parent/worker。这个受控 MuJoCo timing 仍不是硬件 hard-RT 证据；见
[REALTIME_RUNTIME.md](../../REALTIME_RUNTIME.md)。
