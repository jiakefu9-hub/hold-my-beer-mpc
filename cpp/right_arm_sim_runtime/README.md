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

关闭时写带 `kRequestShutdown` 的请求；worker发布同request id的shutdown
响应后退出。EOF也会让worker退出。

## 构建和测试

```bash
chmod +x cpp/right_arm_sim_runtime/build_and_test.sh
cpp/right_arm_sim_runtime/build_and_test.sh
```

测试覆盖固定ABI、seqlock、模型维度、RNEA/mapper/Executor全链、缓存复用、
配置变更、NaN、维度错误、真实fork/exec共享内存与pipe握手，以及shutdown。

布局查询不需要scene和pipe：

```bash
/tmp/hold-my-beer-mpc-right-arm-sim-runtime-build/\
right_arm_sim_runtime_worker --print-layout
```

当前协议数组上限是 `nq/nv/nu/nbody=64/64/64/64`。worker启动时从scene
读取真实维度，每份请求必须严格相等；当前scene为 `30/29/23/27`。
