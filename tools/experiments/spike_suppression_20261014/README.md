# 学习 MPC 冲击尖峰：离线研究入口

本目录没有真机命令或 DDS 发布器，普通 MPC/真机入口不加载这里的控制器。
报告见 `docs/g1_field_validation/sessions/20261014_LEARNED_MPC_SPIKE_AUDIT.md`。
当前结论：增加加速度增量代价能减少回放指令反向，但尚未改善闭环综合表现，**不作为上机版本**。

唯一新增控制项：

```
J_delta = lambda * (||a_0 - a_previous||² + Σ[k=1..N-1] ||a_k-a_(k-1)||²)
```

这是加速度增量软代价，没有额外除以 dt，不是硬 jerk 上限。原状态、输入、任务权重和边界保留。
稀疏仿真 QP 增加相邻输入交叉块；硬件离线 QP 经相同状态消元后加入同一个二次型。
`algebra_check()` 用独立差分矩阵核对两种实现，并检查零权重的成本矩阵完全一致。
回放在 packet 被离线接受后，用最终建模加速度更新 a_previous，含启动交接。
仿真使用上一生成动作，因此 mapper 回退/延迟实验中不能声称它是最后物理实现加速度；这项限制保留在报告中。

运行前在仓库根目录构建纯仿真依赖：

```bash
bash cpp/build_runtime.sh
/home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/experiments/spike_suppression_20261014/run_simulation.py \
  tools/experiments/spike_suppression_20261014/cases.json --cpus 2 --keep
/home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/experiments/spike_suppression_20261014/run_simulation.py \
  tools/experiments/spike_suppression_20261014/followup.json --cpus 2 --keep
```

仿真入口使用原完整走停模板和原仿真基线，不把它称为真机 H0 学习预测器；6 ms，9 段，8 s 任务。
同名结果有缓存，复测应给 JSON 中试验名增加后缀，避免覆盖。

真实日志只读回放：

```bash
/home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/experiments/spike_suppression_20261014/run_replay.py \
  --source evaluation/hardware_shadow/commissioning/morning_M01_mpc_learned_zero_payload75_walk_20261014_125333_831147_ee9f/raw.jsonl \
  --output evaluation/spike_suppression_20261014/replay_new --cpu 4
```

网络构造被显式禁止；只在内存中构造、序列化 SDK 数据类型，不初始化 DDS。
输出保留 compact NPZ、每轮结果、输入/控制器哈希；仅删除本次生成的大体积回放 JSONL，不删除任何真机日志。
回放的状态不会响应新指令，因此只能评价数值一致性、约束、指令变化，不能评价实际末端改善。
离线耗时不包括 DDS、进程 IPC、真实调度和落盘，不是 6 ms 真机验收。

`decompose_bias.py RAW [RAW ...]` 用 MuJoCo 条件右臂模型独立分解日志的 b 项，
包含重力、基座线加速度、基座角加速度和速度项；分解之和须与日志闭合到 1e-5 Nm 内。
它分解的是模型命令，不是测得的电机实际力矩或碰撞力。
