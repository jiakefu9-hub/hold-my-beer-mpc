# 延迟预测接入完整离线任务：2026-09-29

基线 `39f201fd1bdf02ddcbdcc0dce99c3d8fd8f2d95b`。
没有 DDS participant／publisher，没有真实输出，未修改仿真控制器或 PID，也没有改系统实时／电源设置。

已完成软件集成：冻结预测器、区间扰动平移、命令时刻状态估计、MPC 与候选力矩、接管／退权、
本地 SDK 消息及 CRC／序列化、float32 最终命令历史、日志和独立审计。
说明见 [HARDWARE_MPC_ROBUSTNESS.md 第 6 节](../../docs/g1_field_validation/HARDWARE_MPC_ROBUSTNESS.md#6-09-29完整离线任务集成)。

## 结论

- 113 项相关测试通过；两轮本地消息审计通过、日志无丢失、最终 weight 与附加前馈均为零。
- 主窗口整链平均 8.545／8.675 ms，p99 为 9.340／9.802 ms；全部主窗口周期超过 6 ms。
- 实际调度大多跳至 12 ms，不代表把控制器改为 12 ms，也不能声称 6 ms 已可用。
- 所有主窗口消息都晚于暂假定的“周期开始＋6 ms”才准备完，该假定作用时刻不具备实机时序有效性。
- 静止反馈是计时输入，不响应这些新命令，不能据此评价加速度跟踪或稳瓶效果。
- weight 的过渡混合和执行延迟均未实机辨识。真实电机响应不通过多做离线模型测试来冒充验收。

`final_summary.json` 保存两轮完整计时、环境和源码哈希；`run*_audit.json` 保存独立消息审计与原始日志哈希。
`preoptimization_summary.json` 保存优化前约 8.94 ms 平均工作时间的一轮，不能用不同批次差异宣称严格的性能因果结论。
`verification.json` 保存测试、完整性与“包准备晚于假定时刻”的计算结果及文件哈希。
原始 JSONL 和逐周期 timing.json 保留在本地 `evaluation/hardware_shadow/commissioning/torque_delay_lifecycle_final_20260929/`。

第一次烟测遇到 SDK 默认构造器是函数、不能直接调用其 deserialize 的软件错误；
已改为从实际消息类型反序列化，随后完整流程和相关测试通过。失败记录仍在本地 `torque_delay_lifecycle_smoke_20260929/`，没有伪装成一轮成功数据。

## 复验

不要同时运行压力试验或重负载测试干扰计时。目录必须尚不存在。

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
conda activate g1_mpc
python tools/g1_commissioning/benchmark_hardware_mpc.py \
  --actuation measured_torque_preview \
  --torque-config configs/hardware_mpc_torque_recovery.yaml \
  --predictor learned_filtered --assumed-command-delay-ms 6 --observation-delay-ms 4 \
  --runs 2 --cpu 2 \
  --output-dir evaluation/hardware_shadow/commissioning/torque_delay_lifecycle_recheck
python tools/g1_commissioning/audit_measured_torque_replay.py \
  evaluation/hardware_shadow/commissioning/torque_delay_lifecycle_recheck/run1
python tools/g1_commissioning/audit_measured_torque_replay.py \
  evaluation/hardware_shadow/commissioning/torque_delay_lifecycle_recheck/run2
```

`--assumed-command-delay-ms` 在这里按“周期开始→假定作用时刻”解释，不是 DDS 往返时间，也不是测得的电机时延。
`--observation-delay-ms` 仅把回放数据延后递交，仍保留原时间戳。现场输出 CLI 没有新增这两个开关。
`status=complete`／进程退出 0 只表示流程结束，**不表示截止时间通过**，必须查看 `primary_5_18.deadline_misses`。

新增重点测试是 `test_mpc_delay_lifecycle`：区间平均的重叠积分、最终消息才入历史、
原始观测与预测初值分别审计、SDK float32 往返、正常退权和异步观测时间。
完整 113 项测试是上一阶段 107 项加上该模块的 6 项，命令：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/g1-mpc-mpl \
  PYTHONPATH=tools/g1_commissioning/tests:tools/g1_commissioning:. \
  /home/fjk/miniforge3/envs/g1_mpc/bin/python -m unittest \
  test_hardware_mpc test_hardware_mpc_analysis test_mpc_host test_mpc_replay_audit \
  test_hardware_health test_hardware_journal test_hardware_runner \
  test_arm_torque_feedback test_hardware_arm_inverse_dynamics \
  test_mpc_inverse_preview test_measured_torque_mpc test_hardware_pid_control test_pid_6ms \
  test_mpc_recovery test_mpc_serialization test_torque_robustness_study \
  test_mpc_delay_preview test_torque_robustness_report test_mpc_delay_lifecycle -q
```
