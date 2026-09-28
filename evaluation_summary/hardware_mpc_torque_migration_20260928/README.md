# 实测状态 MPC／力矩迁移的离线证据

本目录发布可复算的小型模型试验结果、图片和计时摘要；没有真实机器人输出。
完整说明见 [迁移文档](../../docs/g1_field_validation/HARDWARE_MPC_TORQUE_MIGRATION.md)。
基线为 `3e63977a763ea3f905d67359e7d523ead6119f63`。

- `closed_loop/summary.json`：八组闭环模型对照、源码／库／XML 哈希、失败状态和独立可行性检查。
- `closed_loop/*.npz`：每组 6 ms 采样的曲线数字，包含失败组的已完成前缀；不是实机采集。
- `closed_loop/comparison.png`：由上述程序生成；STOPPED 标出不完整的压力测试。
- `timing/summary.json`：两轮完整离线时间，开启冻结预测器，输入为合成静止状态。
- `timing/run1_audit.json`、`run2_audit.json`：独立消息审计及原始 JSONL 哈希。
- 完整计时日志仍在本地 `evaluation/hardware_shadow/commissioning/measured_torque_timing_final_20260928/`，
  不作为真机数据或整批原始 JSONL 提交。

结果状态必须分开理解：数学／消息测试通过；模型匹配的三秒闭环完成；
延迟／负载偏差压力测试未通过；6 ms 截止时间未通过；真实控制未运行。

本轮共 72 项相关测试通过：第一组 54 项，另复查共享代码影响的 18 项 PID／6 ms 测试。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/g1-mpc-mpl \
  PYTHONPATH=tools/g1_commissioning/tests:tools/g1_commissioning:. \
  /home/fjk/miniforge3/envs/g1_mpc/bin/python -m unittest \
  test_hardware_mpc test_hardware_mpc_analysis test_mpc_host test_mpc_replay_audit \
  test_hardware_health test_hardware_journal test_hardware_runner \
  test_arm_torque_feedback test_hardware_arm_inverse_dynamics \
  test_mpc_inverse_preview test_measured_torque_mpc -v

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/g1-mpc-mpl \
  PYTHONPATH=tools/g1_commissioning/tests:tools/g1_commissioning:. \
  /home/fjk/miniforge3/envs/g1_mpc/bin/python -m unittest \
  test_hardware_pid_control test_pid_6ms -v
```

本目录是离线开发阶段证据，不是现场准入证明。压力案例里使用的 4／6 ms 延迟和 +20% 负载偏差
是显式假定，不是对这台 G1 的测量。完整计时未包含 DDS／网络传输或实际电机响应。
