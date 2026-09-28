# 力矩 MPC 鲁棒性研究：离线证据

2026-09-28 开始，09-29 整理。基线 `a73a58a13adda4310cdd87ca51f7eefe848d4f25`。
说明见 [HARDWARE_MPC_ROBUSTNESS.md](../../docs/g1_field_validation/HARDWARE_MPC_ROBUSTNESS.md)。
无机器人连接、DDS publisher 或真实输出。完成的是离线研究阶段，不是现场验收。

## 文件

- `studies/*.json`：七项单因素／组合对照、候选修正、延迟估计偏差和向量化前后结果；保留当次源码／模型／库哈希。
- `comparison/`：从保存的 NPZ 重新核算的图表和 JSON，检查数值与原 summary 一致；明确区分完整试验和失败前缀。
- `failures/*.npz`：原版和仅制动候选的原约束矩阵，供独立 LP 重算。失败状态在相应 `studies/*.json` 中。
- `timing/summary.json`：两轮完整流程计时，合成静止输入＋真实冻结预测器＋制动约束，**不含新延迟预测**。
- `timing/run*_audit.json`：从完整 JSONL 独立重算的消息审计及原文件 SHA256。
- `verification.json`：插值实现前后逐数组对照、测试和证据清单。

完整闭环 NPZ、运行 JSONL 保留在被 Git 忽略的 `evaluation/hardware_shadow/commissioning/`，不批量提交。
`comparison/comparison.json` 记录每个原始 NPZ 和 summary 的路径及 SHA256。
没有覆盖旧阶段报告，也没有把模型输出放进真机轨迹数据集。

## 重跑

以下命令均离线。输出目录必须不存在；不要覆盖本次保存证据。

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
conda activate g1_mpc

# 原版七因素对照；exit=0 代表研究器完成，不代表每个 case 通过。
python tools/g1_commissioning/study_torque_robustness.py --duration 5 --cpu 4 \
  --output-dir evaluation/hardware_shadow/commissioning/torque_baseline_recheck

# 只开制动约束，保留会失败的案例。
python tools/g1_commissioning/study_torque_robustness.py --duration 5 --cpu 4 \
  --torque-config configs/hardware_mpc_torque_recovery.yaml \
  --output-dir evaluation/hardware_shadow/commissioning/torque_guard_recheck

# 加显式延迟假设：绝不从 plant 参数偷偷读取延迟或质量。
python tools/g1_commissioning/study_torque_robustness.py --duration 20 --cpu 4 \
  --torque-config configs/hardware_mpc_torque_recovery.yaml \
  --assumed-command-delay-ms 6 --scenarios both_delays combined \
  --output-dir evaluation/hardware_shadow/commissioning/torque_compensated_recheck

# 数字来自以上 NPZ，非人工编写；失败前缀不与完整20秒混算。
python tools/g1_commissioning/report_torque_robustness.py \
  --study baseline=evaluation/hardware_shadow/commissioning/torque_baseline_recheck \
  --study guard=evaluation/hardware_shadow/commissioning/torque_guard_recheck \
  --study compensated=evaluation/hardware_shadow/commissioning/torque_compensated_recheck \
  --compare-until 2.4 \
  --output-dir evaluation/hardware_shadow/commissioning/torque_report_recheck

# 单独运行，避免上述模型试验／测试抢占 CPU 干扰计时。
python tools/g1_commissioning/benchmark_hardware_mpc.py --cpu 2 --runs 2 \
  --actuation measured_torque_preview --predictor learned_filtered \
  --torque-config configs/hardware_mpc_torque_recovery.yaml \
  --output-dir evaluation/hardware_shadow/commissioning/torque_lifecycle_recheck
python tools/g1_commissioning/audit_measured_torque_replay.py \
  evaluation/hardware_shadow/commissioning/torque_lifecycle_recheck/run1
python tools/g1_commissioning/audit_measured_torque_replay.py \
  evaluation/hardware_shadow/commissioning/torque_lifecycle_recheck/run2
```

仅延迟候选：省去 `--torque-config`，运行 combined 5 秒。
延迟估计误差对照：保留配置，分别设置 `--assumed-command-delay-ms 4` 和 `8`，运行 combined 5 秒。
保存结果中的这些两组是在标量插值版本上运行，不能拿其时间直接与优化后版本排性能名次。

## 测试

完整相关测试共 107 项。覆盖候选前向检查、消息与 PD 算术、原仿真 QP 对照、延迟历史因果性、
独立 SO(3) 插值校验、原约束保留、实际 QP 初值的独立 LP、JSON 数值一致性和共享 PID 路径。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/g1-mpc-mpl \
  PYTHONPATH=tools/g1_commissioning/tests:tools/g1_commissioning:. \
  /home/fjk/miniforge3/envs/g1_mpc/bin/python -m unittest \
  test_hardware_mpc test_hardware_mpc_analysis test_mpc_host test_mpc_replay_audit \
  test_hardware_health test_hardware_journal test_hardware_runner \
  test_arm_torque_feedback test_hardware_arm_inverse_dynamics \
  test_mpc_inverse_preview test_measured_torque_mpc \
  test_hardware_pid_control test_pid_6ms test_mpc_recovery test_mpc_serialization \
  test_torque_robustness_study test_mpc_delay_preview test_torque_robustness_report -q
```

首次完整合跑发现 HiGHS 进程全局线程设置不一致导致 status 4（不是数学无解）；
统一测试与诊断的单线程设置后重跑。保留事实，不能把这个测试错误混成压力案例的 status 2 无解。

未通过项仍包括整条含延迟预测路径的 6 ms 时间验收、真实执行器辨识以及正常／异常交还的物理验证。
