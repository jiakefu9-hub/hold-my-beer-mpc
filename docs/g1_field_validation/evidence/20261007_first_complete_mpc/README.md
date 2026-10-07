# 首次完整真机 MPC 基线（2026-10-07）

本目录冻结成功轮的分析成果；后续修复不覆盖这里的数据。原始日志仍保存在本机
`evaluation/hardware_shadow/commissioning/mpc_torque_walk_reboot_performance_20261007_172732/raw.jsonl`，
SHA-256 为 `375f302a9709cdc2b19ab6f1e1e579621aff5061e50b68ef02a918458ad06f28`。
同名 `.tar.gz` 本地备份包含原始数据、现场配置和分析结果，未将 80 MB 原始日志提交到 Git。

## 实验及版本边界

- `hold_current`，6 ms；3 s 抬臂、2 s 等待、10 s 请求 0.5 m/s 前进、3 s 停车、3 s 退权。
- 左臂既定 PD 姿态；右臂实测状态加速度 MPC、逆动力学、候选力矩检查；右肩 yaw 零参考软 PD 为 2/0.2。
- 末端线加速度、角加速度、角速度代价为零，学习预测器未使用。
- 操作者确认全程平顺、没有外飘；软件正常退权到 weight=0，161 次速度 RPC 均成功。
- `run_provenance.json` 提取自原始日志，保存运行时源码哈希、配置和主机环境；本次提交保留相同控制源码。
- 已知未解决项：RPC 异常曾阻止自动退权；运行中 FSM 已改为不轮询，不能实时确认所有模式变化。

## 结果文件与统一比较

`mpc_summary.json`、`pid_summary.json` 是原离线分析摘要的原样副本；
`endpoint_metrics_h0.png` 为本轮曲线。两者均按全走停 `[5,18)` 计算，不能只截稳态宣称全程性能。
端点来自仿真 XML 的瓶身中心，姿态由实测关节角和身体 IMU 结合模型重建；没有瓶身独立姿态传感器。

| 右瓶相对 H0 竖直方向的倾角 | 6 ms PID（10-05 单次） | MPC（本轮单次） |
| --- | ---: | ---: |
| 平均 | 1.591° | 0.649° |
| RMS | 1.899° | 0.836° |
| P95 | 3.774° | 1.354° |
| 最大 | 6.545° | 4.358° |

RMS 降低约 56%，但不是同日随机交替重复实验，不能当作统计显著优越性。
本轮右瓶线／角加速度 RMS 为 2.188 m/s²、6.528 rad/s²，比 PID 的 1.895、5.816 高；
当前结论是端正改善，并未证实所有动态指标改善。

相邻控制开始间隔中位数为 6.000 ms，约 47.3% 略大于 6 ms；超过 7 ms 为 13/2155（0.603%）。
原 deadline 统计为 10/2155（0.464%），包括计时入队后的完整统计为 12/2155（0.557%）。
deadline 检查的是本拍是否在预定下拍时刻之前做完工作，不是相邻两拍开始间隔是否大于 6 ms。

## 复算

在仓库根目录使用原始日志运行：

```bash
MPLBACKEND=Agg /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/analyze_hardware_mpc.py \
  evaluation/hardware_shadow/commissioning/mpc_torque_walk_reboot_performance_20261007_172732/raw.jsonl \
  --output-dir /tmp/g1_first_mpc_reanalysis
```

软件回归已覆盖健康检查和无 DDS 的运行器正常／失败路径；这份成功基线不承诺所有异常下都能退权。
