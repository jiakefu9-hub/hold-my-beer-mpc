# 配置索引

| 用途 | 配置 |
| --- | --- |
| 三组展示仿真 | `simulation_showcase_20261103/D01_position_pid_balanced.yaml`、`D02_mpc_current_hold.yaml`、`D03_mpc_predictive.yaml` |
| 展示的最终模型覆盖与实验名 | `../tools/experiments/simulation_showcase/cases.json`；明确覆盖为 `scene_payload75.xml` |
| 真机名义负载 | `g1_payload75.yaml` |
| 最终学习／保持 MPC 权重 | `hardware_mpc_learned_acc001_alpha0005_omega1.yaml` |
| 最终 MPC 力矩执行 | `hardware_mpc_torque_learned.yaml` |
| 零位位置 PID / 放宽参考限速 PID | `hardware_position_pid_zero_20261013.yaml` / `hardware_position_pid_zero_relaxed_20261013.yaml` |
| C 前馈 PID | `hardware_acceleration_pid_20261012.yaml`；T 系列独立候选在 `hardware_control_comparison_20261012/` |
| 原冻结仿真 | `g1.yaml` |

`simulation_optimization_20261010/`、`simulation_simple_controllers_20261011/`、
`simulation_acceleration_pid_20261012/` 等是有结果支持的历史研究配置，保留用于复现。
其他现场 YAML 是显式候选或历史实验条件，不能因日期较新就当作推荐参数。
历史现场数据的有效配置在每轮日志及归档快照中；后续改 YAML 不改变已录制结果。
