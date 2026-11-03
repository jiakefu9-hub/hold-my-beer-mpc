# 工具地图

所有命令从仓库根目录执行。环境和构建见 [快速开始](../docs/QUICKSTART.md)。

| 目录／入口 | 用途 |
| --- | --- |
| `experiments/simulation_showcase/replay.py` | D01/D02/D03 保存轨迹的回放；无 SDK、无控制输出 |
| `experiments/simulation_showcase/run_study.py` | 三组展示仿真的重新计算 |
| `g1_commissioning/g1_run_morning_20261013.py` | 最终 M 系列，默认只显示命令，显式选项才能运行 |
| `g1_commissioning/g1_walk_mpc_learned.py` | 真机学习 MPC 独立入口 |
| `g1_commissioning/g1_walk_pid.py` | 真机位置 PID 独立入口 |
| `g1_commissioning/analyze_hardware_pid.py` | 原始日志末端重建与统一指标 |
| `realtime/` | 早期只读采集及 shadow/HIL；不是最终现场控臂入口 |
| `build_*evidence.py`、`archive_*`、`prepare_data_cleanup.py` | 历史归档工具及删除合同；复现展示无需运行 |
| `visualize_bottle_mount.py` | 负载几何与坐标查看 |

## 展示入口为什么依赖其他实验目录

```text
simulation_showcase/run_study.py
  -> simple_controller_comparison/study.py          批运行、计时、保存
  -> c_mpc_followup/simulation_entry.py              当前扰动保持适配
     -> simple_controller_comparison/simulation_entry.py
     -> simple_controller_comparison/controllers.py PID / C / MPC 选择
     -> mpc_comprehensive/strategy.py                研究配置适配
  -> mpc_comprehensive/metrics.py                    统一六项指标
```

这些目录中的 C、q-only MPC、预览修正等扩展只有明确配置才启用。
正式 D01–D03 没有启用额外姿态预测校正或时间前视。
因此保留这些源文件是运行依赖，不应因其研究日期较早而删除。
历史候选 JSON/YAML 用于复现实验，默认只运行明确指定的 case 文件。

现场源码、C++ 工具和停止条件详见 [g1_commissioning](g1_commissioning/README.md)。
