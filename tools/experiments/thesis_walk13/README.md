# 10 s 前进＋3 s 停车仿真

当前论文三组仿真与真机均以前进请求开始为零点，10 s 请求停车、13 s 结束评价。新场景由 `protocol.py` 统一采集、模板与在线运行时序；`full_template.py` 保留原 11＋4 初态设计，`simulation_entry.py` 接入原仿真链和既有当前角速度外推规则，`metric_packager.py` 沿用原五项指标定义。

已完成的结果、限制和三条回放命令见[结果说明](../../../docs/simulation/experiments/20261117_walk10_stop3/README.md)。回放使用 `replay.py D01/D02/D03`，实际时长 13 s；`--check` 核对，`--once` 单次播放。旧仿真入口与数据已清理；当前入口所需共享模块集中保留于本目录。

原生运行库由 `bash cpp/build_runtime.sh` 构建。`run.py` 会采集完整 11＋4 并运行三组，已有最终清单时拒绝覆盖。本目录只有仿真与保存状态回放入口，不导入机器人 SDK 或连接机器人。

如需独立重新采集和运行，使用 `run.py --variant <新名称>`；默认已完成结果拒绝覆盖。
