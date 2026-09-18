# 工具导航

所有命令默认从仓库根目录执行。目录位置不是输出许可，运行前看对应程序的能力说明。

| 位置 | 用途 | 说明 |
| --- | --- | --- |
| [g1_commissioning/](g1_commissioning/README.md) | 独立真机查询、A2/A3、原始行走采集和首次 PID | C++ 默认无 SDK／网络；Python PID 也需独立 profile、许可和人工确认；离线准备不代表真机通过 |
| `realtime/run_hardware_state_inspection.sh` | H1 只读状态检查 | [H1 现场指南](../docs/g1_field_validation/h1/G1_H1_FIELD_RUNBOOK.md) |
| `realtime/audit_hardware_state_trace.py` | 对保存的状态 trace 离线审计 | [H1 结果](../docs/g1_field_validation/sessions/20260911_H1.md) |
| `realtime/run_hardware_shadow.sh` | 共享核心的 shadow／离线阶段入口 | [Shadow 边界](../docs/hardware/HARDWARE_SHADOW.md)；不是控臂实验启动器 |
| `experiments/qa_qalpha/` | 现有仿真参数实验 | 不是真机 PID/MPC 部署工具 |
| 根层 `build_*evidence.py`、`archive_*`、`prepare_data_cleanup.py` | 冻结仿真证据、归档与清理工具 | 本次未运行；不能把整理文档当作授权清理原始数据 |

后续现场程序与测试仍放在 `g1_commissioning/`，不按每次实验另复制一套源码。
实机具体参数随 session 保存；仓库中维护审阅模板，不提交本地原始数据。
