# G1 真机实验入口

这里区分“如何操作”“已经证实什么”“每次实际发生了什么”。
截至 2026-09-16：H1 只读采集、离线 HIL、A2 静态响应、A3 平衡控臂和 IMU 零点观察已有记录；
新相位观察／行走采集程序完成离线准备，**尚未真机执行**。PID/MPC 真机稳杯尚未验证。

## 现在最常用的几份文件

| 想做什么 | 看哪里 |
| --- | --- |
| 快速回忆已知结论和未知项 | [KNOWN_RESULTS.md](KNOWN_RESULTS.md) |
| 查某天实验的成果、参数与证据路径 | [sessions/README.md](sessions/README.md) |
| 下一步看步态相位、采走动时的原始扰动 | [RAW_WALK_CAPTURE.md](RAW_WALK_CAPTURE.md) |
| 查 A1/A2/A3 的完整操作与接口依据 | [RUNBOOK.md](RUNBOOK.md)；这是原来的长篇现场 README |
| 只做首次 H1 状态读取 | [详细指南](h1/G1_H1_FIELD_RUNBOOK.md)、[现场速查](h1/G1_H1_FIELD_CHECKLIST.md) |
| 理解瓶子中心、身体系和世界系 | [ENDPOINT_FRAME.md](ENDPOINT_FRAME.md) |
| 复做 IMU 零点时机实验 | [IMU_ZERO_REFERENCE_TEST.md](IMU_ZERO_REFERENCE_TEST.md) |
| 编译程序、找 profile 和源码 | [工具说明](../../tools/g1_commissioning/README.md) |
| 写下一次 PID/MPC 或采集实验记录 | [SESSION_TEMPLATE.md](SESSION_TEMPLATE.md) |

## 当前能力边界

| 路径 | 状态／不要混淆的地方 |
| --- | --- |
| H1 state-only | 已有真实状态与审计 PASS；不等于所有硬件语义、现场验收均完成 |
| Publisher-absent HIL | 离线输出链记录 would-write；没有给电机发命令 |
| 独立 A2 / A3 | 已实测 `rt/arm_sdk` 静态响应／双臂平衡控臂，不是整机低层接管 |
| 30 秒 phase observer | 只读程序离线就绪；GetPhase 是否可用仍未知 |
| 19 秒 walk collector | 有真实双臂和速度输出能力，仅离线就绪；需独立现场执行 |
| 真机 PID/MPC / production adapter | 未完成真机闭环验证；`cpp/unitree_arm_adapter` 仍禁止真实 publisher |

下一步按原始采集指南先观察相位，再安排定时行走采集。航向目标是 **IMU 世界系 +X（yaw=0）**，
不是起步时的朝向；目前只保存原始数据，不在线构造扰动模板。
本导航不发出运行授权，也不修改任何停止门或模式要求。

## 文件怎么放

`h1/` 放只读现场指南；`sessions/` 放按日期排列的真实记录和离线准备记录；
本层放可复用操作指南、坐标定义和结论索引。代码、原始数据、官方 PDF 不混进来。
原始日志留在本地 `evaluation/`，每次记录引用实际目录和哈希；Git 克隆不自带这些原始数据。
整个项目的分区见 [文档总导航](../README.md)。
