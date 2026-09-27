# G1 真机实验入口

这里区分“如何操作”“已经证实什么”“每次实际发生了什么”。
截至 2026-09-27：H1、离线 HIL、A2/A3 和 IMU 零点观察已有记录；20 ms PID 有实机复验，
6 ms PID 待复验。新真机 MPC 程序、冻结预测模型及对应文档已完成离线准备；尚无 MPC 真机结果。

采集／预测更新（2026-09-25）：旧五条带牵拉轨迹已撤回；十二条固定 H0 新轨迹已完成离线审计、
方法比较与画图。当前入口是[十二条 H0 预测研究](H0_WALK_PREDICTOR_STUDY.md)，不是下方旧五条报告。

## 现在最常用的几份文件

| 想做什么 | 看哪里 |
| --- | --- |
| 看当前十二条 H0 轨迹、预测代码与可复算结果 | [H0_WALK_PREDICTOR_STUDY.md](H0_WALK_PREDICTOR_STUDY.md)：24 页图集、方法比较、下一步模板方案 |
| 快速回忆已知结论和未知项 | [KNOWN_RESULTS.md](KNOWN_RESULTS.md) |
| 查某天实验的成果、参数与证据路径 | [sessions/README.md](sessions/README.md) |
| 下一步看步态相位、采走动时的原始扰动 | [RAW_WALK_CAPTURE.md](RAW_WALK_CAPTURE.md) |
| 做“左臂固定、右臂 PID”的首次真机持瓶实验 | [HARDWARE_PID.md](HARDWARE_PID.md)：程序、完整 `[5,18)` 指标与仿真差异 |
| 6 ms PID 复验后做右臂 MPC | [HARDWARE_MPC.md](HARDWARE_MPC.md)：预测模型、程序对应关系、时间验证和现场命令 |
| 看五条轨迹是否可用、如何对齐及能否预测扰动 | [WALK_DATASET_AUDIT.md](WALK_DATASET_AUDIT.md)：离线审计与世界系模板建议 |
| 比较相位模板、多关节和 IMU 历史预测方法 | [WALK_PREDICTOR_METHOD_STUDY.md](WALK_PREDICTOR_METHOD_STUDY.md)：论文依据与五条实测数据比较 |
| 看五条腿部／身体 IMU 曲线 | [WALK_SIGNAL_GALLERY.md](WALK_SIGNAL_GALLERY.md)：22 张图、HTML 与 PDF |
| 测普通调度下的通信往返、收包和 6 ms 主机抖动 | [TIMING_VALIDATION.md](TIMING_VALIDATION.md)，集成到只读 phase probe；尚无本轮真机数字 |
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
| 30 秒 phase observer | 已实测；本机 `GetPhase` 返回 7301，未取得可用官方相位 |
| 21 秒 walk collector | 十二条固定 H0 轨迹已采集并离线研究；旧五条因吊绳牵拉全部撤回；第 06 条缺完整会话尾标记，单独补充分析 |
| 真机 PID | 20 ms 版已复验；当前 6 ms 版离线准备，下一轮为首次 6 ms 实机验证 |
| 真机 MPC | 独立参考轨迹 MPC、冻结 H0 模型和离线评估入口；尚未完成真机闭环验证 |
| production adapter | 本次不变；`cpp/unitree_arm_adapter` 仍禁止真实 publisher |

旧五条轨迹仅保留历史审计和方法参考，不再作为训练／检验数据。十二条新版轨迹用走前最后两秒平均
朝向定义本轮固定 H0，航向保持以 H0 +X 为目标；原始日志不改写，全部 IMU 样本离线派生到 H0。
冻结模型已接入独立 MPC 候选，研究精度不代表真机控制效果；采集程序不在线构造扰动模板。
本导航不发出运行授权，也不修改任何停止门或模式要求。

## 文件怎么放

`h1/` 放只读现场指南；`sessions/` 放按日期排列的真实记录和离线准备记录；
本层放可复用操作指南、坐标定义和结论索引。代码、原始数据、官方 PDF 不混进来。
原始日志留在本地 `evaluation/`，每次记录引用实际目录和哈希；Git 克隆不自带这些原始数据。
整个项目的分区见 [文档总导航](../README.md)。
