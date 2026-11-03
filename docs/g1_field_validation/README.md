# 真机实验与复现入口

本阶段实验和主对照录像已完成。首先读 [最终实验清单](../FINAL_EXPERIMENTS.md)、
[精选数据归档](evidence/20261014_final_archive/README.md)和[实验状态](EXPERIMENT_STATUS.md)。

| 目的 | 文档 |
| --- | --- |
| 当前学习 MPC 的模型与程序 | [HARDWARE_MPC_LEARNED](HARDWARE_MPC_LEARNED.md) |
| 力矩执行、日志和跟踪分析 | [HARDWARE_MPC_EXECUTION](HARDWARE_MPC_EXECUTION.md) |
| 位置 PID 与评价定义 | [HARDWARE_PID](HARDWARE_PID.md) |
| 末端坐标和瓶轴方向 | [ENDPOINT_FRAME](ENDPOINT_FRAME.md) |
| 学习预测器的输入、标签和验证 | [H0 研究](H0_WALK_PREDICTOR_STUDY.md)、[细化](H0_WALK_PREDICTOR_REFINEMENT.md) |
| 最终六轮和录像实验 | [上午结果及录像](sessions/20261013_MORNING_RESULTS_AND_VIDEO_PLAN.md) |
| C 前馈 PID 对照 | [最终 T 系列](sessions/20261012_FINAL_COMPARISON.md) |
| 逐轮结果、故障和修复 | [session 索引](sessions/README.md) |
| 首次接机、模式与停止条件 | [RUNBOOK](RUNBOOK.md)、[H1 检查](h1/G1_H1_FIELD_CHECKLIST.md) |
| 程序、构建和 profile | [工具说明](../../tools/g1_commissioning/README.md) |
| 追加实验记录格式 | [SESSION_TEMPLATE](SESSION_TEMPLATE.md) |

最终常规实验：双臂零中立位，正常左臂 PD 20/1，右臂 6 ms MPC 或位置 PID，
75 mm／250 g 名义负载模型，PC1 内置腿部行走。实际附件与安装并非每轮完整称量辨识。
学习 MPC 主基线关闭时间对齐、无额外肩 roll 回正；`hold_current` 保留当前扰动及逆动力学补偿。
M08 增益实验、M10 交互暂停、T 系列 C 等必须按各自 session 条件解释。

`tools/g1_commissioning/` 已经有实际控臂／行走输出能力，目录名不表示只读。
参数与操作条件是原机器的已记录配置，不是新机器的自动运行授权。
当前整理只做离线验证；实际运行仍沿用入口的显式 execute、状态检查及停止退权流程。

旧五条受吊绳牵拉轨迹已撤回；相关历史报告移到 Git 恢复点，当前预测器依据十二条固定 H0 数据。
系统辨识没有得到可靠可分离参数，因此没有将拟合值强行写回模型。
完整研究、负结果与边界见 [已知结论](KNOWN_RESULTS.md)。
