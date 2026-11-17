# 文档导航

新人按 [快速开始](QUICKSTART.md) → [架构](../ARCHITECTURE.md) → [最终实验清单](FINAL_EXPERIMENTS.md)
阅读即可。实验已经完成，旧记录里的“明天”“下一步”按记录当时理解。

| 内容 | 当前入口 |
| --- | --- |
| 安装、回放、构建、重跑、离线验证 | [QUICKSTART](QUICKSTART.md) |
| 论文主对照、最终录像对应数据 | [FINAL_EXPERIMENTS](FINAL_EXPERIMENTS.md) |
| MPC / PID 数学 | [MPC](design/MPC_DESIGN.md)、[PID](design/PID_DESIGN.md) |
| 仿真模板、时间、航向 | [Full Task](simulation/FULL_TASK_TEMPLATE.md)、[实时计时](simulation/REALTIME_RUNTIME.md)、[航向](simulation/HEADING_CONTROL.md) |
| 三组展示仿真 | [展示报告](simulation/experiments/20261117_walk10_stop3/README.md) |
| 真机算法、使用条件 | [真机入口](g1_field_validation/README.md) |
| 已有结论及未解决项 | [实验状态](g1_field_validation/EXPERIMENT_STATUS.md)、[已知结果](g1_field_validation/KNOWN_RESULTS.md) |
| 逐次现场记录 | [session 索引](g1_field_validation/sessions/README.md) |
| 精选原始日志、配置、复算数据 | [真机归档](g1_field_validation/evidence/20261014_final_archive/README.md) |
| 预测器研究 | [十二条 H0 轨迹](g1_field_validation/H0_WALK_PREDICTOR_STUDY.md) |
| 工程困难及开发过程 | [Challenge](history/CHALLENGE.md)、[MPC 开发记录](history/MPC_DEVELOPMENT_LOG.md) |
| 删除项、恢复版本、验收结果 | [仓库整理记录](history/REPOSITORY_CLEANUP_20261103.md) |

`simulation/experiments/` 当前只保留 10 s 前进＋3 s 停车的三方法仿真；被替代的仿真数据与专用脚本已按要求清理。
`hardware/` 是早期 shadow/HIL 基础设施的契约；实际输出入口在 `tools/g1_commissioning/`。
`evaluation/` 是本机完整生成数据；精选副本在上述 Git 归档中，视频和完整数据不随克隆提供。
