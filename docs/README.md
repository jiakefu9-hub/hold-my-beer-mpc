# 项目文档导航

从这里找文档；运行命令仍从仓库根目录执行，代码、配置和实验原始数据没有迁移。
第一次阅读项目看 [仓库说明](../README.md) 和 [架构](../ARCHITECTURE.md)；
准备上真机，直接看 [G1 现场实验入口](g1_field_validation/README.md)。

## 文档按用途分区

| 目录 | 放什么 | 主要入口 |
| --- | --- | --- |
| `design/` | 控制算法的设计和数学约定 | [MPC](design/MPC_DESIGN.md)、[PID](design/PID_DESIGN.md)、[LQR](design/LQR_DESIGN.md) |
| `simulation/` | 当前冻结仿真任务、扰动模板、航向和计时 | [固定任务模板](simulation/FULL_TASK_TEMPLATE.md)、[冻结边界](simulation/PRE_HARDWARE_FREEZE.md)、[航向控制](simulation/HEADING_CONTROL.md)、[实时计时](simulation/REALTIME_RUNTIME.md)、[延迟实验](simulation/EXPERIMENTAL_MPC_LATENCY_PLAN.md) |
| `hardware/` | 生产链硬件接口、只读 shadow、离线 HIL 和阶段门 | [集成计划](hardware/HARDWARE_INTEGRATION_PLAN.md)、[Shadow](hardware/HARDWARE_SHADOW.md)、[离线准备](hardware/HARDWARE_OFFLINE_PREPARATION.md) |
| `g1_field_validation/` | 真机怎么操作、已经验证什么、每次实验结果 | [现场入口](g1_field_validation/README.md)、[已知结论](g1_field_validation/KNOWN_RESULTS.md)、[实验记录索引](g1_field_validation/sessions/README.md) |
| `history/` | 旧路线图、问题复盘和开发过程，不作为当前运行指令 | [路线图](history/PLAN.md)、[工程案例](history/CHALLENGE.md)、[MPC 开发日志](history/MPC_DEVELOPMENT_LOG.md)、[早期仿真／真机差异笔记](history/sim_real_diff.md) |

原先根目录的同名设计／硬件文档已移动到上述目录，H1 两份指南移到
`g1_field_validation/h1/`；H1/HIL 结果统一进入 `sessions/`。
原先很长的现场 `README.md` 改为 [RUNBOOK.md](g1_field_validation/RUNBOOK.md)，
现场 README 现在只承担导航。没有留下两份会各自过期的正文副本。

## 代码、配置、数据为什么不一起搬

```text
仓库根目录
├── README.md / ARCHITECTURE.md       项目入口、运行时地图
├── docs/                            设计、手册、结论与实验摘要
├── run.sh / main_sim.py / arm_*.py   现有仿真入口与控制算法（保持路径）
├── configs/                         仿真与 hardware shadow 的 YAML
├── tools/
│   ├── g1_commissioning/             独立现场程序、profile 模板、测试
│   └── realtime/                     H1/shadow launcher 与离线审计
├── right_arm_runtime/ / cpp/         共享运行时、模型执行与硬件边界
├── disturbance_template/            仿真扰动模板代码及选定资产
├── evaluation/                      本地原始实验数据，不提交 Git
├── evaluation_summary/              可提交的轻量证据，现有仿真路径不变
└── local_reference/unitree_g1/       本地官方资料，不提交 Git
```

移动 Python/C++ 入口、模型、配置或 SDK 会影响导入、构建和现场命令，本次不做这类重构。
`/home/fjk/g1_ws/unitree_sdk2` 继续作为仓库外依赖。
既有 `evaluation/hardware_shadow/...` 也保留；名字带 shadow 不表示其中每次实验都只读，
具体能力以 session 记录和程序为准。记录里的原始路径和 SHA-256 不改。

## 后续 PID / MPC 真机实验怎样归档

- 操作流程放 `docs/g1_field_validation/`，算法设计留在 `docs/design/`。
- 每次现场实验在 `sessions/` 新建 `YYYYMMDD_<主题>_<序号>.md`，例如
  `YYYYMMDD_PID_STATIC_01.md`；不要把当天日志不断追加到总 README。
- 使用 [实验记录模板](g1_field_validation/SESSION_TEMPLATE.md)，分清计划、实际命令、
  现场观察、数据结果、未确认项和原始证据位置。
- 只有取得证据后才更新 [已知结论](g1_field_validation/KNOWN_RESULTS.md)；
  “程序已写好”“离线通过”“真机执行过”“物理效果通过”分开写。
- 可复用的现场源码和测试继续放 `tools/g1_commissioning/`；本次使用的实机 profile
  和原始日志一起保存在独立 `evaluation/.../<session>/`，仓库内只保留可审阅模板。
- 以后若正式接入 PID/MPC 生产输出链，再按硬件集成计划单独设计；不因归档位置相邻
  就把 commissioning 当作已经开放的 MPC 输出接口。暂不创建空的 PID/MPC 目录。
