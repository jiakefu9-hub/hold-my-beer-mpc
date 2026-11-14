# 最终实验与展示入口

实验已完成。本页负责把控制器、配置、数据与展示编号对应起来；详细指标由链接中的结果文件维护。
它不是新一轮上机计划。安装和命令见 [快速开始](QUICKSTART.md)。

2026-11-12 新增完整 15 s 仿真：0–12 s 行走、12–15 s 停车，模板由 11 条建模轨迹建立，另用 4 条独立轨迹检查。位置 PID、无未来预览 MPC 与模板预览 MPC 三组均录满 15 s，验收通过，无 QP 回退或执行救援。当前默认回放为 `final_template11x4`，规划加速度上限为 8 rad/s²、执行验收上限为 15 rad/s²。配置、模板、指标和回放见[15 s 场景结果](simulation/experiments/20261111_walk12_stop3/README.md)；下方旧 8 s 配置与记录继续作为历史证据。

## 三组仿真

| 展示编号 | 右臂控制器 | 配置 |
| --- | --- | --- |
| D01 | 位置 PID | [D01_position_pid_balanced.yaml](../configs/simulation_showcase_20261103/D01_position_pid_balanced.yaml) |
| D02 | No-Preview MPC：当前测得扰动在预测窗内保持 | [D02_mpc_current_hold.yaml](../configs/simulation_showcase_20261103/D02_mpc_current_hold.yaml) |
| D03 | Disturbance-Preview MPC：冻结仿真扰动模板 | [D03_mpc_predictive.yaml](../configs/simulation_showcase_20261103/D03_mpc_predictive.yaml) |

共同：左臂名义 PD Kp=20、Kd=1，75 mm／250 g 圆环负载，2 ms 物理步长、6 ms 手臂控制。
MPC 为 9 个预测区间；不启用额外姿态预测校正或区间前视。D02 与 D03 的配置只差
`research_hold_future_disturbance`。D02 仍反馈当前状态和当前扰动，不能称为开环或全部前馈关闭。
实际使用的场景由 [cases.json](../tools/experiments/simulation_showcase/cases.json) 覆盖为
`scene_payload75.xml`，复现时应使用研究入口，不能只拿单个 YAML 推断场景。

- [展示说明与全程指标](simulation/experiments/20261103_showcase/README.md)
- [逐项指标 CSV](simulation/experiments/20261103_showcase/metrics.csv)
- [配置、模型、轨迹哈希清单](simulation/experiments/20261103_showcase/manifest.json)
- [回放程序](../tools/experiments/simulation_showcase/replay.py)：分别运行 `D01`、`D02`、`D03`。

`D01_OLD`、`D01_SOFT`、`*_R50` 和 `stiff_left_baseline/` 是保留的历史对照，不属于最终三组。
正面放大和斜前方视角可以来自同一轨迹。说明线与放大镜头只改变显示，不放大真实角度。

## 真机重复实验和录像记录

主比较是 10 月 13 日上午 M01/M06 学习 MPC、M02/M05 当前扰动保持 MPC、M03/M04 位置 PID，
每类两轮；[六轮结果](g1_field_validation/sessions/20261013_MORNING_RESULTS_AND_VIDEO_PLAN.md)
同时报告完整窗口与初始收敛影响。精选原始日志、摘要、NPZ 在
[最终归档](g1_field_validation/evidence/20261014_final_archive/README.md)。

下表是下午重录批次的数据标识，完整运行目录与有效性见归档 [RUNS.md](g1_field_validation/evidence/20261014_final_archive/RUNS.md)。
这些标识对应实验日志；录像媒体没有纳入 Git，也没有在这里校验视频文件与日志的逐帧同步。

| 录像编号 | 控制内容 | 复用实验入口 | 最终批次日志时间及后缀 |
| --- | --- | --- | --- |
| V01 | 零中立位位置 PID，6 ms | M03 | `20261013_164407_686365_5c1c` |
| V02 | 当前扰动保持 MPC，6 ms | M02 | `20261013_164656_767428_c0d8` |
| V03 | 学习预测 MPC，6 ms | M01 | `20261013_164838_063152_fd65` |
| V04 | 参考速度／加速度放宽的位置 PID | M07 | `20261013_165125_196608_32b3` |

这些最终对照使用双臂零中立位、75 mm／250 g 模型、正常左臂 PD 20/1，时间对齐关闭。
V03 的 MPC 是 `hardware_mpc_learned_acc001_alpha0005_omega1.yaml` 与
`hardware_mpc_torque_learned.yaml`；历史 `164546` 非零位基线单独保留。
使用 [M 系列入口](../tools/g1_commissioning/g1_run_morning_20261013.py) 查阅参数；本页不提供自动上机流程。

[MPC 与 C 前馈 PID 真机比较](g1_field_validation/sessions/20261012_FINAL_COMPARISON.md)
保留为强对照补充；C 在部分动态指标上的优势与已观察到的接触都没有因整理而删除。
中止、线缆干扰和 DDS 初始化失败的记录保留原有效性备注，不混入成功轮次。

## 使用结果时的边界

- 仿真采用仓库 RL 行走策略，真机采用 PC1 内置策略。即使左臂名义增益相同，身体激励和执行系统也不同。
- 真机学习预测器使用 H0 数据，仿真采用另一套扰动模板；二者的收益分别报告。
- 竖直倾角评价绝对端正程度；相对平均瓶轴的摆动 RMS 评价动态摆幅。不要用标量倾角去均值替代后者。
- 完整、行走、起停阶段和滤波口径必须注明；不能只选优势阶段。线加速度与冲击改善有限也是结论。
- Git 包含 20 轮完整压缩原始日志、29 条运行记录和精选仿真轨迹；本机其他 `evaluation/` 数据及视频仍需独立备份。
