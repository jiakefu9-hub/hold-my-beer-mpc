# 10 秒行走、3 秒停车：当前仿真对照

2026-11-17 按用户要求将仿真与真机评价时长统一。两者均以开始请求前进为 0 s，10 s 请求停车，13 s 结束评价。旧 12＋3 s 及其他被替代的仿真数据已按后续要求清理。

## 模板与对照条件

重新采集 11 条建模轨迹（名义初态＋五对正负下肢 q/dq 偏移），另外 4 条检查轨迹不参与模板均值。15 条采集全部通过原检查。模板按相同任务时间跨轨迹平均，没有裁剪、循环或拼接旧模板。物理记录至 13.06 s，为最后 54 ms 预测提供尾部；严格 pre-step 最后一行是 13.058 s，回放保存 0–13 s 共 6501 帧。

三组为位置 PID、当前角速度姿态外推的无未来预览 MPC、模板预览 MPC。无预览仍保持当前 a/omega/alpha，不读取模板未来值；模板预览不加当前实测偏差修正。两种 MPC 的配置仅 `research_current_omega_pose` 不同，其他权重、约束、原有执行设置均保持；规划/执行加速度上限为 8/15 rad/s²。仿真不含真机抬臂和接管过程，保留原有 24 ms 数值启动 PD。

## 完整 [0,13) 指标

| 右臂方法 | 倾角均值 ° | 摆动 RMS ° | 线加速度 RMS m/s² | 角加速度 RMS rad/s² | 角速度 RMS rad/s | QP 回退 | 严格验收 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 位置 PID | 0.630263 | 0.700813 | 3.115628 | 12.948407 | 0.248965 | 0 | PASS |
| 无未来预览 MPC（当前角速度姿态外推） | 0.228712 | 0.257525 | 3.276910 | 9.527904 | 0.114509 | 6 | FAIL |
| 模板预览 MPC | 0.154451 | 0.174665 | 2.834524 | 6.709359 | 0.058214 | 2 | FAIL |

三组记录完整、无跌倒、无执行救援。D02/D03 因 6/2 次 QP 回退，严格零回退验收为 FAIL，不能将完整记录写成全部验收通过；没有为了通过而修改容差、放宽约束或重选更有利运行。原始分段统计仅留 JSON，正文只展示完整窗口。

## 回放

在项目根目录运行；仅播放保存的状态，不重新计算控制或连接机器人。

```bash
/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/experiments/thesis_walk13/replay.py D01
/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/experiments/thesis_walk13/replay.py D02
/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/experiments/thesis_walk13/replay.py D03
```

可加 `--once` 单次播放、`--check` 只检查文件和运动学。时长由本清单的 `duration_s=13` 驱动。

## 证据与复现

- [运行清单](manifest.json)、[逐组指标](D01_metrics.json)、[时序与配置核对](verification.json)。核对实际 planned/runtime 前进请求在 9.998 s 为 0.5 m/s、10.000 s 为零；保留航向控制的角速度请求。
- [模板来源清单](template/full_task_template_manifest.json)、[模板资产](template/full_task_template.npz)、[留出检查](template/heldout_metrics.json)、[配置](configs/D01.yaml)。副本与运行时使用的原文件内容一致；清单保留实际运行路径和 SHA-256。
- [采集与运行入口](../../../../tools/experiments/thesis_walk13/run.py)：复用既有采集器、模板数学和仿真控制链，使用独立协议与保存目录。已完成清单拒绝覆盖。

真机 V01_PID/M05/V03_LEARNED 未重跑。论文图 4.3 与图 4.5 只将横轴减去 5 s，仍使用原始任务时间 [5,18) 的完整数据，指标不变。仅修改显示时钟，不改变预测器锚点、控制器内部任务时钟或任何真机记录。

<a id="cleanup"></a>

## 当前保存与清理范围

2026-11-17 按要求仅保留本次三方法仿真及其 11＋4 模板采集。旧 8 s、12＋3 s、姿态外推对照、早期参数探索和相关专用试验脚本已删除；模型、共享控制器、预训练行走策略及全部真机资料保留。当前复现所需支持模块已集中到 `tools/experiments/thesis_walk13/`，不依赖旧试验目录。

[运行时依赖记录](runtime_dependencies.json)区分本次实际运行时的原入口快照与整理后的等价入口；数值函数与类保持一致，未重新运行或改写当前实验结果。
