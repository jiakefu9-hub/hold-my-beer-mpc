# 12 秒行走、3 秒停车仿真

2026-11-12 已恢复原采集设计并重新完成三组对照，当前默认回放为 **`final_template11x4`**。新任务为 0–12 s 前进、12–15 s 停车；原始数据记录到 15.06 s，保存回放为 0–15 s、7501 帧。没有真机抬臂和接管动作，保留仿真原有的 24 ms 数值启动 PD。

模板由 **11 条建模轨迹**（1 条名义初态＋5 对正负下肢初始 q/dq 偏移）建立，另有 **4 条独立检查轨迹**（2 对）。种子和逐条偏移与原 8 s 设计完全核对，所有记录重新采集到 15.06 s，没有延用或拼接旧 8 s 数据；15 次采集均通过检查。4 条检查轨迹不参与模板平均。

模板按**相同绝对任务时间跨轨迹平均**，没有将不同稳定行走周期折叠成一个平均周期。因此不能直接循环旧 8 s 文件来延长任务。周期模板或起步—周期行走—停车拼接属于另一种实现，需要另行验证接缝与预测窗口；本次保留现有完整任务方法。

## 当前三组结果

评价窗口为完整 `[0,15)`，行走 `[0,12)` 与停车 `[12,15)` 的结果另存逐组 JSON。

| 方法 | 右端竖直倾角均值 | 右端方向摆动 RMS | QP 回退 | 执行救援 | 验收 |
| --- | ---: | ---: | ---: | ---: | --- |
| 位置 PID（D01） | 0.6331° | 0.7034° | 0 | 0 | PASS |
| 无未来预览 MPC（D02） | 0.5942° | 0.6931° | 0 | 0 | PASS |
| 模板预览 MPC（D03） | 0.1575° | 0.1773° | 0 | 0 | PASS |

三组均无跌倒、无非有限数据，完整记录和严格验收均通过；两种 MPC 没有 QP 失败/回退或执行层安全救援。两者的配置仅差 `research_hold_future_disturbance`。沿用用户已授权的共同执行验收上限 15 rad/s²，MPC 规划加速度上限仍为 8 rad/s²，其余权重、关节/力矩边界、周期和模型均保持该独立场景原设置。

在仓库根目录分别运行：

```bash
/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/experiments/thesis_walk15/replay.py D01
/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/experiments/thesis_walk15/replay.py D02
/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/experiments/thesis_walk15/replay.py D03
```

可显式指定 `--variant final_template11x4`，`--once` 单次播放，`--check` 检查散列、模型、有限轨迹和运动学。三组均已通过 `--check`；本批未重新打开图形窗口。回放器此前已修复 8 s 循环边界，按所选清单的 15 s 时长播放；空格暂停/继续，R 重播，G 切换方向辅助杆。

数据：[当前清单](variants/final_template11x4/manifest.json)、[详细说明](variants/final_template11x4/README.md)、[PID 指标](variants/final_template11x4/D01_metrics.json)、[无未来预览 MPC 指标](variants/final_template11x4/D02_metrics.json)、[模板预览 MPC 指标](variants/final_template11x4/D03_metrics.json)。精确配置和模板检查记录保存在该版本下的 `configs/` 与 `template/`；完整原始采集/运行目录由清单追溯。

## 与前期 2＋1 版本的关系

前期 `execution_margin15` 使用 2 条建模、1 条检查，三组也已录满，但 D03 有 2 次 QP 回退、严格验收为 FAIL。其全任务右端倾角/摆动为 0.1491°/0.1646°，本次 11＋4 版本为 0.1575°/0.1773°；不以增加数据条数推出姿态指标必然更优。新旧两套模板还在相同的 4 条新留出轨迹上进行离线误差比较，线/角加速度误差略降、角速度/姿态误差略升，详见[同批留出比较](variants/final_template11x4/template/same_heldouts_comparison.json)。本次各方法仍只运行一次，不作统计显著性结论。

旧 2＋1 轨迹、配置、数值及回退记录完整保留，见[当时结果](variants/execution_margin15/README.md)；使用 `replay.py D03 --variant execution_margin15` 可播放该批。原 8 s 展示及原 15 s 配置的 13.98 s 失败记录继续保留。

---

# 原配置历史记录

2026-11-11。用户授权延长仿真任务；未连接或控制机器人，未改变既有真机数据和原 8 秒展示。本轮仍使用旧三组控制器的参数与负载模型，重建了匹配 12 秒停车点的独立仿真模板。

以下是调整前原配置的实际结果，继续保留；当前默认回放使用上文的新配置。

| 组别 | 记录情况 | 验收与限制 |
| --- | --- | --- |
| D01，位置 PID | 0–15 s 完整，已保存回放 | 检查通过，未跌倒，无非有限数据 |
| D02，无未来预览 MPC | 在 13.98 s 退出，未完成 15 s | 执行层 `NO_SAFE_TORQUE`；原配置复核得到相同退出。未扩大验收边界来把它改写成成功 |
| D03，模板预览 MPC | 0–15 s 完整，已保存回放 | 1 次 QP 失败/回退、4 次执行层安全救援，验收状态 FAIL；轨迹完整且未跌倒，回放保留实际行为 |

0–12 s 发送前进命令，12–15 s 请求零平移速度。没有真机抬臂/接管过程；保留原仿真 24 ms 数值启动。原始数据录到 15.06 s，以覆盖最后的预测窗口；窗口回放只显示 15 s 任务。既有真机主对照是程序计时 5–15 s 前进，因此这不是逐秒相同的真机复现。

在仓库根目录打开窗口回放：

```bash
# 位置 PID
/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/experiments/thesis_walk15/replay.py D01 --variant original

# 模板预览 MPC，带上述回退记录
/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/experiments/thesis_walk15/replay.py D03 --variant original

# 无未来预览 MPC，明确标记为 13.98 秒退出的失败回放
/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/experiments/thesis_walk15/replay.py D02_FAILED --variant original
```

默认循环播放；空格暂停/继续，R 重播，G 切换辅助杆，关闭窗口退出。辅助杆显示实际记录的末端 +Z 方向，不放大角度。回放只读取保存的 qpos/qvel 并更新图像，不重新计算控制。

清单：[manifest.json](manifest.json)；逐组指标：[位置 PID](D01_metrics.json)、[模板预览 MPC](D03_metrics.json)。新指标按完整 `[0,15)`、行走 `[0,12)`、停车 `[12,15)` 分开计算；由于 D02 未完成，暂不能形成三组同窗口的定量比较，也不能直接替换论文中的旧 8 秒结果表。

失败原始依据位于 `evaluation/thesis_walk12_stop3_20261111/logs/D02.log` 与 `execution_failure_state.json`。退出前实际轨迹保存在 [D02_partial.npz](failed_replays/D02_partial.npz)，没有补写退出后的状态。退出时 MPC 期望加速度仍在规划范围内，异常发生在加速度到力矩的执行候选验收；不能仅凭此记录就断言机器人必然跌倒，或将它解释为 QP 本身求解失败。完整原始轨迹、配置及新模板保存在同一新场景目录。

已检查三份回放的散列、模型、有限数值与正向运动学，另验证了实际窗口播放和一帧离屏渲染。文件/图形检查不等于控制质量验收，D03 的回退和 D02 的提前退出仍按原始状态报告。

随后用户已明确允许独立仿真配置调整，执行情况见本文开头。原配置的失败、回退和两份完整轨迹继续保留。
