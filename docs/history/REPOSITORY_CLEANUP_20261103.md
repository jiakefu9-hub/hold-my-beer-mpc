# 2026-11-03 仓库交接整理

整理前恢复点：`4a51a52d0958d3560928996622f64d4aaff7b84e`。控制算法、真机参数、物理模型、三组展示轨迹均未修改。
先验证并提交最新展示配置、回放和报告，再清理过期叙述。已完成的实验不是待办。

## 删除及恢复

以下文档从当前目录删除，内容仍在整理前 Git 提交中。当前文档中的历史引用改为固定提交链接。
主要是已被后续结果覆盖的准备计划、旧参考轨迹 MPC 说明，以及已撤回五条数据的报告。
失败结果、C 前馈 PID、辨识局限、制动消融和关键原始证据继续保留。

| 删除的旧文档 | 恢复／浏览 |
| --- | --- |
| `docs/history/PLAN.md` | [整理前内容](https://github.com/jiakefu9-hub/hold-my-beer-mpc/blob/4a51a52d0958d3560928996622f64d4aaff7b84e/docs/history/PLAN.md) |
| `docs/hardware/HARDWARE_INTEGRATION_PLAN.md` | [整理前内容](https://github.com/jiakefu9-hub/hold-my-beer-mpc/blob/4a51a52d0958d3560928996622f64d4aaff7b84e/docs/hardware/HARDWARE_INTEGRATION_PLAN.md) |
| `docs/g1_field_validation/NEXT_FIELD_EXPERIMENTS.md` | [整理前内容](https://github.com/jiakefu9-hub/hold-my-beer-mpc/blob/4a51a52d0958d3560928996622f64d4aaff7b84e/docs/g1_field_validation/NEXT_FIELD_EXPERIMENTS.md) |
| `docs/g1_field_validation/HARDWARE_MPC_REFERENCE_SERVO.md` | [整理前内容](https://github.com/jiakefu9-hub/hold-my-beer-mpc/blob/4a51a52d0958d3560928996622f64d4aaff7b84e/docs/g1_field_validation/HARDWARE_MPC_REFERENCE_SERVO.md) |
| `docs/g1_field_validation/WALK_DATASET_AUDIT.md` | [整理前内容](https://github.com/jiakefu9-hub/hold-my-beer-mpc/blob/4a51a52d0958d3560928996622f64d4aaff7b84e/docs/g1_field_validation/WALK_DATASET_AUDIT.md) |
| `docs/g1_field_validation/WALK_PREDICTOR_METHOD_STUDY.md` | [整理前内容](https://github.com/jiakefu9-hub/hold-my-beer-mpc/blob/4a51a52d0958d3560928996622f64d4aaff7b84e/docs/g1_field_validation/WALK_PREDICTOR_METHOD_STUDY.md) |
| `docs/g1_field_validation/WALK_SIGNAL_GALLERY.md` | [整理前内容](https://github.com/jiakefu9-hub/hold-my-beer-mpc/blob/4a51a52d0958d3560928996622f64d4aaff7b84e/docs/g1_field_validation/WALK_SIGNAL_GALLERY.md) |
| `docs/g1_field_validation/sessions/20260915_A2_PREPARATION.md` | [整理前内容](https://github.com/jiakefu9-hub/hold-my-beer-mpc/blob/4a51a52d0958d3560928996622f64d4aaff7b84e/docs/g1_field_validation/sessions/20260915_A2_PREPARATION.md) |
| `docs/g1_field_validation/sessions/20260915_A3_OFFLINE_PREPARATION.md` | [整理前内容](https://github.com/jiakefu9-hub/hold-my-beer-mpc/blob/4a51a52d0958d3560928996622f64d4aaff7b84e/docs/g1_field_validation/sessions/20260915_A3_OFFLINE_PREPARATION.md) |
| `docs/g1_field_validation/sessions/20260927_HARDWARE_MPC_PREPARATION.md` | [整理前内容](https://github.com/jiakefu9-hub/hold-my-beer-mpc/blob/4a51a52d0958d3560928996622f64d4aaff7b84e/docs/g1_field_validation/sessions/20260927_HARDWARE_MPC_PREPARATION.md) |
| `docs/g1_field_validation/sessions/20261005_MPC_FIELD_PREP.md` | [整理前内容](https://github.com/jiakefu9-hub/hold-my-beer-mpc/blob/4a51a52d0958d3560928996622f64d4aaff7b84e/docs/g1_field_validation/sessions/20261005_MPC_FIELD_PREP.md) |
| `docs/g1_field_validation/sessions/20261009_ZERO_ARM_NEUTRAL_PREPARATION.md` | [整理前内容](https://github.com/jiakefu9-hub/hold-my-beer-mpc/blob/4a51a52d0958d3560928996622f64d4aaff7b84e/docs/g1_field_validation/sessions/20261009_ZERO_ARM_NEUTRAL_PREPARATION.md) |
| `docs/g1_field_validation/sessions/20261011_NUMBERED_MPC_VALIDATION.md` | [整理前内容](https://github.com/jiakefu9-hub/hold-my-beer-mpc/blob/4a51a52d0958d3560928996622f64d4aaff7b84e/docs/g1_field_validation/sessions/20261011_NUMBERED_MPC_VALIDATION.md) |
| `docs/g1_field_validation/sessions/20261012_FIELD_FOLLOWUP.md` | [整理前内容](https://github.com/jiakefu9-hub/hold-my-beer-mpc/blob/4a51a52d0958d3560928996622f64d4aaff7b84e/docs/g1_field_validation/sessions/20261012_FIELD_FOLLOWUP.md) |
| `docs/g1_field_validation/sessions/20261013_MORNING_PLAN.md` | [整理前内容](https://github.com/jiakefu9-hub/hold-my-beer-mpc/blob/4a51a52d0958d3560928996622f64d4aaff7b84e/docs/g1_field_validation/sessions/20261013_MORNING_PLAN.md) |

本机 `cad/` 的三个临时切片 `dange.gcode.3mf`、`dangeh2d.gcode.3mf`、`liangge.gcode.3mf`
已在逐文件 SHA-256 校验的本地备份中保存，然后移出工作目录；可编辑 3MF 与通用 STL 保留。
备份和删除哈希清单位于 `evaluation/repository_cleanup_20261103/`，不随 Git 克隆。

## 保留范围

- `evaluation/` 的既有成功／失败原始实验数据没有删除。
- 真机精选归档和预测模型按 manifest 校验，归档内容没有为清理而重写。
- `tools/experiments/c_mpc_followup/`、`simple_controller_comparison/` 和 `mpc_comprehensive/`
  同时是展示入口的代码依赖，不能按目录日期删除。
- 历史 LQR、shadow/HIL 和执行层保留其实现与必要契约；它们不是最终真机入口。
- 本地原始数据与录像并未因这次提交自动获得远端备份。

## 交接验收

复现说明见 [快速开始](../QUICKSTART.md)，最终结果及录像映射见
[最终实验清单](../FINAL_EXPERIMENTS.md)。机器可读的完整记录见
[VALIDATION.json](REPOSITORY_CLEANUP_20261103_VALIDATION.json)。

- 保存的 8 组回放、配置和模型哈希均通过；D02/D03 配置仅差扰动预览开关。
- 冻结 H0 预测库哈希通过；真机归档 408 个文件逐一核验通过（20 轮完整原始日志、29 条运行记录）。
- 在独立源码目录从归档 NPZ 重新计算 10 轮、2,448 项分阶段指标，全部通过。
- 重新构建四个仿真 C++ 组件、真机运动学库，离线 preflight 通过。
- 比较控制器 7 项、C 路径 5 项、M 系列现场入口离线契约 16 项测试通过，共 28 项。
- 当前非归档 Markdown 相对文件链接检查通过；历史证据原件不为修链而重写。
- 只修改启动环境解析、构建入口、研究运行后缀和校验工具；控制器、模型、参数不改。

### 数值复现限制

共运行 7 次纯仿真：三组新构建验收、两组 MPC 复测、两组整理前运行器＋原编译库对照。
所有任务完整结束，无摔倒／NaN，但新 D02/D03 的严格 Full Task 状态为 FAIL：
分别有 6/1 次 QP 失败回退，执行候选回退为 2/4 次。D01 为 PASS。
整理前与整理后在同一现有环境中的全阶段指标、门槛和关节范围逐值一致，复测也一致；
因此本次清理未引入这一数值差异，但不能宣称新克隆下的严格仿真门已全部通过。
原已保存的三组轨迹和配置全部保留，原记录的 PASS 没有改写。

| 右端，全程 0–8 s | 保存轨迹 D02 | 新复算 D02 | 保存轨迹 D03 | 新复算 D03 |
| --- | ---: | ---: | ---: | ---: |
| 竖直倾角均值 ° | 0.5883 | 0.5903 | 0.3111 | 0.3544 |
| 平均轴摆动 RMS ° | 0.6886 | 0.6911 | 0.1866 | 0.2068 |
| 线加速度 RMS m/s² | 3.3410 | 3.3441 | 2.9986 | 3.0082 |
| 角加速度 RMS rad/s² | 9.8751 | 9.9026 | 7.1663 | 7.2025 |
| 速度 RMS m/s | 0.4580 | 0.4578 | 0.4567 | 0.4561 |
| 角速度 RMS rad/s | 0.1518 | 0.1521 | 0.0653 | 0.0660 |

旧保存运行与当前重新计算差异的根因尚未定位；不能直接归因于操作系统调度或滤波。
本次未通过提高容差、改权重或替换轨迹来让严格检查过关。
回放可按哈希完全重现保存画面，重算动力学是另一层验收。

### 验证环境与保存范围

测试复用现有 Linux/Python 3.10 环境；依赖版本配方已记录，但没有验证联网全新安装。
因磁盘余量不足，独立检出共享 Git 对象并省略压缩原始日志；源码、模型和 NPZ 为独立文件，
原始日志哈希在主工作目录验证。仿真输出临时放在内存文件系统；验收摘要、配置和日志保存在
本机 `evaluation/repository_cleanup_20261103/`，机器可读摘要已随 Git 提交。
临时检出与新验收产生的大文件在记录保存后清除，既有实验数据不删除。
没有接入机器人，没有真实输出，也没有重新录制或更改精选展示。
