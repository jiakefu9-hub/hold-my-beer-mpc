# 12 秒行走、3 秒停车仿真

当前最终版本为 **`final_template11x4`**，恢复与原 8 s 相同的 11 条建模＋4 条独立检查采集设计，重新采集完整 15 s 任务。三组均已重跑、录满并通过验收，无 QP 回退或执行救援。数据、指标、历史版本和三条回放命令见[结果说明](../../../docs/simulation/experiments/20261111_walk12_stop3/README.md)。本目录仅用于 MuJoCo 仿真及保存状态回放，不连接机器人。

任务为 0–12 s 行走、12–15 s 请求零速，记录至 15.06 s，为最后 54 ms 预测窗口提供尾部。无真机抬臂/接管动作，保留原仿真 24 ms 数值启动 PD。回放按清单的 15 s 时长播放，`--once` 单次播放，`--check` 检查文件和运动学。

`full_template.py` 重用共享固定 PD 采集器、完整任务构建与留出检查算法，逐条核对旧 11＋4 的种子和正负下肢 q/dq 偏移。均值按相同任务时间跨轨迹计算，不循环稳定步态，不拼接旧 8 s 模板。`run.py` 默认选择这一完整设计，要求用独立 variant 保存，不覆盖旧数据；复用已采集数据时核对配置、源代码及原始文件散列。清单转换为运行接口要求的协议字段。

```bash
/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/experiments/thesis_walk15/run.py --variant final_template11x4 --execution-max-qacc 15
```

已完成版本拒绝覆盖；需要独立重算时更换 variant 名称。原生后端需已构建（`bash cpp/build_runtime.sh`）。两种 MPC 仅差 `research_hold_future_disturbance`；沿用已授权的执行验收上限 15 rad/s²，规划上限仍为 8 rad/s²。没有更改共享控制器、代价权重、关节/力矩限制或真机程序。`simulation_entry.py` 在隔离进程中接入任务时钟、12 s 停车与名义 6±0.4 m 路程检查，保留其他检查。

初次延长任务时采用的 2 条建模＋1 条检查数据仍保留：原配置有 13.98 s 退出，`execution_margin15` 完整但 D03 有 2 次 QP 回退。只有显式传 `--template-design legacy2plus1` 才复现该缩减设计；历史轨迹使用 `replay.py D03 --variant execution_margin15` 或 `--variant original` 查看，不作为当前最终 11＋4 结果。独立滤波探索同样基于前期 2＋1 模板，不自动成为当前三方法主对照。
