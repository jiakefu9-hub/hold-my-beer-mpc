# 前馈 PID 与位置 MPC 仿真对照

仅仿真，无 SDK、DDS 或真机启动路径。报告与最终配置：

- `docs/simulation/experiments/20261011_simple_controllers/REPORT.md`
- `configs/simulation_simple_controllers_20261011/`

2026-10-12 的 C 参数细化使用独立 `refine_c.py`、`c_round*.json` 和
`evaluation/acceleration_pid_refinement_20261012/`，不覆盖上述首轮对照。
后续报告见 `docs/simulation/experiments/20261012_c_refinement/REPORT.md`。
可选 `simple_apid_bounded_allocation` 将旧 C 的末端逐关节裁剪改为五变量
受约束最小二乘，`simple_apid_predictive_brake` 复用既有提前制动；两项默认关闭。
这些仍是仿真配置，真机的独立 C 配置与力矩约束不可用它们直接替换。

`controllers.py` 实现三个新控制器：

1. `ff_pid`：归一化的二维重力误差 PID，预测身体转动前馈；最终版本同时抑制绕瓶轴自转，输出位置/速度参考给 PD。
2. `position_mpc`：状态只有五维 q，输入是五维位置参考 r，模型 `q_next=(1-beta)q+beta*r+bias`。优化未来 9 段，只执行首段。bias 是根据上一拍预测与本拍测量更新的已知外部模型修正，不是 QP 状态。无逆动力学。
3. `acceleration_pid`：任务空间 PID 加线/角加速度扰动前馈，一次 5×5 阻尼最小二乘求关节加速度；使用原加速度 MPC 的同一力矩执行层。没有滚动优化，是额外的执行层消融对照。

`original_pid` 保留原 `arm_pid.py` 的增益与公式，并统一参考位置边界；不是旧真机 6 ms PID 数据。

## 复现

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
bash cpp/build_runtime.sh
/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/experiments/simple_controller_comparison/test_controllers.py
/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/experiments/simple_controller_comparison/study.py \
  tools/experiments/simple_controller_comparison/reproduce.json --cpus 2 --keep
```

`study.py` 缓存同名结果；需要再次运行时更换 JSON 内的 name，不覆盖旧实验。每组保存完整解析后的配置、来源哈希、结果、日志；调参批次只清理自身的可复现冗余数据，保留 compact_metrics.npz 的实际 qpos/qvel/瓶轴/速度/加速度。正式入选与复核保留完整轨迹。

临时目录清空后先重新执行 `bash cpp/build_runtime.sh`。调参入口会在整个批次之前检查
四个 native 产物，缺失时不启动仿真，也不将环境初始化失败当成控制器失败。

查看其中一个已运行结果（命令由本机结果索引解析，打开 MuJoCo 回放窗口）：

```bash
RUN_PATH=$(/home/fjk/miniforge3/envs/g1_mpc/bin/python -c 'import json; print(json.load(open("evaluation/simple_controller_comparison_20261011/results/final_position_mpc.json"))["path"])')
/home/fjk/miniforge3/envs/g1_mpc/bin/python replay_trajectory.py "$RUN_PATH/trajectory.npz"
```

换成 `final_ff_pid` / `final_acceleration_pid` / `final_mpc_conservative` 可比较其他方案。

## 接入边界

完整走停测试协议原来只接受 `arm_controller=mpc`，隔离入口复用这个协议标签，但真实算法由配置的 `simple_controller` 和 metadata.actual_control_family 标明。位置控制时 acceleration_controller=False，M*a+b 与 MuJoCo 候选力矩检查完全不执行；最终物理仿真、接触和执行器 ±25 Nm 限制继续生效。

为了不修改普通/真机启动器，研究入口对 `main_sim.py` 的少数接口片段做精确断言替换：位置控制没有逆动力学进程，因此不要求该进程的预热、交接力矩或 worker affinity；仍保持 24 ms 启动 PD、6 ms 更新、2 ms 物理步、腿部策略和任务时钟。延迟记录中没有 mapper 更新时明确记 false。源代码接口变化会立即失败，不静默套用。

位置策略的第三输出为兼容协议的 unused 零向量，绝非实际加速度。通用日志中的 “MPC DDQ tracking” 对这些策略没有物理意义；不要用它做结论。实际物理指标与执行器力矩从 MuJoCo 轨迹统一计算。

普通共享代码仅修复 `sim_support.py` 的无逆动力学采样导出：0 次 mapping 调用不再对空数组求 max；6 个新测试和 6 个工厂回归测试覆盖本次改动。
