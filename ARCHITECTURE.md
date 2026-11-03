# 控制链与代码地图

截至 2026-11-03，真机 MPC、位置 PID、前馈加速度 PID（C）已通过独立现场入口完成实验。
`tools/g1_commissioning/` 是这些实验的实际实现；早期共享核心的 production adapter
仍未启用真实输出。两条路径不能混为一谈。

## 仿真

```mermaid
flowchart LR
  RL["Torch 下肢策略 · 20 ms"] -->|关节目标| Sim["MuJoCo · 2 ms"]
  Sim -->|当前状态| Pred["固定任务扰动模板"]
  Sim -->|q / dq| MPC["右臂加速度 MPC · 6 ms"]
  Pred --> MPC
  MPC -->|ddq| Worker["C++ RNEA + MuJoCo 候选验收"]
  Worker -->|力矩| Sim
  PD["左臂固定关节 PD"] --> Sim
```

`run.sh` 启动 `main_sim.py`，自动构建四个原生组件。
Python 主进程拥有唯一 MuJoCo 世界，等待 C++ worker 返回后才推进物理时间。
正式冻结路径固定 CPU 7、数值库单线程；展示研究入口明确记录独立的研究运行条件。

- MPC：`arm_mpc.py`；配置装配：`right_arm_control_setup.py`。
- 预测：`disturbance_predictor.py`、`disturbance_template/`。
- 模型：`kinematics_helper.py`、`robot_model_backend/`。
- 进程：`right_arm_runtime/sim_process.py`、`cpp/right_arm_sim_runtime/`。
- 执行：`cpp/right_arm_rnea/`、`cpp/ddq_torque_mapper/`、`cpp/right_arm_executor/`。
- 展示入口：`tools/experiments/simulation_showcase/`。复用比较研究适配器；详见[工具地图](tools/README.md)。

当前固定任务为 0–6.4 s 请求前进，6.4–8 s 停车；预测共 54 ms。
历史模板已知任务停车时刻，不能据此宣称支持任意未知任务。
三组回放使用保存的 qpos/qvel，仅做正运动学渲染。

## 已完成的真机控制

```mermaid
flowchart LR
  Loco["速度请求"] --> PC1["PC1 内置行走控制"]
  PC1 --> Robot["G1 腿部与身体运动"]
  Robot -->|关节状态 / 身体 IMU| Host["接收与时间戳"]
  Host --> Pred["H0 学习预测 / 当前扰动保持"]
  Host --> Control["右臂 MPC 或 PID"]
  Pred --> Control
  Control --> Exec["现场执行层"]
  Exec -->|rt/arm_sdk| Arms["左右手臂"]
```

真机入口为 `g1_walk_mpc_learned.py`、`g1_walk_mpc.py`、`g1_walk_pid.py`。
`g1_run_morning_20261013.py` 保存最终 M 系列的参数组合；T 系列保存 C 的对照。

| 模块 | 作用 |
| --- | --- |
| `hardware_mpc_learned.py` / `hardware_mpc_predictor.py` | 学习前馈、当前状态与未来区间 |
| `hardware_mpc_control.py` / `hardware_mpc_solver.py` | 实测 q/dq 的 6 ms、9 区间 QP |
| `hardware_arm_inverse_dynamics.py` / `hardware_mpc_torque_control.py` | 耦合右臂 5×5 动力学与力矩执行 |
| `hardware_pid_control.py` | 位置 PID 与关节参考 |
| `hardware_acceleration_pid.py` | C：任务空间 PID 与加速度分配 |
| `g1_walk_mpc.py` / `mpc_compute_process.py` | 生命周期、独立计算进程、命令与日志 |
| `analyze_hardware_pid.py` | H0 下末端重建和评价 |

上述文件除特别说明均在 `tools/g1_commissioning/`。左臂固定关节 PD；腰只有一个真实自由度。
力矩模型包含实测移动基座姿态与运动输入，没有显式求解双腿、脚地接触或 PC1 内部状态。
现场默认完整流程为约 3 s 抬臂、2 s 基线、10 s 行走、3 s 停车及至少 3 s 退权；M10 等独立实验另有定义。

## 保留的基础设施

`run_hardware_shadow.py`、`cpp/unitree_arm_adapter/` 和 `tools/realtime/`
用于只读采集及无 publisher 的 HIL。它们保留协议、超时、状态配对和输出边界验证，
不属于最终真机实验的实际输出链。其存在不能替代现场验证，也不是待完成的论文主任务。

详细依据：[MPC 数学](docs/design/MPC_DESIGN.md)、[仿真模板](docs/simulation/FULL_TASK_TEMPLATE.md)、
[真机学习 MPC](docs/g1_field_validation/HARDWARE_MPC_LEARNED.md)、
[最终实验](docs/FINAL_EXPERIMENTS.md)、[shadow/HIL](docs/hardware/HARDWARE_OFFLINE_PREPARATION.md)。
