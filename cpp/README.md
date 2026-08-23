# C++ 右臂运行时与平台边界

这个目录保存 Python 共享控制核心之下的原生计算和平台适配层。扰动模板、
continuous-H、QP 组装、权重和 full-task task clock 不在这里复制。

| 目录 | 运行位置 | 职责 |
|---|---|---|
| `right_arm_rnea` | 仿真在线 | Pinocchio RNEA 名义前馈的 C ABI |
| `ddq_torque_mapper` | 仿真在线 | MuJoCo 局部 DDQ→力矩映射、候选验收与救援 |
| `right_arm_executor` | 仿真在线，也是真机语义基准 | 2 ms PD、限幅、超时与 NaN 保护 |
| `right_arm_sim_runtime` | MuJoCo worker | external-step IPC，组合 RNEA、mapper 和 executor，返回已认证 `final_tau` |
| `unitree_arm_adapter` | 硬件适配器边界 | state-only LowState/torso-IMU bridge、2 ms publisher-absent HIL 与 future-output supervisor |

`run.sh` 会增量构建正式仿真需要的共享库和
`right_arm_sim_runtime` worker。完整数值回归由各子目录的
`build_and_test.sh` 独立执行。真机适配器可显式构建只有 subscriber 的 state bridge；
publisher-absent HIL 不链接 Unitree SDK。真实命令 publisher target 已移除，
`UNITREE_ARM_ADAPTER_BUILD_DDS=ON` 会在 CMake 配置阶段 fail closed。

正式 simulation adapter 是
`main_sim.py -> RightArmSimProcess -> right_arm_sim_runtime`；hardware shadow
adapter 是 `unitree_arm_state_bridge -> read-only Python shadow`。二者共享 MPC、
运动学和 predictor 接口，但不共享平台 payload，也不维护两份 MPC。

仿真 `ddq_torque_mapper` 依赖 MuJoCo 当前接触求解器，不得将它误称为真机可
直接复用的接触验收。真机 floating-base RNEA 还需要经过验证的 base
pose/twist/contact estimator；在此之前，2 ms Unitree 适配器只负责状态/命令
协议、安全保护和通信。hardware shadow 仍只读、只兼容 legacy phase template，
不支持最终 full-task v2 + 24 ms handoff，全部硬件输出均为
hardware-unverified。
