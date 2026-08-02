# C++ 右臂运行时

这个目录是 Python MPC 之下的原生计算和安全层，不包含扰动模板、QP 组装、调参或画图。

| 目录 | 运行位置 | 职责 |
|---|---|---|
| `right_arm_rnea` | 仿真在线 | Pinocchio RNEA 名义前馈的 C ABI |
| `ddq_torque_mapper` | 仿真在线 | MuJoCo 局部 DDQ→力矩映射、候选验收与救援 |
| `right_arm_executor` | 仿真在线，也是真机语义基准 | 2 ms PD、限幅、超时与 NaN 保护 |
| `unitree_arm_adapter` | 真机独立进程 | LowState/arm SDK、2 ms 周期、共享内存与最终安全闸 |

`run.sh` 会增量构建仿真使用的前三个共享库。全部数值回归和微基准由各子目录的 `build_and_test.sh` 独立执行。真机适配器默认只构建 dry-run；启用 Unitree SDK2 编译需显式设置 `UNITREE_ARM_BUILD_DDS=ON`，编译成功也不会自动发送命令。

仿真中 `ddq_torque_mapper` 依赖 MuJoCo 当前接触求解器，不得将它误称为真机可直接复用的接触验收。真机 floating-base RNEA 还需要经验证的 base pose/twist 和接触状态估计；在此之前，2 ms 适配器只执行最新命令、安全保护和通信。
