# 机器人模型后端

该目录把 MPC 预测所需的末端位姿、Jacobian 和 Jacobian 导数统一到同一接口下，便于在 MuJoCo 参考实现与 Pinocchio 实现之间切换。Pinocchio 还可为 DDQ 执行链提供名义 RNEA 力矩；仿真中的接触、浮动基反作用、候选验收和安全回退仍由 MuJoCo 负责。

## 安装

项目当前在 `g1_mpc` Conda 环境中使用 conda-forge 的 Pinocchio 4.1：

```bash
conda install -n g1_mpc -c conda-forge pinocchio=4.1
```

不在 `requirements-mpc.txt` 中写入同名 PyPI 包，避免误装与机器人动力学库无关的软件包。

## 配置

`configs/g1.yaml` 中的两个开关相互独立：

```yaml
mpc_prediction_kinematics_backend: cpp_pinocchio
ddq_nominal_inverse_dynamics_backend: cpp_pinocchio
```

第二项还可设为 `mujoco` 或 `pinocchio_shadow`；shadow 模式先完成 MuJoCo 主路径，再记录 `right_arm_tau_ff_shadow`、差值、有效标记和额外耗时，最终仍执行 MuJoCo 名义力矩。Pinocchio shadow 异常只使该对照无效，不改变控制输出。
当两项都设为 `cpp_pinocchio` 时，预测窗口与 RNEA 复用同一个原生 handle：预测窗口一次批量计算，RNEA 每次按最新状态计算。`pinocchio` 仍保留为逐项数值回归基准。

active Pinocchio 在仿真中采用 fail-fast，且当前未把运行时 `qfrc_applied/xfrc_applied` 映射为 RNEA 外力；本项目当前仿真的这两项为零。

## 一致性与微基准

运行：

```bash
conda run -n g1_mpc python robot_model_backend/validate_and_benchmark.py
conda run -n g1_mpc python robot_model_backend/validate_inverse_dynamics.py
```

第一个脚本随机改变右臂构型，对比抓持点位姿、世界系 Jacobian 和 Jacobian 导数；第二个脚本独立对比整机 RNEA，不把干摩擦方向近似混入刚体动力学验证。2026-08-02 的 100 个随机构型验证中，预测运动学数值误差处于浮点精度；1000 次单节点调用的平均耗时为 MuJoCo `111.5 us`、Pinocchio `23.8 us`。完整闭环耗时仍应以评估目录中的 timing 记录为准。
