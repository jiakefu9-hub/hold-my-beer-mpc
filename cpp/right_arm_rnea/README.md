# Right-arm Pinocchio RNEA C ABI

该目录把当前 Python `PinocchioPredictionBackend.compute_right_arm_rnea()`
等价迁移为 Pinocchio 4.1 C++ 共享库。它仍读取仿真使用的同一份
`scene.xml`：MuJoCo 只在初始化阶段解析完整 scene、提供可靠的
`qpos/qvel` 索引；实时计算只调用 Pinocchio RNEA，不调用 MuJoCo
动力学或接触求解器。

## 数学与坐标约定

- 输入 `qpos` 使用 MuJoCo 的 `xyz + wxyz`；Pinocchio 使用
  `xyz + xyzw`，初始化时按关节名预计算映射。
- MuJoCo floating-base 平动速度按世界系表达，进入 Pinocchio 前旋到
  body 系；两者角速度均按 body 系表达，直接复制。
- 右臂以外的广义加速度设为零；右臂五维使用传入的 `ddq_des`。
- 输出 `tau_rnea` 是纯刚体动力学力矩，不含 MuJoCo passive 和滑动摩擦。
  为接入现有执行链，C ABI 同时接受右臂 `tau_passive`、`friction_loss`、
  MuJoCo timestep 和 breakaway steps，并计算：

```text
tau_friction = -friction_loss * direction(dq, ddq_des)
tau_ff = tau_rnea - tau_passive - tau_friction
```

同一个 handle 复用 Pinocchio `Data` 和工作向量，不能并发调用。C ABI
对维度、空指针、非有限值和非法摩擦参数做显式检查，C++ 异常不会跨越
ABI 边界。

## 一键构建、测试和基准

```bash
./cpp/right_arm_rnea/build_and_test.sh
```

默认在 `/tmp/hold-my-beer-mpc-right-arm-rnea-build` 构建，不向仓库写入
产物。测试会随机化整机标量关节、浮动基姿态/速度和右臂 DDQ，逐项比较
Python/C++ 的 `tau_rnea`、摩擦项和 `tau_ff`，随后分别报告 Python wall、
ctypes wall、C++ 映射加 RNEA 以及单独 `pinocchio::rnea` 的耗时。

可调整样本量：

```bash
./cpp/right_arm_rnea/build_and_test.sh --samples 200 --repeats 5000
```
