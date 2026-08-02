# C++ DDQ → torque mapper

这个目录把 `sim_support.py::local_forward_dynamics_torque_mapping` 的完整核心链移到 C++。当 `ddq_forward_dynamics_backend: cpp` 时，Python 主控制流程通过窄 C ABI 调用它；Python 版本继续作为数学参考和回归对照。

## 当前覆盖范围

【核心】一次调用包含：

1. 复制当前 `qpos`、`qvel`、`ctrl`、`qacc_warmstart`、`qfrc_applied` 和 `xfrc_applied`。
2. 名义力矩处执行一次完整 `mj_forward`，得到右臂基线 DDQ 并建立位置/速度缓存。
3. 五个右臂力矩方向各执行一次 `mj_forwardSkip(mjSTAGE_VEL)`，建立局部映射 `G_tau`。
4. 用阻尼 SVD 求力矩修正；候选按局部模型预测排序，至少真实验收两个，必要时继续验收较小比例。
5. 按相同条件触发第二轮重线性化和可选安全救援。
6. 救援失败时，重新验收上一拍执行力矩；安全才采用 hold-last。
7. 返回最终力矩、加速度误差、`G_tau`、奇异值、候选选择、安全状态、MuJoCo 调用数和分阶段耗时。

实现不会把上一拍力矩未经本拍动力学检查就直接复用，也不会省略 `qacc_warmstart`。这两点对接触约束下的数值一致性很重要。

## 一键构建与验证

```bash
./cpp/ddq_torque_mapper/build_and_test.sh
```

可缩短或加长随机测试和基准：

```bash
./cpp/ddq_torque_mapper/build_and_test.sh --samples 20 --repeats 100
./cpp/ddq_torque_mapper/build_and_test.sh --samples 100 --repeats 1000
```

Release 共享库默认生成在：

```text
/tmp/hold-my-beer-mpc-ddq-torque-mapper-build/libddq_torque_mapper.so
```

测试脚本会将相同随机状态分别交给 Python 参考实现与 C++，比较最终力矩、基线/预测/验收 DDQ、局部增益矩阵、奇异值以及二轮、救援、hold-last 分支，并分别统计 Python wall time、`ctypes + C ABI` wall time 和 C++ 内部总时间。

## C ABI

公开接口位于：

```text
include/ddq_torque_mapper/ddq_torque_mapper_c.h
```

基本生命周期：

```text
ddq_torque_mapper_create(scene.xml)
    ↓
ddq_torque_mapper_compute(state, request, params, output)
    ↓
ddq_torque_mapper_destroy(handle)
```

`DdqTorqueMapperState` 接收完整状态数组及显式长度；`DdqTorqueMapperRequest` 只含五维右臂任务；`DdqTorqueMapperOutput` 使用固定长度数组，避免跨 ABI 返回 C++ 容器或动态内存。

同一个 handle 持有一个 MuJoCo scratch `mjData` 和复用缓冲区，因此不能被多个线程同时调用。不同控制线程应各自创建 handle。

## 当前限制

- 只针对项目当前 `resources/g1_description/scene.xml` 的状态结构：`na=nmocap=neq=nuserdata=npluginstate=0`。创建时会显式检查；未来模型增加这些状态后会拒绝启动，而不是静默算错。
- 右臂五个 joint/actuator 按名字解析，并检查 hinge、`gear=1` direct-drive 和有效力矩范围。
- 本目录只是原生计算模块和验证工具，尚未改动 `main_sim.py`、`sim_support.py` 或配置开关。
- MuJoCo 候选验收反映仿真中的浮动基、接触、摩擦和约束；真机最终安全层不能把它误认为真实接触状态的直接测量。
