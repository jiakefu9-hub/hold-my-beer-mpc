# 仿真与真机差异记录

## 1. 四元数来源不同

当前仿真里使用的 `torso_quaternion` 不是从 XML 里的 IMU 传感器直接读出来的，而是直接来自 MuJoCo 引擎状态 `d.xquat[torso_id]`，表示 `torso_link` 相对于 MuJoCo 世界系的姿态。当前模型中的 `imu_in_torso` 只配置了 `gyro` 和 `accelerometer`，没有直接输出 quaternion 的传感器项，所以仿真里的四元数本质上是引擎真值，不是 IMU 原始测量。

真机侧则不同。公开的 G1 状态接口示例显示软件状态中可以拿到 `imu_quaternion` / `imu_euler`。这更适合理解为 IMU 配合姿态解算或状态估计后的输出，而不是原始 IMU 芯片直接给出的裸四元数。它可以用于本项目中的重力方向、倾斜误差和相对姿态控制，但 yaw 的绝对世界零点不应过度假设。

## 2. heading 参考方向的定义不同

仿真里如果目标就是沿 MuJoCo 世界系 `x` 方向行走，那么参考前进方向直接由仿真世界系定义，不需要再从 warm-up 平均一个新的世界方向；如果走偏，只需要手动微调 walking command 的 turning 分量即可。真机里则不应默认状态估计器的世界系 `x` 轴天然等于机器人起步前方，因此更稳妥的做法是：以前3个周期内相对稳定时段的 IMU heading 平均值定义一个参考 heading frame，其中 `z` 轴仍与真实重力方向一致，`x` 轴取该平均 heading 方向；从后续控制开始，扰动前馈和 heading 相关分析都相对于这个参考 frame 解释。