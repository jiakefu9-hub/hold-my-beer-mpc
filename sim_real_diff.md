# 仿真与真机差异记录

## 1. 四元数来源不同

当前仿真里用到的 `torso_quaternion` 不是从 XML 中定义的 IMU 传感器直接读出来的，而是直接来自 MuJoCo 引擎状态 `d.xquat[torso_id]`，表示 `torso_link` 相对于 MuJoCo 世界系的姿态。当前 XML 里的 `imu_in_torso` 只配置了 `gyro` 和 `accelerometer`，没有直接输出 quaternion 的传感器项。

真机侧则不同。根据公开的 G1 状态接口示例，软件状态里可以拿到 `imu_quaternion` 和 `imu_euler`。这通常应理解为 IMU 配合姿态解算 / 状态估计后的输出，而不是原始 IMU 芯片直接给出的“裸四元数”。对本项目来说，这个四元数可以用于重力方向、倾斜误差和相对姿态控制，但其 yaw 的绝对世界零点不应过度假设，实际使用时更适合配合第一帧或启动稳定段作为参考。