# 仿真与真机差异记录

## 1. 四元数与 IMU 运动量来源不同

当前仿真里使用的 `torso_quaternion` 不是从 XML 里的 IMU 传感器直接读出来的，而是直接来自 MuJoCo 引擎状态 `d.xquat[torso_id]`，表示 `torso_link` 相对于 MuJoCo 世界系的姿态。当前模型中的 `imu_in_torso` 只配置了 `gyro` 和 `accelerometer`，没有直接输出 quaternion 的传感器项，所以仿真里的四元数本质上是引擎真值，不是 IMU 原始测量。

与之类似，`main_sim.py` 里当前使用的躯干 / IMU 运动量也主要来自 MuJoCo 引擎真值：`torso_ang_vel` 和 `torso_lin_vel` 是通过 `get_site_vel(m, d, imu_site_id)` 从 site Jacobian 与 `qvel` 计算出来的世界系速度；随后 `torso_acc = d/dt(torso_lin_vel)`、`torso_alpha = d/dt(torso_ang_vel)` 由差分得到。也就是说，当前仿真主循环里真正送入控制器和扰动模型的是 `torso_acc / torso_omega / torso_alpha`，其中 `torso_lin_vel` 只是为了在仿真里更干净地构造 `torso_acc` 的中间量，本身并没有直接进入当前 PID / LQR 公式。

真机侧则不同。公开的 G1 状态接口示例显示软件状态中可以拿到 `imu_quaternion` / `imu_euler`、`imu_angular_velocity` 和 `imu_linear_acceleration`。其中角速度和线加速度可以看作 IMU + 状态接口给出的可直接使用量；但线速度通常不是 IMU 直接可测量量，若需要 base 线速度，一般要靠状态估计、外部里程计或对位置数据做估计。在本项目当前控制链里，这不是硬需求，因为公式真正需要的是 base 的角速度、角加速度和线加速度，而不直接需要线速度；因此真机更合理的对应方式是：直接使用 `imu_angular_velocity`，用差分估计 `alpha`，用 `imu_quaternion` 把 `imu_linear_acceleration` 旋到参考系后再去除重力得到近似 `acc_world`，而不是先去构造一个 IMU 线速度再差分。
## 2. heading 参考方向的定义不同

仿真里如果目标就是沿 MuJoCo 世界系 `x` 方向行走，那么参考前进方向直接由仿真世界系定义，不需要再从 warm-up 平均一个新的世界方向；如果走偏，只需要手动微调 walking command 的 turning 分量即可。真机里则不应默认状态估计器的世界系 `x` 轴天然等于机器人起步前方，因此更稳妥的做法是：以前3个周期内相对稳定时段的 IMU heading 平均值定义一个参考 heading frame，其中 `z` 轴仍与真实重力方向一致，`x` 轴取该平均 heading 方向；从后续控制开始，扰动前馈和 heading 相关分析都相对于这个参考 frame 解释。