# 仿真与真机差异记录

## 1. 四元数与 IMU 运动量来源不同

当前仿真里使用的 `torso_quaternion` 不是从 XML 里的 IMU 传感器直接读出来的，而是直接来自 MuJoCo 引擎状态 `d.xquat[torso_id]`，表示 `torso_link` 相对于 MuJoCo 世界系的姿态。当前模型中的 `imu_in_torso` 只配置了 `gyro` 和 `accelerometer`，没有直接输出 quaternion 的传感器项，所以仿真里的四元数本质上是引擎真值，不是 IMU 原始测量。

与之类似，`main_sim.py` 里当前使用的躯干 / IMU 运动量也主要来自 MuJoCo 引擎真值：`torso_ang_vel` 和 `torso_lin_vel` 是通过 `get_site_vel(m, d, imu_site_id)` 从 site Jacobian 与 `qvel` 计算出来的世界系速度；随后 `torso_acc = d/dt(torso_lin_vel)`、`torso_alpha = d/dt(torso_ang_vel)` 由差分得到。也就是说，当前仿真主循环里真正送入控制器和扰动模型的是 `torso_acc / torso_omega / torso_alpha`，其中 `torso_lin_vel` 只是为了在仿真里更干净地构造 `torso_acc` 的中间量，本身并没有直接进入当前 PID / LQR 公式。

真机侧则不同。公开的 G1 状态接口示例显示软件状态中可以拿到 `imu_quaternion` / `imu_euler`、`imu_angular_velocity` 和 `imu_linear_acceleration`。其中角速度和线加速度可以看作 IMU + 状态接口给出的可直接使用量；但线速度通常不是 IMU 直接可测量量，若需要 base 线速度，一般要靠状态估计、外部里程计或对位置数据做估计。在本项目当前控制链里，这不是硬需求，因为公式真正需要的是 base 的角速度、角加速度和线加速度，而不直接需要线速度；因此真机更合理的对应方式是：直接使用 `imu_angular_velocity`，用差分估计 `alpha`，用 `imu_quaternion` 把 `imu_linear_acceleration` 旋到参考系后再去除重力得到近似 `acc_world`，而不是先去构造一个 IMU 线速度再差分。

## 2. heading 参考方向的定义不同

仿真里如果目标就是沿 MuJoCo 世界系 `x` 方向行走，那么参考前进方向直接由仿真世界系定义，不需要再从 warm-up 平均一个新的世界方向；如果走偏，只需要手动微调 walking command 的 turning 分量即可。真机里则不应默认状态估计器的世界系 `x` 轴天然等于机器人起步前方，因此更稳妥的做法是：以前3个周期内相对稳定时段的 IMU heading 平均值定义一个参考 heading frame，其中 `z` 轴仍与真实重力方向一致，`x` 轴取该平均 heading 方向；从后续控制开始，扰动前馈和 heading 相关分析都相对于这个参考 frame 解释。

## 3. 时间与实时性判断不同

`main_sim.py` 里的 `PerformanceMonitor` 使用 `time.perf_counter()` 统计的是现实世界墙钟时间，而不是 MuJoCo 仿真时间。`simulation_dt` 是仿真每一步推进的时间，同时也被当作实时控制预算：例如 `simulation_dt=0.002` 表示目标 500Hz，每步现实计算时间最好小于 2ms。若仿真 10s 需要现实 40s 才跑完，说明当前程序整体只能约 0.25x real-time 运行，原因可能来自 viewer 渲染、视频 renderer、数据记录、MuJoCo step、Python 循环或控制算法本身。

性能日志里的 `arm avg/max` 和 `arm overruns` 只覆盖右臂控制计算段，也就是 `build_helpers(...)` 与 `arm_policy.compute_action(...)`，其中包含当前使用的 MuJoCo scratch 运动学 / 雅可比计算。若 `arm overruns=0`，只能说明在当前仿真电脑和当前 Python 环境下，右臂控制算法本身没有超过单步预算，是一个积极信号；它不能直接证明真机上也一定实时，因为真机还会多出 IMU / 编码器读取、电机通信、安全检查、滤波、线程调度和操作系统实时性等开销。`loop avg/max` 和 `loop overruns` 覆盖整个仿真主循环，包括控制、`mj_step`、评估记录、视频渲染、debug axes 和 `viewer.sync()`。如果 `arm overruns` 很少但 `loop overruns` 很多，通常说明瓶颈不在右臂算法，而在仿真、渲染、录像或记录数据。

真机上应该单独做实时性 benchmark，不要直接用带 viewer / 录像 / 完整 MuJoCo 仿真的结果判断。建议在真机控制电脑上用 `time.perf_counter_ns()` 或系统提供的单调高精度时钟，在控制线程里记录以下分段耗时：

1. `cycle_total`：一次真实控制周期总耗时，从本周期开始取传感器状态，到本周期命令发送完成。
2. `state_read`：读取 IMU、关节角、关节速度、电机状态的耗时；如果读取接口是异步回调，则记录拿到最新状态与当前控制时刻之间的状态年龄 `state_age`。
3. `state_preprocess`：四元数归一化、坐标系转换、去重力、角加速度差分、滤波、构造 `DisturbanceInput` / `arm_obs` 的耗时。
4. `kinematics`：如果真机仍使用 MuJoCo scratch，则记录写入 base/torso 姿态、右臂关节角、调用 `mj_forward`、计算 Jacobian / dJ 的耗时；如果改用 Pinocchio 或其他运动学库，也记录同一段。
5. `control_solve`：LQR / MPC / PID 真正计算 `q_ref`、`dq_ref` 或 torque command 的耗时。对 MPC 要额外记录求解器迭代次数、是否 warm start、是否求解失败。
6. `safety_postprocess`：关节限幅、速度限幅、力矩限幅、平滑器、急停检查、NaN 检查等安全后处理耗时。
7. `command_send`：把控制命令写到电机 / SDK / CAN / DDS / UDP 等接口的耗时。
8. `period_jitter`：相邻两次控制周期实际开始时间的间隔，用来判断控制线程是否稳定。例如目标 100Hz 时理想间隔是 10ms，目标 500Hz 时理想间隔是 2ms。

判断时不要只看平均值，还要看 `max`、`p95`、`p99` 和 overrun 次数。建议标准是：`cycle_total` 的 `p99` 小于目标周期的 70%-80%，`max` 偶发超过可以接受但不能连续超过；`control_solve + kinematics` 应明显小于目标周期，为通信和系统调度留余量。例如右臂高层 LQR 若跑 100Hz，周期预算是 10ms，最好 `kinematics + control_solve` 稳定在 2-4ms 以内；若强行跑 500Hz，周期预算只有 2ms，就必须确认 `p99` 仍低于约 1.5ms，且 `cycle_total` 几乎没有 overrun。

真机 benchmark 的推荐流程是：先不接电机或不给使能，只读取真实状态并计算控制输出，记录上述耗时；然后接入命令发送但保持机器人悬空或安全支撑，继续记录 `command_send` 和 `cycle_total`；最后在低速、低增益、有限幅的真实运动中记录完整链路。只有在真机控制电脑、真机通信链路和真实线程调度条件下 `cycle_total`、`period_jitter`、`state_age` 都满足预算，才能认为该控制频率对真机是比较稳的。
