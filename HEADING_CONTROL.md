# 世界系航向闭环

## 1. 作用与控制层级

航向闭环位于腿部 RL locomotion policy 的命令输入层，用于抑制机器人在直线行走时的长期偏航。它不属于右臂 LQR，也不改变右臂的 `ddq_des -> tau` 执行链。

- MuJoCo 仿真周期：`simulation_dt = 0.002 s`，即 500 Hz
- 腿部 RL policy 更新周期：`control_decimation = 10`
- 航向闭环更新周期：`0.002 * 10 = 0.02 s`，即 50 Hz
- 右臂 LQR 周期仍为：`0.002 * 3 = 0.006 s`，即 166.7 Hz

航向闭环只修改 RL policy 的第三个行走命令 `cmd[2]`，前向速度和横向速度命令保持不变。

## 2. 参考方向与测量坐标系

参考航向 `psi_ref` 是闭环第一次更新时 torso 在世界坐标系中的 yaw，之后保持固定。因此“直行”表示保持世界坐标系中的固定方向，而不是让参考方向跟随机器人一起旋转。

当前航向来自 MuJoCo `torso_link` 的世界系四元数。代码先把 yaw 展开为连续角度，避免经过 `+pi/-pi` 时产生跳变。真机上应使用 IMU/状态估计器输出的世界系或局部导航系 heading，并保持同样的角度展开逻辑。

## 3. 去除步态周期摆动

行走时 torso yaw 会随左右迈步产生周期摆动。控制器不直接使用瞬时 yaw，而是在 50 Hz 下保存一个步态周期的数据：

- 步态周期：`T_gait = 0.8 s`
- 窗口样本数：`0.8 / 0.02 = 40`
- `bar(psi)`：窗口内连续 yaw 的均值
- `bar(omega_z)`：窗口内世界系 yaw rate 的均值

一个完整周期可以削弱主要周期分量，平均群延迟约为 0.4 s。使用两个周期会把延迟增至约 0.8 s，目前对慢偏航修正没有必要。

## 4. 控制律

定义航向误差：

```text
e_psi = psi_ref - mean(psi)
```

RL policy 的 yaw-rate 命令为：

```text
yaw_rate_cmd = clip(
    yaw_rate_feedforward + Kp * e_psi - Kd * mean(omega_z),
    -yaw_rate_limit,
    +yaw_rate_limit,
)
```

当前参数：

| 参数 | 数值 | 含义 |
|---|---:|---|
| `yaw_rate_feedforward` | `0.0127 rad/s` | 原有固定补偿，来自 `cmd_init[2]` |
| `heading_kp` | `0.6` | 世界系航向误差反馈 |
| `heading_kd` | `0.1` | 平均 yaw rate 阻尼 |
| `heading_max_yaw_rate` | `0.25 rad/s` | 最终命令限幅 |
| `heading_filter_cycles` | `1.0` | 滑动均值窗口长度，以步态周期计 |

evaluation 结束后的 cooldown 阶段会继续计算闭环，但最终三维行走命令统一线性衰减到零。

## 5. 配置与代码位置

配置位于 `configs/g1.yaml`：

```yaml
heading_control_enabled: true
heading_filter_cycles: 1.0
heading_kp: 0.6
heading_kd: 0.1
heading_max_yaw_rate: 0.25
cmd_init: [0.5, 0, 0.0127]
```

- `sim_support.py::HeadingHoldController`：角度展开、周期均值和反馈控制律
- `main_sim.py`：50 Hz 调用、RL observation 命令写入和运行数据记录

## 6. 记录与验收指标

每次运行会在 `trajectory.npz` 和 `control_preview.csv` 中记录：

- `heading_reference_world`
- `heading_yaw_unwrapped`
- `heading_yaw_filtered`
- `heading_yaw_error`
- `heading_yaw_rate_filtered`
- `heading_yaw_rate_correction`
- `heading_yaw_rate_command`
- `heading_command_saturated`

同时生成：

- `heading_control.png`
- `heading_control_diagnostics.json`

首次闭环实验 `evaluation/left_fixed_right_lqr/20260722_213701` 中，torso yaw slope 从开环的 `-0.01232 rad/s` 降到 `-0.00033 rad/s`，evaluation 末漂移从 `-0.0860 rad` 降到 `+0.0050 rad`，且 yaw-rate 命令未触及限幅。因此当前参数作为后续 LQR/MPC 实验的固定行走基线。

## 7. 边界

该闭环只能让底盘保持世界系航向，不能保证机器人严格沿世界系直线移动；若还需要消除横向位置漂移，应另加世界系横向位置/速度闭环。航向闭环也不负责右臂避碰，右臂安全仍需工作空间或几何距离约束。
