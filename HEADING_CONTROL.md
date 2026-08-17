# 正式 full-task v2 的命令与航向时序

本文只描述当前冻结的 MuJoCo 正式路径：`full_task_template v2 + continuous causal-H v2 + 24 ms startup-PD + 右臂 MPC process`。航向控制属于下肢 locomotion policy 的命令输入层；它既不决定右臂何时接管，也不修改 MPC、DDQ-to-torque 或安全认证逻辑。

代码入口与冻结参数见 [`configs/g1.yaml`](configs/g1.yaml)、[`main_sim.py`](main_sim.py)、[`sim_support.py`](sim_support.py)、[`disturbance_template/full_task_protocol.py`](disturbance_template/full_task_protocol.py) 和 [`disturbance_template/full_task_startup_pd.py`](disturbance_template/full_task_startup_pd.py)。

## 1. 唯一正式时间线

| simulation time | task time | 已冻结行为 |
|---:|---:|---|
| `0.000 s` | `0.000 s` | 建立 task epoch；planned command 立即为 `[0.5, 0, 0.0127]`；template、continuous-H 与 headline 同时从零开始。这里表示命令已可见，不声称机器人已经产生位移。 |
| `[0.000, 0.024) s` | 同左 | 下肢正常执行；右臂只执行配置目标的固定姿态 PD。该 24 ms 前缀属于 headline。 |
| `0.020 s` | `0.020 s` | 下肢 policy 第一次计算并消费新的 runtime command；heading controller 也在这个 20 ms 更新点第一次用实测 torso yaw 建立固定世界系参考。 |
| `0.024 s` | `0.024 s` | 右臂 MPC process 在合法 6 ms 网格的 anchor 4 接管，直接查询绝对 task time `0.024 s` 的模板；task、gait、template 和 H 均不 reset、不重播。 |
| `6.400 s` | `6.400 s` | planned `vx/vy` 在该边界直接变为零，无 ramp；planned `wz` 前馈仍保留，runtime `wz` 继续由 heading controller 给出。 |
| `[0.000, 8.000) s` | 同左 | 唯一 headline；startup-PD、启动、行走和停车全部包含在内。 |

基础网格为 MuJoCo `2 ms`、右臂 MPC `6 ms`、下肢 policy `20 ms`。24 ms handoff 只由固定的 sample/anchor 索引决定；足部接触、torso 加速度和竖直速度仅记录为诊断，不能延迟或提前接管。正式 full-task 路径也不启用 cooldown 衰减：`[0,8)` 内 command scale 始终为 `1`。

## 2. Planned command 与 runtime command

两者必须分开理解和记录：

- `planned_command` 是绝对 task-time direct-step schedule。`0 <= t < 6.4 s` 使用配置的 nominal command；`t >= 6.4 s` 只把 `vx/vy` 直接置零，保留 nominal `wz=0.0127 rad/s` 作为航向前馈。
- `runtime_command` 是下肢 policy 真正消费的命令。在每个 20 ms policy 更新点，它先复制 planned command；heading 开启时只把第三维 `wz` 替换为闭环输出，`vx/vy` 不受 heading controller 改写。

因此，`t=6.4 s` 之后“停止”专指平移命令归零，不代表 runtime `wz` 必须为零。planned/runtime 字段的 schema 定义与记录合同见 [`disturbance_template/full_task_recording.py`](disturbance_template/full_task_recording.py)。

## 3. Heading controller

冻结配置为：

```yaml
heading_control_enabled: true
heading_filter_cycles: 1.0
heading_kp: 0.6
heading_kd: 0.1
heading_max_yaw_rate: 0.25
cmd_init: [0.5, 0, 0.0127]
```

`HeadingHoldController` 以 20 ms、即 50 Hz 更新。第一次更新时，它把 MuJoCo `torso_link` 的世界系 yaw 设为固定参考 `psi_ref`；之后先展开 `+pi/-pi` 跳变，再对 yaw 与世界系 yaw rate 分别做一个 `0.8 s` 步态周期的移动算术均值。窗口容量为 `0.8 / 0.02 = 40` 个样本；窗口未满时只使用已经到达的因果前缀。

控制律为：

```text
e_psi = psi_ref - mean(yaw_unwrapped)
correction = 0.6 * e_psi - 0.1 * mean(yaw_rate_world)
runtime_wz = clip(0.0127 + correction, -0.25, +0.25)
```

这里的移动平均是现有 heading controller 自身的抗步态摆动机制；当前方案没有增加第二层 heading 低通。heading 全程启用，但它只是生成下肢 runtime `wz`，不会门控 20 ms policy 更新、24 ms 右臂 handoff 或 MPC 输出。

## 4. Heading controller 与 continuous-H 不是同一个状态

两者都涉及 torso yaw，但职责、采样网格和算法不同，不能混用：

| 项目 | Heading controller | Full-task continuous-H v2 |
|---|---|---|
| 用途 | 生成下肢 policy 的 runtime `wz` | 定义模板扰动的 heading 坐标系并完成在线世界系变换 |
| 更新网格 | `20 ms` policy 网格 | `6 ms` MPC anchor 网格 |
| 角度处理 | 展开 yaw 后做算术移动平均，同时平均 yaw rate | 对 yaw 的 `sin/cos` 做圆周平均 |
| 因果窗口 | 最多一个 `0.8 s` 周期 | `t<0.8 s` 用因果前缀；`0.8<=t<6.4 s` 用过去 `0.8 s`；`t>=6.4 s` 冻结最后一个停车前 H |
| 对控制的作用 | 改写 runtime command 的第三维 | 不生成 locomotion command |

continuous-H 在 `t=0` 就使用第一个有效 6 ms anchor 的实测 yaw；每个 54 ms 预测窗口内固定为查询 anchor 已经确定的 H，禁止 future yaw leakage。其 reset 与 task epoch 绑定，而不是与 24 ms handoff 绑定。完整定义见 [`FULL_TASK_TEMPLATE.md`](FULL_TASK_TEMPLATE.md)。

## 5. 记录与受控仿真证据

正式记录保留以下 heading 字段：

- `heading_reference_world`
- `heading_yaw_unwrapped`
- `heading_yaw_filtered`
- `heading_yaw_error`
- `heading_yaw_rate_filtered`
- `heading_yaw_rate_correction`
- `heading_yaw_rate_command`
- `heading_command_saturated`
- `planned_command` 与 `runtime_command`

CPU 7 受控证据包位于 [`evaluation_summary/full_task_template_v2_final_freeze/`](evaluation_summary/full_task_template_v2_final_freeze/)。其中三次 nominal 的 heading error RMS/最大绝对值均为 `0.011451/0.029599 rad`，三次 `heldout_pair_02_minus` 均为 `0.010899/0.029098 rad`；六条运行的 command saturation fraction 均为 `0`。这些数字来自各运行目录的 `heading_control_diagnostics.json`，而 planned/runtime 的时序图位于 [`representative_plots/`](evaluation_summary/full_task_template_v2_final_freeze/representative_plots/)。它们是受控 MuJoCo 仿真事实，不是真机硬实时或实物航向稳定性证据。

## 6. 平台边界与限制

当前 yaw 测量来自 MuJoCo torso 世界系姿态，yaw rate 来自仿真 torso 状态。未来真机适配必须由 IMU/状态估计器提供坐标定义一致、时间戳可信的 heading 与 yaw rate，并重新验证 20 ms 调度、饱和行为和 locomotion policy 响应。

现有 hardware shadow 是只读验证路径，尚未接入并验证最终 `full_task v2 + 24 ms handoff` 的真机主动输出。航向闭环只能抑制世界系偏航；它不保证横向位置严格为零，也不承担右臂避碰或最终力矩安全认证。
