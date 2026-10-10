# 共同零中立位：左固定 PD 对比右 IMU-MPC

本候选使用一个新的显式开关 `--zero-arm-neutral`，将左右手臂各五个关节的中立目标同时设为
`[0°,0°,0°,0°,0°]`。左臂仍只使用关节 q/dq 的固定 PD，不读取身体 IMU；右臂 MPC 使用实测
身体姿态、角速度、线加速度和学习预测。它检验的是：从相同、与站立 IMU 倾斜无关的几何零位
出发，IMU-MPC 能否比固定关节 PD 更好地保持瓶轴竖直和抑制动态运动。

原 164546 代码路径、配置文件和现场 profile 均未改写。开关只在运行时生成有效目标，并在会话头
记录完整零位数组。左臂默认 Kp/Kd 为 `20/1`；显式增益实验统一使用 `--left-pd-gain-scale`。

## 精确配置

- predictor：`learned_filtered`
- MPC：`configs/hardware_mpc_learned_acc001_alpha0005_omega1.yaml`
- torque：`configs/hardware_mpc_torque_learned.yaml`，无 roll 回正
- `q_ee_acc=0.01`、`q_ee_alpha=0.0005`、`q_ee_omega=1.0`、`q_ee_vel=0`
- 左臂 Kp/Kd 保持正常 `20/1`，目标关节角固定
- 左右十个手臂中立目标均为零；腰与两个无效槽仍为零
- 右臂 `q_posture` 的参考构型、预测内 pitch 回正目标及 yaw 固定参考同步变为零

身体理论竖直且两臂全零时，当前 XML 中左右瓶轴相对世界竖直的倾角均为 `0.0046°`。
左右瓶中心相对身体 IMU 约为 `[0.3593, +0.1159, -0.0967] m` 与
`[0.3593, -0.1114, -0.0967] m`。真机有限增益和 250 g 瓶重会造成静态下垂，实际姿态须由本轮测量。

## 离线验证

无网络证据位于 `evaluation/hardware_shadow/commissioning/zero_arm_neutral_offline_20261009/`：

- 精确 CLI preflight 通过，`dds_initialized=false`、`publisher_created=false`、`robot_connected=false`；
  右臂 posture、pitch、roll 和 yaw 名义参考均记录为零。
- 使用 164546 完整 raw 做因果回放，2445 拍活跃控制全部完成，最终 `weight=0`，无计算拒绝。
  2162 拍整链计算 P99 `3.252 ms`、最大 `3.689 ms`，`over_6ms_count=0`。
- 回放中右臂发包力矩绝对峰值依次为 `[5.285,3.038,1.598,4.421,0.275] Nm`，低于既有
  `[10,6,4,7,1.5] Nm` 限值。记录状态来自旧 164546 命令，不会响应新零位，不能代替物理验证。
- 准备阶段的零位、计算子进程、学习 MPC、执行路径和速度候选相关测试通过。

## 直接行走命令

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
sudo -v
MPC_TRIAL_OUT="evaluation/hardware_shadow/commissioning/mpc_zero_neutral_164546_walk_$(date +%Y%m%d_%H%M%S)"
taskset -c 0-17 /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/g1_walk_mpc_learned.py enx6c1ff701509c \
  --execute --task walk --predictor learned_filtered \
  --mpc-config configs/hardware_mpc_learned_acc001_alpha0005_omega1.yaml \
  --torque-config configs/hardware_mpc_torque_learned.yaml \
  --zero-arm-neutral \
  --cpu 7 --rt-priority 20 --compute-process \
  --profile evaluation/hardware_shadow/commissioning/g1_pid_6ms_20261005_132052/arm_profile.conf \
  --output-dir "$MPC_TRIAL_OUT" --pid-6ms-validated --torque-stationary-validated \
  --permit-real-output MPC_WALK_H0_CAPTURE --allow-first-torque-field-trial
```

保留既有现场交互确认。此次准备没有连接机器人、创建 DDS publisher 或发送真实指令。按操作者约定，
现场未主动报告吊绳、碰撞、异响或异常停车即视为有效。分析时先核对两边在 3～5 s 的实测 q、
瓶轴倾角和位置，再比较 `[5,18)` 的左右加速度、角运动、力矩余量、计算时序和完整退权。
