# 双臂原地保持末端 Z 轴向上

独立入口 `tools/g1_commissioning/g1_dual_upright.py`。双臂都使用身体 IMU 和实测关节角，控制目标为模型 `left_grasp_site` / `right_grasp_site` 的局部 +Z 在 IMU 导航世界系中指向 `[0,0,1]`。杆的安装方向需与相应模型 +Z 一致；程序没有摄像头或杆上传感器，不能自动识别安装偏差。

不改变 M01～M06、已有 MPC/PID 参数或模型。新增配置为 `configs/hardware_dual_upright.yaml`。这是原地姿态对准工具，不是行走对照的新控制器，不控制末端世界位置。

## 实现和时序

- 每 6 ms，使用 torso IMU 四元数与左右各五关节反馈计算两根末端 Z 轴；以其世界 X/Y 分量为误差，通过左右各自的解析雅可比和阻尼伪逆计算关节参考速度。
- 主动补偿关节为肩 pitch、肘 pitch、腕 roll。肩 roll/yaw 保持零位参考，入场末尾的微小残差受同样限速平滑归零；不使用肩 roll 大幅向内补偿。控制的是向上半球，倒置状态不能当成误差为零。
- 限速 0.12 rad/s，参考加速度上限 0.4 rad/s²；底层 PD 仍为 20/1，不发额外力矩前馈。加入左右实测关节范围及 PD 模型力矩包络检查，不构成无碰撞或实际电机力矩保证。
- 0～3 s 从实测姿态渐入零位参考；3 s 后持续进行双臂竖直反馈，没有自动结束时间。操作员按一次 `Ctrl+C` 后，程序冻结最后一帧关节参考并执行至少 3 s 渐进退权，退权完成后才退出。
- 原有站立 FSM 500、反馈新鲜度、CRC、遥控器、停机与退权流程复用；程序不切换机器人模式。持续原地控制在最终 FSM=500 检查后不获取或刷新 Loco 速度租约，不调用周期性 `SetVelocity(0)`，也不行走。
- 终端每秒显示一次左右倾角（IMU/模型估计）。结束时按一次 `Ctrl+C`，等待终端显示退权完成；不要连续按、关闭终端或使用强制杀进程。

## 离线检查与运行

默认无参数或 `--preflight` 只执行离线验证，不初始化 DDS。

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
/home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/g1_dual_upright.py --preflight
```

机器人已原地站稳、连接网线后，在同一终端运行：

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
sudo -v
taskset -c 0-17 /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/g1_dual_upright.py enx6c1ff701509c \
  --execute --permit-real-output DUAL_UPRIGHT_STATIONARY --cpu 7 --no-capture
```

确认文字仍为 `EXECUTE SCHOOL_G1_20260915`。上面的 `--no-capture` 不创建输出目录，也不保存原始数据；去掉它时，输出自动创建为 `evaluation/hardware_shadow/commissioning/dual_upright_stationary_时间戳_随机后缀/`，保存模型、配置、原profile、有效零位profile、源文件hash和双侧倾角/关节诊断，不覆盖原数据。

## 验证边界

离线覆盖两臂解析雅可比与有限差分、左右映射、yaw不变性、四元数符号、向上/倒置区分、限速限加速度、迟到拍不放大步长、接管连续性、超过原18 s节点后仍持续控制、PD包络、无执行许可拒绝输出，以及共享反馈健康检查和共享渐退权。简化一阶位置执行模型另测试静止、身体前倾14°、roll/pitch组合、固定负载下垂和慢速身体摆动；这不是全身 MuJoCo 动力学，也没有替代真机验证。

证据目录：`evaluation/hardware_shadow/commissioning/dual_upright_preparation_20261013/`。原七个基线/共享文件hash单独保存并复核；现有控制路径未修改。
