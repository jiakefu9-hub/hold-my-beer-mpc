# 加速度扰动前馈时间调整：独立候选，尚未真机验证

基线提交 `50907c986cd475e3d44c03cb6fb4974350955bfc` 在开发前已与 `origin/main` 一致。
原 `164546` 权重／torque 配置、默认运行行为和全部成功原始数据保留。
本次只做无网络分析、开发与验证，没有连接机器人、初始化 DDS 或发送实际命令。

## 改了什么

在原 learned 入口增加默认关闭的 `--align-accel-forecast`。仅准入 `164546` 的
`q_ee_acc=0.01、q_ee_alpha=0.0005、q_ee_omega=1.0` 与无 roll torque 配置，
预测器为 `learned_filtered`；支持既有非零中立位和共同零中立位。

四轮正常实验中，predictor query 到主机 Write 完成的中位数为
`3.396、3.469、3.430、3.468 ms`。因此固定主机预算为 `3.5 ms`，每拍再加实际记录的
`query - anchor`（0～2 ms），总前移量为 `3.5～5.5 ms`。
这是主机时序估计，不是已辨识的 DDS 单程／电机响应延迟，也不是滤波器延迟补偿。
下一轮仍须检查新增计算与记录开销后的实际 query-to-Write 分布。

- 只重采样未来九个区间的 **线加速度和角加速度**。原预测是 6 ms 区间均值，
  按区间重叠长度加权；例如前移 3 ms，第一格为原第 1、2 格均值各一半。
- 保留实测 q/dq、当前 node zero、当前 `M,b`、全部角速度和姿态预测。
  不把未来姿态冒充当前实测姿态，也不启用原先的整套 `--assumed-command-delay-ms` 状态外推。
- 因而这是**局部的加速度前馈预览调整**，不是把整套状态／身体姿态／逆动力学同步搬到命令生效时刻。
  关节状态年龄、当前动力学偏差、角速度／姿态延迟和执行器响应仍未补偿，不能宣称已解决完整时延问题。
- 最后一格超过原 54 ms 学习范围的部分保持最后一个区间均值，在日志中明确记录外延时长；
  没有新增训练出来的未来预测，也没有读取未来实测数据。
- 只在原本启用 learned 的阶段生效，即 5 s 后且 H0 已冻结；抬臂、接管、静止模式和退权沿用原行为。
- 原权重、关节／速度／加速度／力矩限制、左臂固定 PD 和输出前复查保持原值。

代码为 `tools/g1_commissioning/hardware_mpc_forecast_timing.py`；只修改共享入口的显式分派，
未修改原始预测器、MPC 数学、力矩映射或控制 YAML。
会在 session、core metadata 和逐拍 predictor 诊断中记录选项、预算、实际前移量、
新加速度区间时间戳及实际使用的 acc/alpha。原 `forecast_y` 和原时间戳保留用于复核。

## 用现有真机数据筛选

对保存下来的因果预测，分别比较前移 0/3/6/9/12 ms，以及本次的 `query+3.5 ms` 方案。
所有候选对照的是**同一份**后验标签：主机 Write 完成后 6 ms 区间的实测 IMU 15 Hz 因果滤波均值。
原始 IMU 只用于离线标签，运行输入没有读取未来数据；另报告 Write 后 18～24 ms 区间及行走／停车分段。

标签按原 2 ms 网格重建，和原日志 `current_y` 核对：前三轮舍入到六位均为零；
`180354` 各分量 RMSE 最大约 `0.00483`，保留为回调可用时机／重建的有限差异。
这些历史轮次此前已看过，不称作新的盲测。

完整 `[5,18)` 的**扰动预测向量 RMSE**如下，绝不是新控制器的末端运动 RMS：

| 源实验 | 线加速度预测误差 原／新 m/s² | 降幅 | 角加速度预测误差 原／新 rad/s² | 降幅 |
| --- | ---: | ---: | ---: | ---: |
| 164546 | 1.1933 / 0.8823 | 26.06% | 3.5032 / 2.2943 | 34.51% |
| 155844 | 1.0824 / 0.8006 | 26.04% | 3.2020 / 2.1509 | 32.83% |
| 174504 | 1.1496 / 0.8346 | 27.40% | 3.5333 / 2.4502 | 30.65% |
| 180354 | 1.2180 / 0.8762 | 28.06% | 3.5655 / 2.3059 | 35.33% |

首个区间的行走和停车分段，四轮的上述两项向量误差也均下降；提前 9～12 ms 则开始恶化。
3.5 ms 预算来自软件时序，而不是对某个末端性能结果调参拟合。
误差改善只支持选为下一次实验候选，不能转述为“真机末端加速度降低 26%”。

复现工具：`tools/g1_commissioning/analyze_mpc_forecast_timing.py`，接受重复的
`--source <raw.jsonl>` 和新的 `--output-dir`。完整扫描、分段、源数据与源码哈希位于
`evaluation/forecast_timing_20261010/labels/{summary.json,REPORT.md}`。

## 离线验证

- 数学、入口默认值／拒绝组合、零位、进程计算一致性、进程故障后的父进程退权，以及
  原有边界／力矩／时序容错／序列化相关测试，共 88 项通过。
- 从提交 `50907c9` 取出原运行入口，与修改后关闭选项的入口逐拍对照；
  非零／零中立位合计 200 拍，q/dq 参考、期望加速度及前馈力矩数值差异为 **0**。
  证据：`evaluation/forecast_timing_20261010/baseline_regression.json`。
- 对上述四轮真实状态，原／新各完整回放一遍，共八组。复用既有 packet／CRC／约束检查、
  停车后的本地退权路径；每组 `2440～2448` 拍活跃控制均通过，最终 `weight=0`，
  无 QP／力矩包络拒绝。模型净加速度闭合误差均小于 `4.77e-7 rad/s²`。
  回放工具为 `validate_mpc_forecast_timing.py`；
  使用 `--math-only` 将离线 QP 时间 allowance 设为 100 ms，**不是放宽现场限制或证明 6 ms 实时通过**。
  状态不会随新命令改变，所以不是物理闭环仿真或真机验收。
  结果在 `evaluation/forecast_timing_20261010/replay_verified/summary.json`。
  新方案离线主窗口计算＋构包 P99 为 `3.55～3.99 ms`，相对各自旧方案平均增加
  `0.11～0.25 ms`，未包含现场 DDS、计算进程往返和审计 IO，不能称为完整现场周期耗时。
- 新选项的零中立位 preflight 使用实际模型、QP、IDL/CRC，通过且记录
  `dds_initialized=false、publisher_created=false、robot_connected=false`。
  证据：`evaluation/forecast_timing_20261010/preflight_aligned.json`。

`evaluation/` 仍被 Git 忽略；本文件保留关键结果，代码提交不会备份这些原始日志。

## 下一次现场运行与回退

以下沿用近期正常左臂 `20/1`、共同零中立位、原成功负载模型和同一套权重，
只加入前馈时间调整。不增加新的静止接管实验或另一套行走流程。
本命令准备时未检查实时网线状态，真正运行仍由操作者现场发起并输入原确认口令。

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
sudo -v
MPC_TRIAL_OUT="evaluation/hardware_shadow/commissioning/mpc_zero_accel_aligned_walk_$(date +%Y%m%d_%H%M%S)"
taskset -c 0-17 /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/g1_walk_mpc_learned.py enx6c1ff701509c \
  --execute --task walk --predictor learned_filtered \
  --zero-arm-neutral --align-accel-forecast \
  --mpc-config configs/hardware_mpc_learned_acc001_alpha0005_omega1.yaml \
  --torque-config configs/hardware_mpc_torque_learned.yaml \
  --cpu 7 --rt-priority 20 --compute-process \
  --profile evaluation/hardware_shadow/commissioning/g1_pid_6ms_20261005_132052/arm_profile.conf \
  --output-dir "$MPC_TRIAL_OUT" \
  --pid-6ms-validated --torque-stationary-validated \
  --permit-real-output MPC_WALK_H0_CAPTURE --allow-first-torque-field-trial
```

**直接回退：删除 `--align-accel-forecast`，并使用新的输出目录名，例如
`mpc_zero_baseline_walk_$(date +%Y%m%d_%H%M%S)`。** 不需要回滚 Git 或更改 YAML。
这恢复的是近期共同零位基线；若要复现 164546 的非零位，还需去掉 `--zero-arm-neutral`。

若现场已经换成 75 mm 打印负载，应在新旧两组都显式使用
`--controller-config configs/g1_payload75.yaml`，先承认负载已改变，不能把它与旧瓶数据的
差异全部归给时间调整。本次完整实测状态回放使用原负载模型。

下一轮分析同时报告完整窗口、行走、停车的两臂指标、身体激励、力矩／关节余量、
实际 query-to-Write 与预测取样时刻偏差、完整 6 ms 拍耗时、停车和退权结果。
没有真机收益前，原成功基线继续保留，不晋升新方案。
