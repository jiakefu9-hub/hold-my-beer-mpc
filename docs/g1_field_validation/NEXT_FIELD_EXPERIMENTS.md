# 下一次现场实验计划

**最新覆盖说明（2026-10-09）：** 近期已完成
[共同零中立位：左固定 PD 对比右 IMU-MPC](sessions/20261009_ZERO_ARM_NEUTRAL_PREPARATION.md)。
左右各五个关节目标都为零；左侧正常 `20/1` 固定 PD，右侧使用完整 164546 MPC 和
无 roll torque 配置。`155844`、`174504`、`180354` 已完成；下一轮尚未选定。
链接内包含精确设置、离线结果和直接 `walk` 命令。本页下面原计划保留为历史说明。

准备日期：2026-10-09。C 组完成并分析后更新。
今天只开发与离线验证，没有连接机器人或发送真实指令。已验证代码基线为
`11e19ea28adb089c4f105602bc3990fb7dc4e68a`；改动前 HEAD 与 origin/main 一致、工作树干净。
旧配置和成功原始数据保留；新增配置是待真机验证候选。

## 明天优先做什么

B、C 已按计划使用 `learned_filtered`、正常左臂 Kp/Kd `20/1` 和无 roll torque 配置完成。
B 的 `q_ee_vel=0.1` 未降低目标相对速度；C 的 Y 向权重只带来很小的行走收益且停车变差，
两者均不晋升为基线。下一轮尚未由操作者选定。

| 顺序 | 实验 | 本组唯一变化 | 目的 |
| --- | --- | --- | --- |
| A（已完成） | 三代价 MPC + roll 回正 + 左臂 PD 2× | 操作者确认无吊绳牵拉或碰撞；组合无收益 | 不晋升基线；不能从本组严格分离 roll 与左臂增益因果 |
| B（已完成） | 164546 + 末端相对线速度代价 | 只加 `q_ee_vel=0.1`；无 roll；左臂正常 `20/1` | 目标速度未改善，不晋升基线 |
| C（已完成） | 164546 + Y 向加速度代价 | `q_ee_acc=[0.01,0.015,0.01]`；线速度代价为零、无 roll | 行走 Y 向小幅改善，但整体不足，不晋升基线 |
| C2（可选） | 164546 + 更高 Y 向加速度代价 | `q_ee_acc=[0.01,0.02,0.01]`；其余保持 C | 若还要验证权重趋势，可单独做；预期视觉收益有限 |
| D | 收近持瓶构型 | 先改中立角、保持 A 的代价 | 先在仿真筛选更短前伸距离、近竖直且有关节余量的构型；通过后另行准备真机 profile |
| E | 收近构型下轻降中立位代价 | 固定 D 构型，只将 `q_posture` 降至其 80% | 判断姿态自由度增大是否带来收益；与 D 比较，不同时再改加速度或端平代价 |
| 独立附加组 | 左臂 PD 刚度／阻尼敏感性 | 显式 `--left-pd-gain-scale 2`：左臂 Kp 20→40、Kd 1→2 | 观察更硬左臂的传振、姿态和整机反作用；明确标注增益，不充当原 PD 基线 |

若选择 C，直接对照 164546；不把一次略优称为稳定优势。
若 B 有效，以后再单独比较 B 与 B+C。D、E 目前是后续计划，尚未生成可执行的新姿态配置。
暂不新增液体二阶模型、不改成液体激励比例目标、不重调 PID。

主研究对照可以是同一右臂的正常 PID 与 MPC；两者保持相同瓶子、构型、步行流程和评价窗口。
左臂作为当天身体扰动的辅助参照。若改了构型，要给 PID/MPC 相同构型后才能归因于控制器。
仅右臂收近、左臂不变可以研究“控制器+构型”的组合效果，但要说明条件不同。

## 新增线速度代价具体是什么

参数 `q_ee_vel` 默认 **0**；B 候选为 **0.1**（XYZ 等权，单位尺度按 m/s）。
已有 `q_vel=0.08` 是五关节的关节速度代价，二者不相同。

当前 IMU/学习预测接口给出姿态、角速度、角加速度和去重力线加速度，没有可靠的绝对平移速度。
本轮选择可由现有输入计算的量：

```text
v_rel = v_endpoint - v_torso_IMU
      = omega_IMU × r_IMU_endpoint + J_v(q) dq       （均在冻结 H0 中表达）
J_velocity = Σ v_rel(k)^T Q_ee_vel v_rel(k)
             + terminal_scale * v_rel(N)^T Q_ee_vel v_rel(N)
```

这里减去的是 IMU 原点的平移速度，坐标轴不随躯干旋转；因此身体转动的杆臂项仍被保留。
它不是只惩罚 `J_v*dq`，也不是要求行走时瓶子在世界中停住。没有积分 IMU 来猜长期速度，
没有把行走速度指令当成实际速度。模型仍使用已有 `right_grasp_site` 末端点。
每个预测节点包括终端节点都有此代价，沿用现有局部 Jacobian 近似。

`0.1` 是小步试验起点，不是已证明的最优值。例如 `|v_rel|=0.1 m/s` 时该节点贡献为 `0.001`。
它主要尝试抑制身体转动与关节运动造成的末端摆动；身体整体平移引起的冲击并不直接进入此速度误差，
所以不能保证水平加速度降低，也可能限制有用的补偿。原来的绝对末端线加速度代价仍在。
新增代价不是机械阻尼，也不表示已实现冲击能量吸收。

每拍日志 `mpc.one_step_prediction` 新增
`ee_lin_vel_relative_imu_h0_m_s`、对应偏置／Jacobian 和
`cost_terms.linear_velocity_relative_imu`；权重与量的定义也进入元数据。
旧配置为零时不增加这项数学代价，也不计算新增速度任务项。

## 配置与命令

| 组 | `--mpc-config` |
| --- | --- |
| A | `configs/hardware_mpc_learned_acc001_alpha0005_omega1.yaml` |
| B | `configs/hardware_mpc_learned_omega1_vel01.yaml` |
| C | `configs/hardware_mpc_learned_omega1_acc_y0015.yaml` |

以下本地预检不创建 DDS，可用于核对候选 B：

```bash
taskset -c 0-17 /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/g1_walk_mpc_learned.py --preflight --cpu 7 \
  --mpc-config configs/hardware_mpc_learned_omega1_vel01.yaml \
  --torque-config configs/hardware_mpc_torque_learned.yaml
```

2026-10-09 已按上面精确组合通过 preflight：`solver_status=solved`、
`q_ee_vel=0.1`、不含 roll 回正，且
`dds_initialized=false`、`publisher_created=false`、`robot_connected=false`。

下面是**真实输出模板**，只在操作者明确连接机器人并要求运行后使用。
它保留现有 profile 核验、真实输出许可、现场确认、停车与退权流程。
填写组对应的配置和标签；B 第一次先令 `MPC_TRIAL_TASK=stationary`，确认接管正常后再用 `walk`。
不要因为下面出现命令就自动执行。

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
sudo -v
MPC_TRIAL_CONFIG=configs/hardware_mpc_learned_omega1_vel01.yaml
MPC_TRIAL_LABEL=vel01_noroll
MPC_TRIAL_TASK=stationary
MPC_TRIAL_OUT="evaluation/hardware_shadow/commissioning/mpc_${MPC_TRIAL_LABEL}_${MPC_TRIAL_TASK}_$(date +%Y%m%d_%H%M%S)"
taskset -c 0-17 /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/g1_walk_mpc_learned.py enx6c1ff701509c \
  --execute --task "$MPC_TRIAL_TASK" --predictor learned_filtered \
  --mpc-config "$MPC_TRIAL_CONFIG" \
  --torque-config configs/hardware_mpc_torque_learned.yaml \
  --cpu 7 --rt-priority 20 --compute-process \
  --profile evaluation/hardware_shadow/commissioning/g1_pid_6ms_20261005_132052/arm_profile.conf \
  --output-dir "$MPC_TRIAL_OUT" --pid-6ms-validated --torque-stationary-validated \
  --permit-real-output MPC_WALK_H0_CAPTURE --allow-first-torque-field-trial
```

回到历史成功版本：使用 `configs/hardware_mpc_learned_acc001_alpha0005_omega1.yaml`
与 `configs/hardware_mpc_torque_learned.yaml`，左臂使用默认 `--left-pd-gain-scale 1`。

## 每轮看哪些结果

主窗口仍是完整走停 `[5,18)`；另外分开看行走 `[5,15)` 与停车 `[15,18)`。
报告左右瓶轴倾角、H0 XYZ／水平／三维线加速度、角速度和角加速度的 RMS/P95/峰值，
并同步查看身体 IMU 扰动、右肩 roll 漂移、pitch/yaw 边界余量、力矩余量、求解失败／回退、
6 ms 耗时与超期、停车和最终 `weight=0`。B 额外看新的末端相对线速度。
视频机位、瓶子水量和固定方式保持一致，不用加速度单项代替液面视觉观察。
吊绳牵拉整轮标作无效；出现擦碰／异响或异常退出先定位原因，不自动重跑。

## 左臂增益实验的边界

仿真 200% 使左端水平加速度约增加 5.6%，同时倾角反而减小，且发生一次右 MPC QP 回退。
这是单次完整机器人仿真的现象，不能断定实机加硬也按此比例变化。
仿真原值为 `Kp=[80,80,60,80,30]、Kd=[5,5,3,2,1]`；真机左臂已验证值为各关节 `Kp=20、Kd=1`。
**不能将仿真的 200% 数值直接搬到真机。**

当前 profile 加载器仍先核验上述真机已验证增益，原 profile 文件不改。按操作者 2026-10-09
的明确决定，学习 MPC 入口保留显式 `--left-pd-gain-scale 2`：核验后只将左臂五个 Arm SDK
槽位的发包 Kp/Kd 变为 `40/2`，右臂、腰和无该参数时的默认行为不变。会话日志记录原 profile
哈希、有效增益和倍率。倍率可选 `1、1.5、2`；默认 `1`，原 2 倍快捷参数已删除。

`122828` 已完成这一敏感性实验：左瓶倾角更小、水平加速度更大，左右视觉姿态差距缩小。
提高左臂加速度让右臂“显得更好”，并不证明右臂绝对效果改善。主 PID/MPC 对照、已完成 B
和后续主要候选继续使用正常参数。

## 本地验证记录

见 `evaluation/mpc_velocity_preparation_20261009/` 的离线回放与汇总，
以及 `tools/g1_commissioning/tests/test_mpc_linear_velocity.py`。
验证涵盖运动基座几何有限差分、逐点／批量与终端节点一致性、代价二次型、零权重兼容、
参数隔离及计算子进程一致性。回放使用 164546 成功日志、禁止网络，包含接管和完整退权。
这些检查只证明软件和给定日志上的数学／力矩检查通过，不是新的真机物理效果或实时性验证。

相关测试共 92 项通过：原硬件 MPC／学习版 27 项，以及新增速度、力矩约束、执行、
实测状态力矩、现场准入和计时容错 65 项。可复现命令（不连接机器人）：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/g1_mpc_matplotlib \
  PYTHONPATH=.:tools/g1_commissioning:tools/g1_commissioning/tests \
  /home/fjk/miniforge3/envs/g1_mpc/bin/python -m unittest \
  test_hardware_mpc test_mpc_learned test_mpc_linear_velocity \
  test_mpc_actuation_constraints test_mpc_execution test_measured_torque_mpc \
  test_mpc_field test_mpc_timing_grace -q
```

已完成回放目录：`evaluation/mpc_velocity_preparation_20261009/run_20261009_010708/`。
源日志 SHA256 为 `c1400bd114cba1890278a4d27d4726e95c9c670fe92d0941eb83974a80b90919`。
该批回放生成于 A 完成前，三组共用当时的 roll 配置，主要验证新增速度数学、日志和执行链；
它不是下一轮无 roll B 的物理效果证据。三组各完成 2445 个活跃 MPC 控制拍，最终 `weight=0`，无计算拒绝；
计划加速度与最终模型力矩反算的最大差异均小于 `4.8e-7 rad/s²`。

| 回放组 | 完整局部计算 P99 / 最大值 | `[5,18)` 局部计算超 6 ms |
| --- | --- | --- |
| A | 5.105 / 5.859 ms | 0 / 2162 |
| B | 3.338 / 3.768 ms | 0 / 2162 |
| C | 3.469 / 5.817 ms | 0 / 2162 |

这次运行环境允许的 CPU 集合不含 7，所以回放选择可用 CPU 0；没有改变主机 affinity 配置。
不同组的时间差包含调度／预热差异，不证明新增代价让求解更快；不包括真实 DDS、IPC 和日志 IO。
明天仍需按既有现场准备检查 CPU 7 与整链耗时。

同一日志状态下，B 相对 A 的单关节总力矩最大变化约 `0.00854 N·m`，C 为 `0.17689 N·m`。
因此 B 的 `0.1` 是很轻的起点，不能期待已证明明显收益；这两项是命令差异，
不是末端加速度的物理改善量。若现场 B 变化太小，先保留完整结果，再决定下一独立权重。
