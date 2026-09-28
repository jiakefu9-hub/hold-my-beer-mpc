# 真机 MPC 逆动力学候选：先补执行模型，再验证真实响应

日期：2026-09-28。保留基线 `3e63977a763ea3f905d67359e7d523ead6119f63`。

**本文保留早期 `inverse_dynamics_preview` 对照及其历史测量。当前主线见
[实测状态与多候选力矩迁移](HARDWARE_MPC_TORQUE_MIGRATION.md)。** 新版不再采用本文的持续参考状态，
已加入条件正动力学候选筛选和离线过渡设计；两种力矩候选仍都禁止真实输出。
本文末尾 62 项测试及时间数字对应该阶段记录，不表示后续修改后的重新测量。

**当前是离线候选，不是已经开放的真机力矩控制程序。** 已实现 MPC 加速度到逆动力学力矩的
计算、离线消息构造、模型对照、旧实测力矩分析；没有发送非零力矩，没有让机器人自行标定。
原有 `reference_servo` 和 PID 的真实输出行为保持不变。

## 为什么改这个方向

用户记得的“积分一拍再交给 PD，手臂无力”有仓库记录：
[历史问题](../history/CHALLENGE.md)描述的是早期 LQR，每拍目标紧贴实测状态，PD 误差很小，
连重力支撑也不足。当前真机 reference-servo MPC 持续保存参考，不是那个逐拍重置版本；
但它仍未标定真实伺服动态，因此不能保证预测的加速度在真机实现。

本次选择的方向是 **逆动力学前馈 + PD 纠偏**，不是取消所有反馈的开环力矩控制：

```text
MPC 参考加速度 → 参考约束后的加速度
                         ↓
实测关节 q/dq + 身体 IMU 运动 + 负载模型 → 逆动力学 tau_ff
q_ref / dq_ref / Kp / Kd ───────────────→ 固件 PD（只加一次）
```

候选计算 `tau_ff = M(q)·ddq_ref + h(q,dq,身体运动)`，
预期固件总力矩为 `tau_ff + Kp(q_ref-q) + Kd(dq_ref-dq)`。
PD 项只用于记录预计总量，**没有再次塞入 tau_ff**。
因此即使逆动力学完全准确，PD 纠偏也会改变实际加速度；不能把两者同时存在解释成精确加速度源。

当前候选仍复用已有参考轨迹 QP、目标、约束与跟踪误差几何修正，只替换／检查其执行层假设；
没有伪称已经实现经实机辨识的关节加速度闭环。

## 官方支持与不能照搬的部分

- [官方 G1 Arm5 示例](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/g1/high_level/g1_arm5_sdk_dds_example.cpp)
  发布 `q/dq/kp/kd/tau` 到 `rt/arm_sdk`，其示例 tau 默认仍为零。
- [官方 G1_23 遥操作控制器](https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/robot_arm.py)
  的 motion 分支使用 `rt/arm_sdk`，并写入 `tauff_target`；
  [对应 IK](https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/robot_arm_ik.py)
  使用 Pinocchio RNEA 计算前馈。它的零速度／零加速度重力补偿不等于本项目行走扰动补偿已验证。
- 仿真还使用 MuJoCo 局部正动力学试算、接触约束以及力矩修正。
  真机没有真实全状态／接触解算器，**不能用仿真器试算值冒充实测加速度**。
  本文早期候选未移植这层修正；后续迁移版已加入条件模型修正，仍不复制仿真摩擦参数。

这些资料支持接口与设计方向，不证明这台机器的 tau 比例、精度或当前固件响应。
不需要因为采用上肢力矩前馈就改成全身 `rt/lowcmd` 或进入 debug。

## 程序对应关系

| 文件 | 用途 |
| --- | --- |
| [hardware_arm_inverse_dynamics.py](../../tools/g1_commissioning/hardware_arm_inverse_dynamics.py) | 身体 IMU 移动基座 → 右臂 C++ RNEA；无 SDK |
| [hardware_mpc_inverse_preview.py](../../tools/g1_commissioning/hardware_mpc_inverse_preview.py) | 复用 MPC，计算前馈、单独记录 PD，构造仅供离线查看的非零 tau 消息 |
| [g1_walk_mpc.py](../../tools/g1_commissioning/g1_walk_mpc.py) | `--actuation inverse_dynamics_preview` 仅允许 preflight；`--execute` 拒绝此候选 |
| [benchmark_hardware_mpc.py](../../tools/g1_commissioning/benchmark_hardware_mpc.py) | 回放旧轨迹，测量新增逆动力学后的完整离线时间 |
| [analyze_arm_torque_feedback.py](../../tools/g1_commissioning/analyze_arm_torque_feedback.py) | 离线核对 tau_est、PD 估算、静差与模型重力矩；不自动拟合／启用补偿 |

目前没有真机力矩接管／退出曲线，也没有经现场核对的力矩幅度与变化率参数。
离线候选在非 MPC 阶段的 tau 为零，因此它的阶段拼接**不能拿来直接向机器人发送**。
除了命令行拒绝，底层 `run_device` 也拒绝 offline-only 控制器；原有真实消息构造仍固定 tau=0。

## 坐标、负载与力矩计算

逆动力学使用当前节点的 H0 姿态、角速度、角加速度，以及 **torso IMU 点去掉重力后的线加速度**。
这些量来自已有因果滤波器，仍有滤波延迟；不把未来预测值当成当前真实状态。

把上游腰／腿关节冻结在虚拟模型里，同时重构浮动根运动，使模型 torso IMU 的姿态和运动与输入一致。
不是假定真实腰、腿静止，也没有假造下肢加速度。IMU 到虚拟根的杠杆项显式计算：

`a_root = a_IMU − alpha × r − omega × (omega × r)`。

身体运动已包含腿部反作用对躯干的影响，但右臂未来反作用如何影响内置步态仍是未建模的闭环耦合。
手部额外接触力假定为零。

从当前 XML 读取瓶子质量 **0.25 kg／只**、质心和惯量；用户已明确实物左右各绑一个 250 g 水瓶。
质量相符不代表实物绑法、质心及惯量已标定。
XML 的五关节 armature 均为 `0.01 kg·m²`，也没有经过真机辨识。
原生 RNEA 已包含该项；结果单列 `tau_rigid_nm / tau_armature_nm / tau_model_nm`，不重复添加。

## 旧数据实际告诉了我们什么

对当前十二条 H0 数据中的 trial01、06、12 做可复现抽样：每 100 条 LowState 取一条，
与此前最近的命令和 torso IMU 做时间戳匹配（不使用未来样本；两者年龄均 ≤100 ms）。
仅使用 weight=1 的 `[3,18)`；不是全采样率尖峰审计。

- 这三个记录中 tau_est 明显非零；抽到的值落在 `0.0625 N·m` 网格上。
  这是采样观察，**不是精度 ±0.0625 的官方保证**。
- 3.5–5 秒低关节速度的站立样本中，右肩目标 −4°、实测均值约 +1.67～+1.91°；
  右肘目标 −7.8°、实测约 −0.59～−0.49°。旧 PD 确实有明显静差。
- 同段肘关节 tau_est 均值约 −2.37 N·m，0.25 kg 模型重力矩约 −1.83 N·m，差约 0.54 N·m。
  不能直接把这个差拟合成“电机少输出 23%”：负载、惯量、摩擦、估计偏差等都可能混在其中。
- 抽到的 `ddq_raw` 全为零；**不能用它声称真实加速度为零或验证 MPC 加速度跟踪**。
  后续应从带时间戳的 dq 做有噪声／有延迟说明的加速度估计。

重要后果：原有持瓶目标是带着 PD 静差调出来的。加入重力补偿后，同一组数字可能让手臂移到另一姿态。
因此不能把旧目标、满额前馈和原有 PD 直接叠加，就宣称“动力学修好了”。
真正接管前必须核对名义实测持瓶姿态，并设计不突然改变总力矩的过渡。

三轮的抽样数分别为 156、154、155；原始日志 SHA256、抽样条件及各关节统计保存在
[力矩反馈摘要](../../evaluation_summary/hardware_mpc_inverse_20260928/torque_feedback.json)。

## 没有测力计，能否自动辨识

**可以做动态辨识，不等于完成绝对力矩标定。** 后续可在静止平衡、人员监护下，
用小幅、低速、可停止的参考动作记录 q/dq、tau_est、tau_ff 与时间戳，估计响应延迟、
重力静差和部分摩擦／惯量参数。无需先购买测力计才能开始这类工作。

但单靠机器人自身的 tau_est，无法独立验证它自己的绝对力矩尺度；不能把“命令20、估计18”
直接变成自动乘以20/18的增益。已知质量／力臂、多静态姿态、加载／卸载比较能提供额外约束；
实物总重可先用普通秤确认，不必是测力计。需要高精度绝对力矩时才进一步考虑独立测力设备。

本次只实现离线分析，没有写入自动激励／自学习补偿，更没有让机器人自己动。
实物负载质量已经明确。后续现场仍应从**静止的重力补偿渐入／渐出与响应检查**开始，
通过后才接入完整加速度前馈与行走；原有 PID 对照保留。

## 本机离线复验

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
conda activate g1_mpc
python tools/g1_commissioning/g1_walk_mpc.py \
  --preflight --actuation inverse_dynamics_preview --cpu 2

python tools/g1_commissioning/benchmark_hardware_mpc.py \
  --actuation inverse_dynamics_preview \
  --source-npz evaluation/hardware_shadow/commissioning/walk_h0_predictor_study_20260925/data/trial10.npz \
  --output-dir evaluation/hardware_shadow/commissioning/inverse_preview_recheck \
  --runs 2 --cpu 2

python tools/g1_commissioning/analyze_arm_torque_feedback.py \
  evaluation/hardware_shadow/commissioning/g1_walk_h0_heading18_20260918_150839/raw.jsonl \
  evaluation/hardware_shadow/commissioning/g1_walk_h0_heading18_20260918_151919/raw.jsonl \
  evaluation/hardware_shadow/commissioning/g1_walk_h0_heading18_20260918_153034/raw.jsonl \
  --stride 100 --output evaluation/hardware_shadow/commissioning/torque_feedback_recheck.json
```

输出目录必须是新目录。回放反馈不会响应本次计算的力矩，所以只能测计算与消息路径，不能证明控制效果。

## 本轮验证结果与停止位置

**62 项相关测试通过**，其中新增十项。包括 50 组随机状态下 C++ RNEA 对独立 MuJoCo 质量矩阵／
偏置力的对照（容差 `1e-10 N·m`），及同模型、给定基座运动下加速度反算对照（容差 `1e-9 rad/s²`）。
也验证了 H0 yaw 旋转不改变关节力矩、IMU 杠杆项、无重复 PD、候选不能进入真实运行器。
这些是数学一致性容差，不是 G1 电机精度。

使用 trial10、CPU 2、BLAS 单线程进行两轮完整定时回放：

| `[5,18)` 指标 | 第 1 轮 | 第 2 轮 |
| --- | ---: | ---: |
| 逆动力学单独耗时 p99 | 0.251 ms | 0.230 ms |
| MPC 控制器（含逆动力学）p99 | 2.734 ms | 2.726 ms |
| 完整离线单拍工作 p99 | 4.866 ms | 4.647 ms |
| 完整离线单拍工作最大值 | 7.821 ms | 11.808 ms |
| 实际周期最大值 | 12.021 ms | 13.241 ms |
| 超过 6 ms 截止时间／样本数 | 3 / 2163 | 5 / 2161 |

两轮均正常完成，日志无失败／丢弃，参考轨迹独立审计通过；共 8 / 4324 拍超时，约 0.185%。
时间包括本地消息序列化／CRC和日志入队，不包括 DDS 收发、真实 RPC 竞争或电机响应。
主机内核 PREEMPT_RT，但进程仍为普通 SCHED_OTHER；没有修改系统设置。

离线候选的右臂前馈最大绝对值约为 `[4.03,3.49,1.80,3.79,0.15] N·m`；
若与冻结旧反馈下的 PD 直接叠加，肘关节预计总量可达约 `8.93 N·m`。
**这是回放中的假设合成值，不是实机力矩，也不是批准的安全限值**；旧反馈不会响应新命令，
不能用它预测真实闭环峰值。但它再次说明不能直接把旧 PD 与新前馈拼接成现场控制器。

preflight 实际构造了包含非零右臂 tau 的 1004 字节消息，CRC 检查通过；未创建发布器。
源码／原始日志哈希和完整结果见[验证摘要](../../evaluation_summary/hardware_mpc_inverse_20260928/validation.json)。
旧现场输出计算方式不变，但当前 CLI 默认已改为离线 `measured_torque_preview`；
旧参考基线需显式选择。本阶段只证明**可计算、可复验的离线执行层候选**，不证明真机加速度跟踪完成。
