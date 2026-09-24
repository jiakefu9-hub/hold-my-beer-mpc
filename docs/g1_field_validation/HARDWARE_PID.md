# G1 真机 PID 持瓶实验

这是一条独立的首次真机 PID 路径，不改 production adapter，也不启用 `rt/lowcmd`、MPC 或
整机底层接管。2026-09-18 的首次真机调试在约 5.35 秒中止：右臂明显抖动，速度 RPC
故障后又发生了过快交还。随后完成了两轮 `0.08 rad/s` 修复复验和一轮 `0.07 rad/s` 单变量复验，
现场均未见抖动、突变或突然放下；日志也显示正常三秒退权。位置边界减速、航向保持延续到 18 秒及
当前 `0.07 rad/s` 上限都已有一次完整真机运行记录。分析结束后仍恢复 `FIELD_OUTPUT_LOCKED=True`，
避免误重跑。

失败轮保存了逐周期目标角／目标速度、实测角度／速度、weight、IMU、LowState、PID 重力误差、
DDS 与 RPC 时间和故障事件。原始记录位于
`evaluation/hardware_shadow/commissioning/g1_pid_20260918_153938/run3/raw.jsonl`。

## 这次到底控制什么

- 机器人由操作者先放到双脚着地、原地自主平衡的 Regular Motion Mode（查询值 FSM 500）。
  程序不切模式。
- 左臂一直用关节 PD 保持已经实测过的非零 A3 姿态：肩 pitch −4°、肩 roll −1°、
  肘 pitch −8.1°，其余臂轴为 0°。
- 右臂采用仿真中的 `ArmPIDPolicy`：每 20 ms 读取当前身体 IMU 和右臂关节角，使用与仿真相同的
  MuJoCo XML、`right_grasp_site` 瓶身中点及重力方向误差，生成右臂五关节 `q_ref/dq_ref`。
  右臂的名义姿态不是全零，而是肩 pitch −4°、肩 roll +1°、肘 pitch −7.8°。
- 单腰 yaw 固定为 0°。Arm SDK 有效槽继续使用现场已经验证过的 `kp=20、kd=1`。
- 右臂生成的位置参考被限制在上述名义姿态每个关节 ±5° 内；这只是命令范围，不是“检测到某个
  瞬时关节速度就退出”的门槛。程序没有新增关节速度停止条件。
- 仿真 PID 原始输出仍保留 `pid_max_dq=0.48 rad/s`，但真机输出前新增独立 governor：
  当前复验后采用 `hardware_pid_max_dq=0.07 rad/s`、`hardware_pid_max_ddq=0.20 rad/s²`。
  前一轮完整复验使用的是 `0.08 rad/s`；这次只把最终速度上限降低 12.5%，其余 PID 增益不变。它同时限制速度和
  相邻周期速度变化，并从进入 PID 时的实测关节角起步。逐帧同时记录原始 PID 输出和 governor
  之后真正准备发送的输出。

第一版直接读取当前仿真参数：任务误差 `Kp=[1.2,1.2]`、`Kd=[1.2,1.2]`、`Ki=[0,0]`，
五关节姿态正则为 `[1.15,1.15,2.10,1.15,0.95]`，阻尼伪逆系数 0.15。也就是说代码框架是 PID，
但首次真机基线仍关闭积分项，实际先做任务空间 PD；不会在现场临时猜一组新增益。

首次故障数据的离线回放共有 115 个 PID 周期。旧版速度命令中 61.2% 达到 ±0.48 rad/s，
五关节合计出现 137 次符号反转，单周期目标角最大跳 0.550°。相同实测输入经过真机 governor 后，
当时 `0.08 rad/s` 修复版回放的最大速度为 0.076 rad/s、最大加速度 0.20 rad/s²、反转 12 次、单周期目标角最大跳 0.087°，
全部目标仍在名义角 ±5° 内。该结果只验证命令生成被平滑，不能代替下一次真机验证。

右臂误差仍是瓶子相对重力竖直方向的二维误差：

`e_g = (R_H0E^T · [0, 0, -9.81])[:2]`，其中 `R_H0E` 把末端向量转到 H0。

这里固定 H0 的 Z 轴就是重力竖直方向，所以 yaw0 不会改变该竖直误差；H0 仍用于统一记录姿态、
线加速度、角加速度和行走方向。FK 的输入来自真机实测关节角与身体 IMU，不是直接读取一个不存在的
“真机末端姿态传感器”。

## 一轮程序的时间表

| 程序时间 | 动作与控制 |
| --- | --- |
| 0–3 秒 | 双臂进入上述 A3 姿态，weight 从 0 到 1；速度为零 |
| 3–5 秒 | 左臂固定，右臂 PID 开始；原地记录 yaw，得到本轮 H0 |
| 5–15 秒 | 请求 0.5 m/s 向 H0 +X 行走，并做航向保持；左右臂控制继续 |
| 15–18 秒 | 前进速度为零，但航向保持继续工作；双臂继续控制并记录减速和停车扰动 |
| 18 秒后 | 航向修正归零；收到零速度回复后冻结手臂参考，weight 至少用三秒从 1 降到 0 |

主评价区间只有一个：**`[5,18)`，从开始走到停车等待结束的全过程。** 它包含起步、正常行走、
下发零前进速度后的减速和三秒 stop-settle，不会像部分仿真报告那样只挑中间稳定段。航向保持
在 `[15,18)` 仍可发送非零 yaw-rate 修正，到 18 秒才归零。`[5,7)`、
`[7,15)`、`[15,18)` 只生成辅助诊断，不能代替主结果。零速度 RPC 成功和三秒等待并不能单独证明
机器人在物理上已经完全静止，因此原始腿部状态也一直保留。

## 与仿真 PID 的主要区别

| 项目 | 仿真 | 这版真机程序 |
| --- | --- | --- |
| PID 更新周期 | 6 ms | 20 ms，先与已验证的 Arm SDK 频率一致 |
| 身体／关节状态 | MuJoCo 真值 | `rt/secondary_imu` 与 `rt/lowstate` 实测值 |
| 末端姿态 | MuJoCo 可直接读 site | 用同一 XML 对实测关节做 FK，再与身体 IMU 合成 |
| 左臂／右臂名义角 | 仿真配置决定 | 使用当前实机 A3 非零持瓶姿态 |
| 下层执行 | 仿真执行器 | `rt/arm_sdk` 的 q/dq + `kp=20,kd=1`；不发扭矩、不发 `rt/lowcmd` |
| 末端加速度 | MuJoCo ground truth | IMU + FK 离线估计，需重采样和滤波 |
| 评价区间 | 既有报告常取中间 evaluation | 固定采用完整 `[5,18)` |

所以这次结果可用于判断“真机 PID 是否改善持瓶”，但真机滤波后的加速度数值不能直接当作与 MuJoCo
ground truth 完全相同的传感器。第一轮先验证闭环方向、是否平顺、是否经常触及 ±5° 参考范围，
再决定是否调整增益或提高频率。

## 现场运行

程序依赖当前 `/home/fjk/g1_ws/unitree_sdk2_python` 和 `g1_mpc` 环境。先复制
[`pid_walk_capture.template`](../../tools/g1_commissioning/profiles/pid_walk_capture.template)，只填写
本次真实确认过的项目。模板故意是 `DRAFT`，不能直接运行，也不要机械地把确认项全改成 true。

机器人已经处于 FSM 500、原地自主平衡，且有线只读状态正常后：

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
NIC=enx6c1ff701509c                 # 以当天实际机器人网卡为准
FIELD_PROFILE=/path/to/reviewed_pid_walk_profile.conf
OUT="evaluation/hardware_shadow/commissioning/pid_$(date +%Y%m%d_%H%M%S)"

/home/fjk/miniforge3/bin/conda run -n g1_mpc \
  python tools/g1_commissioning/g1_walk_pid.py "$NIC" \
  --profile "$FIELD_PROFILE" \
  --controller-config configs/g1.yaml \
  --output-dir "$OUT" \
  --permit-real-output PID_WALK_H0_CAPTURE
```

程序先只接收状态并连续检查 FSM 500；终端要求输入精确的 `EXECUTE <robot_id>` 后再次检查新鲜状态，
之后才创建 `rt/arm_sdk` publisher。它只注册 FSM getter 和速度 API 7105，不注册模式 setter。
L2+B、FSM 离开 500、CRC／状态／IMU 失效、tick 真回退、DDS 写失败或日志溢出都会停止正常流程。
这仍是软件互锁，不是独立急停或安全认证。

当前输出锁开启时，上述命令会在 DDS 初始化和 publisher 创建之前拒绝执行。以后解除锁之前，应先
审核本文件所述 governor、三秒退权、位置边界减速和 18 秒航向保持测试。

## 停止与故障退权规则

- 正常结束、Ctrl-C 和速度 RPC 失败：先请求零行走速度，冻结最后一次已经写出的手臂目标，随后把
  当时的 weight 用完整三秒线性降到零。代码不再发送单帧 `weight=0` 故障帧。
- 退权同时具有逐帧下降上限；即使操作系统在退出中途卡顿，恢复后的下一帧也不能直接跳到零，
  只会把总退权时间延长到三秒以上。
- 如果零速度回复一秒内未确认，记录该事实后仍执行完整三秒退权，避免无限保持手臂 ownership；
  这不等于物理停车已经得到证明。
- 如果 FSM 离开 500、检测到 L2+B、LowState 失效或 DDS 写入失败，程序停止发布，不继续与机器人的
  阻尼／模式切换争抢。断网、进程被强杀或 DDS 本身不可写时，软件客观上无法保证继续发送三秒序列；
  因此“永不瞬时主动写零”可以由代码保证，“任何物理故障都能完成三秒退权”不能虚假保证。

每轮输出目录至少包含：

- `raw.jsonl`：完整 LowState、身体 IMU、FSM、速度请求、H0、逐帧 PID／关节目标和写入耗时；
- `arm_profile.conf`：本轮实际 profile；
- `controller_config.yaml`：本轮实际 PID 参数；
- 两个输入文件的 SHA-256 写在 session header 中；实际导入的 SDK2 Python 路径及关键模块哈希也入日志。

## 离线评价

```bash
/home/fjk/miniforge3/bin/conda run -n g1_mpc \
  python tools/g1_commissioning/analyze_hardware_pid.py "$OUT/raw.jsonl" \
  --output-dir "$OUT/analysis"
```

分析器把身体 IMU、左右瓶中点的姿态、线加速度和角加速度统一转换到本轮固定 H0。输出包括：

- `summary.json`：`[5,18)` 主指标及三个辅助分段；
- `metrics.npz`：完整数值数组；
- `metrics.csv`：便于人工查看的逐时刻结果；
- `endpoint_metrics_h0.png`：左右瓶加速度、角加速度和竖直倾角曲线。

每只手的主指标包括瓶中点三轴线加速度模、对液面横向晃动更直接的 H0 水平线加速度模、角加速度模、
瓶轴偏离 H0 竖直的角度和 upright alignment；
右臂还统计 PID 重力误差、五关节跟踪误差、±5° 参考范围触边比例、控制计算耗时和 DDS 写调用耗时。
默认把接收数据重采样到 200 Hz，再用 105 ms 三次 Savitzky–Golay 窗口估计导数；滤波设置和局限会
写入 `summary.json`，以后做 PD/PID/MPC 对比时必须保持一致。这个居中滤波是离线、非因果评价，
不会喂给在线 PID。所有原始记录仍保留，离线结果不会覆盖 `raw.jsonl`。

## 当前验证状态

离线测试覆盖 H0 跨 ±π 平均、固定参考、非零左臂姿态、右臂 FK/PID、±5° q-reference 约束、
3/2/10/3/3 秒阶段，以及主指标必须完整覆盖 `[5,18)`。运行：

```bash
/home/fjk/miniforge3/bin/conda run -n g1_mpc \
  python tools/g1_commissioning/tests/test_hardware_pid_control.py
/home/fjk/miniforge3/bin/conda run -n g1_mpc \
  python tools/g1_commissioning/tests/test_analyze_hardware_pid.py
```

这些测试没有连接机器人。三轮修复版真机复验均完整运行且现场未见抖动和突然放下；右瓶竖直倾角
稳定优于左瓶，但动态指标只小幅改善。当前 `0.07 rad/s` 单变量候选已通过一轮真机复验，仍不能把
单轮候选当作完整统计结论；继续比较 PID/MPC 时应保留原始记录并使用相同指标。
