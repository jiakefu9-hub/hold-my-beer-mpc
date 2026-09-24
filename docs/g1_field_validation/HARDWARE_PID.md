# G1 真机 PID 持瓶实验

这是一条独立的首次真机 PID 路径，不改 production adapter，也不启用 `rt/lowcmd`、MPC 或
整机底层接管。2026-09-18 的首次真机调试在约 5.35 秒中止：右臂明显抖动，速度 RPC
故障后又发生了过快交还。随后完成了两轮 `0.08 rad/s` 修复复验和一轮 `0.07 rad/s` 单变量复验，
现场均未见抖动、突变或突然放下；日志也显示正常三秒退权。位置边界减速、航向保持延续到 18 秒及
当前 `0.07 rad/s` 上限都已有一次完整真机运行记录。分析结束后仍恢复 `FIELD_OUTPUT_LOCKED=True`，
避免误重跑。这些均为 **20 ms** 实机结果，保存于提交 `a1d0197cbe28849eeec69b19a9d41d19eb5818d5`。
2026-09-24 当前代码已准备为 **6 ms**，通过离线回放和几何／时序测试；尚未进行 6 ms 实机复验。

失败轮保存了逐周期目标角／目标速度、实测角度／速度、weight、IMU、LowState、PID 重力误差、
DDS 与 RPC 时间和故障事件。原始记录位于
`evaluation/hardware_shadow/commissioning/g1_pid_20260918_153938/run3/raw.jsonl`。

## 这次到底控制什么

- 机器人由操作者先放到双脚着地、原地自主平衡的 Regular Motion Mode（查询值 FSM 500）。
  程序不切模式。
- 左臂一直用关节 PD 保持已经实测过的非零 A3 姿态：肩 pitch −4°、肩 roll −1°、
  肘 pitch −8.1°，其余臂轴为 0°。
- 右臂采用仿真中的 `ArmPIDPolicy`：目标每 6 ms 读取最新身体 IMU 和右臂关节角，使用与仿真相同的
  MuJoCo XML、`right_grasp_site` 瓶身中点及重力方向误差，生成右臂五关节 `q_ref/dq_ref`。
  右臂的名义姿态不是全零，而是肩 pitch −4°、肩 roll +1°、肘 pitch −7.8°。
- 单腰 yaw 固定为 0°。Arm SDK 有效槽继续使用现场已经验证过的 `kp=20、kd=1`。
- 右臂生成的位置参考被限制在上述名义姿态每个关节 ±5° 内；这只是命令范围，不是“检测到某个
  瞬时关节速度就退出”的门槛。程序没有新增关节速度停止条件。
- 仿真 PID 原始输出仍保留 `pid_max_dq=0.48 rad/s`，但真机输出前新增独立 governor：
  当前复验后采用 `hardware_pid_max_dq=0.07 rad/s`、`hardware_pid_max_ddq=0.20 rad/s²`。
  20 ms 前两轮完整复验使用的是 `0.08 rad/s`；第三轮降低到 `0.07 rad/s`。它同时限制速度和
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
| PID 更新周期 | 6 ms | 目标 6 ms；已实测版本为 20 ms，新版待实机验证 |
| PID 时间语义 | 原任务修正除以 6 ms；滤波 alpha=0.07/周期 | 保留已验证 20 ms 控制强度与滤波时间常数，见下文；不是完全相同的增益／带宽 |
| 身体／关节状态 | MuJoCo 真值 | `rt/secondary_imu` 与 `rt/lowstate` 实测值 |
| 末端姿态 | MuJoCo 可直接读 site | 用同一 XML 对实测关节做 FK，再与身体 IMU 合成 |
| 左臂／右臂名义角 | 仿真配置决定 | 使用当前实机 A3 非零持瓶姿态 |
| 下层执行 | 仿真执行器 | `rt/arm_sdk` 的 q/dq + `kp=20,kd=1`；不发扭矩、不发 `rt/lowcmd` |
| 末端加速度 | MuJoCo ground truth | IMU + FK 离线估计，需重采样和滤波 |
| 评价区间 | 既有报告常取中间 evaluation | 固定采用完整 `[5,18)` |

所以这次结果可用于判断“真机 PID 是否改善持瓶”，但真机滤波后的加速度数值不能直接当作与 MuJoCo
ground truth 完全相同的传感器。第一轮先验证闭环方向、是否平顺、是否经常触及 ±5° 参考范围，
不追求通过反复调参使 PID 达到 MPC 的效果。

## 6 ms 版具体改了什么

1. 每周期一次 `mj_kinematics + mj_comPos + mj_jacSite`，同时得到误差与解析雅可比，代替原来
   多次扰动关节做中心差分。公式是 `J_g = (g_E × J_omega_E)[:2]`；120 个随机姿态与中心差分对照。
   依据 [MuJoCo Jacobian API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jacsite)。
2. 保留普通任务空间 PID 结构，Ki 仍为 0。旧实现的任务修正量再除以 dt；真机现在固定除以
   0.020 s，使改变刷新周期不会额外放大任务速度 3.33 倍。误差差分与积分使用实际循环间隔。
3. `alpha(dt)=1-(1-0.07)^(dt/0.020)`，6 ms 时约 0.021535，保留原约 0.276 秒滤波时间常数。
   仿真入口没有启用这两项时间换算，行为保持原样。相同 6 ms 只统一更新频率，不表示物理带宽完全一致。
4. 使用单调时钟的固定时间槽；超时就跳过过期槽，不追赶补发。命令积分间隔至多为 6 ms，
   卡顿后也不放大单帧位移。位置 ±5°、目标速度 0.07 rad/s、目标加速度 0.20 rad/s² 保留。
5. 正常、Ctrl-C、可处理故障继续至少三秒退权。逐帧最大 weight 降幅改为 `0.006/3=0.002`，
   卡顿只延长退权；失联等既有例外仍见下文。
6. 数值库单线程，控制线程绑一个 CPU（`--cpu`；默认优先 7，否则可用列表首个），保持普通 Linux
   调度，不修改系统实时策略。既有 DDS／日志线程独立运行。当前离线环境不允许 CPU 7，验证使用 CPU 2；
   因此现场命令示例显式给出 `--cpu 2`。它与仿真的 CPU 7 不是同核测量，不冒充严格同环境性能对比。

全程新增 `g1_pid_timing_v1`：实际间隔、唤醒延迟、含命令日志入队的循环耗时、deadline miss、
跳过周期、状态／IMU 在写入结束时的接收年龄、两者时间差、重复使用测量的次数、DDS 写调用耗时。
单独 timing 行入队不计入 work 指标，但实际相邻循环间隔会反映它；DDS write 返回不代表电机已执行。
原始 LowState 完整电机记录仍约 20 ms 一条，躯干 IMU 约 5 ms 一条；每条 6 ms 控制命令额外保存
所用双臂实测 q/dq、输入时间戳、目标和 PID 中间量。名义 6 ms 不保证每次都有新传感器包。

下一步只需一轮 6 ms 实机复验：继续观察平顺性、停车和退权；离线看完整 `[5,18)` 指标、
实际周期／漏周期、状态年龄和进程正常退出。若现场循环大部分时间仍超出 6 ms，才定位 DDS 回调等
耗时并考虑 C++；当前不增加迁移、实时内核或新控制器。

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
  --cpu 2 \
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
审核本文件的 6 ms 离线结果，并将下一轮作为 6 ms 首次实机复验。

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

这些测试没有连接机器人。三轮 20 ms 修复版真机复验均完整运行且现场未见抖动和突然放下；右瓶竖直倾角
稳定优于左瓶，但动态指标只小幅改善。当前 `0.07 rad/s` 单变量候选已通过一轮真机复验，仍不能把
单轮候选当作完整统计结论；继续比较 PID/MPC 时应保留原始记录并使用相同指标。

6 ms 离线复现（不创建 DDS participant，不连接机器人，只用 IDL 构造和序列化包）：

```bash
/home/fjk/miniforge3/envs/g1_mpc/bin/python -m unittest discover \
  -s tools/g1_commissioning/tests -p 'test_*pid*.py'

/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/g1_commissioning/benchmark_hardware_pid.py \
  evaluation/hardware_shadow/commissioning/g1_pid_tune_dq007_20260918_170950/raw.jsonl \
  --cpu 2 --output-dir /tmp/g1-pid-6ms-new-replay
```

输出目录必须不存在；完整结果及限制见 [6 ms 离线记录](sessions/20260924_PID_6MS_OFFLINE.md)。
