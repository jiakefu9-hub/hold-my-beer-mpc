# 2026-10-13 上午：零位位置 PID 与 MPC 学习预测消融

2026-10-13执行结果：以下六轮均已完成，结果和后续录像命令见[六轮结果与最终录像](20261013_MORNING_RESULTS_AND_VIDEO_PLAN.md)。下文保留原准备依据和安排，不代表仍待执行。

目的：在当前展示负载和零中立位下补齐位置 PID 对照；只改变预测模式，比较 MPC 学习预测与当前扰动保持。六轮结束后再分析，不在此预定下午录制或新增调参。

## 顺序与固定条件

| 编号 | 右臂控制 | 相比谁、改什么 |
| --- | --- | --- |
| M01 | 原成功 MPC，learned_filtered | 当天基线，原权重，时间对齐关闭 |
| M02 | 相同 MPC，hold_current | 相比 M01 只改预测器模式 |
| M03 | 零中立位位置 PID | 保留旧 6 ms 位置 PID 增益、参考限速、参考加速度限制 |
| M04 | 重复 M03 | 同配置复测 |
| M05 | 重复 M02 | 当前扰动保持复测 |
| M06 | 重复 M01 | 学习预测复测；检查上午前后变化 |

M01/M06、M02/M05、M03/M04分别完全相同，输出目录独立。这一前后对称顺序有助于观察漂移，但六轮仍是小样本，不当作随机化统计试验。

## 上机前补充离线核查：2026-10-13

保留以上六个编号，不加入新权重、4 ms周期或滤波参数变更。新增分析不修改任何在线控制路径。

- 完整记录输入回放：学习MPC、保持当前扰动MPC、零位位置PID均通过，最终退权到0。没有DDS连接；不能据此声称真机闭环效果或实时性已通过。
- 对10月12日12轮有效记录重算75/105/135 ms评价窗口，共36组。105 ms复算与原指标一致；三种窗口下仍是MPC姿态较好、C角加速度/角速度较低，线加速度差别较小。所有组保持相同评价方法，未选择性改滤波展示优势。
- 三轮原MPC的写出窗口预测核查：对齐使滤波标签的线加速度误差降低23.3%～31.3%、角加速度误差降低27.6%～34.4%；这不是末端闭环改善证据，MPC继续关闭对齐。
- 执行核查：12轮记录模型/指令力矩未超配置界限，无fallback；不能据此证明物理力矩安全或无接触。原MPC三轮期望关节加速度每关节触顶约0.8%～2.6%（各轮比例平均），不支持“几乎一直被加速度上限卡住”。
- 将模型加速度积分与实测dq使用同一离线导数算子比较，肘/腕等关节的运动与模型期望明显不一致；三个窗口均出现差异。不能把幅值比当作执行效率，也未分离模型、摩擦、接触、反馈滤波与传输延迟，暂不据此修改力矩映射。LowState记录的ddq通道在这些段落均为0，不作为实测加速度。
- 原MPC三轮完整工作P99为5.003/5.430/5.509 ms，28/6473拍截止超期；反馈接收时间不等于传感器源时间。保持6 ms，继续逐轮检查停车退权。
- 本次重新运行26项测试全部通过：9项编号/零位PID/入口隔离测试，17项过期反馈、CRC/模式变化、超期、计算进程异常及父进程退权测试。均为离线软件检查，不等于真机故障注入验证。

证据：[完整离线报告](../../../evaluation/hardware_shadow/commissioning/engineering_readiness_20261013/audit/REPORT.md)、[三路径回放](../../../evaluation/hardware_shadow/commissioning/engineering_readiness_20261013/replay/summary.json)。T05继续按既定规则纳入并记录接触，原绊线、初始化失败和窄边界中止排除项不变。

复现审计（输出目录必须是新目录）：

```bash
/home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/analyze_field_readiness_20261013.py \
  --output-dir evaluation/hardware_shadow/commissioning/engineering_readiness_20261013/new_audit
```

## 固定条件

- 共同：6 ms，双臂零中立位，左臂正常固定参考PD 20/1，无人工扰动，75 mm/250 g负载模型，相同实际负载、安装及路线。
- 所有本批实验关闭 `--align-accel-forecast`；位置PID本来没有学习预测或此对齐功能。
- 继续原走停程序：0～3 s抬臂、5～15 s请求0.5 m/s、15～18 s停稳，确认零速后渐退权。位置PID从既有3 s时刻开始姿态反馈；MPC仍是3.3～4.8 s的1.5 s切入。无需新增单独静止接管实验。
- MPC使用原 `hardware_mpc_learned_acc001_alpha0005_omega1.yaml` 和 `hardware_mpc_torque_learned.yaml`；0.01/0.0005/1.0权重不变。
- **不修改成功MPC关节边界、力矩/速度/加速度限制、PD增益或pitch/yaw内部回中。** 不恢复此前导致中止的窄roll边界。
- 原T01～T12入口、164546配置和旧非零位PID保持可用，新入口不替代它们。

`hold_current`只关闭学习预测：保持当前滤波线加速度、角速度、角加速度用于未来；姿态按保持的角速度推进。当前IMU、姿态反馈、重力与移动基座逆动力学均保留。论文称“学习预测消融”，不能称“全部前馈置零”。

## 零位位置PID到底改了什么

新增显式 `--zero-arm-neutral --position-pid-config configs/hardware_position_pid_zero_20261013.yaml`，旧入口默认不变。

1. 两臂中立位和PID内部回中目标为零，不按初始IMU倾角预置非零目标。
2. 原姿态反馈Kp/Kd `[1.2,1.2]`、Ki `[0,0]`，回中增益 `[1.15,1.15,2.1,1.15,0.95]`、滤波与20 ms物理参考时间保持不变。
3. 参考速度上限仍为0.07 rad/s，参考加速度仍为0.2 rad/s²；底层PD20/1、无额外力矩前馈。
4. 不直接将参考角盒缩为零位±5°。旧PID受载时实际用到肘约−10°的参考命令，因此新盒是“旧非零中立位±5°”与“零位±5°”的并集：

| 参考角，度 | 肩pitch | 肩roll | 肩yaw | 肘pitch | 腕roll |
| --- | ---: | ---: | ---: | ---: | ---: |
| 下界 | -9 | -5 | -5 | -12.8 | -5 |
| 上界 | +5 | +6 | +5 | +5 | +5 |

这是**位置PID命令参考角范围**，不等于实测关节安全范围，不是MPC边界。这是零中立位下重新验证的位置PID，不是132052原样重跑；中午同时检查参考限幅，不能因姿态差就默认控制算法必然差。有效profile副本、参考范围和新配置哈希随新数据保存，原profile不写回。

## 接触与数据

按操作者最新要求，轻微手臂—躯干接触不自动作废，MPC和PID/C统一处理，记录发生在哪段，保留实际任务指标；不宣称全部运行无接触。T05 165900继续保留对照地位并注明视频观察到接触。原有控制保护与真实故障处理保持，软件模型接触标记不新增为退出条件。

中午统一 `[5,18)` 完整走停，另分行走和停车段；左右均分析：竖直倾角均值、相对平均瓶轴方向摆动RMS、三维线加速度、角加速度、相对线速度及角速度RMS。附关节轨迹/参考限幅、力矩包络、完整6 ms工作计时及超期、停车退权。角度和动态量来自q/IMU重建，非液体或光学真值。

## 与411次仿真研究的关系

那次为411次尝试、398次完整结束，最终是保守范围、加速度综合、姿态综合三个**仿真**组合。10月12日真机只测试其中部分思路：T02/T04的时间对齐，T10把线加速度权重0.01改为0.02（关闭对齐）；均未带来值得替换原方案的综合收益。

没有原样上机三套组合，也没有加入仿真的姿态预测误差校正、较大权重组合、不同运动范围或OSQP设置。因此结论仅为“昨日已试候选中保留原MPC”，不是“三套仿真方案已经真机失败”，更不是原MPC全局最优。

## 运行

同一终端先设置：

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
sudo -v
G1_PYTHON=/home/fjk/miniforge3/envs/g1_mpc/bin/python
G1_MORNING=tools/g1_commissioning/g1_run_morning_20261013.py
```

每次只运行当前编号的一条，完成停车退权后再手动选择下一条。没有自动连续运行六组的循环。入口自动保留taskset/CPU及原现场确认：

```bash
"$G1_PYTHON" "$G1_MORNING" M01 --execute
```

```bash
"$G1_PYTHON" "$G1_MORNING" M02 --execute
```

```bash
"$G1_PYTHON" "$G1_MORNING" M03 --execute
```

```bash
"$G1_PYTHON" "$G1_MORNING" M04 --execute
```

```bash
"$G1_PYTHON" "$G1_MORNING" M05 --execute
```

```bash
"$G1_PYTHON" "$G1_MORNING" M06 --execute
```

仍会要求输入原确认文字 `EXECUTE SCHOOL_G1_20260915`。默认网卡 `enx6c1ff701509c`，不自动连接或改变机器人模式。

不加`--execute`只显示离线检查命令；加`--preflight`只做离线检查，不连接机器人。输出为 `evaluation/hardware_shadow/commissioning/morning_Mxx_...`，微秒时间戳和随机后缀避免覆盖已有数据。

## 离线准备与回退

改动前备份位于 `evaluation/hardware_shadow/commissioning/morning_preparation_20261013/baseline/`，包括原控制配置、PID源文件及哈希。旧运行命令仍可使用；原MPC权重/模型/力矩文件同时由新入口核对哈希。

相关测试和完整记录输入回放不初始化DDS，不能替代新零位PID的真机复验。离线复现命令：

本轮离线结果：68项相关测试通过，包括hold模式计算子进程与直接计算一致、子进程退出后父进程仍可退权。学习/保持两种MPC各2434拍活跃控制，位置PID2485拍，三者均完成离线退权到0。旧PID与改动前保存源代码对照200拍，q/dq输出最大差异0、诊断一致；10个MPC基线文件哈希未变。证据在 `morning_preparation_20261013/validation_v2/summary.json` 和 `baseline_parity.json`。早一轮回放在汇总JSON编码阶段失败，修正编码后已完整重跑；`validation/`是该次未完成汇总目录，不作为最终通过记录。

输入来自MPC闭环历史，回放PID输出不会改变这些输入；回放中参考触限不能当成新PID真实运行的触限率，更不能据此估计新PID末端指标。

```bash
taskset -c 0-17 /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/validate_morning_20261013.py \
  --source evaluation/hardware_shadow/commissioning/comparison_T01_mpc_original_timing_zero_payload75_walk_20261012_174738_286024_948b/raw.jsonl \
  --output-dir evaluation/hardware_shadow/commissioning/morning_preparation_20261013/new_validation \
  --cpu 7
```

输出目录必须是新目录。输入回放不传播机器人状态，不把回放输出当成末端性能预测。
