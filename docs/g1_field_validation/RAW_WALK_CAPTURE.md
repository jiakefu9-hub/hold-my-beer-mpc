# 真机步态观察与原始扰动采集

2026-09-17：两个程序已完成真机运行。`GetPhase` 未取得有效值；行走采集第一次因吊绳卡住排除，
随后五次由操作者确认有效。详见[现场记录](sessions/20260917_WALK_CAPTURE.md)。这些轨迹采用旧的
“IMU 世界 yaw=0”航向目标，现保留为历史研究数据，**不与新的固定 H0 轨迹混合建模板**。
本轮只采数据，不构建模板，不运行 MPC，也不改变已有 H1、A2、A3 的入口。

采集后的[五条轨迹离线审计](WALK_DATASET_AUDIT.md)已完成：记录可用，同腿周期约 1.014 秒，
需要按腿部事件对齐；存在预测收益，但尚无可直接上机的扰动模板。以下时间表不变，航向参考和
坐标派生方式已改为每轮固定 H0。

## 现场先做什么

1. 你先让机器人双脚着地，在 Regular Motion Mode（FSM 500）下原地自主平衡。
   两个程序都不会替你切模式，不进入 debug、不释放内置运控。
2. 核对本次有线连接和只读状态接收；沿用已有现场连接流程。程序本身不改电脑网络。
3. 先运行下面的 **30 秒观察**。看到 `READY` 后，用遥控器走几步、停下来。
   结束后查看相位返回和腿部／IMU 数据，再决定是否开始电脑发速度指令的采集。
4. 第二个程序会真的驱动双臂并请求行走，不是只读实验。运行前留出行走、停车余量，
   安排好防跌和网线随行；站在原地的 A3 成功不等于行走采集已验证。

## 1. 30 秒只读观察：`g1_phase_probe`

只订阅 `rt/secondary_imu`、`rt/lowstate`，查询 FSM 和 `GetPhase`。
它不创建关节 command publisher，不发送速度或模式指令。
查询 RPC 会发数据请求，“只读”不是完全没有网络发送。

检测到 FSM 500 和新鲜状态后打印 `READY`，然后开始计满 30 秒。
这段时间你用遥控器操作，电脑每半秒显示一次当前 FSM、相位返回码／数组和收包数量。
记录的是每次收到的原始消息，不是半秒打印一次的数据。

2026-09-17 起，这个只读入口还默认记录 CPU 7 上的 6 ms probe 和查询往返／收包计时。
支持线程使用其他物理核，调度保持 SCHED_OTHER，不改系统实时设置；不要将整个程序 taskset 到单核。
报告命令和测量边界见 [计时验证说明](TIMING_VALIDATION.md)。这些指标不是单向 DDS／电机响应延迟。

```bash
# 仓库根目录；NIC 按本次只读连通检查的结果设置
NIC=enx6c1ff701509c
mkdir -p evaluation/hardware_shadow/commissioning
/tmp/g1-capture-device/g1_phase_probe "$NIC" \
  --output-dir "evaluation/hardware_shadow/commissioning/phase_$(date +%Y%m%d_%H%M%S)" \
  --permit-read-only PHASE_30S_READ_ONLY --timing-cpu 7
```

`GetPhase` 在本机 SDK 中仍存在，但标为 deprecated，API ID 是 7006。
是否被这台固件支持、数组各项对应哪条腿、单位／回绕范围是什么，都留待实测。
正常回复不等于已证明“周期为 0.8 秒”，常数数组也不等于有可用步态相位。
需要看走路时是否连续变化、是否重复回绕、与腿部状态的对应及请求延迟。
失败回复同样保留；接口不可用时，仍能取得 IMU 和腿部状态作后续分析。
该程序不自动估计周期、不自动宣布相位可用。

## 2. 定时行走采集：`g1_walk_capture`

| 程序时间 | 机器人收到的目标 |
| --- | --- |
| 0–3 秒 | 双臂／单腰 yaw 从实测姿态平滑进入目标，weight 从 0 到 1；速度为零 |
| 3–5 秒 | 双臂 PD 保持，weight=1；原地等待 |
| 5–15 秒 | 向前速度 0.5 m/s，启用航向保持；双臂继续 PD 保持 |
| 15–18 秒 | 前进和转向目标都为零；继续保持双臂 |
| 18–21 秒 | 保持关节目标，weight 从 1 降到 0，交还双臂 |

时间以日志里的 `task_epoch_monotonic_ns` 为第 0 秒。全程记录，包含准备期、
0–21 秒及收尾；后续可单独选取 5–18 秒，但不在采集时丢掉起止片段。
目前停车严格按主机时间，不按步数／相位；即使相位查询成功，也不会擅自换成数周期。
0.5×10=5 米只是速度目标的积分，**不是实测距离，也不是最大停车距离**。

### 航向：每轮固定 H0，取走前最后两秒平均朝向

程序在 3–5 秒原地等待阶段采集躯干 yaw，用圆周平均（可正确处理 ±π 回绕）得到本轮 `yaw0`；
到第 5 秒、第一条非零速度请求前冻结。H0 的 Z 与 IMU 导航世界系同向，H0 的 +X 指向 `yaw0`。
此后 H0 固定，不会随机器人当前朝向转动，也不会在进入某个步态周期时重置。

因此不再要求操作者把每轮机器人肉眼摆到 IMU 世界 yaw=0；只需在开始时让机器人朝向本轮希望的
走廊方向。速度接口仍给机器人“向前走”的目标，航向控制只纠正相对该起步方向的偏移，
不是世界系位置导航，也不会纠正横向位置。

冻结前必须得到至少 1 秒跨度的有效样本，否则程序拒绝开始走。控制器对最近 1 秒的 yaw 做圆周
平均、对绕竖直方向角速度做平均，用 PD 请求有界转向（kp=1.0、kd=0.1、最大绝对转速
0.25 rad/s）。1 秒只是反馈滤波窗口，不把它当作精确步态相位。`yaw0`、相对 H0 的 yaw、
误差、修正量及每次速度请求都入日志。第一条新轨迹要先看是否出现过冲或摆动，再决定是否继续采集。

### 双臂姿态

沿用最后一次 A3 的静态近竖直目标，不再用全部零度：

| 关节 | 左臂 | 右臂 |
| --- | --- | --- |
| shoulder pitch | −4° | −4° |
| shoulder roll | −1° | +1° |
| elbow pitch | −8.1° | −7.8° |
| 其余臂轴 | 0° | 0° |

唯一腰 yaw 为 0°；有效关节 kp=20、kd=1。双臂都是固定目标 PD，
不会随身体摇晃实时修正瓶子姿态，也没有末端 FK／MPC。带瓶后的姿态误差仍需看数据。
profile 采用已有 A3 的审阅机制；没有新增“腕部某个瞬时 dq 过大就退出”的门槛。

将 [`timed_walk_capture.template`](../../tools/g1_commissioning/profiles/timed_walk_capture.template)
复制为本次现场 profile，沿用已确认的本机身份／映射资料，核对本次现场条件后填写审阅项。
模板故意保持 `DRAFT`；不能机械地把所有确认项改成 true。
它的 `hold_s=15` 指 weight=1 的整个保持段（3–18 秒），不是走 15 秒。
`weight_rate_per_s=1/3`、`max_weight=1`、`total_timeout_s=22`。

```bash
# FIELD_PROFILE 指向这次已审阅的 profile；不直接使用 DRAFT 模板
FIELD_PROFILE=/path/to/reviewed_walk_profile.conf
/tmp/g1-capture-device/g1_walk_capture "$NIC" \
  --profile "$FIELD_PROFILE" \
  --output-dir "evaluation/hardware_shadow/commissioning/walk_$(date +%Y%m%d_%H%M%S)" \
  --permit-real-output TIMED_WALK_RAW_CAPTURE
```

它会先读状态，再要求终端输入 `EXECUTE <robot_id>`；输入后重检新鲜状态，
通过才创建 `rt/arm_sdk` publisher 并开始计时。不会创建 `rt/lowcmd`。
行走客户端只注册速度 API 7105，不注册切模式／启动／释放服务接口。

### 停车与退出

- 正常时约每 50 ms 刷新速度，单次请求 duration 最长 0.2 秒；临近第 15 秒缩短请求时长。
  第 15 秒开始发零速度；第 18 秒退权前要求已收到停车阶段的零速度成功回复。
  请求／回复时间都保存，便于检查实际调度和延迟。
- Ctrl-C 请求零速度，冻结当时手臂目标／weight，保持 3 秒后用 3 秒退权，结束本轮。
- 状态／IMU 超时、CRC／tick 真回退、非有限数据、FSM 离开 500、L2+B、输出失败或
  记录队列溢出等会中止继续正常输出。模式／遥控停止已触发时不再补发手臂命令；
  其他异常沿用已有 best-effort 零权重处理。
- 零速度 RPC 成功、有限 duration 和一次退权写入都**不能证明实物已经停稳**，
  也不能保证失联／进程崩溃后的行为。落地时 L2+B 会失去主动平衡，不能把悬挂实验的
  “直接阻尼”照搬成无支撑行走的正常停车；保留现场防跌和人工异常处置。

## 保存了什么

每次要求一个全新的输出目录，已有目录拒绝覆盖。主要文件是 `raw.jsonl`，
行走采集还会保存实际使用的 `arm_profile.conf`。

| 日志类型 | 内容 |
| --- | --- |
| `g1_torso_imu_raw_v1` | 身体／躯干 `rt/secondary_imu` 的四元数 wxyz、RPY、三轴角速度、原始加速度、温度 |
| `g1_lowstate_raw_v1` | tick、原始 mode、CRC、遥控字节、骨盆 IMU、全部 35 个消息槽的关节 q/dq/ddq/力矩等；包括腿部 |
| `phase_reply`、`fsm_reply` | 请求和回复时间、返回码；相位原文、解析数组，FSM 数值 |
| `task_epoch`、`task_stage` | 程序零时刻、抬臂／等待／行走／停车／退权的边界 |
| 控制记录 | 逐帧上肢目标与 weight、DDS 写入结果和时间；速度／航向请求与回复 |

躯干 IMU 与骨盆 IMU 分开保存，不混为同一个测点。35 个消息槽不代表本机有 35 个实体电机。
`raw.jsonl` 是不可替代的原始证据：**四元数／欧拉角保留消息值；角速度、加速度保留 IMU 局部分量。**
不减重力、不平滑原始加速度。仿真所需的角加速度不是 IMU 直接输出，
后续用这批带时间的原始角速度离线估计；不能拿电机 `ddq` 冒充身体角加速度。

每个消息带主机单调时钟接收时间和本地回调序号；相位 RPC 另记请求／回复时间。
这些不是硬件同步时间，不能把一次 RPC 的回复时刻当作准确足底触地时刻。
当前 `secondary_imu` 类型没有源时间戳／源序号，不能声称已证明网络零丢包。
记录器保存每次送到回调的消息，异步落盘；缓冲区溢出计数并报错，不悄悄覆盖旧数据。
程序另存唯一的 `heading_reference_frozen` 事件，给出本轮 `yaw0`、实际样本数和观察跨度。
采集结束后运行下面的离线工具，把**全部**躯干和骨盆 IMU 样本（包括冻结 yaw0 之前的样本）
转换到同一个固定 H0；原始文件不被覆盖：

```bash
CAPTURE_DIR=evaluation/hardware_shadow/commissioning/walk_YYYYMMDD_HHMMSS
/home/fjk/miniforge3/bin/conda run -n g1_mpc \
  python tools/g1_commissioning/derive_walk_h0.py "$CAPTURE_DIR/raw.jsonl" \
  --output-dir "$CAPTURE_DIR/h0"
```

输出 `h0/h0_imu.npz` 和 `h0/manifest.json`。前者同时包含躯干／骨盆相对 H0 的四元数、RPY、
角速度、保留重力的 specific force，以及减去重力后的线加速度；后者记录原始文件哈希、yaw0、
公式和样本数。这里的 H0 是本轮固定坐标系，不是不断跟随身体的局部系。转换不在线生成模板，
角加速度仍留给后续因果滤波／差分步骤。

## 离线构建与当前验证

以下只构建／测试，不连接机器人。默认构建仍不依赖 SDK／DDS，网络目标全部显式开启。
若只准备只读观察，把 `G1_COMMISSIONING_BUILD_WALK_CAPTURE` 改为 `OFF`。

```bash
cmake -S tools/g1_commissioning -B /tmp/g1-capture-device \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DG1_COMMISSIONING_BUILD_DEVICE_QUERY=ON \
  -DG1_COMMISSIONING_BUILD_REAL_OUTPUT=OFF \
  -DG1_COMMISSIONING_BUILD_MODE_STEP=OFF \
  -DG1_COMMISSIONING_BUILD_WALK_CAPTURE=ON \
  -DUNITREE_SDK2_DIR=/home/fjk/g1_ws/unitree_sdk2
cmake --build /tmp/g1-capture-device --parallel 4
ctest --test-dir /tmp/g1-capture-device --output-on-failure
```

本轮使用外部 SDK2 commit `fa925bf6bb3fff439000266d70bde32eb5cd3597`。
默认 SDK-free 构建通过 9/9 项测试；按上面命令开启查询和采集入口的设备编译检查通过
11/11 项测试；H0 Python 转换测试在 `g1_mpc` 环境通过 2/2。
检查覆盖 21 秒边界、weight、走前 yaw0 冻结／跨 ±π 平均、H0 全样本转换、速度请求时限、
原始值保存、队列完整排空、拒绝覆盖、无效参数／DRAFT 拒绝及只读源码边界。
离线测试不会调用正常的设备入口；编译通过不代表该固件的相位或行走接口已实测。

接口依据：本机同版本官方
[LocoClient](https://github.com/unitreerobotics/unitree_sdk2/blob/fa925bf6bb3fff439000266d70bde32eb5cd3597/include/unitree/robot/g1/loco/g1_loco_client.hpp)
及 [API 定义](https://github.com/unitreerobotics/unitree_sdk2/blob/fa925bf6bb3fff439000266d70bde32eb5cd3597/include/unitree/robot/g1/loco/g1_loco_api.hpp)。
上述时间表、PD 参数、日志及停止流程是本项目试验设计，不是官方认证的安全功能。
