# 2026-09-15 A3 离线准备记录

## 结论

- 状态：`OFFLINE_READY / HARDWARE_NOT_RUN`。
- 用户现场观察：机器人曾由锁定站立切到 Regular Motion Mode，并能双脚着地、原地自主平衡。
  当时没有保存软件 getter 读回，因此尚不能把该观察登记成已证实 `FSM=500`；下次 A3 前读取。
- 本轮没有连接机器人、没有创建 DDS publisher、没有发送任何真机命令。
- 新增独立 `g1_arm_balance_hold_execute`。它保留内置腿部运控，只发布 `rt/arm_sdk`，
  不调用模式 setter、`ReleaseMode`、Loco 速度接口或 `rt/lowcmd`。

## A3 固定试验定义

- 适用状态：双脚着地、原地自主平衡、软件持续读回 `GetFsmId()==500`。
- 有效目标：左臂 5、右臂 5、单腰 yaw 1，共 11 个；waist roll/pitch 两个协议槽保持无效全零。
- 目标角度：11 个有效关节均为 SDK 坐标 `0 rad`。
- 轨迹：3 秒内从最新实测姿态插值到零位，同时 weight `0→1`；保持 5 秒；
  零位目标不变，3 秒内 weight `1→0`。总长 11 秒，超时 12 秒。
- 初始控制参数：所有有效槽 `kp=20, kd=1, dq=0, tau_ff=0`。
  这些是 A2 已使用过的较低硬件增益，不复制仿真配置中的更高增益。
- A3 的 weight 上限 1 只存在于 A3 schema；A2 仍拒绝超过 0.5。

## 独立准入与停止条件

- A3 profile schema：`g1_arm_balance_hold_site_v1`。
- CLI 许可词：`A3_GROUNDED_BALANCE_HOLD_ONLY`。
- profile 与 executable 不匹配时拒绝；synthetic fixture 永久拒绝真实输出。
- publisher 前：field-reviewed profile、连续新鲜状态、FSM 500、人工
  `EXECUTE <robot_id>`，随后再收集一轮新状态。
- 运行时检查 FSM 500、遥控器 L2+B、CRC、100 ms 状态新鲜度、tick 回退、有限数值和
  DDS write。A3 不以反馈速度、跟踪误差、raw mode、统一 ±1.2 rad 角度范围或单次
  deadline 迟到中止；这些值仅记录。A2 的既有 gate 未改变。
- L2+B 或 FSM 离开 500 时停止正常写入且不再发最后一帧；这仍是软件联锁，
  不是独立硬件急停或固件失联安全认证。

## 离线验证

- 默认无 SDK/DDS 构建：成功；CTest `6/6` 通过。
- 显式设备构建（SDK2 `/home/fjk/g1_ws/unitree_sdk2`）：A1/A2/A3 均编译成功；
  CTest `7/7` 通过。
- synthetic A3 preview：553 行（1 行 profile 摘要 + 552 帧），计划时长 11 秒，
  明确报告 `real_output_profile_gate_passed=false`。
- 覆盖：3/5/3 秒相位、weight 0/0.5/1/0、从非零启动姿态插值到零、
  A3 weight 超限、非零目标、FSM 非 500、A2/A3 联锁条件；另用极端但有限的
  `q=5 rad`、`dq=100 rad/s` 和 raw mode 255 验证这些值不会触发 A3 gate。

## 下次现场尚需取得的事实

1. 在实际原地自主平衡状态查询并保存 `fsm_id`；只有读回 500 才使用本 A3 入口。
2. 保存该状态的新 LowState snapshot，用于离线 preview；raw mode 只记录，不再作为 A3 gate。
3. 从模板复制新的 field profile，逐项审阅后再做离线 preview；模板自身保持 DRAFT。
4. A3 真机结果须另建 session，记录视频、profile、JSONL、实际关节跟踪、是否平顺交还；
   本记录不能当作 A3 硬件通过证明。

## 下次附加 IMU yaw 实验

- 开机时整机朝场地左向；网络可用后尽早开始既有 H1 state-only 长时采集。
- 依次保留开机朝左、FSM 500 朝左、内置运控转到前向、朝前运行 A3 四个稳定窗口。
- H1 保存 `rt/secondary_imu` torso 四元数/RPY/gyro/raw accel 与关节状态；A3 JSONL
  保存同机 monotonic 时间，现场视频提供物理朝向标签。二者事后按时间对齐。
- 重点比较 wrap 后的 torso yaw 差值及 waist yaw；不使用 IMU 推断平移位置原点。
- 官方资料未确认目标固件的 yaw 零点建立时刻；结果按“支持／不支持某假设”记录，
  不把一次约 90°变化直接写成已经证明世界坐标系定义。
