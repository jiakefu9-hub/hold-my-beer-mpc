# 2026-09-17 G1 原始行走采集

## 结论

- 第一次运行 `g1_walk_trial01_20260917_1429` 因吊绳卡住、部分时间无法前进，操作者明确判定无效；原始数据保留，但排除在五条轨迹之外。
- 第二次运行 `g1_walk_trial02_20260917_1443` 由操作者确认有效，记为计划五条轨迹中的 **第 1 条有效轨迹**。
- 第三次运行 `g1_walk_trial03_20260917_1455` 由操作者确认有效，记为计划五条轨迹中的 **第 2 条有效轨迹**；当前进度为 **2/5**。
- 操作者随后直接在终端运行 `g1_walk_20260917_150150` 并确认有效，记为 **第 3 条有效轨迹**；当前进度为 **3/5**。
- 操作者直接运行 `g1_walk_20260917_150634` 并确认有效，记为 **第 4 条有效轨迹**；当前进度为 **4/5**。
- 操作者直接运行 `g1_walk_20260917_151055` 并确认有效，记为 **第 5 条有效轨迹**；计划的五条原始轨迹已采齐。
- 本轮只采原始数据，没有生成扰动模板，也没有运行 PID/MPC。

## 第二次有效运行

- 开始前：专用有线接口 `enx6c1ff701509c` 为 `192.168.123.99/24`；只读查询收到 LowState，CRC 拒绝 0，FSM 为 500。
- 时间表：0–3 秒双臂升权，3–5 秒等待，5–15 秒以 `0.5 m/s` 请求前进，15–18 秒零速度保持，18–21 秒退权。
- 航向目标：IMU 导航世界系 `+X`（yaw=0）。
- 程序阶段实际记录：`forward_walk` 约从 5.018 秒至 15.014 秒；21.015 秒进入 complete。
- 速度请求：200 次非零前进请求；全部 422 次速度 RPC 回复返回 0。
- 状态：FSM 查询始终为 500；无 CRC 错误、无日志队列丢失、无 `session_fault`。
- 数据量：躯干 IMU 26,968 条、LowState 26,971 条、上肢 DDS 写入记录 1,049 帧。
- 正常结束：`normal_release_completed`，最终 weight=0；日志不能单独证明物理停车，实物运行有效性来自操作者现场确认。
- `GetPhase`：52 次查询均返回 7301，没有取得可直接使用的官方步态相位。

## 本地证据

- 有效轨迹目录：`evaluation/hardware_shadow/commissioning/g1_walk_trial02_20260917_1443/`
- `raw.jsonl` SHA-256：`0d2cdef8b79aa41781248cff39e64a51b6c9c03407ee42b0279d17d7c330a1aa`
- `arm_profile.conf` SHA-256：`f802ca080f499dfdac26a2d7772788c03ea2ee328e66241d4c885e6a08da4191`
- 无效轨迹目录：`evaluation/hardware_shadow/commissioning/g1_walk_trial01_20260917_1429/`
- 无效轨迹 `raw.jsonl` SHA-256：`6105c9d21bf1a021172aca49d621bd5ad2ad7c6b6f71c55d82ad8ab170fbc0b1`

### 第 2 条有效轨迹

- 目录：`evaluation/hardware_shadow/commissioning/g1_walk_trial03_20260917_1455/`
- 操作者现场确认：有效，吊绳未造成该轮无效。
- 实际阶段：约 5.018–15.013 秒前进，21.015 秒 complete。
- 200 次非零速度请求；全部 422 次速度 RPC 回复返回 0。
- FSM 始终为 500；CRC 错误 0、队列丢失 0、`session_fault` 0。
- 躯干 IMU 26,858 条、LowState 26,858 条、上肢 DDS 写入记录 1,049 帧。
- `raw.jsonl` SHA-256：`77e6ce8fedf79afb29fbdf416a9c5651e36b3f5f9cbaec86f8f93a8e47b55a58`
- `arm_profile.conf` SHA-256：`11ecb96e3d889ef58ea2738b7552195af390064bdd1fc7cbd53e6d4683bc1ee8`

### 第 3 条有效轨迹

- 目录：`evaluation/hardware_shadow/commissioning/g1_walk_20260917_150150/`
- 操作者现场确认：有效。
- 实际阶段：约 5.007–15.015 秒前进，21.010 秒 complete。
- 200 次非零速度请求；全部 421 次速度 RPC 回复返回 0。
- FSM 始终为 500；CRC 错误 0、队列丢失 0、`session_fault` 0。
- 躯干 IMU 27,410 条、LowState 27,410 条、上肢 DDS 写入记录 1,047 帧。
- `raw.jsonl` SHA-256：`6585919ba9b645ee2fe81cf68f259d8a98054f6057cff8b56ffdd3fcd76a7683`
- `arm_profile.conf` SHA-256：`11ecb96e3d889ef58ea2738b7552195af390064bdd1fc7cbd53e6d4683bc1ee8`

### 第 4 条有效轨迹

- 目录：`evaluation/hardware_shadow/commissioning/g1_walk_20260917_150634/`
- 操作者现场确认：有效。
- 实际阶段：约 5.006–15.000 秒前进，21.009 秒 complete。
- 200 次非零速度请求；全部 421 次速度 RPC 回复返回 0。
- FSM 始终为 500；CRC 错误 0、队列丢失 0、`session_fault` 0。
- 躯干 IMU 27,040 条、LowState 27,141 条、上肢 DDS 写入记录 1,047 帧。
- `raw.jsonl` SHA-256：`4d94ca932eade54fef5493b4cea03d24eb51d51325211c825261bb02b57c1c40`
- `arm_profile.conf` SHA-256：`11ecb96e3d889ef58ea2738b7552195af390064bdd1fc7cbd53e6d4683bc1ee8`

### 第 5 条有效轨迹

- 目录：`evaluation/hardware_shadow/commissioning/g1_walk_20260917_151055/`
- 操作者现场确认：有效。
- 实际阶段：约 5.002–15.010 秒前进，21.019 秒 complete。
- 200 次非零速度请求；全部 422 次速度 RPC 回复返回 0。
- FSM 始终为 500；CRC 错误 0、队列丢失 0、`session_fault` 0。
- 躯干 IMU 26,763 条、LowState 26,765 条、上肢 DDS 写入记录 1,048 帧。
- `raw.jsonl` SHA-256：`ef47dc421703c96d019b9ea2edc28b7bb19986be5bf394b2a9337e091cce095c`
- `arm_profile.conf` SHA-256：`11ecb96e3d889ef58ea2738b7552195af390064bdd1fc7cbd53e6d4683bc1ee8`

## 本批次状态

五条有效轨迹已经采齐，另有一条因吊绳卡住而明确排除。下一步是离线做跨轨迹完整性、
时间对齐、重复性和步态事件可预测性检查；采集结束本身不代表扰动模板可用。

### 随后的离线审计（同日）

已完成上述检查，见[五条轨迹审计](../WALK_DATASET_AUDIT.md)。五条均可用于离线研究；
同腿周期约 1.014 秒，第 1 条与后四条相位约差半周期，不可按任务秒数直接平均。
前三条拟合、后两条检验支持继续研究周期辅助预测，但起步、冲击和停车尚不能保证准确前馈；未导出上机模板。

操作者补充确认水瓶条件一致，吊绳可能有轻微拖拽；缺少张力观测，不能确认其影响可忽略，
不能把这批数据自动等同于完全无拖拽行走。原始有效／无效判定及原始文件保持不变。

原始目录受 `evaluation/` 忽略规则保护，不随 Git 克隆分发；仓库只保存本记录与摘要哈希。
