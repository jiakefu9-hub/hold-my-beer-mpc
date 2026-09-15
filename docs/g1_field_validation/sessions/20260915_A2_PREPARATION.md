# G1 A2 准备记录：输出前暂停（2026-09-15）

> 本文保留 12:35 阶段的历史事实。后续已验证遥控器切阻尼／锁定站立，并加入
> [停止联锁](20260915_A2_STOP_INTERLOCK.md)；不能继续将本文的“只掌握拔网线”
> 描述当作当前状态。后续仍没有执行真实控臂。

**结果：只读复查、A2 构建与离线预览已完成；没有执行 A2 控臂程序，没有创建手臂
命令 publisher。** 当前停止原因是本机异常停止／断线恢复方法尚未得到可靠确认，
不是网络、FSM 或编译失败，也不是用户未授权 A2。

## 1. 现场信息与停止依据

用户要求直接执行 A2；沿用已报告的悬挂、脚离地条件。询问异常处置与竞争控制源后，
用户答复：“他之前就给我说的是拔网线，没有其他东西在控制手臂”。

- 现场已口头确认没有其他程序控制手臂，作为用户提供的信息记录；本机相关进程检查也未见竞争输出程序。
- 目前得到的应急说明仅为拔网线，没有本机已验证的独立停止／恢复方法。
- 当前 A2 正常退出依赖继续发送降低 weight 的消息；状态异常时只尝试发送一帧零权重。
  拔网线会同时切断这些消息的传输路径，不能确保机器人收到交还指令。
- 目标固件在用户程序断线后如何处理此前的 Arm SDK 目标／权重尚未确认。
  不声称它必然继续动作，也不声称它必然安全停止；已有 A1a/A1b 均未验证此行为。
- 悬挂减少跌倒风险，但不消除手臂与自身、吊具或附近人员碰撞的风险。

因此没有把 `loss_recovery_confirmed`、`emergency_procedure_confirmed` 等未知项
直接设为 true，没有绕过 profile gate，也没有使用官方大幅摆臂示例替代现有小幅测试。

## 2. 本轮已完成的安全准备

- 有线 `enx6c1ff701509c` 仍连接，地址 `192.168.123.99/24`。
- 12:35:58（Europe/Berlin，UTC+02:00）独立 getter 查询报告落盘：
  `GetFsmId rc=0,value=4`，仍为锁定站立；`GetFsmMode rc=0,value=0`；
  `CheckMode rc=0,form="0",name="ai"`；`GetBalanceMode rc=7301`，没有有效 balance 值。
- 查询 LowState 274 帧，CRC 拒绝 0；原始 `version=[0,0]`、`mode_pr=0`、`mode_machine=4`。
- 从当前代码构建 `/tmp/g1-a2-build-20260915/`，开启独立 A2 编译目标，
  `G1_COMMISSIONING_BUILD_MODE_STEP=OFF`。**编译不等于运行。**
- CTest **4/4 通过**；没有启动 `g1_arm_static_execute`，没有进入其人工确认或 publisher 创建阶段。
- 使用未修改的 `site_profile.template` 与本轮新快照运行无 SDK 的离线 preview，
  `offline_preview_completed=true`，但 `real_output_profile_gate_passed=false`。

## 3. 离线预览的具体动作（没有发送给机器人）

| 项目 | 预览结果 |
| --- | --- |
| 动作关节 | 右肩 pitch，Arm slot 5 / motor 22 |
| 目标角 | 本轮 q0=0.290353805 → 0.310353805 → q0 rad |
| 偏移 | +0.02 rad，约 1.15° |
| 目标变化速度 | 0.02 rad/s |
| 其他有效槽位 | 左臂五关节、右臂其余四关节和腰 yaw 均以各自 q0 为固定目标 |
| PD / 前馈 | kp=20、kd=1；dq=0、tau_ff=0；无效腰 roll/pitch 槽保持零 |
| weight | 2 秒 0→0.1，保持完成动作，2 秒 0.1→0 |
| 总轨迹时间 / 记录数 | 6.2 秒 / 312 帧 |
| 最后一帧 | phase=complete、terminal=true、weight=0 |
| 全部帧事件 | `offline_would_write`，不是 DDS 写入 |

模板仍为 DRAFT。预览的 readiness_errors 是**未填写模板的完整检查列表**，
不能据此否定已经取得的 FSM=4 或用户关于无竞争程序的口头确认；但也不能把模板直接
重标为已现场审核。未来执行前须据实整理独立 profile，并重新采集新鲜姿态。

## 4. 证据与恢复条件

证据目录：
[`evaluation/hardware_shadow/commissioning/g1_a2_20260915_1235/`](../../../evaluation/hardware_shadow/commissioning/g1_a2_20260915_1235/)。

- `pre_query.json`、`pre_state.conf`、`pre_query_console.log`：本轮只读结果。
- `build_and_tests.log`：A2 编译与离线测试。
- `template_preview.jsonl`、`preview_console.log`：离线轨迹及未通过的真实输出 profile gate。
- `evidence.sha256`：快照、轨迹、代码、模板与构建产物校验。

代码、模板和既有 H1/A1b/A2 gate 均未改动；原始 evidence 继续被 Git 忽略。
本次只补记录和索引，没有 commit/push，没有切换当前模式。

继续 A2 前，先由带教／Unitree 确认本机可用的异常停止及断线后的处置方法，
而不是只保证“拔网线即可”。不照搬尚未核实的遥控器按键，也不以真实控臂故障注入
来试出停止方法。**本轮停在输出前，不能记为 A2 实机通过或动作失败。**
