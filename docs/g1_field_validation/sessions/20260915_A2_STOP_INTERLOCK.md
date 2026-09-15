# 2026-09-15：A2 停止联锁、只读验证与输出前拒绝

> 本文是联锁实现和当时 v1 profile 拒绝的历史记录。后续用户明确授权审查准入条件，
> 已改为 [v2 首次试验方案审核](../../../tools/g1_commissioning/profiles/PRETEST_REVIEW.md)。
> 不改写本文当时没有真实输出的事实；v2 修订也未执行真实输出。

## 结果

用户明确要求“改代码，并运行 A2”。已完成停止联锁实现、离线测试、设备编译以及
真机只读监测。随后调用新版 A2 入口时，现有 DRAFT 模板被原有 profile gate
拒绝，退出码 1，拒绝发生在 DDS 初始化、人工 EXECUTE 提示和 publisher 创建之前。
**没有给机器人发送关节命令，不能记为 A2 动作通过或动作失败。**

## 本次新增证据与已有事实

- 用户保持悬挂、脚离地。用户先按 L2+B，模式读回 `FSM=1, rc=0`，并明确报告
  看见手臂下垂；后来按 L2+↑、报告“摆好了”，读回 `FSM=4, rc=0`。
- 两次 `mode_machine` 都是 4，证明该字段不能代替 FSM 判别阻尼／锁定站立。
- 用户明确说没有其他程序控制手臂；这是现场口头信息，不伪装为 PC2 进程审计。
- 原始记录：`evaluation/hardware_shadow/commissioning/g1_remote_damp_check_20260915_125311/`
  及 `g1_remote_stand_check_20260915_125803/`。
- 不能继续把“只知道拔网线，没有任何已验证按键”作为当前停止依据。

## 修改范围

- `arm_stop_interlock.hpp`：SDK-free、单向锁存的停止判定。FSM getter 错误、模式非 4、
  时间异常、样本过期或观测间隙均拒绝继续；L2+B 使用 SDK 按键位图解码并锁存。
- `device_fsm_monitor.hpp`：只注册 GetFsmId，独立线程查询；不注册 setter。
  SDK RPC timeout 100 ms，调用之间等待 50 ms，模式数据从请求起算最多 200 ms。
- `arm_static_execute_main.cpp`：启动阶段、publisher 创建前、正常轨迹与 SIGINT
  交还过程都检查联锁；写出前再次检查。联锁触发后不自动恢复，不切换模式，
  不发送终止帧，退出码 4。已有 CRC、tick、状态新鲜度、速度／跟踪／deadline
  条件及其 best-effort fault frame 未放宽。
- `g1_arm_stop_observe`：三秒只读观察，复用相同状态回调和 FSM 监测器，没有
  关节命令 publisher；用于在真实输出前检查新增监测路径是否可运行。
- 没有修改 H1、生产 adapter、腿部控制、MPC、A3/A4；没有修改 profile 的准入条件。

200 ms 是软件新鲜度限制，不是物理急停时延保证。异步接收与写出之间可能已有
在途消息，软件不能撤回它；断网也会影响状态反馈。联锁不替代机器人自身保护，
不证明 Arm SDK 失联后的固件行为，也不等同于独立硬件急停。

## 测试与实际调用

| 检查 | 结果 |
| --- | --- |
| 默认 SDK-free 构建 `/tmp/g1-a2-interlock-default-20260915` | 5/5 CTest 通过 |
| opt-in 设备构建 `/tmp/g1-a2-interlock-20260915` | 6/6 CTest 通过 |
| 按键位图、非 4 模式、RPC 失败、迟到／过期／观测间隙、锁存不恢复 | 离线通过；包含全部 65536 种按钮位组合 |
| SDK LowState 的实际 CRC／遥控字节回调 | 使用本地构造消息离线通过；无 DDS 初始化 |
| 真机三秒 publisher-free stop observe | 通过；150 条观察记录，启动初次等待之后 gate 健康，LowState 均有效 |
| 新版 A2 搭配现有模板启动 | profile rejected，exit=1；未初始化 DDS，未建 publisher |

只读观察未要求用户再次按键，没有在真实运动中做故障注入；不能把离线短脉冲
解码测试当成真机遥控器接管时延测试。

证据目录：
[`g1_a2_stop_interlock_20260915`](../../../evaluation/hardware_shadow/commissioning/g1_a2_stop_interlock_20260915/)：
`stop_observe.jsonl`、`default_tests.log`、`device_tests.log`、`attempt_result.json`。

## 剩余问题必须如实处理

原有 `site_profile.template` 仍为 DRAFT、身份字段为 UNSET、确认项均为 false。
模板报出的全部缺项不是“现场一项都没通过”：锁定站立、基本遥控器退阻尼、
用户关于无竞争程序的陈述已有依据；映射和权重接口也有 SDK／官方资料依据。
但尚未形成逐项列明依据的实机执行 profile，不能为了让 gate 通过而把未知项
一律填 true。尤其新联锁本身不证明目标固件失联后的权重处理与恢复行为。

若现有准入流程把“首次输出才可观测的结果”也要求为输出前已验证，必须单独审查
该准入设计，区分官方接口合同、已确认的现场处置和待测结果，不能暗中更改含义。
这次仅完成获准的停止联锁修正，没有借机重定义或绕过 profile gate。
