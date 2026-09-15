# 2026-09-15：首次 A2 已执行，未证实手臂响应

## 结论

首次 A2 完成了本地 DDS 写入和 weight 回零，但**没有验证通过手臂控制**。
用户现场反馈“没有看出动作”；关节反馈也未显示右肩跟随目标。
`normal_release_completed` 仅表示程序完成其发送流程，不证明固件接受了接管或交还。

随后按用户要求调查无响应原因。本轮只恢复电脑已有有线连接、查询状态、发现 DDS
端点和订阅已有命令；没有切模式、创建关节命令 publisher、提高参数或重跑 A2。
尚未确定可修复的根因，不把猜测写成已解决。

## 首次实际执行结果

原始证据（本地、Git 忽略）：
`evaluation/hardware_shadow/commissioning/g1_a2_execute_20260915_133007/`。
其中 `profile_review.md`、`site_profile.conf`、`pre_query.json`、`execute.jsonl`、
`post_query.json` 分别保留审阅依据、实际参数、前后查询和逐帧命令／反馈。

| 项目 | 实测或实际设置 |
| --- | --- |
| 前后 FSM | 均为 4，查询返回 0；motion switcher 均为 form=0、name=ai |
| 状态 | 前／后分别收到 235／255 帧，CRC 拒绝均为 0 |
| 输出 | 仅 `rt/arm_sdk`；311 条 `dds_write`，约 6.200 秒 |
| 动作 | 右肩 pitch，motor 22；其余有效上肢／腰 yaw 维持起始目标 |
| 参数 | 偏移 0.02 rad（约 1.146°），最大 weight=0.1，kp=20、kd=1、前馈力矩=0 |
| 目标右肩 | 起点 0.288328469 rad，最高 0.308328469 rad |
| 实测右肩 | 最小 0.288292527，最大 0.288340449 rad；全程范围约 **0.002746°** |
| 结束 | 最后一帧 weight=0；日志 outcome=`normal_release_completed` |
| 现场观察 | 用户没有看出手臂动作 |

跟踪误差约 0.02 rad 没有超过本次 0.1 rad 的保护阈值，所以程序没有报故障。
**保护阈值未超限不等于动作成功。** 本次未进入 debug、未发布 `rt/lowcmd`、未执行 MPC。

证据 SHA-256：

```text
execute.jsonl     446b83dc871b50876d0def85df189dc80e1df2c5960598df5597f808ac135452
site_profile.conf 82ad0ebab0d9f544b0327c8f8f1502e7ee5031c9a21391c1c9316f18e96a3809
执行二进制        386c81f2c8a31063bdcc1f76123aba49069685db4844da1b9cb0b27210fd7a19
```

## 无响应调查：哪些解释成立，哪些尚未证实

### 1. 不能认定“锁定站立一定不支持 Arm SDK”

重新渲染阅读本地官方截图型 PDF：

- `local_reference/unitree_g1/6_高层运动开发.pdf` 第 4 页：Arm SDK 由内置运动服务提供，
  无需 debug，建议悬挂并在锁定站立下测试。
- `local_reference/unitree_g1/5_软件服务接口.pdf` 第 6 页 DDS 部分：列出的可用状态包括
  **锁定站立、运控 1、运控 2**；motor 29.q 为 weight。

这说明原 A2 的模式选择有官方资料依据；仍不能保证资料覆盖本机实际固件。
[官方 XR Motion 指南](https://github.com/unitreerobotics/xr_teleoperate/wiki/Motion)
展示运动模式下的协同路径，但不能据此反推锁定站立必然无效。
不把切换常规运动模式当成已经确认的修复，尤其不能在脚离地时随意启用平衡／行走。

### 2. 用户提供的 GitHub 问题确实存在，但不是已确认的固件缺陷

[SDK2 Python issue #155](https://github.com/unitreerobotics/unitree_sdk2_python/issues/155)
和 [#156](https://github.com/unitreerobotics/unitree_sdk2_python/issues/156)
是同一作者于 2026-05-25 提交的重复报告，本轮查询仍为 open。
报告称 23DoF 腰能动、双臂不动；但作者**已经尝试 R1+X 进入 regular mode，weight 到 1**，
仍无响应。因此该报告不支持“只需从 FSM=4 切到运动模式就能修好”。
报告还称 sport／arm RPC 不可达（3102），而本机 sport FSM getter 可用，情况并不完全一致。
未获得官方确认、目标固件版本或修复方案，不能据此给本机判定固件 bug。

[issue #170](https://github.com/unitreerobotics/unitree_sdk2_python/issues/170)
关于 R2+A／FSM=802 的说法来自另一台 **29DoF EDU+、1.5.3** 的用户，不能直接套用本机。

### 3. 权重与动作小，是可能因素，不是已查明的原因

0.1 接管权重及约 1.15° 的目标偏移不足以构成强响应验证；但不能将
“目标偏移乘 weight”当成真实关节必然达到的角度，也不能证明必须 weight=1 才生效。
本次不先提高 weight、角度或刚度，不以反复放大动作排查。

### 4. 映射正确；消息格式有差异，但没有证据证明因此被拒绝

实际依赖 SDK2：`/home/fjk/g1_ws/unitree_sdk2`，commit
`fa925bf6bb3fff439000266d70bde32eb5cd3597`。
对照其 `example/g1/high_level/g1_arm5_sdk_dds_example.cpp`：
左臂 15–19、右臂 22–26、weight 29 的映射一致；本机仅有效腰 yaw=12。

该官方 C++ 示例保留默认的 mode_machine、motor.mode 和 CRC；当前 A2 会填入
mode_machine=4、有效 motor.mode=1 并计算 CRC。这是待核对差异，不是已证实的错误；
不能仅因示例省略字段便盲目删除。官方不同实现的填写方式也不完全相同。
SDK `Write()` 成功表示本地 DDS 写操作返回，不提供机器人接受或执行命令的确认。

## 本轮最新只读现场结果

证据目录：`evaluation/hardware_shadow/commissioning/g1_a2_no_response_20260915/`。

- 开始检查时有线网卡有 carrier，但没有 IPv4。恢复原有 `g1-h1-readonly` 连接后，
  网卡 `enx6c1ff701509c` 使用既有 `192.168.123.99/24`；没有改网络配置或默认路由。
  这是本轮查询前的链路问题，**不是首次 A2 无响应的已知原因**。
- `current_query.json`：FSM getter 返回 0，**当前 FSM=0**，已不是首次 A2 的 FSM=4。
  收到 236 条状态、CRC 拒绝 0；motion switcher 仍为 form=0、name=ai。
  balance getter 仍返回 7301；不能把无效的 raw_value=0 当成有效平衡状态。
- LowState.version 仍为 `[0,0]`，不能由此识别系统／运控固件。
- `dds_discovery_restored.log`：发现 `rt/arm_sdk` 的 `LowCmd_` 接收端，表明对应 DDS
  reader 存在，**不证明当前模式允许执行，也不证明首次 A2 的消息被接收**。
- 还发现同主题发布端，其 participant 同时关联 arm action 服务主题。
  随后的 `existing_arm_commands.log`：只读订阅 3 秒收到 **0 条**命令。
  因此没有持续竞争发布的实测证据；不能证明首次 A2 期间没有竞争。

诊断临时程序只创建发现／订阅端，不回放消息，均已自然退出。

## 下一步

1. 先取得本机 App 显示的系统／运控版本或版本截图；**不升级固件**。
   现有 getter 和 LowState 没有回答本机固件版本问题，不能声称已经确认其支持矩阵。
2. 当前 FSM=0，不能直接重跑仅允许 FSM=4 的 A2，也不修改 gate 放行。
   下一次需恢复到已验证的悬挂锁定站立状态，并重新读回确认。
3. 在获得版本依据或明确可验证的修正后，再设计单变量、有界的 A2 重试。
   如果最终需要验证运动模式支持，那是 A3 的模式／平衡条件变化，不能伪装成原 A2 重跑。

当前状态：**首次 A2 执行已完成、手臂响应未通过；无响应根因仍未确认，本轮未重跑。**
