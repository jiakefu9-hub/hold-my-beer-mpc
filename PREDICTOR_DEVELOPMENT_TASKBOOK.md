# 扰动预测器开发任务书（动态版）

更新时间：2026-08-15

## 1. 这份文件怎么用

这不是一组必须机械执行到底的固定提示词，而是一份长期路线和候选提示词。

工作方式如下：

1. 开发窗口每次只执行“当前阶段”，阶段内部可以自主完成实现、测试、仿真和画图。
2. 阶段结束后必须停止，不得自动进入下一阶段。
3. 用户把本阶段结果交给“总控/理解窗口”审查。
4. 总控窗口根据实际结果决定：通过、补充诊断、返工，或者进入下一阶段。
5. 下方后续阶段提示词只是草案；真正执行前，应由总控窗口结合最新结果重新确认或改写。

这样既避免每个小修改都要询问，也避免一个早期错误被后续阶段不断放大。

## 2. 当前开发起点

任务书创建前现场核对到：

```text
repository: /home/fjk/g1_ws/disturbance-lab
branch:     feat/predictor-interface
HEAD:       afc126e
worktree:   clean
```

创建本文件后，工作区预期只多出本任务书。开发窗口必须重新执行
`git status --short --branch`；如果还有其他未说明的改动，不得 reset、stash 或覆盖，
应先报告。

若状态完全符合预期且 `feat/full-task-predictors` 尚不存在，开发窗口在开始 T1 前创建并
切换到该分支；如果同名分支已经存在，不得覆盖，应先报告。

未经用户明确要求，不要 commit、push、删除历史结果或修改无关文件。

## 3. 已冻结的总体目标

项目保留两个候选方向，共四个大验收阶段：

1. **T1：纯模板离线构建**——收集多次完整任务数据，建立一条完整时间模板并画图。
2. **T2：纯模板闭环**——把完整时间模板接入 MPC，从开始到停车统一评价。
3. **N1：完整神经网络离线预测**——训练从第一帧工作的 GRU，并展示预测—真实图。
4. **N2：完整神经网络闭环**——接入 MPC，与纯模板进行同协议比较。

只在这四个节点做人为验收，不为普通代码小步增加额外审批。

现有 `template`、`neural`、`hybrid_residual`、`zoh` 必须保留原行为和历史产物。
新方法使用新名字，例如 `full_task_template` 和 `full_neural`。

所有结论均为 MuJoCo simulation-validated、hardware-unverified。

## 4. 统一任务协议（暂定，T1 首次 smoke 后冻结）

历史正式协议总长 10.4 s，机器人全程 XY 起终点位移约 4.78 m。新协议暂定：

```text
0.0 <= t < 6.4 s   直接发布并保持当前 nominal locomotion command
t = 6.4 s          计划 vx/vy 一拍直接切换为零，不做线性减速
6.4 <= t < 8.0 s   保持零计划命令，观察直接停车后的响应
0.0 <= t < 8.0 s   唯一 headline 评价区间
record_end >= 8.06 s  仅为最后一个 54 ms horizon 保留标签尾段
```

选择 8.0 s 的原因：

- 6.4 s 行走正好覆盖 8 个 0.8 s 步态周期；前三周期启动后，仍保留多个后续周期。
- 停车后保留 1.6 s，即两个步态周期，用于观察直接跳零的峰值和余振。
- 按历史位移估算约走 3.18 m，比旧实验约 4.78 m 缩短三分之一；实际距离以 T1
  第一次 smoke 为准。

如果 smoke 发现距离或场地仍不合适，总控窗口可在正式批量采集前将协议缩短为
`0–5.6 s` 行走、`5.6–6.4 s` 零命令。正式数据开始采集后不得再修改时序。

“直接切零”指计划平移命令 `vx/vy` 归零。heading control 全程保持开启，因此最终
送入下肢策略的 `wz` 仍可能包含很小的 heading 闭环修正。数据中必须同时记录计划命令
和最终 runtime command。

任务 `t=0` 是 command 已发布且 predictor 可以读取的第一拍；`[0,0,0]` 也是合法命令，
不能用“命令非零”作为 predictor 开关。

## 5. 纯模板的新定义

### 5.1 不再使用旧稳定周期模板

新的 `full_task_template` 完全重新采集和构建。旧稳定模板只保留为历史对照，不能成为
新模板的一部分。

新模板不是“第一周期 + 第二周期 + 第三周期 + 稳态周期”的拼接，也不把第四、第五、
第六周期混在同一个 phase bin。它按绝对任务时间工作：

```text
template[t] = 多次实验在同一个 task time t 的完整未来扰动窗口平均
```

因此第一、第二、第三以及后续每个周期都保留自己的时间位置，没有周期之间的人工接缝。
代价是它只适用于这一套固定命令和固定停车时刻；这应被称为固定任务 baseline，而不是
通用扰动预测器。

### 5.2 最简产物

第一版只需要：

- 每次实验一个 2 ms 原始 episode；
- 一个 `full_task_template.npz`；
- 一个 `manifest.json`；
- 一套个体轨迹、均值/离散带和 held-out 对比图。

不要同时建立 raw、half-smoothed、fully-smoothed 等多个版本。先看未经人为平滑的平均结果，
确实发现问题后再由总控窗口决定是否处理。

每个 6 ms MPC anchor 直接保存完整未来窗口，而不是先平均单点再临时拼接：

```text
nodes:     [anchor_count, 10, ...]
intervals: [anchor_count,  9, ...]
```

node 0 在线时始终由当前实测覆盖。最后一个 headline anchor 仍需要未来 54 ms，因此原始
采集必须延长到至少 8.054 s，但额外尾段不进入 headline。

### 5.3 H 系

- 第一周期：使用从任务开始到当前 anchor 的 yaw 因果圆周均值；第一帧就是当前 yaw。
- 后续周期：使用上一完整周期的 yaw 圆周均值。
- 每个 54 ms horizon 内固定使用 anchor 当时已经可得的 H，禁止 future yaw leakage。
- 姿态使用相对 anchor 的旋转，并采用合法的 SO(3) 处理，不能直接平均出非法旋转矩阵。
- 模板构建、离线检查和在线 predictor 必须调用同一套 H 定义。

### 5.4 固定停车时间带来的信息优势

绝对时间模板天然知道 6.4 s 将要停车，所以在停车前最多 54 ms 就可能读到停车后的模板。
而第一版神经网络只读取当前 command，不使用 future-command preview。

T1/T2 可以把模板作为“固定任务、已知时间表”的强 baseline；但到 N2 正式比较前，必须由
总控窗口决定：

1. 明确披露模板具有固定时间表信息优势；或
2. 给两个方法相同的停车预告信息；或
3. 限制模板在零命令实际到达前不得跨入停车段。

不得把信息条件不同的结果表述成完全公平的预测器准确率比较。

## 6. 重复实验怎样避免一模一样

完全确定性的 MuJoCo 在相同初态、相同命令和相同控制器下会生成相同轨迹。仅仅更换一个
没有实际作用于状态的 seed，再运行 18 次，没有统计意义。

### 6.1 构建模板时保持不变的量

- nominal command、启动时刻和直接停车时刻；
- gait phase/time origin；
- 右臂 MPC 结构、权重和执行链；
- heading controller 的增益和参考定义；
- payload、地面参数和模型参数。

不要通过混合不同 MPC 权重或 heading 增益来“制造数据差异”。那会把多个不同控制系统的
轨迹平均成一个模板，最终模板不再对应实际使用的任何一个控制器。此类变化只适合后续单独
做鲁棒性实验。

### 6.2 第一版允许变化的量

优先复用仓库已经使用过的安全初态扰动：

- 下肢 12 个关节初始位置：零均值小扰动，标准差约 `0.006 rad`，限制在
  `[-0.018, 0.018] rad`；
- 下肢 12 个关节初始速度：零均值小扰动，标准差约 `0.01 rad/s`，限制在
  `[-0.03, 0.03] rad/s`。

正式 build 集合使用成对的 `+delta/-delta`，再加一个 nominal episode，使平均初态不发生
偏移。建议先做 11 条 build（nominal + 5 对）和 4 条 held-out（2 对）；是否增加数量，
根据实际离散程度决定，不能认为“运行越多自动越好”。

初始 yaw 可以使用小幅成对变化，但 heading reference 应随初始 yaw 一起旋转，避免人为制造
heading correction 冲击。由于 H 系会消除大部分纯 yaw 差异，yaw 变化主要用于验证坐标不变性，
不是主要的数据多样性来源。

如果上述 q/dq 扰动产生的轨迹仍几乎完全相同，不要盲目增加 episode 数。应先停止并交给
总控窗口判断，下一层候选才是经过安全 smoke 的小幅 torso roll/pitch、base linear/angular
velocity 扰动。第一版不要同时引入摩擦、执行器强度、传感器噪声和多个控制参数。

### 6.3 必须展示的数据多样性证据

T1 图中必须同时显示：

- 每条 build 轨迹的浅色曲线；
- 跨 episode 均值和标准差/分位带；
- 各 episode 起始 q/dq 扰动摘要；
- 任意两条轨迹的最大差异或跨 episode 方差；
- held-out 真值与模板预测。

如果轨迹仍完全重合，应明确报告“重复采集没有新增信息”，而不是把零方差包装成模板很准。

## 7. 模型与推理档位

`Sol / Terra / Luna` 是模型档位；`High / Extra High(xhigh) / Max` 是推理强度。
Codex `Ultra` 更接近多代理并行模式，不是简单的“比 Max 再高一级”。

当前环境提供的组合为：

- Sol：High、xhigh、Max、Ultra；
- Terra：High、xhigh、Max、Ultra；
- Luna：High、xhigh、Max；当前没有 Luna Ultra。

官方 API 文档列出的 GPT-5.6 reasoning effort 是 `none/low/medium/high/xhigh/max`；
这里的 Ultra 是 Codex 工作模式。模型定位与 effort 说明见
[OpenAI Model guidance](https://developers.openai.com/api/docs/guides/latest-model)。

推荐路由：

| 工作 | 推荐模型 |
|---|---|
| 明确步骤的检查、补跑一条 smoke、固定格式画图 | Luna High |
| 常规模块实现、实验汇总、小范围修正 | Terra High |
| 完整时间模板构建和普通疑难诊断 | Terra xhigh |
| MPC horizon 接入、GRU/姿态链、在线 neural 接入 | Sol xhigh |
| xhigh 无法定位的跨层根因诊断 | Sol Max |
| 最终把数据泄漏、控制接口、统计分开独立终审 | Sol Ultra；预算敏感可 Terra Ultra |

Token 充足并不意味着每一步都应使用 Sol Ultra。重复运行和固定画图交给 Luna/Terra，真正涉及
坐标、因果、SO(3) 和控制接口时使用 Sol xhigh。

## 8. 当前阶段：T1 纯模板离线构建

状态：**待开始**

推荐模型：**GPT-5.6 Terra，Extra High / xhigh**

### 给开发窗口的当前提示词

```text
请读取仓库根目录 PREDICTOR_DEVELOPMENT_TASKBOOK.md，并只执行其中的 T1：
“纯模板离线构建”。不要进入T2、N1或N2。

先复核git状态。预期基线是feat/predictor-interface@afc126e，另外只有新建但尚未
提交的PREDICTOR_DEVELOPMENT_TASKBOOK.md；如有其他改动，停止并报告，不要覆盖。
若状态完全符合且feat/full-task-predictors尚不存在，创建并切换到这个新分支；
若同名分支已经存在，不要覆盖，先报告。

本阶段目标很简单：用多次完整任务轨迹建立一条按绝对task time索引的新模板，
并把模板长什么样展示出来。不要使用或拼接旧稳定周期模板，也不要接入正式MPC控制路径。

使用任务书暂定的direct-step协议：0–6.4 s直接保持nominal command，6.4 s计划
vx/vy一拍跳零，6.4–8.0 s保持零命令；heading control全程开启；headline为0–8.0 s；
为最后54 ms标签继续采集到至少8.06 s。计划命令和heading修正后的runtime command都要记录。

先实现一条distance smoke并报告实际XY起终点位移。如果仿真安全且距离没有明显超出约
3.2 m的预期，可以在同一阶段继续；若距离或稳定性明显不合适，立即停止，不要批量采集。

命令、右臂MPC参数、heading参数、payload、物理模型和gait time origin全部固定。
只改变小幅、可复现的初始下肢q/dq，并使用+delta/-delta成对设计。先做11条build
（nominal+5对）和4条held-out（2对）。如果不同run仍完全重合，不要盲目继续增加数量，
应停止并报告离散程度。

每2 ms保存原始数据。对每个6 ms anchor，直接在build episodes之间平均完整的
10-node/9-interval未来窗口。第一周期H使用0..t的yaw圆周均值，后续使用上一完整
周期均值，一个54 ms窗口内冻结anchor H，禁止future leakage。node0后续在线时必须
由实测锚定，姿态平均必须保持合法SO(3)。

第一版只生成一个full_task_template.npz、一个manifest和必要图片，不创建多套平滑模板。
图片至少包括：所有个体轨迹+均值/离散带、0–2.4 s启动放大、6.2–8.0 s直接停车放大、
held-out真值对比、初态扰动与轨迹离散程度、实际行走距离。

增加足够证明时间索引、H系、窗口shape、SO(3)、build/held-out隔离和可复现性的测试。
不要修改MPC权重、现有predictor模式或正式闭环评价；不要commit或push。

完成后必须停止，并只交付：
1. 是否达到T1验收条件；
2. 改动文件和运行命令；
3. 测试与distance smoke结果；
4. 模板、manifest和图片的绝对路径；
5. 多次轨迹是否真的不同，以及离散程度；
6. held-out误差和发现的问题；
7. 你建议的下一阶段名称、模型和理由，但不要自行开始下一阶段。
```

### T1 验收门槛

- 实际运行没有继续使用旧稳定模板数据；
- 任务时序确实是直接启动、直接切零，没有隐藏 ramp/cooldown；
- heading control 开启；
- 多次轨迹不是毫无意义的完全复制，初态扰动零均值且安全；
- 完整模板和 held-out 图可以直接查看；
- 无非法旋转、NaN/Inf、时间错位或 future yaw leakage；
- 尚未接入正式闭环。

## 9. 后续阶段草案（不得机械自动执行）

以下内容用于提前说明方向。每个阶段真正开始前，由总控窗口结合上一阶段结果生成最终提示词。

### T2 草案：纯模板闭环

推荐模型：Sol xhigh。若只补跑固定 seed 或重画图，可改用 Luna High。

目标：新增独立 `full_task_template` predictor，首拍有效、node 0 实测、完整提供
10 nodes/9 intervals，并在统一 0–8.0 s 协议下与旧 baseline 做配对闭环比较。

必须先解决或披露“绝对时间模板提前知道停车”的信息条件；不得自动调 MPC 权重来改善结果。
headline 只使用全程，分段仅作诊断。报告水杯/右 EE tilt RMS、p95、max，acc/alpha，QP、
fallback、完整 6 ms timing 和行走距离。

### N1 草案：完整 GRU 离线预测

推荐模型：Sol xhigh。训练管线冻结后的重复训练和画图可使用 Terra High。

每帧仍为 50 维；有效历史长度从第一帧 `L=1` 逐渐增加到最多 `L=34`。采用支持
length/mask 的变长窗口 GRU，不能等待 34 帧。正常预测路径不用模板，也不使用 future
command preview。

第一版可预测 absolute 9×6 `acc_H/alpha_H`，从当前实测 omega/rotation 出发，通过一致的
积分构造未来 omega/rotation；先做 oracle-alpha rollout，分离积分误差和模型误差。

正式数据不仅包含固定直线任务，还应覆盖策略安全范围内的启动、直接停车、零命令、不同
vx、正负 vy 和正负 wz；按完整 episode/seed 划分 train/val/test。结束时必须给出从第一帧
开始的预测—真实图、前三周期和停车放大图、horizon-wise误差、姿态误差和CPU推理时间。

### N2 草案：完整神经网络闭环

推荐模型：Sol xhigh。固定实现后的批量配对实验可使用 Terra High。

新增 `full_neural`，保留旧模式。第一帧 `L=1` 就必须输出，正常路径不读取模板；异常安全
fallback必须有reason code，正式headline内fallback应为零。完成offline-online replay parity、
reset/时间断点/SO(3)/旧模式回归和短smoke后，才运行统一全程闭环比较。

结果必须直接与通过验收的纯模板比较；如果神经网络姿态或水杯指标更差，应如实报告并停止，
不能通过更换评价窗口或偷偷调MPC来解释掉。

## 10. 每阶段统一回报格式

开发窗口在任何阶段结束后均按以下格式回报，并停止：

```text
阶段：
状态：PASS / PARTIAL / FAIL

改动文件：
运行命令：
测试结果：
关键数字：
产物绝对路径：
发现的问题：
尚未验证的内容：

建议下一阶段：
建议模型与推理档位：
建议理由：

我尚未开始下一阶段。
```

开发窗口可以提出建议，但是否继续以及下一条最终指令由总控窗口决定。
