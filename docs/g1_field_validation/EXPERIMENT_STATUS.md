# 当前实验进度

更新：2026-10-08 收尾／新对话交接；每完成一轮，更新对应一句。早期 session 文档中的“下一步”是当时计划，当前顺序以本文件为准。

**当前：学习前馈 MPC 的线加速度、角加速度、角速度代价均已完成真机实验；保留 `0.01 / 0.0005 / 1.0` 为当前行走基线。下一轮只验证新增轻度肩 roll 回正的行走效果。**

- 6 ms PID：已完成真机行走、停车和退权，保留为对比基线，无需重跑作为前置步骤（[记录](sessions/20261005_PID_6MS_FIELD.md)）。
- 旧版不加学习前馈 MPC：已首次完整走停、平顺退权，成功版本和结果已保存（[记录](evidence/20261007_first_complete_mpc/README.md)）。
- 右臂小幅系统辨识：正式三轮 `121118`、`121256`、`121407` 均平顺、完整退权，联合分析已完成；重复性较好，但不足以可靠分离惯量／摩擦／力矩估计偏差，**暂不写回 MPC 参数，也不再重复同配置采集**（[结论与证据](sessions/20261008_ARM_IDENTIFICATION.md)）。
- 新版微调 MPC、不加学习前馈：首次运行在停车阶段因肩 pitch 边界退出；加入预测内轻度 pitch 回正后，`130817` 已完整走停、正常退权且操作者确认平顺，右肩 pitch 到 `-15°` 边界仍有 3.59° 余量（[失败与修正](sessions/20261008_MPC_REFINED_HOLD_FIELD.md)，[成功对照](sessions/20261008_MPC_REFINED_AND_LEARNED_SUCCESS.md)）。
- 新版 MPC、加学习前馈：`131606` 已首次完整走停、正常退权且操作者确认平顺并看到右臂随身体补偿；右瓶 H0 竖直倾角 RMS 0.552°，但动态加速度未全面优于 hold/PID，一对运行不作统计性结论（[成功对照](sessions/20261008_MPC_REFINED_AND_LEARNED_SUCCESS.md)）。
- 学习前馈＋末端线加速度代价 0.01：独立配置 `145841` 已完整走停并正常退权；RMS/P95 有改善但峰值和 pitch 余量变差。连续 stationary 表明 MPC 将右瓶倾角从约 5.68° 调到 0.26°，但 0.8 s 接管仍显得快。直接在 3.0 s 接管时真实右肘尚以约 -0.66 rad/s 追赶，导致力矩包络退出；现依据实测改为 3.3～4.8 s 平滑接管。
- 学习前馈＋`q_ee_acc=0.01`＋`q_ee_alpha=0.0005`：独立配置 `162906` 已完整走停、正常退权且操作者确认顺利。右末端全流程线/角加速度 RMS 为 `1.894 m/s²` / `5.650 rad/s²`，瓶轴倾角 RMS `0.776°`；收益主要出现在停车阶段，单轮对比不足以把整体行走波动都归因于角加速度代价。
- 学习前馈＋三项动态代价，`q_ee_omega=1.0`：独立配置 `164546` 已完整走停并正常退权。相比 `162906`，右末端角速度、角加速度和瓶轴倾角 RMS 分别降低约 `18.5%`、`7.2%` 和 `8.6%`，线加速度基本不变；力矩与肩 pitch 边界余量同时改善。
- `q_ee_omega=2.0`：`165549` 已完整走停、正常退权，但右末端线／角加速度 RMS 为 `2.318 m/s² / 6.315 rad/s²`，差于 `1.0` 的 `1.912 / 5.243`；操作者报告疑似异响，是否实际擦碰未确认，不选作下一轮基线。
- `q_ee_omega=0.5`：`170732` 已完整走停、正常退权，右末端线／角加速度 RMS 为 `1.987 m/s² / 5.274 rad/s²`，倾角 RMS `0.697°`；并非所有指标都差于 `1.0`，但综合当前单轮结果仍保留 `1.0`，不宣称已证明全局最优。
- `q_ee_omega=1.0`＋轻度肩 roll 回正：`171551` 静止实验已正常结束、最终 `weight=0`，右瓶倾角 RMS `0.326°`；roll 目标为既有中立位 `+1°`，预测内增益 `kp=1.0、kd=0.1`，保持原发包 PD 增益和物理限值。
- roll 回正行走：`172126` 虽执行完，但操作者确认吊绳多次牵拉，**整轮作废，不用于评价控制效果或声明行走验证通过**；原始目录已按批准的清理清单删除，需补一轮有效行走。
- 十二条 H0 行走轨迹及扰动预测研究：已完成采集和离线分析并接入新版 MPC；首次未见训练的真机运行表明滤波扰动包络基本对准、幅值偏保守（[研究](H0_WALK_PREDICTOR_STUDY.md)，[真机结果](sessions/20261008_MPC_REFINED_AND_LEARNED_SUCCESS.md)）。

## 接下来要做的事

1. 下次现场在已使用的 Regular Motion/FSM 500、自主平衡静止状态下，运行 `learned_filtered`＋`q_ee_acc=0.01、q_ee_alpha=0.0005、q_ee_omega=1.0`＋roll 回正的一轮行走；静止轮已完成，无新增异常时不再将 PID、辨识或静止测试作为重复前置步骤。
2. 使用独立新目录保存结果，现场确认吊绳无牵拉、手臂无擦碰／异响及停车放臂情况；如出现异响，先检查原因，不把完整跑完当作无异常证明。
3. 与 `164546` 的无 roll 回正成功轮比较完整走停 `[5,18)`：同时报告左／右瓶倾角、线加速度、角加速度、角速度，以及肩 roll 漂移、pitch/yaw 边界余量、力矩余量和计算耗时。
4. 若 roll 漂移改善且未明显损害端正／动态指标，则保存为下一成功基线；再依据数据决定下一项小幅调参。当前不再增加新代价，不同时改三个权重，也不预先放宽约束。

## 新窗口最小交接

- 仓库：`/home/fjk/g1_ws/hold-my-beer-mpc`；先读本文件，再按任务深度选择下面的材料，无需重新扫描全部历史、官方 PDF 或已结束的磁盘清理任务。
- 机器人：G1 23DoF、每臂五个电机、腰只有一个实际关节；双手各绑 250 g 水瓶。左臂按既有非零中立角 PD 固定，右臂运行 6 ms 加速度 MPC＋耦合的五关节逆动力学力矩执行；腿部使用内置行走，手臂使用 `rt/arm_sdk`。
- 坐标与评价：H0 的 yaw 零点来自走前 3～5 s 的平均朝向，冻结后不随身体转动；主指标是瓶身中心／瓶轴在 H0 的完整走停表现，原始 IMU 仍保留原值。
- 已有轻度 pitch/yaw 回正在预测内，新 roll 回正也在预测内；新版是直接 `M*a+b`，没有多候选力矩搜索。系统辨识已完成，但未得到可靠可分离参数，因此没有将辨识结果强行写回质量、惯量或摩擦模型。
- 程序入口：`tools/g1_commissioning/g1_walk_mpc_learned.py`；共享运行入口 `g1_walk_mpc.py`；预测／回正逻辑 `hardware_mpc_learned.py`（均位于 `tools/g1_commissioning/`）。
- 下一轮必须显式选择 `configs/hardware_mpc_learned_acc001_alpha0005_omega1.yaml` 和 `configs/hardware_mpc_torque_learned_roll_center.yaml`；仅使用入口默认配置并不等于三项动态代价版本。退回已成功行走版时，保留前者，只把 torque 配置改为 `configs/hardware_mpc_torque_learned.yaml`。
- 既有流程：0～3 s 抬臂；3.3～4.8 s 平滑切入 MPC；5～15 s 请求 `0.5 m/s` 前进；15～18 s 停车并继续航向保持；随后约三秒渐退权。速度×时间不是实测距离或距离限位。
- 环境：Python `/home/fjk/miniforge3/envs/g1_mpc/bin/python`，网卡 `enx6c1ff701509c`，主机连接脚本 `tools/g1_commissioning/connect_g1_network.sh`；沿用 `--cpu 7 --rt-priority 20 --compute-process` 和既有主机 performance 准备，不能仅凭历史记录假设权限／网络此刻有效。
- 现场 profile：`evaluation/hardware_shadow/commissioning/g1_pid_6ms_20261005_132052/arm_profile.conf`。收到下一次现场运行指令再执行真实输出；仅交接或查看进度不启动机器人。

### 建议阅读深度

1. **只为继续下一轮现场实验（默认）**：读本文件、检查 `git status`／`git log -1`，再读 [学习 MPC 程序说明](HARDWARE_MPC_LEARNED.md) 的运行入口、配置和 fail-safe 部分，以及 [2026-10-08 成功对照](sessions/20261008_MPC_REFINED_AND_LEARNED_SUCCESS.md)。这已经足够生成下一轮命令和分析数据。
2. **需要理解仿真到真机的控制逻辑**：再读 [仿真 MPC 设计](../design/MPC_DESIGN.md) 和 [仿真完整任务](../simulation/FULL_TASK_TEMPLATE.md)，对照 `main_sim.py`、`arm_mpc.py`、`sim_support.py`。重点区分：仿真使用 MuJoCo 全模型做候选力矩／正动力学检查；真机当前使用实测 `q/dq`、移动基座 IMU 和耦合 5×5 手臂模型直接计算 `M*a+b`，没有在线复制完整接触仿真。
3. **需要修改真机 MPC 数学或执行模型**：完整阅读 [学习 MPC 程序说明](HARDWARE_MPC_LEARNED.md)，然后检查 `g1_walk_mpc.py`、`hardware_mpc_learned.py`、`hardware_arm_inverse_dynamics.py` 及相应测试；必须先保留当前成功配置，单次只改一个可解释因素。
4. **需要与 PID 比较**：读 [真机 PID 说明](HARDWARE_PID.md)、[6 ms 真机记录](sessions/20261005_PID_6MS_FIELD.md)，对照 `arm_pid.py` 和 `g1_walk_pid.py`。PID 已是完成的对照组，不要把重新调 PID 或重跑 PID 当作 MPC 下一轮的前置条件。
5. `MPC_DEVELOPMENT_LOG.md`、早期失败 session 和旧仿真结果只在追溯具体设计理由或故障时按关键词查阅，不建议新窗口从头通读。

## 当前证据与版本状态

所有下列实验目录均位于 `evaluation/hardware_shadow/commissioning/`；分析可读各自的 `endpoint_analysis/summary.json`、`execution_analysis/summary.json` 和 `field_review/summary.json`。

| 用途 | 实验目录 | 原始数据状态 |
| --- | --- | --- |
| 当前三代价成功行走基线 | `mpc_learned_acc001_alpha0005_omega1_walk_20261008_164546` | raw 与分析均保留 |
| 下一候选的已完成静止验证 | `mpc_learned_omega1_roll_center_stationary_20261008_171551` | raw 与分析均保留 |
| omega=2 对照 | `mpc_learned_acc001_alpha0005_omega2_walk_20261008_165549` | raw 已清理，配置和分析保留 |
| omega=0.5 对照 | `mpc_learned_acc001_alpha0005_omega05_walk_20261008_170732` | raw 已清理，配置和分析保留 |

本次交接核对的 raw SHA256：`164546` 为 `c1400bd114cba1890278a4d27d4726e95c9c670fe92d0941eb83974a80b90919`；`171551` 为 `bb7642bb1b1af623ad3808380942b3be17e71d8c5ed326b368a629d47588efe4`，均与已存分析一致。

omega=1 成功基线最初发布于 `ac270363acb99d0b29c421f858beb7e0913ef7c4`。本轮交接提交继续保存：

- `tools/g1_commissioning/g1_walk_mpc.py`、`hardware_mpc_learned.py`、`tests/test_mpc_learned.py` 中的独立配置准入、预测内可选 roll 回正及相应测试。
- `configs/hardware_mpc_learned_acc001_alpha0005_omega05.yaml`、`configs/hardware_mpc_learned_acc001_alpha0005_omega2.yaml` 两个已完成对照配置，以及下一轮使用的 `configs/hardware_mpc_torque_learned_roll_center.yaml`。
- 本文件中的当前结论和下一步。交接提交不代表 roll 回正行走已验证；目前只有静止验证有效，`172126` 行走因吊绳牵拉作废。
- 新窗口仍应先执行 `git status --short --branch` 和 `git log -1 --oneline`，以仓库实际 HEAD 和工作树为准。
- 项目及系统缓存清理已完成，系统盘最近检查剩余约 25 GiB，无需继续清理。十二条 H0 原始轨迹、五轮辨识、6 ms PID、主要成功 MPC 和 `assets/g1_hardware_mpc_predictor` 已保留；早期离线回放与预测研究中间 NPZ 等已删除，历史文档中的部分路径不再存在。
- `evaluation/` 被 Git 忽略，原始真机数据主要在本机；代码提交不等于 raw 备份。跨电脑继续时需要另行带上成功原始记录；上述配置随 Git 同步。同一电脑的新对话可直接读取本地数据。
