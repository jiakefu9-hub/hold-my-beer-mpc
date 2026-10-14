# 关键真机证据归档：论文与手机交接

2026-10-15归档，实验与分析截至10月14日。实验已完成；这里是阅读和复核入口，不是新的真机运行指令。
项目解释及可复制提示词见[手机交接](../../../PROJECT_HANDOFF_FOR_MOBILE.md)。

## 手机上先读这些

| 问题 | 已随Git提交的证据 |
|---|---|
| 起步、中途与停车到底有什么差别？ | [分段完整报告](reports/phase_comparison_20261014/REPORT.md)、[六项指标图](reports/phase_comparison_20261014/phase_metrics.png)、[分段JSON](reports/phase_comparison_20261014/summary.json) |
| 三种主方案的完整任务表现？ | [上午六轮报告](reports/morning_comparison_20261013_final/REPORT.md)、[完整JSON](reports/morning_comparison_20261013_final/summary.json) |
| MPC与强前馈PID C谁更好，时间对齐有没有用？ | [12轮逐轮结果](reports/final_comparison_20261012/REPORT.md)、[参数/计时/汇总JSON](reports/final_comparison_20261012/summary.json)、[解释与接触说明](../../sessions/20261012_FINAL_COMPARISON.md) |
| 滤波、执行跟踪、完整循环计时是否检查过？ | [工程审计](reports/engineering_readiness_20261013/audit/REPORT.md)、[审计数据](reports/engineering_readiness_20261013/audit/summary.json)、[离线回放结果](reports/engineering_readiness_20261013/replay/summary.json) |
| 预测更准确为什么不等于控制更好？ | [时间对齐误差](reports/forecast_timing_20261010/labels/REPORT.md)、[误差数据](reports/forecast_timing_20261010/labels/summary.json)、[原版输出一致性](reports/forecast_timing_20261010/baseline_regression.json) |
| 右手落脚振动是不是更大？ | [解释](../../sessions/20261014_LEARNED_MPC_SPIKE_AUDIT.md)、[五轮尖峰及频谱数据](reports/spike_review_20261014/summary.json)、[最新波形](reports/spike_review_20261014/latest_linear_spikes.png) |
| 是否是为了保持姿态而产生大动作？ | [姿态/尖峰机制检查](reports/posture_spike_causality_20261014/REPORT.md)、[固定反馈消融数据](reports/posture_spike_causality_20261014/ablation.json)、[98个事件时序](reports/posture_spike_causality_20261014/event_timing.json) |
| 能不能预测落脚后硬切PD？ | [预测可用性报告](reports/impact_predictability_20261014/REPORT.md)、[数据](reports/impact_predictability_20261014/summary.json)；当前没有可替代成功基线的候选 |
| 预测器本身如何验证？ | [H0研究](../../H0_WALK_PREDICTOR_STUDY.md)、[汇总CSV](reports/walk_h0_predictor_study_20260925/benchmark/metrics_aggregate.csv)、[逐轨迹CSV](reports/walk_h0_predictor_study_20260925/benchmark/metrics_per_trial.csv) |
| 想找某轮具体数据？ | [运行索引](RUNS.md)、[机器可读清单](run_index.json)、[跨组关键指标CSV](key_metrics.csv) |

## 保存了什么，没保存什么

- 29条运行记录：12轮T系列、6轮上午主对照、4轮最终录像、2轮历史基线、2轮10月14日尖峰运行、3轮明确排除记录。
- 其中20轮保存**完整、无损压缩的原始`raw.jsonl.xz`**：上午6轮、录像4轮、原MPC/C各3次共6轮、历史2轮、最新2轮。
- 所选运行保留实际配置、profile、生命周期事件、控制命令与计时列。已有的末端完整200 Hz NPZ和图表保持原文件字节；10月14日两轮使用尖峰审计内的重建NPZ，没有虚构旧版`endpoint_analysis`。
- 其他6轮T探索/对齐及3轮排除记录保留配置、事件、可用指标和来源哈希，**没有在本归档保存完整原始日志**。所以本包不是全量实验备份。
- 不含手机拍摄视频、全部H0训练采集原始日志、全部仿真大轨迹或生成的完整离线回放JSONL。仿真的四套研究摘要和配置原本已在[仿真实验目录](../../../simulation/experiments/)内，无需重复上传。
- 原本机`evaluation/`不移动、不删除，仍被Git忽略。此次是明确选取后复制到可跟踪目录，不是取消全局忽略。未加入打印切片文件或无关资料。
- 原日志保留运行主机路径、网卡名与软硬件配置等技术元数据，并非匿名数据集；没有上传登录凭据。

文件逐项SHA-256、原路径、来源哈希、是否原样复制/派生/无损压缩，见[manifest.json](manifest.json)。原日志归档总约268 MB；整个证据目录约573 MB（十进制，含逐时刻数据）。均为普通Git文件，无需Git LFS。手机优先读取小报告、JSON、CSV和图片，不必下载或解压全部日志。

## 关键结论及比较边界

上午每组两轮，统一`[5,18)`，右端竖直倾角均值/平均轴摆动RMS：学习MPC`0.6308°/0.5819°`，保持MPC`0.6263°/0.5742°`，位置PID`2.3550°/2.9576°`。对应线加速度`2.0938/2.1178/1.8649 m/s²`。MPC姿态较好，不能写成线加速度也全面优于位置PID。

学习MPC中途左/右摆动`0.6346°/0.5178°`，停车`1.8519°/0.6215°`：相对左手优势在停车更明显，但右手自身停车不一定更稳。原位置PID起步含初始纠偏；放宽限速PID作为独立组，不混入原PID重复。

T系列原MPC与C各三轮：MPC两种姿态指标更好，C部分角运动指标更低，线加速度差异小于轮间波动；两套工作配置的时间对齐开关不同。完整同对齐对照也保存，不能把架构与设置差异混为一谈。

起停主题是实验后正在考虑的论文定位，并非预注册任务或导师确认题目。必须保留持续行走、完整任务、不利指标和失败记录。`hold_current`仅关闭学习预测，仍有当前IMU和动力学前馈；C是受约束控制分配的强对照。

## 指标与文件怎么读

| 文件/指标 | 含义 |
|---|---|
| `endpoint_analysis/metrics.npz`，历史PID为`analysis/metrics.npz` | 原有完整200 Hz重建数据，保留完整时间序列，含关节、身体和末端姿态/加速度；不是独立末端传感器真值 |
| `trajectory_preview_50hz.csv` | 从上述数组在任务3～21秒每4点取1点，方便手机浏览；**不用于峰值分析或重新计算论文指标**；列后缀0/1/2为x/y/z，四元数0/1/2/3为w/x/y/z |
| `commands.csv` | 每个记录到的`dds_write`选定字段展开，没有采样降频；原始字段位置关系和完整嵌套诊断以raw为准。五维右臂顺序pitch/roll/yaw/elbow/wrist，较长指令数组遵循原SDK/profile槽位，不将两类索引混用 |
| `timing.csv` | 保留原有timing及complete-cycle行，可能同一拍有两种记录，按schema区分，不能把两行当成两拍 |
| `events.json` | 生命周期与配置概要，以及原始schema计数；是摘录，不代表所有事件 |
| 竖直倾角均值 | 每帧末端+Z与H0竖直方向的夹角再平均 |
| 平均轴摆动RMS | 每帧+Z与本窗口归一化平均轴的角距离RMS；不同于无符号竖直倾角标量标准差 |
| 线/角加速度和角速度 | 三维向量模RMS；统一105 ms三阶SG派生，75/135 ms敏感性另存 |
| 相对速度 | 末端相对torso IMU位置在H0中的导数，去窗口均值；不是世界系绝对行走速度 |

合并起步/停车先在完整连续数据上求导，再选两个窗口的样本；不会把两段拼接后求导。主摆动使用两段共同平均轴，包含阶段间中心变化；另存各段分别定中心的结果。每轮先算指标，再按运行平均，不把时间采样当成独立重复。

## 有效性与保留冲突

- T01 160554旧窄边界中止、T03 163955操作者报告线缆绊扰、T12 174249 DDS初始化失败：均不纳入控制效果主比较，来源记录保留。
- T05 165900操作者从录像观察到肩/上臂接触躯干；按操作者评价口径保留，但必须标注接触，不称确认无接触。
- **T02 163351目录有旧`INVALID_RUN.md`，与后来最终纳入清单冲突。**当时用户明确说“不算”的是T03，T03的`operator_validity.json`及后续最终汇总也排除T03而纳入T02。本包保留两份原始标记，保持最终汇总原值并显式披露冲突；不能把T02当作无争议的开对齐证据，更不能静默删除旧标记。若以后发表需对T02做排除敏感性检查。此冲突不影响上午六轮和最新分段的主结论。
- M08左增益加倍、M10交互暂停和人工左臂扰动不混入正常左PD主对照。尖峰报告保留的M08/M10诊断数组是辅助资料，本包不包含它们完整raw。

## 无需原电脑的离线复核

在任意克隆目录，文件校验仅需Python标准库；加`--deep`逐个解压核对20轮原始日志哈希，内存流式处理，不写文件：

```bash
python3 docs/g1_field_validation/evidence/20261014_final_archive/verify_archive.py --deep
```

已安装NumPy/SciPy时，可以仅用归档NPZ重算10轮上午/录像的全部分段六指标，并核对汇总；无需原`evaluation/`、MuJoCo或机器人SDK：

```bash
python3 docs/g1_field_validation/evidence/20261014_final_archive/recompute_phase.py
```

`raw.jsonl.xz`是标准XZ流，解压后与本机原始文件逐字节一致。历史分析脚本在`reports/`中原样保存，部分带原电脑绝对路径，作为方法来源；若要重跑整个原始状态到运动学重建，应先按`manifest.json`恢复所需文件、安装原分析依赖并显式调整路径。不能声称所有历史分析在空白云环境无需准备就能运行。

[build_archive.py](build_archive.py)记录选取和导出规则；它仅供仍拥有本机源数据时重新导出，不是手机使用的前置步骤。[验证记录](VALIDATION.md)说明本次实际通过的检查。
