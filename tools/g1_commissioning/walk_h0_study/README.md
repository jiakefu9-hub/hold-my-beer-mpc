# H0 真机扰动预测研究：程序与复算入口

这是**离线研究**文件夹，没有 SDK 初始化、网络连接或机器人输出。旧五条带牵拉的轨迹已撤回，
新模型不使用它们。旧程序和原始日志不删除，便于审计先前数字。

本次结果和结论见 [H0 十二条研究报告](../../../docs/g1_field_validation/H0_WALK_PREDICTOR_STUDY.md)。
后续探索见[误差百分比与改进报告](../../../docs/g1_field_validation/H0_WALK_PREDICTOR_REFINEMENT.md)。

## 每个文件负责什么

| 文件 | 职责 |
| --- | --- |
| `prepare.py` | 明确十二条来源和整条轨迹划分；复用原始审计器，检查 CRC／时间／丢帧／RPC／结束记录；转固定 H0，因果重采样和滤波 |
| `methods.py` | 每种方法的输入、预测函数、标准化、线性回归、近邻权重、相位估计和预测目标定义 |
| `benchmark.py` | 只在开发轨迹上选参数，再评估保留轨迹；保存训练模型、逐点预测／真值和完整误差表 |
| `plots.py` | 十二条信号图、开发集关联分析、运动学事件时序分析、HTML／PDF |
| `plot_predictions.py` | 从保存的预测／误差表画比较图，不重新拟合、不手填分数 |
| `verify_results.py` | 从保存模型重建新预测和标签、复算误差、导出一个八近邻计算例子；第 06 条单独补充检验 |
| `verify_legacy.py` | 独立重建旧五条的 24 维近邻预测，核对旧三个数字的来源；不是新模型的训练步骤 |
| `test_*.py` | 合成数据测试：坐标变换、因果性、标签区间、模型和误差公式 |

复用的 `../analyze_walk_dataset.py` 只负责旧有通用原始日志审计和因果滤波；新划分、H0 转换和
模型实验都在本文件夹。没有把已有采集／PID／MPC 控制程序迁进来或改变其行为。

## 方法到底怎样算

下面的方法均由 `methods.py` 实现；不是根据印象填一张比较表。

| 名称／程序键 | 实际计算 |
| --- | --- |
| 沿用当前测量 `zoh` | 未来各点都等于当前因果滤波后的身体数据 |
| 按时间平均 `clock_average` | 训练轨迹同一程序时刻的未来数据取平均，不识别腿部动作 |
| 相位平均 `phase_average` | 从左髋 pitch − 右髋 pitch 的同向过零识别重复动作；训练完整周期平均成 128 格；在线只用过去最多三周期估计进度；尚无足够事件时用当前值 |
| 双髋角＋时间 `hips_clock_ridge` | 两个髋 pitch 角＋程序时间及其二、三次项，拟合带正则的线性模型；这里是程序时间，**不是已知官方步态相位** |
| 双髋／双膝角＋角速度 `hips_qdq_knn` / `knees_qdq_knn` | 四维状态找相似历史时刻，按距离加权其未来数据；旧报告对应双膝，新研究两者都明确保留 |
| 全部下肢角＋角速度 `legs_qdq_knn` | 十二个腿关节的角和速度，共 24 维，查历史相似状态的未来扰动 |
| 近期 IMU `imu_history_ridge` | 现在以及过去 12、30、60、120 ms 的 H0 身体数据，拟合多提前量线性预测 |
| 近期 IMU＋腿部历史 `imu_legs_history_ridge` | 在上述输入加入同五个时刻的十二腿关节角／速度 |
| 近远期分工 `hybrid_switch` | 近处用历史线性预测，远处用全腿查表；切换提前量只由开发集交叉验证决定，不看保留轨迹选它 |

### “非线性查表”不是一个黑箱名称

1. 对训练库每个时刻保存输入 `x=[q_leg, dq_leg]` 和随后九个提前量的真实扰动 `Y`。
2. 各输入维用**训练数据**的均值和标准差变为可比较尺度：`z=(x-mean)/std`。
3. 查询时计算标准化后的欧氏距离，找 k 个近邻。
4. 权重 `w_i=1/max(distance_i,0.001)`，归一化后预测 `Y_hat=sum(w_i*Y_i)`。
5. k 在 8、24、64 中通过开发集整条留一验证选择。保留轨迹不能进入库、标准化或选参。

因此它不是“预测未来的腿角再求 IMU”，也不需要官方相位。角度相同但屈伸方向不同，角速度会帮助区分。
`models/*.npz` 保存标准化参数、训练输入／输出与原始轨迹及行号索引，能追到每个邻居的来源。

### 目标与误差的固定口径

- H0 每条只建立一次：走前 3–5 秒的平均 yaw 是 +X，Z 继承导航系竖直；不是身体跟随坐标系。
- 原始数据按主机回调时刻取“最近已收到的值”，2 ms 网格；不从未来做插值。
- 加速度／角速度使用因果 15 Hz 一阶滤波；角加速度是角速度后向差分后再滤波，非直接传感器通道。
- 输入历史不超过预测时刻；九个预测提前量为 6、12、…、54 ms。
- 新研究与仿真对齐：加速度／角加速度在每个 6 ms 区间取三个 **pre-step 左端点**，例如 0、2、4 ms；角速度／姿态取区间末端节点。旧报告用 2、4、6 ms，不能不注明差异直接比绝对数字。
- 三轴分量 RMSE：`sqrt(sum((预测-实测)**2)/(样本数*3))`。单位分别为 m/s²、rad/s、rad/s²；不是三维误差向量长度的 RMS。
- 姿态有 RPY 诊断及旋转角误差；论文式的预测比较尚不是部署用四元数／旋转矩阵合同。
- 主比较覆盖起步、走动、停车指令后的平衡段 `[5,18)`，而非只挑稳态。为保证 H0 已冻结及最长预测标签不越入退权段，具体预测锚点保存在 NPZ 中，首点略晚于 5 秒，末点早于 17.946 秒。
- 滤波输出可预测，不等于原始落地尖峰可预测。另存同一预测相对未滤波加速度的误差；不冒充已完成原始目标重新训练。

## 一次完整复算

从仓库根目录运行，使用已安装的 `g1_mpc` 环境；无需重装依赖。输出路径应选新的目录，benchmark
拒绝覆盖已有结果。以下是已保存结果的同一流程，`recheck` 只是复算目录示例：

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/g1-h0-study-mpl
PY=/home/fjk/miniforge3/envs/g1_mpc/bin/python
OUT=evaluation/hardware_shadow/commissioning/walk_h0_study_recheck

"$PY" tools/g1_commissioning/walk_h0_study/prepare.py --out "$OUT"
"$PY" tools/g1_commissioning/walk_h0_study/benchmark.py \
  --data-dir "$OUT/data" --output-dir "$OUT/benchmark"
"$PY" tools/g1_commissioning/walk_h0_study/plots.py \
  --input-dir "$OUT/data" --output-dir "$OUT"
"$PY" tools/g1_commissioning/walk_h0_study/plot_predictions.py --study-dir "$OUT"
"$PY" tools/g1_commissioning/walk_h0_study/verify_results.py --study-dir "$OUT"
"$PY" tools/g1_commissioning/walk_h0_study/verify_legacy.py \
  --output "$OUT/legacy_verification.json"
"$PY" -m unittest discover -s tools/g1_commissioning/walk_h0_study -p 'test_*.py' -v
```

最后一个旧数字核查步骤独立于新模型；没有旧原始资料的分享包可以跳过它。
当前生成结果在 `evaluation/hardware_shadow/commissioning/walk_h0_predictor_study_20260925/`。
`evaluation/` 被 Git 忽略，**提交代码不等于备份数据／模型／图**；分享或换电脑复算还要复制原始
`raw.jsonl`、profile、status 和相应派生结果。记录中的 SHA256 用于核对是不是同一份输入。

## 后续改进的复算

`refine_local.py`、`refine_nonlinear.py`、`refine_blend.py`、`refine_innovation.py` 分别保存局部模型、
非线性历史、连续混合和当前 IMU 偏差衰减；`summarize_refinements.py` 从保存预测复算表格并画图。
比较轨迹 09–12 已被看过，所有新结果均标为探索性，不重新称作盲测。旧输出不覆盖。

接上面的解释器变量，从仓库根目录运行；各方法输出目录必须不存在：

```bash
BASE=evaluation/hardware_shadow/commissioning/walk_h0_predictor_study_20260925
REF=evaluation/hardware_shadow/commissioning/walk_h0_refinement_recheck
"$PY" tools/g1_commissioning/walk_h0_study/refine_local.py --baseline-dir "$BASE" --output-dir "$REF/local"
"$PY" tools/g1_commissioning/walk_h0_study/refine_nonlinear.py --data-dir "$BASE/data" --output-dir "$REF/nonlinear"
"$PY" tools/g1_commissioning/walk_h0_study/refine_blend.py --study-dir "$BASE" --output-dir "$REF/blend"
"$PY" tools/g1_commissioning/walk_h0_study/refine_innovation.py --baseline-dir "$BASE" --local-dir "$REF/local" --output-dir "$REF/innovation"
"$PY" tools/g1_commissioning/walk_h0_study/summarize_refinements.py --baseline-dir "$BASE" --refinement-dir "$REF"
```
