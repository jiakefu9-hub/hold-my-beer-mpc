# Disturbance learning pipeline

本目录实现从 MuJoCo 因果采集到 absolute/residual MLP、闭环消融和 realtime 验证的
最小流水线。模型设计和时间语义见
[DISTURBANCE_PREDICTOR.md](../DISTURBANCE_PREDICTOR.md)，冻结实验结论见
[PRE_HARDWARE_FREEZE.md](../PRE_HARDWARE_FREEZE.md)。

## 1. 目录职责

| 文件 | 职责 |
|---|---|
| `collect_dataset.py` | 采集一个 pre-step-aligned episode，生成 raw/windows/validation |
| `collect_episodes.py` | 按 `mlp_baseline.yaml` 采集 18 个 seeds 并写 manifest |
| `dataset.py` | 构造 34 x 50 history、9 x 6 target，并检查 causality/alignment |
| `command_schedule.py` | 训练 schedule 与 unseen generalization profiles |
| `mlp_model.py` | 共享的小型 MLP 定义与 CPU checkpoint loader |
| `train_mlp.py` | absolute MLP：划分 episode、归一化、overfit、训练、baseline 对比和 timing |
| `train_residual_mlp.py` | residual MLP：针对在线等价 template-with-slow-bias 训练 |
| `run_closed_loop_ablation.py` | `template/neural/hybrid_residual` 第一轮闭环消融 |
| `run_generalization_ablation.py` | unseen schedules、seeds 和小 payload 重复验证 |
| `run_readiness_validation.py` | safety、payload/model-mismatch 与完整 timing 验证 |
| `run_realtime_timing_ablation.py` | target runtime repeated timing gate 的底层 runner |

## 2. 数据定义

采集器在每次 2 ms `mj_step` 前记录 torso、下肢策略和 command 状态。每个 6 ms
MPC anchor `t` 构造：

```text
history: 34 x 50, t-198 ms ... t
target:   9 x 6,  [t,t+6 ms) ... [t+48,t+54 ms)
```

50 个输入是 H-frame torso omega/acc、torso-frame gravity direction、12 维下肢
`q/dq/policy target`、3 维 runtime command 和 phase sin/cos。6 个 target 通道是
H-frame interval acc xyz 与 alpha xyz。

`validate_supervised_windows()` 检查 pre-step 时间、单调索引、6 ms stride、history
末端不晚于 anchor、target 严格位于未来、上一完整周期 H-frame、shape 和有限性。
Train/validation/test 由训练脚本按完整 episode 划分，不能把相邻窗口随机拆开。

## 3. 本地产物与 Git 边界

以下目录已由仓库 `.gitignore` 排除：

```text
disturbance_learning/data/       raw episodes、window NPZ、validation、manifest
disturbance_learning/artifacts/  checkpoint、normalization、完整训练日志
evaluation/                      每次闭环的原始日志、图、CSV、视频
```

不要用 `git add -f` 提交这些内容。可审查的轻量 JSON/CSV 摘要写入已跟踪的
`evaluation_summary/`。Checkpoint 内包含 normalization、feature/target names、
episode split 和 shape；residual checkpoint 还包含 prediction mode、H-frame、
control dt、template variant 及 slow-bias metadata。

由于 checkpoint 有意不进 Git，fresh clone 的 `configs/g1.yaml` 安全默认使用
`template`。要运行 `neural` 或 `hybrid_residual`，必须先在本地重建对应 artifact，
或从受控的独立 artifact 备份恢复并验证 metadata；不能伪造空 checkpoint。

## 4. 采集与训练

以下命令都从仓库根目录运行，并假设现有 `g1_mpc` 环境已经能运行 MuJoCo、Torch
和项目 C++ 后端。`requirements-mpc.txt` 只补充 MPC 的 OSQP 依赖，不是完整锁定
环境。仓库已跟踪 locomotion policy、MJCF 和 400-bin template。

### 4.1 单 episode smoke/对齐验证

```bash
python disturbance_learning/collect_dataset.py g1.yaml \
  --episode-id alignment_smoke \
  --seed 0 \
  --output-prefix disturbance_learning/data/alignment_smoke
```

成功时会生成 `_raw.npz`、`_windows.npz`、`_validation.json`，并在终端打印同一份
validation report。所有文件都保持本地。

### 4.2 18-episode dataset

```bash
python disturbance_learning/collect_episodes.py
```

中断后可以验证并保留已经完成的 episodes：

```bash
python disturbance_learning/collect_episodes.py --reuse-existing
```

episode 数、seed、输出目录和 split 位于 `mlp_baseline.yaml`。改变实验时应复制一份
新的 YAML，而不是覆盖已有摘要所对应的定义。

### 4.3 Absolute MLP

```bash
python disturbance_learning/train_mlp.py
```

脚本先做 64-sample overfit sanity check，再按 episode split 训练，报告 train/val/
test 以及 start/steady/velocity-change/stop/stopped 的 RMSE，比较 ZOH/template/MLP，
执行 batch-1 CPU timing，并验证 checkpoint save/reload parity。

默认本地产物：

```text
disturbance_learning/artifacts/b2_mlp_baseline/mlp_checkpoint.pt
evaluation_summary/b2_mlp_baseline/summary.json
```

### 4.4 Residual MLP

```bash
python disturbance_learning/train_residual_mlp.py
```

它按 episode 顺序运行与在线相同的 template 和 slow bias，再用
`absolute target - template interval target` 训练，不读取 absolute MLP 输出作为
residual。默认 checkpoint 为：

```text
disturbance_learning/artifacts/hybrid_residual_mlp/residual_mlp_checkpoint.pt
```

## 5. 测试、消融与复现顺序

先运行快速测试：

```bash
python -m pytest \
  tests/test_disturbance_dataset.py \
  tests/test_mlp_baseline.py \
  tests/test_neural_disturbance_predictor.py
```

有本地 checkpoint 后，按风险和成本递增运行：

```bash
python disturbance_learning/run_closed_loop_ablation.py \
  --group neural_closed_loop_reproduction

python disturbance_learning/run_generalization_ablation.py \
  --group hybrid_generalization_reproduction
```

已有 group 中断后可加 `--resume`。原始 runs 写入 `evaluation/<group>/`，默认轻量
摘要分别更新 `evaluation_summary/neural_closed_loop_ablation/summary.json` 和
`evaluation_summary/hybrid_generalization_validation/summary.json`。

Realtime gate 不应直接调用 Python runner；在满足 PREEMPT_RT、CPU/IRQ isolation
和 governor 条件后，使用仓库包装脚本：

```bash
./tools/realtime/run_target_timing_gate.sh \
  --control-cpu 7 \
  --group target_rt_reproduction
```

完整环境和恢复方法见 [REALTIME_RUNTIME.md](../REALTIME_RUNTIME.md)。

## 6. Fresh-clone 最小复现

1. Checkout `feat/predictor-interface`，确认 `policy/motion.pt`、
   `resources/g1_description/scene.xml` 和
   `disturbance_model_new_heading/templates_heading_interval/` 存在。
2. 在已验证的 `g1_mpc` 环境运行上述三个 predictor/dataset tests。
3. 先以 `template` 运行仿真，确认不依赖本地 checkpoint。
4. 运行单 episode alignment smoke，检查 validation JSON 全部通过。
5. 采集 18 episodes；不要从相邻窗口重新随机划分。
6. 依次训练 absolute 和 residual MLP，检查 overfit、split、checkpoint parity 和
   CPU timing。
7. 运行闭环与 unseen-schedule ablation，结果只能与同一 checkout、配置和 schedule
   定义下的摘要比较。
8. 只有目标 realtime checker PASS 后才运行 repeated timing gate。

轻量摘要可以复核历史结论，但不能替代重新生成 gitignored 数据/checkpoint，也不
构成真机验证。
