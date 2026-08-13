# Torso 扰动预测器设计

状态：**仿真已验证 / 硬件未验证（hardware-unverified）**。本文描述当前代码，
不是未来提案。系统级事实与最终结果以
[PRE_HARDWARE_FREEZE.md](PRE_HARDWARE_FREEZE.md) 为准。

## 1. 目标与设计选择

右臂 MPC 需要预测未来 torso 运动对末端稳定任务的影响。现有周期模板在稳态行走
中可靠，但启动、停止、速度变化、非严格周期扰动和实际落脚相位偏差并不完全服从
固定模板。因此当前方案不是删除模板，而是：

```text
hybrid_residual = periodic template + neural residual correction
```

优先级依次是闭环稳杯效果、完整 6 ms 计算预算、fail-closed 安全和实现可解释性。
模型为小型 CPU MLP，一次 inference 输出完整 9-step horizon；当前没有 GRU、
Transformer 或逐步自回归调用。

## 2. 统一 predictor 接口

`main_sim.py` 只依赖 `disturbance_predictor.py` 中的统一协议和 factory：

```python
class DisturbancePredictor(Protocol):
    def reset(self) -> None: ...
    def update(self, observation: DisturbancePredictorObservation) -> None: ...
    def predict(self, horizon: int, dt: float) -> DisturbanceHorizon: ...
```

factory 根据 `configs/g1.yaml` 的 `disturbance_predictor` 构造四种真实实现：

| 配置值 | 实现 | 作用 |
|---|---|---|
| `template` | `TemplateDisturbancePredictor` | adapter 复用成熟 `PhaseDisturbancePredictor`，不改变其数值路径 |
| `neural` | `NeuralDisturbancePredictor` | absolute MLP 预测 interval acc/alpha，其余量实测 ZOH |
| `hybrid_residual` | `ResidualHybridPredictor` | template preview 加 residual MLP 修正 |
| `zoh` | `ZeroOrderHoldPredictor` | 全时域保持当前实测扰动 |

`DisturbancePredictorObservation` 汇集当前实测扰动、gravity direction、下肢
`q/dq`、policy target、runtime command、gait phase，以及可选的上一拍 QP/完整
控制区间质量信息。缺失的学习特征不会被猜测：neural/hybrid 记录 fallback 并退回
其安全基线。

接口输出继续使用 `kinematics_helper.py` 的既有结构：

```text
DisturbanceInput
  acc_world, omega_world, alpha_world, rot_world_body

DisturbanceHorizon
  nodes:     horizon + 1 个 DisturbanceInput
  intervals: horizon     个 DisturbanceInput
```

因此 `KinematicsHelper` 和 `arm_mpc.py` 不知道内部是 template、MLP 或 ZOH；B0
也没有修改 MPC 核心数学路径。

## 3. H-frame、node 与 interval 语义

### 3.1 因果 H-frame

步态周期 `C[j]` 使用**上一完整周期** `C[j-1]` 内 torso yaw 的圆周均值定义
heading `yaw_H`。`W_R_H = Rz(yaw_H)`，向量转换为：

```text
v_H = W_R_H^T v_W
v_W = W_R_H v_H
```

`z_H` 与世界重力轴一致。`yaw_H` 只在周期边界更新，并在当前周期内冻结。模板
生成、B1 dataset 和在线 neural inference 使用同一个定义；不会用当前未完成周期
的未来 yaw。第一完整周期只用于建立 heading，尚未就绪时返回实测 ZOH。

### 3.2 时间语义

当前 `dt=6 ms`、`N=9`：

```mermaid
flowchart LR
    N0((node 0<br/>t，实测)) -->|interval 0<br/>[t,t+6 ms)| N1((node 1<br/>t+6 ms))
    N1 -->|...| N8((node 8<br/>t+48 ms))
    N8 -->|interval 8<br/>[t+48,t+54 ms)| N9((node 9<br/>t+54 ms))
```

- `nodes[k]` 是 `t+k*dt` 的瞬时扰动，长度 `N+1`；`nodes[0]` 必须严格等于
  当前 measured disturbance。
- `intervals[k]` 是 `[t+k*dt,t+(k+1)*dt)` 的区间扰动，长度 `N`；`u_k`
  与同下标 interval 配对。
- 阶段 acc/alpha 任务使用 interval；阶段 omega、rotation、gravity/tilt 使用
  node。终端只消费 `nodes[N]` 的 omega/rotation，没有终端加速度项。
- 当前 `KinematicsHelper` 不读取 interval rotation，但 predictor 仍保持完整结构。

输出给 MPC 前全部恢复为世界系；H-frame 是模板和学习数据的航向归一化坐标，
不是改变 MPC 数学坐标系。

## 4. 400-bin 周期模板

模板位于 `disturbance_model_new_heading/templates_heading_interval/`，schema v2：

- 步态周期 `0.8 s`，400 个相隔 `2 ms` 的 phase start bins；
- 每个 bin 同时保存瞬时 node 模板和从该相位开始的未来 `6 ms` interval 模板；
- interval acc/alpha 先在同一世界系用速度端点差生成，再旋入区间起点 H-frame；
- interval omega 用三个 2 ms 子区间的复合梯形积分生成；
- 对齐 6 ms 网格时使用预展开 LUT；off-grid 请求在相邻周期 bin 间插值；
- 姿态以当前实测 `rot_world_body` 锚定模板相对旋转；
- slow bias 以 `tau=0.4 s` 估计持续测量—模板偏差，保留快速周期波形；
- 第一完整周期前整个 horizon 使用 measured ZOH。

`TemplateDisturbancePredictor` 是 adapter，不重新实现以上逻辑。B0 回归测试覆盖第一
周期 ZOH、模板激活、node 0 实测锚点、`N+1/N` 数量、slow bias、on-grid、
off-grid，以及 adapter 与原 `PhaseDisturbancePredictor` 的逐项数值一致性。

## 5. 200 ms neural 数据管线

原始样本在每个 2 ms MuJoCo `mj_step` **之前**采集。dataset 以 6 ms MPC anchor
构造训练窗口：34 个历史样本覆盖 `t-198 ms ... t`，名义窗口约 204 ms；输出
覆盖随后 9 个 6 ms interval，共 54 ms。

每个历史时刻有 50 个特征：

| 特征 | 维数 | 坐标/单位 |
|---|---:|---|
| torso angular velocity | 3 | H-frame，rad/s |
| torso linear acceleration | 3 | H-frame，m/s² |
| gravity direction | 3 | torso frame |
| lower-body `q`, `dq` | 12 + 12 | rad，rad/s |
| lower-body policy target positions | 12 | rad |
| runtime command `vx,vy,wz` | 3 | 策略命令 |
| gait phase `sin,cos` | 2 | 无量纲 |

absolute target 的每一行是未来 interval 的
`[acc_H xyz, alpha_H xyz]`，shape 为 `9 x 6`。

### 5.1 无 future leakage 的约束

- history 最后索引就是 anchor `t`，任何输入时间戳不得大于 `t`；
- target `k` 只用 `t+k*6 ms` 与 `t+(k+1)*6 ms` 的速度端点；
- H-frame 只使用在 anchor 前已经完整结束的周期；
- train/validation/test 按 episode 划分为 12/3/3，相邻窗口不能跨集合；
- normalization 只用 train episodes 拟合并随 checkpoint 保存；
- collector 保存 pre-step 时间、原始索引、anchor、segment 和 heading 元数据，
  dataset validation 会检查单调时间、固定 stride、history/target 边界和有限性。

采集、训练和 fresh-clone 流程见
[disturbance_learning/README.md](disturbance_learning/README.md)。

## 6. Absolute MLP 与 residual MLP

两种 checkpoint 使用相同的小型网络：

```text
34 x 50 history
  -> flatten 1700
  -> Linear(1700,128) + ReLU
  -> Linear(128,128) + ReLU
  -> Linear(128,54)
  -> reshape 9 x 6
```

参数量为 241,206。在线固定 CPU、`eval()`、`torch.inference_mode()` 和单线程，
每拍只调用一次模型。

### 6.1 Absolute MLP

`train_mlp.py` 直接学习未来 absolute H-frame interval acc/alpha。在线 `neural`
模式把输出旋回世界系并填入 9 个 interval；nodes 和 interval omega/rotation 明确
使用当前 measured ZOH。这样可做独立消融，但会丢失模板的未来姿态结构。

### 6.2 Residual MLP

Hybrid 没有把 absolute checkpoint 含糊地叠加到模板上。`train_residual_mlp.py`
重新训练：

```text
residual target
  = absolute future interval acc/alpha in H
  - sequential online-equivalent template interval acc/alpha in H
    (包含相同 phase state 与 slow bias)
```

在线 `hybrid_residual` 先取得完整 template `DisturbanceHorizon`，仅对 9 个 interval
的 acc/alpha 加 residual；全部 nodes 以及 interval omega/rotation 原样来自 template。
checkpoint metadata 会校验 `control_dt`、horizon、H-frame 定义、template variant、
slow-bias 开关和时间常数，语义不匹配时拒绝加载。

## 7. 四种模式与安全 fallback

| 模式 | nodes | interval acc/alpha | interval omega/rotation | fallback |
|---|---|---|---|---|
| `template` | 模板，node 0 实测 | 模板 + slow bias | 模板 + slow bias | 首周期 measured ZOH |
| `neural` | measured ZOH | absolute MLP | measured ZOH | measured ZOH |
| `hybrid_residual` | 完整 template | template + residual MLP | 完整 template | 完整 template |
| `zoh` | measured ZOH | measured ZOH | measured ZOH | 本身即 ZOH |

Hybrid safety gate 在以下任一条件成立时整拍退回完整 template preview：

- H-frame/history 未就绪或历史时间断点；
- 输入、归一化输入、模型输出或物理 correction 非有限；
- 归一化输入绝对值/RMS、归一化输出或 acc/alpha correction norm 超过配置可信范围；
- 上一个完整 6 ms 控制区间 overrun，进入一拍 template cooldown；
- residual 生效后连续两拍 QP 失败，插入一拍 template-only probe。

Fallback 具有计数、原因码和 timing diagnostics。它只隔离 neural residual；若
OSQP 本身失败，MPC 另行使用有界回中和速度制动。hardware shadow 在 predictor
之前还有 joint/IMU/时间戳/stale-state contract gate；该硬件契约仍未验证。

## 8. 主要实验结论

以下均为仿真或本机 CPU 结果：

1. **B2 absolute MLP**：18 episodes、11,232 windows，train/validation/test 为
   7,488/1,872/1,872；test acc/alpha RMSE 为 `0.1612/0.7184`。batch-1
   inference mean/p99/max 为 `0.0377/0.0492/0.0751 ms`。
2. **Residual MLP 离线**：test hybrid acc/alpha RMSE 为 `0.1791/0.9043`；
   residual 定义经过 checkpoint metadata 和在线等价模板验证。
3. **Unseen schedule 闭环**：3 个新 schedule profiles x 2 seeds。Hybrid 相对
   template 的 acc/alpha 在 6/6 条件改善，配对平均为 `4.91%/8.78%`；tilt
   平均改善 `2.68%`，5/6 改善，最差整体为 `-0.25%`。
4. **模式取舍**：neural-only 的 aggregate acc/alpha 最低，但 tilt RMS
   `0.11269`，比 template `0.06186` 高约 82%；hybrid 的 tilt 为 `0.06021`，
   同时保留 acc/alpha 收益，因此被选为当前综合最佳模式。
5. **Target runtime**：PREEMPT_RT repeated gate 中 predictor mean/mean-p99/
   worst-max 为 `0.499/0.627/0.866 ms`；完整控制路径 worst max `4.006 ms`，
   9,588 个区间中没有 6 ms overrun。

轻量结果位于 `evaluation_summary/`；原始 episode、checkpoint 和运行日志被 Git
忽略。上述结果不能证明真机预测收益。

## 9. 实现与验证入口

| 文件 | 职责 |
|---|---|
| `disturbance_predictor.py` | 统一接口、factory、四种模式、history、MLP 和 safety gate |
| `sim_support.py` | 成熟 `PhaseDisturbancePredictor` 模板实现 |
| `kinematics_helper.py` | `DisturbanceInput`、`DisturbanceHorizon` 和 MPC 任务仿射项 |
| `main_sim.py` | predictor-neutral 控制循环与 timing/diagnostics |
| `disturbance_learning/` | 因果采集、dataset、训练和闭环消融 |
| `tests/test_disturbance_predictor.py` | B0 template 行为冻结 |
| `tests/test_disturbance_dataset.py` | 时间对齐与 leakage 检查 |
| `tests/test_neural_disturbance_predictor.py` | absolute/hybrid、metadata 和 fallback |

当前第一次硬件只读/shadow 应先使用 `template`。只有 locomotion context、H-frame
相位和全部硬件契约在目标 G1 上确认后，才有资格验证 `hybrid_residual`；任何
有效控制输出都不属于本文所述已验证范围。
