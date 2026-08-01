# 右臂 MPC 稳定控制系统设计

## 1. 文档范围与当前阶段

本项目使用人形机器人的右臂稳定抓持点，降低行走过程中末端的线加速度、角速度、角加速度和倾斜误差。系统分工如下：

- 下肢继续使用现有 RL locomotion 策略。
- 左臂保持固定 PD 基线。
- 右臂 MPC 优化 5 个受控关节的期望关节加速度 `ddq_des`。
- 下层执行器将 `ddq_des` 转换为最终关节力矩。

当前实现支持两种可配置的 base 预测：

1. **零阶保持。** 每个控制周期读取当前 base 运动量，并在本轮预测时域内保持不变。
2. **phase-based 扰动前馈。** 从三个离线 H 系模板中选择一个，运行时先转换到 MuJoCo 世界系，再分别生成节点扰动和未来 6 ms 控制区间扰动；快速周期变化来自模板，当前测量只提供精确的 $k=0$ 节点和慢变化偏差。

当前 MPC 不进行整机接触力优化，也不把浮动基完整刚体动力学放进 QP。它是一个以关节加速度为输入、使用局部运动学观测模型的稀疏线性时变 QP。

当前仓库已经实现 MPC、可选扰动前馈和第 10 节的 `ddq_des` 到力矩执行链。本文同时作为数学定义与当前实现说明。

---

## 2. 坐标系、任务点与关节定义

### 2.1 坐标系

- $\{W\}$：世界惯性坐标系。
- $\{H\}$：与重力对齐的 heading 坐标系；$x_H$ 指向上一完整步态周期的 torso 平均方位角，$z_H=z_W$。
- $\{B\}$：torso/IMU 坐标系。
- $\{E\}$：右手抓持点坐标系。

旋转矩阵约定：

$$
{}^W R_B:\{B\}\rightarrow\{W\},
\qquad
{}^W R_H:\{H\}\rightarrow\{W\},
\qquad
{}^E R_W=({}^W R_E)^T .
$$

当前仿真中的 $\{W\}$ 就是 MuJoCo 世界系。MPC 接收的完整扰动量及第 5 节的末端运动学始终使用 $\{W\}$ 表达；$\{H\}$ 只用于保存与绝对世界航向无关的离线模板。运行时根据上一完整步态周期确定 ${}^W R_H$，把模板旋回 $\{W\}$ 后再送入 MPC。

### 2.2 任务末端

稳定控制和评估使用 `right_grasp_site`，而不是 `right_wrist_roll_rubber_hand` 的 body 原点。抓持点位于瓶身大圆柱中心，与手腕末端刚性固连，其相对手腕的局部平移为

$$
\begin{bmatrix}
0.2 & 0.035 & 0
\end{bmatrix}^T\ \mathrm{m}.
$$

末端位置、姿态、Jacobian、加速度和可视化均以该抓持点为准。

### 2.3 右臂关节顺序

右臂受控维数为 $n=5$，关节顺序固定为：

1. `right_shoulder_pitch_joint`
2. `right_shoulder_roll_joint`
3. `right_shoulder_yaw_joint`
4. `right_elbow_joint`
5. `right_wrist_roll_joint`

本文所有关节向量、权重和上下界都必须采用这个顺序。

在当前 MuJoCo 模型的符号下：

- shoulder roll 为正时，右臂向身体中线内收；
- shoulder yaw 为正时，右手也向身体中线移动；
- shoulder pitch 为负时，抓持点主要向前、向上移动；为正时主要向下、向后收回。

这些符号不能从“左/右手”的直觉直接猜测，真机迁移时必须重新核对编码器方向。

---

## 3. 状态、输入与总体数据流

### 3.1 状态和输入

定义状态、输入和选择矩阵：

$$
x_k=
\begin{bmatrix}
q_k\\
\dot q_k
\end{bmatrix}
\in\mathbb R^{2n},
\qquad
u_k=\ddot q_k\in\mathbb R^n,
$$

$$
S_q=
\begin{bmatrix}
I_n&0
\end{bmatrix},
\qquad
S_v=
\begin{bmatrix}
0&I_n
\end{bmatrix},
$$

$$
q_k=S_qx_k,\qquad \dot q_k=S_vx_k.
$$

本文在每次 MPC 求解开始时都重新编号：$k=0$ 表示当前测量点，$k=1,\ldots,N$ 表示本轮预测点。当前配置 $N=12$，因此有 12 个控制输入 $u_0,\ldots,u_{11}$ 和 12 个控制区间，但状态点为 $x_0,\ldots,x_{12}$，共 13 个；$x_{12}$ 是只计算终端代价和状态约束的终端点。本文不再为当前真实控制时刻另引入下标 $t$。

### 3.2 为什么仍以 `ddq` 为优化输入

当前 MPC 保留 $u_k=\ddot q_k$，原因是：

- 末端线加速度和角加速度与 $\ddot q$ 直接相关，核心防晃目标容易写成凸二次代价。
- $q$ 和 $\dot q$ 可以通过二阶积分器预测，状态方程简单且稀疏。
- 关节角、速度和加速度上下界都可以直接写成线性约束。
- QP 不必显式包含整机惯量矩阵、接触力和摩擦锥，便于先完成在线实现与调试。
- 当前已有 DDQ 到力矩的数值前向动力学映射，可用于隔离上层 MPC 与下层执行误差。

该选择也有明确局限：

- QP 中满足的 $u_k$ 是期望关节加速度，不等于接触和限位作用下必然实现的真实 `qacc`。
- 力矩上限不直接出现在当前预测模型中，最终仍依赖下层力矩限幅与前向动力学验收。
- 二阶积分器不描述浮动基、腿部和右臂之间的动力学耦合。
- 摩擦、接触和关节限位发生离散切换时，局部 DDQ 到力矩映射可能失效。

因此，上层 MPC 负责生成受约束的 `ddq_des`，下层执行链负责寻找当前整机状态下尽可能准确且安全的力矩。

### 3.3 多速率数据流

当前仿真建议沿用以下周期：

- MuJoCo physics step：$0.002\ \mathrm{s}$。
- 右臂 MPC 更新周期：$\Delta t=0.006\ \mathrm{s}$。
- `ddq_des` 在三个 physics step 内保持。
- 逆动力学和 DDQ 到力矩映射可在每个 $0.002\ \mathrm{s}$ physics step 使用最新整机状态重新计算。

---

## 4. 预测模型与工作轨迹

### 4.1 二阶积分器

采用分段常加速度离散模型：

$$
A=
\begin{bmatrix}
I_n&\Delta t I_n\\
0&I_n
\end{bmatrix},
\qquad
B=
\begin{bmatrix}
\frac12\Delta t^2I_n\\
\Delta t I_n
\end{bmatrix},
$$

$$
x_{k+1}=Ax_k+Bu_k,\qquad k=0,\ldots,N-1.
$$

这里不加入额外扰动项。base 运动不直接改变关节二阶积分器，而是进入第 5 节的末端观测模型。

### 4.2 base 扰动预测与配置

#### 4.2.1 节点量与控制区间量

每次求解同时维护两条序列。节点量位于 $t_k$：

$$
d_k^{\mathrm{node}}=
\left(
{}^Wa_{B,k},
{}^W\omega_{B,k},
{}^W\alpha_{B,k},
{}^WR_{B,k}
\right),
\qquad k=0,\ldots,N.
$$

控制区间量对应随后整个 $[t_k,t_{k+1})$：

$$
\bar d_k^{\mathrm{int}}=
\left(
{}^W\bar a_{B,k},
{}^W\bar\omega_{B,k},
{}^W\bar\alpha_{B,k}
\right),
\qquad k=0,\ldots,N-1.
$$

因此当前 $N=12$ 时有 13 个节点量和 12 个区间量。$u_k$ 与同下标的 $\bar d_k^{\mathrm{int}}$ 配对；角速度状态代价、重力误差、姿态和终端项使用 $d_k^{\mathrm{node}}$。终端没有 $u_N$，也没有第 13 个控制区间。

配置 `mpc_disturbance_feedforward_enabled: false` 时，两条序列都对当前测量做零阶保持。这里的当前加速度和角加速度已经经过安全限幅与 MPC 专用低延迟滤波，当前默认滤波系数为 0.5；角速度和姿态直接来自当前 torso/IMU 状态。下一控制周期重新测量并刷新。

#### 4.2.2 未来 6 ms H 系区间模板

开启前馈时从 `disturbance_model_new_heading/templates_heading_interval` 读取模板。生成入口为：

```bash
./disturbance_model_new_heading/run_interval_all.sh
```

采集器在每次 `mj_step` 前按 $2\ \mathrm{ms}$ 记录 IMU site 的世界系线速度、角速度、姿态和传感器加速度，避免旧数据中“step 前相位、step 后状态”造成的固定 $2\ \mathrm{ms}$ 标签偏移。对起点 $t$ 和 $\Delta t=6\ \mathrm{ms}$，先在同一个世界系中构造：

$$
{}^W\bar a_B(t)=
\frac{{}^Wv_B(t+\Delta t)-{}^Wv_B(t)}{\Delta t},
\qquad
{}^W\bar\alpha_B(t)=
\frac{{}^W\omega_B(t+\Delta t)-{}^W\omega_B(t)}{\Delta t},
$$

$$
{}^W\bar\omega_B(t)=
\frac{1}{\Delta t}\int_t^{t+\Delta t}{}^W\omega_B(s)\,ds.
$$

角速度积分由三个 $2\ \mathrm{ms}$ 子区间的复合梯形公式计算。随后使用该区间起点所属的 ${}^HR_W(t)$ 把三个世界系向量旋入 H 系。不能直接对两个不同周期 H 系下的端点速度做差，否则跨周期时会把坐标系变化误当成物理加速度。

模板周期为 $T=0.8\ \mathrm{s}$，使用 400 个均匀的区间起点，相邻起点相差 $2\ \mathrm{ms}$。每个区间模板值表示“从该相位开始的未来 6 ms 平均扰动”，而不是把整个周期切成互不重叠的 6 ms 块。运行时在相邻 2 ms 起点间做周期插值，所以当前控制拍即使落在某个 6 ms 窗口内部，也可以按实际连续相位得到对应的滑动区间预测，不要求控制时钟与模板的固定 6 ms 边界重合。

每个文件显式保存节点模板、区间模板、姿态四元数及其旋转矩阵：

- `raw`：节点和区间量都使用相位 bin 原始均值；
- `half_smoothed`：节点角速度、节点角加速度和姿态做 21 点、约 42 ms 的环形平滑，节点线加速度和三个区间量保持原始；
- `fully_smoothed`：在 half 的基础上平滑节点线加速度，并对三个区间量再做 3 点、约 6 ms 的环形平滑。

新区间模板的六轮闭环对照中，`raw` 得到最低的统一任务评分，当前默认使用 `raw`。`fully_smoothed` 的区间预测误差更低，但闭环效果和 DDQ 安全指标反而更差，因此模板选择以同配置闭环效果为准，不能只比较模板 RMS。

#### 4.2.3 H 系定义与 H 到 W 的转换

把第 $j$ 个步态周期记为 $C_j=[jT,(j+1)T)$。对 $j\ge1$，当前周期使用的 heading yaw 来自上一完整周期 $C_{j-1}$ 内 torso yaw 的圆周均值：

$$
\psi_{H,j}
=\operatorname{atan2}
\left(
\sum_{t\in C_{j-1}}\sin\psi_B^W(t),
\sum_{t\in C_{j-1}}\cos\psi_B^W(t)
\right),
$$

$$
{}^WR_{H,j}=R_z(\psi_{H,j}),
\qquad
{}^HR_{W,j}=R_z(-\psi_{H,j}).
$$

${}^WR_{H,j}$ 在整个 $C_j$ 内保持不变，只在周期边界用刚结束的完整周期更新。第一个周期尚无历史，预测自动退化为零阶保持；约 $0.8\ \mathrm{s}$ 后才启用模板。当前 evaluation 前有 3 个 warm-up 周期，因此正式评价开始时 H 已初始化。

对任一 H 系模板向量 $z\in\{a_B,\omega_B,\alpha_B\}$：

$$
{}^WT_z(\phi)={}^WR_H\,{}^HT_z(\phi).
$$

姿态模板同样先转换到世界系：

$$
{}^WR_{B,\mathrm{tpl}}(\phi)
={}^WR_H\,{}^HR_B^{\mathrm{tpl}}(\phi).
$$

若本轮 72 ms 预测窗口跨过步态周期边界，本轮仍冻结当前 ${}^WR_H$；边界后的下一次求解再使用新完成周期计算的 H 系。

#### 4.2.4 高频模板加低频测量偏差

用 $i$ 表示真实 MPC 控制拍，用 $k$ 表示该拍内部的预测步。对三个向量分别维护慢偏差 $b_{z,i}$：

$$
\beta=1-\exp\left(-\frac{\Delta t}{\tau_b}\right),
$$

$$
b_{z,i}
=(1-\beta)b_{z,i-1}
+\beta\left({}^Wz_i^{\mathrm{meas}}-{}^WT_z^{\mathrm{node}}(\phi_i)\right).
$$

当前 `mpc_disturbance_slow_bias_enabled: true`，且默认 $\tau_b=0.4\ \mathrm{s}$。快速、周期性的落脚冲击由模板直接提供；这个量更直观地理解为“缓慢更新的长期模板修正量”，只补偿模板和当前机器人之间持续存在的均值差，不再用带延迟的瞬时 $d_0$ 误差平移整条未来曲线。

以任一标量分量为例，$\Delta t=0.006\ \mathrm{s}$、$\tau_b=0.4\ \mathrm{s}$ 时 $\beta\approx0.0149$。若原修正量 $b_0=0$，只有一次冲击使“测量减模板”等于 $4$，则

$$
b_1=0.9851\times0+0.0149\times4=0.0596.
$$

若下一拍差值恢复为零，则 $b_2=0.9851\times0.0596=0.0587$，之后继续衰减；因此单次冲击不会把 $4$ 整体加到未来模板。

若“测量减模板”长期恒等于 $1$，则每隔 $6\ \mathrm{ms}$ 更新

$$
b_i=0.9851b_{i-1}+0.0149,
$$

因此 $b_1=0.0149$、$b_2=0.0296$、$b_3=0.0440$；一般地

$$
b_n=1-0.9851^n
=1-\exp\left(-\frac{0.006n}{0.4}\right).
$$

经过 $60\ \mathrm{ms}$（10 拍）有 $b_{10}=0.1393$，经过约 $0.4\ \mathrm{s}$（67 拍）有 $b_{67}\approx0.634$，时间足够长时 $b_n\rightarrow1$。例如未来模板值为 $2$，则对应预测依次被修正为 $2.0149$、$2.0296$、$2.0440$，长时间后趋近 $3$。

节点 $k=0$ 严格取本拍测量；未来节点和控制区间分别为：

$$
{}^W\hat z_{0|i}^{\mathrm{node}}
={}^Wz_i^{\mathrm{meas}},
$$

$$
{}^W\hat z_{k|i}^{\mathrm{node}}
={}^WT_z^{\mathrm{node}}(\phi_{i,k})+b_{z,i},
\qquad k=1,\ldots,N,
$$

$$
{}^W\hat{\bar z}_{k|i}^{\mathrm{int}}
={}^WT_z^{\mathrm{int}}(\phi_{i,k})+b_{z,i},
\qquad k=0,\ldots,N-1.
$$

这里“高频模板预测 + 低频测量偏差”就是上面两项的相加：模板负责未来随步态快速变化的波形，慢偏差负责不同运行回合之间缓慢变化的基线。

姿态不能使用向量加法。程序对模板四元数做最短路径 SLERP，并以当前实测姿态锚定模板相对转动：

$$
{}^WR_{B,k}
={}^WR_{B,0}
\left({}^WR_{B,\mathrm{tpl}}(\phi_0)\right)^T
{}^WR_{B,\mathrm{tpl}}(\phi_k).
$$

因此 $R_{B,0}$ 严格等于当前测量，未来步只复用模板相对姿态变化，不会把采集时的绝对航向强加给当前机器人。最终，预测器一次返回 $N+1$ 个 `nodes` 和 $N$ 个 `intervals`，避免在两个独立调用中重复更新 H 系或慢偏差状态。

### 4.3 逐步线性化与实时迭代

每个预测步都围绕工作点

$$
\bar x_k=
\begin{bmatrix}
\bar q_k\\
\dot{\bar q}_k
\end{bmatrix}
$$

冻结运动学系数。初次启动时用当前状态和零输入生成工作轨迹；之后先平移上一拍的最优输入：

$$
\bar u_k\leftarrow u^\star_{k+1,\mathrm{prev}},
\qquad k=0,\ldots,N-2,
$$

$$
\bar u_{N-1}\leftarrow u^\star_{N-1,\mathrm{prev}},
\qquad
\bar x_0\leftarrow x_{\mathrm{meas}},
$$

再按

$$
\bar x_{k+1}=A\bar x_k+B\bar u_k,
\qquad k=0,\ldots,N-1
$$

向前滚动整条状态工作轨迹。这样得到的轨迹与积分器动力学一致，但仍需检查并裁剪输入、确认状态边界；OSQP 的 primal warm-start 使用这条轨迹，dual warm-start 使用上一拍对偶解的对应平移。

也可以把末端输入补齐为零而不是重复上一拍最后一个输入，但一种策略确定后应保持一致并记录。本文初版采用“重复最后一个输入”，以避免预测末端出现人为突变。

初版采用一次实时迭代：每个控制周期只围绕当前工作轨迹组装并求解一个 QP，不在同一周期内执行多轮 SQP。下一周期再用新状态和上一拍解重新线性化。

---

## 5. 末端物理量的局部仿射模型

以下系数都带预测步下标 $k$，并在工作状态 $\bar x_k$ 处冻结。加速度任务把 $u_k$ 与同一控制区间的 ${}^W\bar a_{B,k},{}^W\bar\omega_{B,k},{}^W\bar\alpha_{B,k}$ 配对；运动学构型、base 姿态、角速度状态任务和二维重力任务使用节点量 $q_k,{}^WR_{B,k},{}^W\omega_{B,k}$。

### 5.1 末端线加速度

末端世界系位置为

$$
{}^Wp_E={}^Wp_B+{}^WR_B\,{}^Bp_E(q).
$$

严格求导可得：

$$
\begin{aligned}
{}^Wa_E={}&{}^Wa_B
+{}^W\alpha_B\times({}^WR_B{}^Bp_E)
+{}^W\omega_B\times\left({}^W\omega_B\times({}^WR_B{}^Bp_E)\right)\\
&+2{}^W\omega_B\times({}^WR_B{}^BJ_v\dot q)
+{}^WR_B\left({}^B\dot J_v\dot q+{}^BJ_v\ddot q\right).
\end{aligned}
$$

冻结区间起点的 $p_E,J_v,\dot J_v$ 后，用未来 6 ms base 区间平均量近似该控制区间的末端加速度：

$$
{}^Wa_{E,k}\approx
D_{a,k}+C_{a,k}S_vx_k+B_{a,k}u_k,
$$

其中

$$
\begin{aligned}
D_{a,k}={}&{}^W\bar a_{B,k}
+{}^W\bar\alpha_{B,k}\times({}^WR_{B,k}{}^Bp_E(\bar q_k))\\
&+{}^W\bar\omega_{B,k}\times
\left({}^W\bar\omega_{B,k}\times
({}^WR_{B,k}{}^Bp_E(\bar q_k))\right),
\end{aligned}
$$

$$
C_{a,k}=
2[{}^W\bar\omega_{B,k}]_\times{}^WR_{B,k}{}^BJ_v(\bar q_k)
+{}^WR_{B,k}{}^B\dot J_v(\bar q_k,\dot{\bar q}_k),
$$

$$
B_{a,k}={}^WR_{B,k}{}^BJ_v(\bar q_k).
$$

这里不是把整个非线性运动学严格积分 6 ms，而是采用“区间平均 base 扰动 + 区间起点运动学”的局部近似。它修正了原实现用起点瞬时、滞后滤波的 $a_B,\alpha_B$ 去比较随后 6 ms 平均末端加速度的主要时序错误，同时保持 QP 对 $x_k,u_k$ 的仿射结构。

实现中直接沿用 LQR 的世界系位置差。对工作构型 $\bar q_k$ 做 MuJoCo scratch 前向运动学后，有

$$
{}^Wr_{BE,k}^{\mathrm{scratch}}
={}^Wp_E^{\mathrm{scratch}}(\bar q_k)
-{}^Wp_B^{\mathrm{scratch}} .
$$

$\mathrm{scratch}$ 只表示该量来自临时的 MuJoCo `MjData`，不是新的坐标系。每次 MPC 求解开始时，代码把当前整机 `qpos` 复制到 scratch；计算各预测步时只把右臂关节替换为 $\bar q_k$，其余关节以及 floating base 的位置、姿态均保持为 $k=0$ 的实测值。因此在同一轮预测中 ${}^Wp_B^{\mathrm{scratch}}$ 和 ${}^WR_{B,0}$ 不随 $k$ 变化，而 ${}^Wp_E^{\mathrm{scratch}}(\bar q_k)$、Jacobian 和末端姿态随 $\bar q_k$ 变化。下一次 MPC 求解会重新复制整机实测状态。

$r_{BE}$ 只是上述位置差的简称，不是新的状态或模板通道。世界系平移在相减时严格抵消，因此模板不需要预测 $p_B$。

scratch 中的 base 姿态保持为当前测量姿态 ${}^WR_{B,0}$。若第 $k$ 步使用预测姿态 ${}^WR_{B,k}$，则只需用相对旋转修正位置差：

$$
\Delta R_k={}^WR_{B,k}({}^WR_{B,0})^T,
$$

$$
{}^Wr_{BE,k}(\bar q_k)
=\Delta R_k\,{}^Wr_{BE,k}^{\mathrm{scratch}}.
$$

它与严格公式 ${}^Wr_{BE,k}={}^WR_{B,k}{}^Bp_E(\bar q_k)$ 完全等价，但实现更直接，也无需显式构造 ${}^Bp_E$。关闭前馈时 $R_{B,k}=R_{B,0}$、$\Delta R_k=I$，因此退化为 LQR 的 ${}^Wp_E-{}^Wp_B$ 写法。世界系 $J_v,\dot J_v,J_\omega,\dot J_\omega$ 和末端姿态使用同一个 $\Delta R_k$ 修正，不能只旋转位置差而遗漏这些量。

### 5.2 末端角速度与角加速度

末端绝对角速度和角加速度为

$$
{}^W\omega_E={}^W\omega_B+{}^WR_B{}^BJ_\omega\dot q,
$$

因此角速度在工作点附近可直接写成关于状态的仿射形式：

$$
{}^W\omega_{E,k}\approx
D_{\omega,k}+C_{\omega,k}S_vx_k,
$$

$$
D_{\omega,k}={}^W\omega_{B,k},
\qquad
C_{\omega,k}={}^WR_{B,k}{}^BJ_\omega(\bar q_k).
$$

这里冻结工作点的 $J_\omega(\bar q_k)$，只保留角速度对 $\dot q_k$ 的直接依赖，没有加入 $\partial(J_\omega\dot q)/\partial q$；这是当前实现采用的简化局部模型。该式也没有关于当前 $u_k$ 的直接项；$u_k$ 先通过二阶积分器改变后续 $\dot q_{k+1}$，再影响后续阶段和终端的角速度。加入该项的原因是：仅令 ${}^W\alpha_E$ 接近零只能阻止角速度继续变化，不能消除已经存在的 ${}^W\omega_E$。

$$
{}^W\alpha_E=
{}^W\alpha_B+
{}^W\omega_B\times({}^WR_B{}^BJ_\omega\dot q)
+{}^WR_B({}^B\dot J_\omega\dot q+{}^BJ_\omega\ddot q).
$$

局部仿射形式为

$$
{}^W\alpha_{E,k}\approx
D_{\alpha,k}+C_{\alpha,k}S_vx_k+B_{\alpha,k}u_k,
$$

$$
D_{\alpha,k}={}^W\bar\alpha_{B,k},
$$

$$
C_{\alpha,k}=
[{}^W\bar\omega_{B,k}]_\times{}^WR_{B,k}{}^BJ_\omega(\bar q_k)
+{}^WR_{B,k}{}^B\dot J_\omega(\bar q_k,\dot{\bar q}_k),
$$

$$
B_{\alpha,k}={}^WR_{B,k}{}^BJ_\omega(\bar q_k).
$$

### 5.3 有符号二维重力误差

定义末端系中的重力向量：

$$
g^E(q;{}^WR_{B,k})={}^ER_W(q;{}^WR_{B,k})g^W.
$$

只选取末端系 $x,y$ 两个分量：

$$
S_{xy}=
\begin{bmatrix}
1&0&0\\
0&1&0
\end{bmatrix},
\qquad
r_g(q;{}^WR_{B,k})=S_{xy}g^E(q;{}^WR_{B,k})\in\mathbb R^2.
$$

正立时 $r_g=0$。该二维误差保留倾斜方向和符号，但正立与精确倒立都会得到零误差。当前保守关节角盒不允许右臂把瓶体翻转到倒立构型，因此初版接受这一歧义，不再为排除不可达构型计算第三维误差。

在 $\bar q_k$ 附近线性化：

$$
r_{g,k}\approx d_{g,k}+G_{g,k}x_k,
$$

这里的 $d_{g,k}$ 和 $G_{g,k}$ 就是把二维重力误差写成关于状态 $x_k=[q_k;\dot q_k]$ 的局部仿射函数时所需的常数项和状态系数。程序按以下顺序获得 $r_g$ 和 $J_g$。

首先，把 scratch 设置为工作构型 $\bar q_k$。scratch 中的 floating base 仍是本轮 $k=0$ 的实测姿态，因此直接读取的末端姿态和右臂角 Jacobian 记为

$$
{}^WR_{E,k}^{\mathrm{scratch},0},
\qquad
{}^WJ_{\omega,k}^{\mathrm{scratch},0}.
$$

它们还不对应预测的 ${}^WR_{B,k}$。程序使用与第 5.1 节相同的相对旋转

$$
\Delta R_k={}^WR_{B,k}({}^WR_{B,0})^T
$$

同时修正末端姿态和角 Jacobian：

$$
{}^WR_{E,k}
=\Delta R_k\,{}^WR_{E,k}^{\mathrm{scratch},0},
\qquad
{}^WJ_{\omega,k}
=\Delta R_k\,{}^WJ_{\omega,k}^{\mathrm{scratch},0}.
$$

因此，scratch 虽然没有真的把 floating base 改成第 $k$ 步姿态，但随后计算 $r_g$ 和 $J_g$ 时使用的已经是预测姿态下的等价结果。程序再计算

$$
g_k^E=({}^WR_{E,k})^Tg^W,
\qquad
r_{g,k}=S_{xy}g_k^E,
$$

并用解析式获得右臂关节对二维重力误差的 Jacobian：

$$
J_{g,k}
=S_{xy}({}^WR_{E,k})^T[g^W]_\times{}^WJ_{\omega,k}
=
\left.
\frac{\partial r_g(q;{}^WR_{B,k})}{\partial q}
\right|_{\bar q_k}.
$$

代码中的 `W_R_E.T @ gravity_world` 对应 $g_k^E$，取前两项对应 $S_{xy}g_k^E$；`(W_R_E.T @ skew(gravity_world) @ J_w)[:2, :]` 对应上式的 $J_{g,k}$。这里使用解析 Jacobian，不需要再对五个关节逐一做重力误差中心差分。

最后，由于 $r_g$ 只直接依赖状态中的关节角 $q$，不直接依赖 $\dot q$，所以

$$
G_{g,k}=J_{g,k}S_q
=
\begin{bmatrix}
J_{g,k}&0
\end{bmatrix},
$$

$$
d_{g,k}=r_g(\bar q_k;{}^WR_{B,k})-J_{g,k}\bar q_k.
$$

$r_g,d_{g,k}\in\mathbb R^2$，$J_{g,k}\in\mathbb R^{2\times n}$，$G_{g,k}\in\mathbb R^{2\times2n}$，对应的 $Q_g\in\mathbb R^{2\times2}$。与三维写法相比，QP 少一个重力残差分量；末端姿态和 Jacobian 仍然需要计算，因此计算量下降有限，主要收益是删除当前不可达构型所需的冗余判别。

当前 MPC 不定义 torso-relative 末端位置残差，也不构造相应的软代价或硬约束。${}^Bp_E$ 和线 Jacobian 仍需用于第 5.1 节的末端线加速度计算，这不表示存在位置保持任务。

---

## 6. 代价函数

### 6.1 各代价项的职责

每个代价项解决不同问题：

- $\|a_E\|_{Q_a}^2$：降低抓持点绝对线加速度。
- $\|\alpha_E\|_{Q_\alpha}^2$：降低抓持点绝对角加速度。
- $\|\omega_E\|_{Q_\omega}^2$：消除已经存在的世界系末端角速度，补足“角加速度为零但仍持续转动”的目标缺口。
- $\|r_g\|_{Q_g}^2$：用有符号二维倾斜误差保持杯体正立。
- $\|q-q_{\mathrm{nom}}\|_{Q_q}^2$：关节姿态正则，使关节提前远离安全盒边缘，并打破近似等价解。
- $\|\dot q\|_{Q_v}^2$：抑制持续关节运动和漂移。
- $\|u\|_R^2$：抑制过大的 `ddq_des`。

关节姿态正则不是重力姿态误差。前者让五个电机角度接近 $q_{\mathrm{nom}}=0$，后者让瓶体相对世界重力保持正立。

当前实验为 shoulder pitch 设置 $[-5^\circ,5^\circ]$ 外层安全盒，为 shoulder roll 设置更严格的非对称外层安全盒 $[-5^\circ,3^\circ]$，并通过第 7.1 节的 $1^\circ$ 内缩裕量让两者分别正常运行在 $[-4^\circ,4^\circ]$ 和 $[-4^\circ,2^\circ]$。因此不再主要依赖姿态正则把肩部推离边缘，分关节权重取

$$
Q_q=\operatorname{diag}(4,\ 4,\ 1.5,\ 0.2,\ 0.2),
$$

其中肩部正则仍强于肘部和腕部，使后两者承担更多补偿，但比上一轮略微降低。该项仍需与重力和加速度任务的实际代价对照，不能仅靠继续增大 $Q_q$ 解决已经发生的硬约束不可行。

当前仿真配置显式使用的主要权重为

$$
Q_a=I_3,\qquad
Q_\alpha=0.075I_3,\qquad
Q_\omega=2I_3,\qquad
Q_g=30I_2,\qquad
R=0.25I_5.
$$

$Q_g$ 的两个对角元素可以分别配置，但它们对应右手末端局部系中的 $x,y$ 倾斜分量，而不是两个独立的世界系安全方向。六轮闭环实验中，提高 $y$ 权重能够降低右手 $y$ 分量，却会把一部分误差转移到 $x$，并增加关节盒恢复和 QP 回退。因此当前默认仍保留 $Q_g=30I_2$，用二维范数评价整体正立程度；$Q_\omega=2I_3$ 则用于补上已经存在的末端角速度。

`configs/g1.yaml` 中的 `mpc_q_ee_omega: 2.0` 对应上述 $Q_\omega$。控制器 API 在缺少该配置项时仍回退到 $Q_\omega=0$，用于兼容旧配置；这不等于当前实验关闭了角速度项。

### 6.2 单步代价

对 $k=0,\ldots,N-1$：

$$
\begin{aligned}
\ell_k(x_k,u_k)={}&
\|{}^Wa_{E,k}\|_{Q_a}^2
+\|{}^W\alpha_{E,k}\|_{Q_\alpha}^2
+\|{}^W\omega_{E,k}\|_{Q_\omega}^2
+\|r_{g,k}\|_{Q_g}^2\\
&+\|S_qx_k-q_{\mathrm{nom}}\|_{Q_q}^2
+\|S_vx_k\|_{Q_v}^2
+\|u_k\|_R^2.
\end{aligned}
$$

其中 $a_{E,k}$ 和 $\alpha_{E,k}$ 使用第 5 节的“同下标 6 ms 区间扰动 + 区间起点运动学”模型；$\omega_{E,k}$ 和 $r_{g,k}$ 是节点量。这样阶段输入 $u_k$ 与它实际覆盖的未来控制区间使用同一时间定义。

所有权重矩阵都应满足半正定；$R$ 应正定，以保证输入方向具有足够正则。

### 6.3 终端代价

终端没有 $u_N$，也不直接惩罚末端加速度；仍惩罚终端角速度、二维重力误差和关节状态：

$$
\ell_N(x_N)=
\|{}^W\omega_{E,N}\|_{Q_{\omega,N}}^2
+\|r_{g,N}\|_{Q_{g,N}}^2
+\|S_qx_N-q_{\mathrm{nom}}\|_{Q_{q,N}}^2
+\|S_vx_N\|_{Q_{v,N}}^2.
$$

初版可取

$$
Q_{\omega,N}=2Q_\omega,\quad
Q_{g,N}=2Q_g,\quad
Q_{q,N}=2Q_q,\quad
Q_{v,N}=2Q_v.
$$

终端关节姿态项继续使用相同的分关节相对权重，不应借助终端缩放重新变成主导目标。

### 6.4 单步二次型

定义

$$
E_{a,k}=C_{a,k}S_v,
\qquad
E_{\alpha,k}=C_{\alpha,k}S_v,
\qquad
E_{\omega,k}=C_{\omega,k}S_v.
$$

忽略与决策变量无关的常数后，单步代价可写为

$$
\ell_k=
x_k^TQ_{xx,k}x_k
+2x_k^TQ_{xu,k}u_k
+u_k^TQ_{uu,k}u_k
+2f_{x,k}^Tx_k
+2f_{u,k}^Tu_k.
$$

其中

$$
\begin{aligned}
Q_{xx,k}={}&
E_{a,k}^TQ_aE_{a,k}
+E_{\alpha,k}^TQ_\alpha E_{\alpha,k}
+E_{\omega,k}^TQ_\omega E_{\omega,k}
+G_{g,k}^TQ_gG_{g,k}\\
&+S_q^TQ_qS_q+S_v^TQ_vS_v,
\end{aligned}
$$

$$
Q_{xu,k}=
E_{a,k}^TQ_aB_{a,k}
+E_{\alpha,k}^TQ_\alpha B_{\alpha,k},
$$

$$
Q_{uu,k}=
B_{a,k}^TQ_aB_{a,k}
+B_{\alpha,k}^TQ_\alpha B_{\alpha,k}
+R,
$$

$$
\begin{aligned}
f_{x,k}={}&
E_{a,k}^TQ_aD_{a,k}
+E_{\alpha,k}^TQ_\alpha D_{\alpha,k}
+E_{\omega,k}^TQ_\omega D_{\omega,k}
+G_{g,k}^TQ_gd_{g,k}\\
&-S_q^TQ_qq_{\mathrm{nom}},
\end{aligned}
$$

$$
f_{u,k}=
B_{a,k}^TQ_aD_{a,k}
+B_{\alpha,k}^TQ_\alpha D_{\alpha,k}.
$$

对局部变量

$$
z_k=
\begin{bmatrix}
x_k\\u_k
\end{bmatrix},
$$

与 OSQP 形式

$$
\frac12z_k^TH_kz_k+h_k^Tz_k
$$

对齐可得

$$
H_k=
2
\begin{bmatrix}
Q_{xx,k}&Q_{xu,k}\\
Q_{xu,k}^T&Q_{uu,k}
\end{bmatrix},
\qquad
h_k=
2
\begin{bmatrix}
f_{x,k}\\f_{u,k}
\end{bmatrix}.
$$

$Q_{xu,k}$、$Q_{uu,k}$ 和 $f_{u,k}$ 中没有角速度项，因为当前阶段的 ${}^W\omega_{E,k}$ 不直接依赖 $u_k$。终端代价使用同样的仿射展开方法得到 $H_N$ 和 $h_N$。

---

## 7. 硬约束

### 7.1 关节角约束

当前模型中，简单对称的 $\pm20^\circ$ 关节盒不能保证无碰撞：shoulder roll/yaw 同时向内时，多关节组合会使肘部接近 torso，甚至使左右瓶体相交。

当前对照实验使用以下外层安全范围：

| 关节 | 角度范围 | 弧度范围 | 设计考虑 |
|---|---:|---:|---|
| shoulder pitch | $[-5^\circ,\ 5^\circ]$ | $[-0.087,\ 0.087]$ | 缩小肩部 pitch 运动，主要由 elbow pitch 补偿 |
| shoulder roll | $[-5^\circ,\ 3^\circ]$ | $[-0.087,\ 0.052]$ | 仿真重放发现正向约 $3.35^\circ$ 起可能出现 torso 与 right shoulder yaw link 接触，因此正向边界更严格 |
| shoulder yaw | $[-20^\circ,\ 5^\circ]$ | $[-0.349,\ 0.087]$ | 正方向将手推向身体中线，故正向更严格 |
| elbow pitch | $[-40^\circ,\ 40^\circ]$ | $[-0.698,\ 0.698]$ | 承担收紧 shoulder pitch 后的主要 pitch 补偿 |
| wrist roll | $[-40^\circ,\ 40^\circ]$ | $[-0.698,\ 0.698]$ | 承担收紧 shoulder roll 后的主要 roll 调姿 |

为 shoulder pitch/roll 预留 $1^\circ$ 执行误差余量。定义

$$
m_q=[1^\circ,\ 1^\circ,\ 0,\ 0,\ 0]^T,
$$

$$
q_{\min}^{\mathrm{op}}=q_{\min}^{\mathrm{safe}}+m_q,
\qquad
q_{\max}^{\mathrm{op}}=q_{\max}^{\mathrm{safe}}-m_q.
$$

因此 shoulder pitch 正常运行在 $[-4^\circ,4^\circ]$、外层安全盒为 $[-5^\circ,5^\circ]$；shoulder roll 正常运行在 $[-4^\circ,2^\circ]$、外层安全盒为 $[-5^\circ,3^\circ]$。正常状态下写成：

$$
q_{\min}^{\mathrm{op}}\le S_qx_k\le q_{\max}^{\mathrm{op}},
\qquad k=1,\ldots,N.
$$

$x_0$ 已由实测状态锁定，不能被当前 QP 改变，因此不把 $x_0$ 再放进关节角硬边界。若实测状态位于正常运行盒外，或按当前向外速度即将越过运行盒，程序先估计当前向外速度的制动包络：

$$
\begin{aligned}
q^{\mathrm{stop},+}
&=q_0+\frac{\max(\dot q_0,0)^2}{2(0.8u_{\max})}+\epsilon_q,\\
q^{\mathrm{stop},-}
&=q_0-\frac{\max(-\dot q_0,0)^2}{2(0.8u_{\max})}-\epsilon_q.
\end{aligned}
$$

其中使用 $80\%$ 的 DDQ 上限估算制动距离，使真正的最大制动仍有 $20\%$ 可行性余量；$\epsilon_q=10^{-4}\ \mathrm{rad}$ 只用于数值容差。

对设置了 $1^\circ$ 内层裕量的 shoulder，若某一方向需要恢复，则该方向在本轮预测中临时开放到对应的外层安全边界，另一方向仍保持内层正常边界。程序同时生成满足 DQ/DDQ 上限的向内制动轨迹作为 OSQP warm-start；其中可能出现的最大向内 DDQ 只用于构造工作轨迹，不会被直接当作控制命令执行。下一真实控制拍重新根据新的 $q_0,\dot q_0$ 计算边界；当状态和制动包络都回到正常运行盒时，约束自动恢复到内层边界。因此允许的是跨多个真实控制拍逐渐恢复，而不是要求 $x_1$ 在 6 ms 内瞬间回到内层运行盒。

若当前状态仍在外层安全盒内，恢复边界不得穿过该安全盒。若当前状态已经越过外层安全盒，则任何 QP 都无法改变既成的 $x_0$；此时只把对应方向扩到制动包络，容纳受当前速度和 DDQ 上限影响而无法瞬间消除的偏差。这里使用的仍是每拍更新数值的硬约束，不是带权重的软约束，也没有向 QP 增加松弛变量。

此前较窄的 elbow/wrist 范围做过离线构型抽样；本次将两者放宽到 $\pm40^\circ$ 后，旧抽样结论不能直接沿用。当前先通过完整步行仿真的接触记录检查控制轨迹，后续仍需对新五维盒重新做离线碰撞抽样。关节盒不是对所有连续构型、模型误差和真机柔性的数学安全证明。

抽样中仍可能出现 `right_shoulder_yaw_link` 与 `torso_link` 这一相邻结构几何对。需要单独确认它是合理的相邻碰撞排除项，还是实际结构干涉；不能依靠 MPC 关节盒掩盖 XML 几何语义问题。

### 7.2 关节速度约束

初版采用统一约束：

$$
-1.0\le\dot q_{k,j}\le1.0\ \mathrm{rad/s},
\qquad
k=1,\ldots,N.
$$

base 在世界中的运动不会直接占用该限制，因为这里约束的是关节相对速度，而不是末端世界速度。$1\ \mathrm{rad/s}$ 对当前 $\pm5^\circ\sim\pm20^\circ$ 的小关节盒已经足够宽，同时可以防止优化器用高关节速度穿过允许区域。

若实验中该约束长期激活并明显损害末端稳定，再根据记录放宽；初版不建议完全取消速度约束。

### 7.3 关节加速度约束

初版采用：

$$
-8.0\le u_{k,j}\le8.0\ \mathrm{rad/s^2},
\qquad
k=0,\ldots,N-1.
$$

该值与当前执行层候选验收使用的瞬时加速度上限对齐，避免 MPC 主动要求明显超过执行安全筛选范围的 `ddq_des`。它约束的是期望加速度；实际 `qacc` 仍必须由第 10 节的前向动力学验收检查。

初版不再额外增加 DDQ rate limit 或 jerk 约束。先记录加速度约束激活率、相邻 `ddq_des` 跳变量和最终力矩平滑度；只有确认存在问题时再增加 $\Delta u$ 代价或约束。

### 7.4 当前不启用的硬约束

当前初版明确不加入：

- torso-relative 末端位置硬约束；
- 瓶体、手腕、前臂到 torso/左臂的 signed-distance 约束；
- QP 内的力矩约束；
- 接触力和摩擦锥约束。

torso-relative 位置也不进入代价函数。碰撞风险当前依赖保守关节角盒、离线构型抽样和仿真接触记录控制。

如果后续出现以下任一情况，就不能继续假设关节盒足够，必须重新引入几何距离约束或独立安全层：

- 瓶体、手腕、肘部或前臂出现非相邻接触；
- 真机几何、负载或柔性与 MuJoCo 模型差异明显；
- 为提高控制能力而放宽 shoulder roll/yaw 的向内边界；
- 多关节组合在盒内仍出现过小安全间隙。

---

## 8. 全局稀疏 QP

### 8.1 全局决策变量

预测时域为 $N$。定义

$$
n_x=2n,\qquad n_u=n,
$$

并采用交错排列：

$$
Y=
\begin{bmatrix}
x_0\\u_0\\x_1\\u_1\\\vdots\\x_{N-1}\\u_{N-1}\\x_N
\end{bmatrix}
\in\mathbb R^M,
$$

$$
M=(N+1)n_x+Nn_u.
$$

状态和输入块的列偏移为

$$
c_x(k)=k(n_x+n_u),\qquad k=0,\ldots,N-1,
$$

$$
c_u(k)=c_x(k)+n_x,
$$

$$
c_x(N)=N(n_x+n_u).
$$

### 8.2 全局目标函数

OSQP 目标为

$$
\min_Y\frac12Y^TPY+q_{\mathrm{osqp}}^TY.
$$

对 $k=0,\ldots,N-1$，把单步 $H_k$ 放入对应的 $(x_k,u_k)$ 主对角块，把 $h_k$ 放入对应向量段；最后放入终端 $H_N,h_N$：

$$
P=\operatorname{blkdiag}(H_0,H_1,\ldots,H_{N-1},H_N),
$$

$$
q_{\mathrm{osqp}}=
\begin{bmatrix}
h_0\\h_1\\\vdots\\h_{N-1}\\h_N
\end{bmatrix}.
$$

这里每个 $H_k$ 本身包含 $x_k$ 与 $u_k$ 的交叉块，因此“分块对角”不表示状态和输入之间没有耦合。

数值实现时应对 Hessian 做对称化，并保留很小的正则：

$$
P\leftarrow\frac12(P+P^T)+\epsilon I,
\qquad \epsilon>0.
$$

### 8.3 全局动力学等式约束

初始状态约束：

$$
x_0=x_{\mathrm{meas}}.
$$

每一步转移约束：

$$
-Ax_k-Bu_k+x_{k+1}=0,
\qquad k=0,\ldots,N-1.
$$

组装为

$$
A_{\mathrm{dyn}}Y=b_{\mathrm{dyn}},
$$

其中

$$
A_{\mathrm{dyn}}=
\begin{bmatrix}
I&0&0&0&\cdots&0\\
-A&-B&I&0&\cdots&0\\
0&0&-A&-B&\ddots&\vdots\\
\vdots&\vdots&\ddots&\ddots&\ddots&0\\
0&0&\cdots&-A&-B&I
\end{bmatrix},
$$

$$
b_{\mathrm{dyn}}=
\begin{bmatrix}
x_{\mathrm{meas}}\\
0\\
\vdots\\
0
\end{bmatrix}
\in\mathbb R^{(N+1)n_x}.
$$

第一块行在 $c_x(0)$ 插入 $I$。第 $k+1$ 个块行分别在 $c_x(k),c_u(k),c_x(k+1)$ 插入 $-A,-B,I$。这样可以无歧义地连接 $x_k,u_k,x_{k+1}$。

### 8.4 全局边界约束

不再使用一个对所有 $Y$ 分量都施加有限上下界的简单单位阵，而是按变量类型构造选择矩阵：

- 对 $x_1,\ldots,x_N$ 插入 $S_q$，施加关节角边界；
- 对 $x_1,\ldots,x_N$ 插入 $S_v$，施加关节速度边界；
- 对 $u_0,\ldots,u_{N-1}$ 插入 $I_n$，施加关节加速度边界。

记组装结果为

$$
l_{\mathrm{box}}\le A_{\mathrm{box}}Y\le u_{\mathrm{box}}.
$$

当前共有 $3Nn$ 行。正常状态使用固定运行盒；进入恢复时，只更新相同约束行的上下界：

$$
\begin{aligned}
\underline q_k&\le S_qx_k\le\overline q_k,&&k=1,\ldots,N,\\
\dot q_{\min}&\le S_vx_k\le\dot q_{\max},&&k=1,\ldots,N,\\
u_{\min}&\le u_k\le u_{\max},&&k=0,\ldots,N-1.
\end{aligned}
$$

正常状态下 $\underline q_k=q_{\min}^{\mathrm{op}}$、$\overline q_k=q_{\max}^{\mathrm{op}}$；恢复状态下采用第 7.1 节本轮冻结、下一真实控制拍重新计算的恢复边界。

### 8.5 OSQP 最终组装

最终约束写成

$$
l\le A_{\mathrm{cons}}Y\le u,
$$

$$
A_{\mathrm{cons}}=
\begin{bmatrix}
A_{\mathrm{dyn}}\\
A_{\mathrm{box}}
\end{bmatrix},
$$

$$
l=
\begin{bmatrix}
b_{\mathrm{dyn}}\\
l_{\mathrm{box}}
\end{bmatrix},
\qquad
u=
\begin{bmatrix}
b_{\mathrm{dyn}}\\
u_{\mathrm{box}}
\end{bmatrix}.
$$

因此：

$$
A_{\mathrm{cons}}\in
\mathbb R^{((N+1)n_x+3Nn)\times M}.
$$

动力学等式通过相同的上下界 $b_{\mathrm{dyn}}$ 表示。当前没有位置、碰撞或松弛变量；正常运行与逐步恢复共用相同的 $A_{\mathrm{cons}}$ 稀疏结构，只在线更新关节角约束的 $l,u$。如果以后增加一般线性化约束，只需继续向 $A_{\mathrm{cons}},l,u$ 追加行。

---

## 9. 在线滚动运行流程

每个 $0.006\ \mathrm{s}$ MPC 控制周期执行：

1. 读取当前右臂 $q,\dot q$，设置 $x_{\mathrm{meas}}$。
2. 读取当前 base 的 $a_B,\omega_B,\alpha_B,R_B$；对 $a_B,\alpha_B$ 限幅并使用 MPC 专用低延迟通道滤波。
3. 前馈关闭时生成 $N+1$ 个节点零阶保持量和 $N$ 个区间零阶保持量；前馈开启时按未来连续相位插值节点模板与 6 ms 区间模板，用模板加慢偏差生成三个向量，并以 $R_{B,0}$ 锚定模板相对姿态变化。
4. 平移上一拍最优解，生成本轮工作轨迹和 OSQP warm-start；首次运行使用零输入滚动。
5. 对每个预测步根据 $\bar q_k$ 计算 scratch 中的 $p_E-p_B$，再按 $R_{B,k}R_{B,0}^T$ 一致修正位置差、Jacobian、Jacobian 导数和末端姿态；线/角加速度模型使用同下标 6 ms 区间扰动，角速度和二维重力模型使用节点扰动。
6. 构造每步与终端代价，更新稀疏 $P,q_{\mathrm{osqp}}$。
7. 更新初始状态、动力学等式和关节角/速度/加速度边界；若当前状态或制动包络位于正常运行盒外，则开放对应方向的恢复边界，并用有界制动轨迹 warm-start。
8. 调用 OSQP。
9. 检查 solver status、原始/对偶残差和约束违反量。
10. 求解成功时只执行第一拍

$$
\ddot q_{\mathrm{des}}=u_0^\star.
$$

11. 下一个控制周期重新测量、重新线性化并重新求解。

OSQP 的稀疏结构在 horizon 和约束类型不变时保持固定。初始化时建立一次 sparsity pattern，在线只更新非零数值、上下界和 warm-start，避免每拍重新分配矩阵。

当前二阶积分器的 $\tfrac12\Delta t^2=1.8\times10^{-5}$，DDQ 对单步关节位置的系数很小，因此 `rho`、`adaptive_rho` 和 `scaled_termination` 都保留为配置项。当前完整时变 QP 继续使用 OSQP 自适应 `rho`；不能只根据删除末端任务后的简化边缘算例固定 `rho`，必须以完整回合的成功率、残差和耗时共同选择数值设置。这些设置不改变 QP 的物理约束。

### 9.1 求解失败回退

若 OSQP 超时、报告不可行或数值异常，当前实现使用温和的“姿态回中 + 速度制动”：

$$
u_{\mathrm{fb,raw}}
=4(q_{\mathrm{nom}}-q)-4\dot q,
\qquad
|u_{\mathrm{fb}}|\le0.5u_{\max}.
$$

$q_{\mathrm{nom}}$ 先裁剪到内层运行盒。常规回退最多使用一半 DDQ 上限；随后仍检查下一拍 DQ 和外层安全盒。只有按当前速度下一拍即将越过外层安全盒时，硬安全投影才允许更强的向内加速度。程序记录失败原因、回退输入、回退是否可行和约束违反量。

不应直接执行包含 NaN、未收敛或明显违反边界的 QP 解。

---

## 10. 从 `ddq_des` 到最终执行力矩

MPC 只输出第一拍 `ddq_des`。下层执行链不改变 MPC 解，只负责寻找能在当前整机状态下实现该加速度的力矩。

### 10.1 短时 PD 参考

用一拍积分生成小 PD 修正需要的参考：

$$
\dot q_{\mathrm{ref}}=
\dot q+\ddot q_{\mathrm{des}}\Delta t,
$$

$$
q_{\mathrm{ref}}=
q+\dot q\Delta t+
\frac12\ddot q_{\mathrm{des}}\Delta t^2,
$$

$$
\tau_{\mathrm{pd}}=
K_p(q_{\mathrm{ref}}-q)+
K_d(\dot q_{\mathrm{ref}}-\dot q).
$$

这条积分路径只生成局部反馈参考，不单独承担 DDQ 跟踪。

### 10.2 逆动力学生成名义力矩

在 MuJoCo 临时数据中复制当前整机状态，令非右臂期望加速度为零、右臂期望加速度为 `ddq_des`，调用 `mj_inverse`。

当前采用非摩擦约束感知的名义前馈：

$$
\tau_{\mathrm{ff}}
=
qfrc_{\mathrm{inverse}}
+qfrc_{\mathrm{constraint,nonfriction}}.
$$

它加回 contact、joint/tendon limit 和 equality 等非摩擦约束，使执行器不主动对抗这些约束；`FRICTION_DOF` 和 `FRICTION_TENDON` 不加回，因此摩擦补偿仍保留在逆动力学结果中。

名义力矩为

$$
\tau_{\mathrm{nom}}=
\operatorname{clip}
\left(
\tau_{\mathrm{ff}}+\tau_{\mathrm{pd}},
\tau_{\min},\tau_{\max}
\right).
$$

非右臂加速度为零的假设只用于提供初始名义力矩，最终加速度由完整前向动力学评估。

### 10.3 数值前向动力学局部映射

固定腿、腰和左臂的当前控制力矩，先用 $\tau_{\mathrm{nom}}$ 执行一次完整前向动力学，得到右臂基准加速度 $\ddot q_b$。

分别给 5 个右臂力矩增加小扰动 $\epsilon=0.1\ \mathrm{N\,m}$：

$$
G_\tau[:,j]\approx
\frac{
\ddot q_{\mathrm{right}}(\tau_{\mathrm{nom}}+\epsilon e_j)
-\ddot q_b
}{\epsilon}.
$$

局部模型为

$$
\ddot q_{\mathrm{right}}
\approx
\ddot q_b+G_\tau\Delta\tau.
$$

用阻尼最小二乘求修正：

$$
\Delta\tau=
\arg\min_{\Delta\tau}
\left\|
G_\tau\Delta\tau-
(\ddot q_{\mathrm{des}}-\ddot q_b)
\right\|_2^2
+\lambda\|\Delta\tau\|_2^2.
$$

### 10.4 第一轮完整验收

构造并完整评估四个候选：

$$
\tau(s)=
\operatorname{clip}
(\tau_{\mathrm{nom}}+s\Delta\tau),
\qquad
s\in\{1,0.5,0.25,0.125\}.
$$

每个候选都在临时 `MjData` 中调用完整 `mj_forward`，不推进真实仿真时间。记录：

$$
e(s)=
\|\ddot q_{\mathrm{right}}(\tau(s))-\ddot q_{\mathrm{des}}\|_2,
$$

$$
e_{\max}(s)=
\max_j
|\ddot q_{\mathrm{right},j}(\tau(s))-\ddot q_{\mathrm{des},j}|,
$$

$$
a_{\max}(s)=
\max_j|\ddot q_{\mathrm{right},j}(\tau(s))|.
$$

严格候选同时满足：

1. 总误差小于当前工作点；
2. $e_{\max}(s)\le4\ \mathrm{rad/s^2}$；
3. $a_{\max}(s)\le8\ \mathrm{rad/s^2}$。

对四个比例全部求值，并按以下顺序选择：

1. 若存在严格候选，在严格候选中选择总误差最小者。
2. 若没有严格候选，但存在“总误差改善且满足瞬时加速度上限”的候选，按 $(e_{\max},e)$ 的字典序选择，优先控制最差关节误差。
3. 若连上述候选也不存在，在“仅总误差改善”的候选中选择总误差最小者；若没有任何改善候选，则保留 $\tau_{\mathrm{nom}}$。

第 2、3 类结果不应标记为严格安全候选，必须分别记录 `tracking_safety_satisfied` 和 `qacc_safety_satisfied`。第 3 类只允许在当前 MPC 仿真验证阶段用于暴露执行链问题；真机不得执行已知超过瞬时加速度上限的候选，而应改用经完整前向动力学检查的保持/制动回退，或启用第 10.5 节的安全救援。

### 10.5 第二轮和安全救援的当前策略

当前 MPC 仿真启用按需第二轮重线性化和最多两轮安全救援，同时完整记录：

- 基准、局部预测和完整验收的 DDQ；
- 四个候选的比例与误差；
- 总误差、最大单关节误差和最大绝对加速度；
- 是否存在严格候选；
- 第一轮最终残差；
- 计算耗时。

第二轮不是每拍固定执行；出现以下情况时才触发：

- 第一轮残差范数超过 $5\ \mathrm{rad/s^2}$；
- 没有满足单关节误差条件的严格候选；
- 局部映射跨过摩擦、接触或限位模式后明显失真。

启用时，应在第一轮已接受力矩处重新构建 $G_{\tau,1}$，针对剩余残差再做一次阻尼最小二乘和四候选完整验收，不能直接复用第一轮 Jacobian。

若两轮后仍超过瞬时加速度上限，当前仿真最多再做两轮额外安全救援，并单独记录触发率、成功率和耗时。若安全救援仍失败，程序取上一仿真步实际执行的右臂力矩，在本拍当前整机状态、非右臂力矩和接触条件下重新调用完整 `mj_forward`。只有其最大瞬时关节加速度不超过上限时才保持该力矩；不能因为它上一拍已经执行过，就跳过本拍验收。该候选也不等同于名义力矩，不能退回已经由完整动力学判定为超限的名义力矩。

启用这两层处理的直接原因是旧运行中已经出现“完整前向动力学准确预测瞬时加速度超限，但仍执行不安全候选”的样本。仿真阶段保留两轮验收便于隔离 DDQ 执行误差；实时部署可根据耗时和危险样本统计降级为一轮验收，但保持候选仍必须重新验收。

实时部署时可以固定为一轮验收；这里的“一轮”仍表示评估完整的四个比例候选，而不是未经完整动力学检查直接执行局部线性修正。

### 10.6 最终执行

只把最终选中的一个力矩写入右臂执行器，然后真实仿真只推进一次：

```text
q, dq, measured base motion
        -> one MPC QP every 6 ms
        -> ddq_des = u0*
        -> one-step q_ref, dq_ref -> tau_pd
        -> inverse dynamics -> tau_ff
        -> clipped tau_nom
        -> baseline forward dynamics
        -> finite-difference G_tau
        -> damped least-squares correction
        -> first-pass four-candidate full validation
        -> optional second pass / safety rescue
        -> if still unsafe: validate previous executed torque in current state
        -> tau_cmd
        -> one real mj_step every 2 ms
```

---

## 11. 记录与验证

### 11.1 MPC 求解记录

每个 MPC 更新周期至少记录：

- $x_{\mathrm{meas}}$、工作轨迹和预测状态/输入；
- `ddq_des`；
- OSQP status、迭代次数、primal/dual residual；
- QP 组装时间和求解时间；
- 关节角、速度、加速度约束的最小裕量和激活率；
- 各代价项的单独数值；
- $N+1$ 个节点扰动、$N$ 个 6 ms 区间扰动、模板版本与 SHA-256、慢偏差以及终端姿态预测变化量。

### 11.2 物理效果

继续使用抓持点统计：

- 世界系末端线加速度；
- 世界系末端角速度；
- 世界系末端角加速度；
- 有符号二维重力误差；
- 关节角、速度和 `ddq_des`；
- 力矩大小、饱和率和平滑度。

虽然初版不在 QP 中加入碰撞距离约束，仍必须记录 MuJoCo 接触对，并重点区分：

- 瓶体、手腕、肘部、前臂与 torso/左臂的非相邻接触；
- `right_shoulder_yaw_link` 与 `torso_link` 的相邻结构接触。

### 11.3 DDQ 执行记录

至少记录：

- `ddq_des`；
- 名义力矩对应的 `ddq_baseline`；
- $G_\tau$ 的奇异值和条件数；
- 局部模型预测 `ddq_predicted`；
- 四个候选的完整前向动力学 `ddq_candidate`；
- 最终验收 `ddq_validated`；
- 真实控制区间平均 `ddq_real`；
- 候选比例、拒绝原因和第一轮残差；
- 可选第二轮/安全救援的触发、接受、成功和耗时；
- 上一拍力矩候选是否需要验收、是否通过以及是否最终采用。

### 11.4 三层定位诊断

仅看最终末端加速度不能判断问题位于模板、MPC 还是力矩执行层。当前实现按同一个 6 ms 右臂控制区间记录三层数据。

第一层判断 6 ms 区间扰动是否正确。这里用 $i$ 表示连续的真实右臂控制拍，用 $k$ 表示某一拍内部的 MPC 预测步。在控制拍 $i$ 求解时，`intervals[0]` 明确预测随后区间 $[t_i,t_{i+1})$：

$$
\left(
{}^W\hat{\bar a}_{B,0|i},
{}^W\hat{\bar\omega}_{B,0|i},
{}^W\hat{\bar\alpha}_{B,0|i}
\right).
$$

下一控制拍到来后，程序用相同区间两端的 torso 世界系速度计算真实平均值：

$$
{}^W\bar a_{B,i}^{\mathrm{real}}
=\frac{{}^Wv_B(t_{i+1})-{}^Wv_B(t_i)}{\Delta t},
\qquad
{}^W\bar\alpha_{B,i}^{\mathrm{real}}
=\frac{{}^W\omega_B(t_{i+1})-{}^W\omega_B(t_i)}{\Delta t},
$$

并用区间内角速度的复合梯形平均得到 ${}^W\bar\omega_{B,i}^{\mathrm{real}}$。这样比较的是同一物理量、同一 6 ms 区间和同一世界系，不再把“区间起点滤波值”与“随后区间平均值”混在一起。

`mpc_diagnostics.json` 记录三种区间扰动的总体向量误差、误差相对实际 RMS 的百分比，以及 $\|\bar a_B\|>5\ \mathrm{m/s^2}$ 冲击样本中的误差；`base_disturbance_interval_template_prediction_vs_actual.png` 只画 evaluation 中间一个完整步态周期的 torso/base 未来 6 ms 区间模板预测与实际曲线，便于检查峰值幅度和相位。`base_disturbance_node_template_tracking.png` 另行比较当前节点测量、同相位节点模板以及上一拍对当前节点的 $k=1$ 预测；它们只评价瞬时节点波形，不再作为 $u_k$ 所用区间扰动的主要质量指标。

第二层判断控制器一步模型结果。把实际采用的 $u_0=ddq_{\mathrm{des}}$ 和预测的 $x_1$ 代入仿射任务模型，记录一步末端线加速度、角速度、角加速度、二维重力误差，以及由 $Q_A,Q_\alpha,Q_\omega,Q_G,Q_q,Q_v,R$ 得到的七项加权代价。这组“一步七项代价”是为了对齐同一个真实控制区间而构造的诊断量，不是单独的阶段代价 $\ell_0$，也不是整个时域的 objective：线/角加速度来自 $(x_0,u_0)$，角速度、重力、姿态和关节速度来自 $x_1$，输入代价来自 $u_0$。实际末端角速度在下一次控制更新前直接读取 MuJoCo 世界系 site 角速度，只有线/角加速度使用 6 ms 区间的速度差分。只有 OSQP 成功时，这些量才是 QP 优化结果；求解失败时记录的是回退输入的一步模型预测，诊断图用红色标记回退样本，不能称为“MPC 理想值”。通用 `metrics.png` 在 MPC 实验中也把主重力范数明确画成 $x,y$ 二维，$z$ 只保留为不进入代价的诊断分量。

真正的 MPC 末端任务诊断保存在 `mpc_diagnostics.json`、`mpc_diagnostics_preview.csv` 和 `mpc_end_effector_task_prediction_vs_actual.png`；该图覆盖整个 evaluation，与上述只画中间一个周期的 base 扰动图不是同一组数据的全程/局部关系。`mpc_tracking_diagnostics.json` 与 `mpc_tracking_preview.csv` 是历史命名的通用 DDQ/LQR 兼容输出，主要使用其中的 DDQ tracking；不要把其中未启用的旧 LQR cost 列当成 MPC 七项代价。

第三层区分执行误差和任务模型误差。先由相邻控制拍前的关节速度差得到 $ddq_{\mathrm{real}}$，再把它回代到与 QP 完全相同、且已使用对应 6 ms 区间扰动的加速度仿射模型：

$$
\hat a_E^{\mathrm{real}\,ddq}=c_a+B_a ddq_{\mathrm{real}},
\qquad
\hat\alpha_E^{\mathrm{real}\,ddq}=c_\alpha+B_\alpha ddq_{\mathrm{real}}.
$$

$ddq_{\mathrm{des}}$ 模型结果与上述回代结果的差主要反映 DDQ 执行误差；上述回代结果与 MuJoCo 末端速度差分加速度的差主要反映区间模板误差、区间起点运动学近似、局部线性化和当前未建模的浮动基/接触反作用。两部分相加才是 QP 理想结果与真实末端响应之间的总差，因此不能仅凭总差直接断言力矩执行失败。

当前仿真用这些数据评估已启用的按需第二轮和安全救援是否值得保留；实时版本仍可根据危险样本减少量和耗时实验降级为一轮。

---

## 12. 已知局限与后续扩展

1. phase-based 模板只适用于与采集时相近的下肢策略、$0.8\ \mathrm{s}$ 周期、速度指令和平地条件。当前已使用 H 系模板消除绝对世界航向差异，但它不能补偿步态周期、速度、地形或下肢策略变化造成的扰动波形变化。
2. 关节角盒只在当前模型中做过工程抽样验证，不能替代真机几何标定和独立安全验证。
3. 二维重力误差不能区分正立和倒立。若以后明显放宽关节角盒、允许从任意构型启动或加入更大的姿态自由度，应恢复第三维误差，或增加保证末端朝上的半空间条件。
4. 低权重关节姿态正则只用于数值与构型偏好，碰撞安全主要来自硬边界和验证，不应通过增大 $Q_q$ 间接实现。
5. `ddq_des` 约束不等于真实接触加速度保证，下层前向动力学验收仍然必要。
6. 即使启用两轮验收和安全救援，仍可能出现没有安全改善候选的离散时刻。当前仿真已把“上一仿真步实际执行且在本拍重新验收安全的力矩”纳入候选；若该候选在当前状态下也不安全，仍需在真机部署前增加独立的紧急制动或停机策略。
7. 当前 QP 不显式限制力矩。若以后需要在预测时域内考虑力矩上限，可在工作点冻结

$$
\tau_k\approx M(\bar q_k)u_k+h(\bar q_k,\dot{\bar q}_k),
$$

从而把 $\tau_{\min}\le\tau_k\le\tau_{\max}$ 作为关于 $u_k$ 的线性约束。

8. 如果需要主动接触、推压环境或严格处理浮动基耦合，应升级为包含整机动力学、接触力和摩擦锥的 whole-body MPC，而不是继续扩展当前运动学积分器。

9. 扰动前馈关闭时并不是完全忽略 base 运动，而是每拍测量当前节点并在 72 ms 时域内零阶保持；模板开启后才提供未来节点变化和与每个 $u_k$ 对齐的 6 ms 区间平均扰动。因此开关实验的差值表示未来预测的附加收益，不是全部 base 补偿的收益。
10. 即使模板完全准确，5 个关节 DDQ 也不能同时独立消除三维线加速度、三维角加速度、三维角速度和二维重力误差；关节盒、DQ/DDQ 上限和互相冲突的任务会留下不可消除残差。当前最佳闭环回合中，6 ms 区间线加速度和角加速度预测误差分别约为实际 RMS 的 29% 和 32%，也不是完全已知未来。
11. 当前 MPC 策略计算平均约 12 ms，超过 6 ms 控制更新周期。仿真按模型时间仍可验证控制逻辑，但实时部署前必须通过减少运动学计算、降低更新频率或其他优化满足实际时间预算。
12. 当前角速度模型冻结 $J_\omega(\bar q_k)$，忽略 $\partial(J_\omega\dot q)/\partial q$。若以后角速度任务权重明显增大、关节运动范围放宽或模型/实际角速度开始出现显著偏差，应恢复完整的一阶线性化。
13. 新区间前馈会在落脚前后更积极地使用 DDQ。当前最佳回合仍有 7 个控制区间的真实平均 DDQ 略高于 $8\ \mathrm{rad/s^2}$，最大约为 $8.334\ \mathrm{rad/s^2}$，多数伴随接触数量切换。真机前应为预测和候选验收增加内部安全裕量，并禁止执行已知不严格安全的“仅有进展”候选。

---

## 13. 实现检查清单

- 右臂维数固定为 $n=5$，所有数组顺序一致。
- `right_grasp_site` 是唯一任务末端。
- 状态为 $x=[q;\dot q]$，输入为 $u=\ddot q$。
- 状态方程严格采用 $x_{k+1}=Ax_k+Bu_k$，不加入额外扰动项。
- 可通过 `mpc_disturbance_feedforward_enabled` 在零阶保持与相位模板前馈之间切换。
- `mpc_disturbance_template` 只能选择 `raw`、`half_smoothed` 或 `fully_smoothed`。
- 当前默认模板为 `raw`；模板选择以同配置闭环任务代价和安全指标为主，预测误差为辅。
- 三种模板均从 `disturbance_model_new_heading/templates_heading_interval` 读取，schema 必须为 2，坐标系必须为 `heading`，区间长度必须与 MPC 的 6 ms 控制周期一致。
- $H_j$ 由上一完整周期 $C_{j-1}$ 的 torso yaw 圆周均值定义，并在当前周期内保持；第一个周期自动回退到零阶保持。
- 模板的 $a_B,\omega_B,\alpha_B,R_B$ 先从 H 系转换到 W 系，MPC 接收的节点和区间扰动始终使用世界系表达。
- 每次预测一次性生成 $N+1$ 个节点量和 $N$ 个区间量；节点 $k=0$ 等于当前测量，未来向量使用模板加慢偏差，未来姿态使用当前测量锚定的模板相对转动。
- 每个 $u_k$ 只与同下标、覆盖 $[t_k,t_{k+1})$ 的 6 ms 区间扰动配对；重力、姿态、角速度状态项和终端项使用节点扰动。
- MPC 的加速度与角加速度测量使用独立低延迟滤波通道，当前系数为 0.5；它不改变离线区间模板中的落脚高频波形。
- 离线模板文件不包含 $p_B$，但包含 Markley 均值姿态四元数及其旋转矩阵；完整扰动点中的 $R_{B,k}$ 由当前实测姿态锚定模板相对姿态得到。
- scratch 中直接由该步工作构型计算 ${}^Wp_E-{}^Wp_B$；世界系平移严格相消，$p_B$ 不进入模板。未来步用 $R_{B,k}R_{B,0}^T$ 同步旋转该位置差和各运动学量。
- 每拍只求解一个 QP，并使用上一拍解平移 warm-start。
- $x_0=x_{\mathrm{meas}}$ 被完整写入等式约束。
- 每个 $x_k,u_k,x_{k+1}$ 都由一行块 $[-A,-B,I]$ 连接。
- 关节角、速度和加速度边界按正确阶段写入。
- 二维重力残差及其 Jacobian、权重维数均为 2。
- 世界系末端角速度使用 ${}^W\omega_E={}^W\omega_B+{}^WJ_\omega\dot q$，阶段和终端代价均可由 `mpc_q_ee_omega` 配置。
- 不构造 torso-relative 位置残差、位置代价或位置硬约束。
- 初版不启用 signed-distance 碰撞约束。
- 关节姿态正则使用低权重，不与二维重力误差混淆。
- OSQP 矩阵使用固定稀疏结构，在线只更新数值。
- 只执行第一拍 `ddq_des`。
- 当前仿真启用按需第二轮重线性化、最多两轮安全救援，以及救援失败后的上一拍力矩重新验收；实时部署可根据耗时实验降级为一轮完整候选验收。
- 所有被拒绝候选只在临时动力学数据中计算，不能写入真实执行器。
- 对 MPC、DDQ 执行映射和完整控制循环分别记录耗时。
