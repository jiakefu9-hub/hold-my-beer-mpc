# 单臂 LQR 稳定控制系统设计文档

## 1. 目标

LQR 作为当前项目中 MPC 的简化版，用于在**不显式处理硬约束**的前提下，完成右臂持杯稳定控制。其建模、符号和物理目标尽量与 `MPC_DESIGN.md` 保持一致，便于后续从 LQR 平滑切换到 MPC。

系统分工保持不变：

- 下肢：沿用现有 RL locomotion
- 上肢：右臂用有限时域、时变 LQR 控制
- 扰动前馈：使用预测得到的 base 扰动进入每个时刻的局部模型
- 输出：LQR 优化关节加速度 `u_k = ddq_k`，由 MuJoCo 逆动力学转换为前馈力矩，并叠加 `q_ref, dq_ref` 的 PD 修正

---

## 2. 坐标系与状态定义

坐标系沿用 `MPC_DESIGN.md`：

- `{W}`：世界系
- `{B}`：躯干 / IMU 基座系
- `{E}`：末端抓持点系

关节维数记为 `n = 5`（右臂 5 个关节）。

定义状态与控制：

$
x_k =
\begin{bmatrix}
q_k \\
\dot q_k
\end{bmatrix}
\in \mathbb{R}^{2n},
\qquad
u_k = \ddot q_k \in \mathbb{R}^{n}
$

定义选择矩阵：

$
S_q =
\begin{bmatrix}
I_n & 0
\end{bmatrix},
\qquad
S_v =
\begin{bmatrix}
0 & I_n
\end{bmatrix}
$

因此：

$
q_k = S_q x_k, \qquad \dot q_k = S_v x_k
$

---

## 3. 离散状态方程

采用和 MPC 相同的二阶积分器离散模型：

$
A =
\begin{bmatrix}
I_n & \Delta t I_n \\
0 & I_n
\end{bmatrix},
\qquad
B =
\begin{bmatrix}
\frac{1}{2}\Delta t^2 I_n \\
\Delta t I_n
\end{bmatrix}
$

$
x_{k+1} = A x_k + B u_k
$

说明：

- `A, B` 在初版中可视为常量
- 若后续需要更高精度，也可替换成时变离散化模型 `A_k, B_k`
- LQR 初版不单独引入额外扰动状态，所有 walking/base 扰动都进入观测代价中的仿射项

---

## 4. 末端物理量的局部仿射模型

这一节与 `MPC_DESIGN.md` 完全一致，只是把同样的物理系数拿来服务 LQR。对每个预测步 `k`，都围绕该步工作点

$
\bar x_k = \begin{bmatrix} \bar q_k \\ \dot{\bar q}_k \end{bmatrix}
$

冻结运动学与 base 扰动系数。程序里凡是带下标 `k` 的 `D/C/B/d/G`，都应理解为“在第 `k` 个预测步、围绕 `\bar x_k` 线性化后得到的局部系数”。

### 4.1 末端线加速度

采用与 `MPC_DESIGN.md` 相同的局部仿射形式：

$
{}^W a_{E,k} \approx D_{acc,k} + C_{acc,k} S_v x_k + B_{acc,k} u_k
$

其中

$
D_{acc,k} = {}^W a_{B,k} + {}^W \alpha_{B,k} \times ({}^W R_{B,k} \, {}^B p_E(\bar q_k)) + {}^W \omega_{B,k} \times \bigl({}^W \omega_{B,k} \times ({}^W R_{B,k} \, {}^B p_E(\bar q_k))\bigr)
$

$
C_{acc,k} = 2[{}^W \omega_{B,k}]_\times {}^W R_{B,k} \, {}^B J_v(\bar q_k) + {}^W R_{B,k} \, {}^B \dot J_v(\bar q_k, \dot{\bar q}_k)
$

$
B_{acc,k} = {}^W R_{B,k} \, {}^B J_v(\bar q_k)
$

这里 `D_acc,k` 是 base 已知扰动常数项，`C_acc,k` 是关节速度耦合矩阵，`B_acc,k` 是关节加速度到末端线加速度的线性映射矩阵。

### 4.2 末端角加速度

同理，末端角加速度采用：

$
{}^W \alpha_{E,k} \approx D_{\alpha,k} + C_{\alpha,k} S_v x_k + B_{\alpha,k} u_k
$

其中

$
D_{\alpha,k} = {}^W \alpha_{B,k}
$

$
C_{\alpha,k} = [{}^W \omega_{B,k}]_\times {}^W R_{B,k} \, {}^B J_\omega(\bar q_k) + {}^W R_{B,k} \, {}^B \dot J_\omega(\bar q_k, \dot{\bar q}_k)
$

$
B_{\alpha,k} = {}^W R_{B,k} \, {}^B J_\omega(\bar q_k)
$

### 4.3 重力方向误差

定义末端防洒水误差：

$
r_g(q) = P_{xy}({}^E R_W(q) g^W) \in \mathbb{R}^2
$

在工作点 `\bar q_k` 处做一阶线性化：

$
r_g(q) \approx r_g(\bar q_k) + J_{g,k}(q - \bar q_k), \qquad J_{g,k} = \left.\frac{\partial r_g}{\partial q}\right|_{q=\bar q_k}
$

由于 `q = S_q x`，可改写成本文统一使用的仿射形式：

$
r_{g,k} \approx d_{g,k} + G_{g,k} x_k
$

其中

$
G_{g,k} = J_{g,k} S_q
$

$
d_{g,k} = r_g(\bar q_k) - J_{g,k} \bar q_k
$

### 4.4 程序实现对应关系

后续程序里，每个预测步 `k` 的这些量应按下面方式构造：

- base 预测器提供 `{}^W a_{B,k}, {}^W \omega_{B,k}, {}^W \alpha_{B,k}, {}^W R_{B,k}`
- 运动学 helper 提供 `{}^B p_E(\bar q_k), {}^B J_v(\bar q_k), {}^B J_\omega(\bar q_k)`
- `{}^B \dot J_v, {}^B \dot J_\omega, J_{g,k}` 可由解析法或有限差分法在工作点处计算
- 然后严格按本节公式生成 `D_acc,k, C_acc,k, B_acc,k, D_alpha,k, C_alpha,k, B_alpha,k, d_g,k, G_g,k`

后文第 5 节到第 9 节中的所有 `Q_xx,k, Q_xu,k, Q_uu,k, f_x,k, f_u,k, P_k, p_k` 都建立在这里这些系数之上。因此本节就是后续程序实现时最直接的“物理建模说明书”。

---

## 5. 单步代价函数

每一步代价定义为：

$
\ell_k(x_k, u_k) = \|{}^W a_{E,k}\|_{Q_a}^2 + \|{}^W \alpha_{E,k}\|_{Q_\alpha}^2 + \|r_{g,k}\|_{Q_g}^2 + \|q_k - q_{nom}\|_{Q_q}^2 + \|\dot q_k\|_{Q_v}^2 + \|u_k\|_{R}^2
$

当前项目里可直接取：

$
q_{nom} = 0
$

于是：

$
\|q_k - q_{nom}\|_{Q_q}^2 = \|q_k\|_{Q_q}^2
$

终端代价定义为：

$
\ell_N(x_N) = x_N^T Q_N x_N
$

推荐初版：

$
Q_N = \text{block\_diag}(Q_q^{(N)}, Q_v^{(N)})
$

---

## 6. 展开为标准二次型

将第 4 节中的局部仿射模型代入第 5 节代价，可得单步代价可统一写成：

$
\ell_k(x_k, u_k) = x_k^T Q_{xx,k} x_k + 2 x_k^T Q_{xu,k} u_k + u_k^T Q_{uu,k} u_k + 2 f_{x,k}^T x_k + 2 f_{u,k}^T u_k + \text{const}
$

其中常数项对优化无影响，可直接丢弃。

### 6.1 二次项矩阵

$
Q_{xx,k} = S_v^T C_{acc,k}^T Q_a C_{acc,k} S_v + S_v^T C_{\alpha,k}^T Q_\alpha C_{\alpha,k} S_v + G_{g,k}^T Q_g G_{g,k} + S_q^T Q_q S_q + S_v^T Q_v S_v
$

$
Q_{xu,k} = S_v^T C_{acc,k}^T Q_a B_{acc,k} + S_v^T C_{\alpha,k}^T Q_\alpha B_{\alpha,k}
$

$
Q_{uu,k} = B_{acc,k}^T Q_a B_{acc,k} + B_{\alpha,k}^T Q_\alpha B_{\alpha,k} + R
$

### 6.2 一次项向量

$
f_{x,k} = S_v^T C_{acc,k}^T Q_a D_{acc,k} + S_v^T C_{\alpha,k}^T Q_\alpha D_{\alpha,k} + G_{g,k}^T Q_g d_{g,k} - S_q^T Q_q q_{nom}
$

当前 $q_{nom} = 0$ 时，最后一项直接为 $0$。

$
f_{u,k} = B_{acc,k}^T Q_a D_{acc,k} + B_{\alpha,k}^T Q_\alpha D_{\alpha,k}
$

---

## 7. 有限时域时变 LQR 形式

总目标函数：

$
J = \sum_{k=0}^{N-1} \ell_k(x_k, u_k) + \ell_N(x_N)
$

动力学约束：

$
x_{k+1} = A x_k + B u_k
$

与无限时域稳态 LQR 不同，这里采用**有限时域、时变 LQR**，因为：

- 每个时刻的 `C_acc,k, B_acc,k, D_acc,k`
- `C_alpha,k, B_alpha,k, D_alpha,k`
- `G_g,k, d_g,k`

都来自当前工作点和未来 base 扰动预测，因此是随 `k` 变化的。

---

## 8. Riccati 反向递推

这一节的核心是回答两个问题：

- $V_k(x_k)$ 和 $\mathcal{Q}_k(x_k, u_k)$ 分别是什么
- 为什么对 $\mathcal{Q}_k$ 关于 $u_k$ 求极小，就能得到最优控制律

动态规划里，$V_k(x_k)$ 表示“从第 $k$ 步开始、当前状态为 $x_k$ 时，未来总成本的最小值”；$\mathcal{Q}_k(x_k, u_k)$ 表示“在第 $k$ 步先强行选一个控制 $u_k$，然后从第 $k+1$ 步开始都按最优策略走时，对应的总成本”。因此两者关系是：

$
V_k(x_k) = \min_{u_k} \mathcal{Q}_k(x_k, u_k)
$

### 8.1 终端条件与 Bellman 递推

定义终端值函数：

$
V_N(x_N) = x_N^T P_N x_N + 2 p_N^T x_N + c_N
$

其中：

$
P_N = Q_N, \qquad p_N = 0
$

假设第 $k+1$ 步的最优值函数已知，并保持二次型形式：

$
V_{k+1}(x) = x^T P_{k+1} x + 2 p_{k+1}^T x + c_{k+1}
$

则 Bellman 递推为：

$
\mathcal{Q}_k(x_k, u_k) = \ell_k(x_k, u_k) + V_{k+1}(A x_k + B u_k)
$

其中单步代价为：

$
\ell_k = x_k^T Q_{xx,k} x_k + 2 x_k^T Q_{xu,k} u_k + u_k^T Q_{uu,k} u_k + 2 f_{x,k}^T x_k + 2 f_{u,k}^T u_k + \text{const}
$

### 8.2 将 $V_{k+1}(A x_k + B u_k)$ 展开

把 $x_{k+1} = A x_k + B u_k$ 代入下一步值函数：

$
V_{k+1}(A x_k + B u_k) = (A x_k + B u_k)^T P_{k+1} (A x_k + B u_k) + 2 p_{k+1}^T (A x_k + B u_k) + c_{k+1}
$

展开后得到：

$
x_k^T A^T P_{k+1} A x_k + 2 x_k^T A^T P_{k+1} B u_k + u_k^T B^T P_{k+1} B u_k + 2 p_{k+1}^T A x_k + 2 p_{k+1}^T B u_k + c_{k+1}
$

与单步代价合并，可得 $\mathcal{Q}_k(x_k, u_k)$ 仍然是关于 $x_k, u_k$ 的二次型：

$
\mathcal{Q}_k = x_k^T F_k x_k + 2 x_k^T M_k u_k + u_k^T H_k u_k + 2 h_k^T x_k + 2 g_k^T u_k + \text{const}
$

其中定义：

$
F_k = Q_{xx,k} + A^T P_{k+1} A
$

$
M_k = Q_{xu,k} + A^T P_{k+1} B
$

$
H_k = Q_{uu,k} + B^T P_{k+1} B
$

$
h_k = f_{x,k} + A^T p_{k+1}
$

$
g_k = f_{u,k} + B^T p_{k+1}
$

### 8.3 为什么对 $u_k$ 求导就得到最优控制

对于固定的 $x_k$，上式中只有下面这些项与 $u_k$ 有关：

$
u_k^T H_k u_k + 2 x_k^T M_k u_k + 2 g_k^T u_k
$

若 $H_k$ 正定，则 $\mathcal{Q}_k$ 对 $u_k$ 是严格凸二次型，因此一阶条件就是全局最优条件：

$
\frac{\partial \mathcal{Q}_k}{\partial u_k} = 2 H_k u_k + 2 M_k^T x_k + 2 g_k = 0
$

解得最优控制：

$
u_k^\star = - H_k^{-1} M_k^T x_k - H_k^{-1} g_k
$

记：

$
K_k = H_k^{-1} M_k^T
$

$
k_k = H_k^{-1} g_k
$

则有：

$
u_k^\star = -K_k x_k - k_k
$

这里 $K_k$ 是状态反馈增益，$k_k$ 是由仿射项带来的前馈项。

### 8.4 把最优控制代回去，得到新的值函数

由于 $V_k(x_k) = \min_{u_k} \mathcal{Q}_k(x_k, u_k) = \mathcal{Q}_k(x_k, u_k^\star)$，把最优控制

$
u_k^\star = - H_k^{-1} M_k^T x_k - H_k^{-1} g_k
$

代回

$
\mathcal{Q}_k = x_k^T F_k x_k + 2 x_k^T M_k u_k + u_k^T H_k u_k + 2 h_k^T x_k + 2 g_k^T u_k + \text{const}
$

后，可直接按关于 $x_k$ 的二次项、一次项和常数项重新收集系数。代入并合并后有：

$
V_k(x_k) = x_k^T \bigl(F_k - M_k H_k^{-1} M_k^T\bigr) x_k + 2 \bigl(h_k - M_k H_k^{-1} g_k\bigr)^T x_k + c_k
$

另一方面，我们一开始假设值函数保持二次型形式：

$
V_k(x_k) = x_k^T P_k x_k + 2 p_k^T x_k + c_k
$

把这两个表达式逐项对比：

- $x_k^T(\cdot)x_k$ 前面的矩阵系数必须相等，因此得到 $P_k$
- $2(\cdot)^T x_k$ 前面的向量系数必须相等，因此得到 $p_k$

于是：

$
P_k = F_k - M_k H_k^{-1} M_k^T = Q_{xx,k} + A^T P_{k+1} A - M_k H_k^{-1} M_k^T
$

$
p_k = h_k - M_k H_k^{-1} g_k = f_{x,k} + A^T p_{k+1} - M_k H_k^{-1} g_k
$

这一步本质上就是“把最优控制代回 Bellman 递推后的 Q 函数，并与值函数的标准二次型模板逐项对号入座”。常数项 $c_k$ 对控制输出无影响，程序里可以不显式保存。

---

## 9. 在线控制输出

这一节回答的问题是：上面推导出的 `Q_xx,k`、`Q_xu,k`、`Q_uu,k`、`f_x,k`、`f_u,k`、`P_k`、`p_k`，在真实控制周期里到底按什么顺序计算，最后又怎样变成当前这一拍真正执行的控制量。

### 9.1 当前控制周期的输入

在当前时刻 `t`，LQR 控制器拿到三类输入：

1. 当前真实状态 `x_0 = [q^T, \dot q^T]^T`
2. 当前工作点附近的运动学线性化结果
3. 未来 `N` 步的 base 扰动预测

其中第 2 类和第 3 类共同决定了每个预测步 `k` 上的局部物理系数，也就是第 4 节明确定义的
`D_acc,k, C_acc,k, B_acc,k, D_alpha,k, C_alpha,k, B_alpha,k, d_g,k, G_g,k`

### 9.2 在线求解的逻辑顺序

在一个控制周期内，推荐严格按下面顺序执行：

1. 读取当前右臂状态，组装 `x_0`
2. 基于当前 `q, \dot q` 和未来 base 扰动预测，对 `k = 0, ..., N-1` 的每一步建立局部模型
3. 对每一步计算单步代价的二次项和一次项：`Q_xx,k, Q_xu,k, Q_uu,k, f_x,k, f_u,k`
4. 设置终端条件：`P_N = Q_N, p_N = 0`
5. 从 `k = N-1` 反向递推到 `0`，依次计算 `F_k, M_k, H_k, h_k, g_k`
6. 由 `H_k, M_k, g_k` 得到反馈增益 `K_k` 和前馈项 `k_k`
7. 再由 `K_k, k_k` 更新 `P_k, p_k`
8. 递推结束后，只取当前步控制律

$
u_0^\star = -K_0 x_0 - k_0
$

9. 将 `u_0^\star` 作为当前控制周期真正执行的最优关节加速度命令
10. 用逆动力学把它转换成前馈力矩，同时生成 PD 修正所需的 `q_ref, dq_ref`
11. 只执行这一拍；下一个控制周期重新感知、重新线性化、重新预测、重新递推

这就是 receding-horizon LQR 的核心思想：虽然为未来 `N` 步都算出了策略，但每次只执行第 `0` 步，然后立刻滚动到下一拍重新求解。

### 9.3 为什么只执行第一步

原因是未来的 base 扰动预测、当前线性化工作点以及真实机械臂状态都会不断变化。如果把整段 `u_0, u_1, ..., u_{N-1}` 一次性全部执行掉，那么后面时刻的模型就可能已经不再匹配真实系统。因此正确做法是：

- 当前拍只执行 `u_0^\star`
- 下一拍重新测量真实状态
- 重新生成新的预测和新的局部模型
- 再求新的 `u_0^\star`

这样做虽然每拍都要重复求解，但鲁棒性更好，也更符合当前项目“逐时刻重线性化、逐时刻更新扰动预测”的设定。

### 9.4 与程序实现一一对应的伪代码

下面给出建议的在线执行流程伪代码，变量名尽量与本文前面的公式保持一致：

```text
input: x0, disturbance_prediction[0:N-1], horizon N

for k in 0..N-1:
    choose operating point xbar_k = [qbar_k; dqbar_k]
    read predicted base terms aB_k, omegaB_k, alphaB_k, RWB_k
    compute pE(qbar_k), Jv(qbar_k), Jw(qbar_k)
    compute Jvdot(qbar_k, dqbar_k), Jwdot(qbar_k, dqbar_k), Jg_k
    construct D_acc,k, C_acc,k, B_acc,k from Section 4.1
    construct D_alpha,k, C_alpha,k, B_alpha,k from Section 4.2
    construct d_g,k, G_g,k from Section 4.3
    compute Q_xx,k, Q_xu,k, Q_uu,k, f_x,k, f_u,k

P_N = Q_N
p_N = 0

for k in N-1..0:
    F_k = Q_xx,k + A^T P_{k+1} A
    M_k = Q_xu,k + A^T P_{k+1} B
    H_k = Q_uu,k + B^T P_{k+1} B
    h_k = f_x,k + A^T p_{k+1}
    g_k = f_u,k + B^T p_{k+1}

    K_k = H_k^{-1} M_k^T
    k_k = H_k^{-1} g_k

    P_k = F_k - M_k H_k^{-1} M_k^T
    p_k = h_k - M_k H_k^{-1} g_k

u_star = -K_0 x0 - k_0
return u_star
```

### 9.5 输出如何接到当前项目

对当前项目而言，第 9.4 节返回的 `u_star` 就是右臂 `ddq_des`。控制层同时执行两条路径：

- 逆动力学路径：用当前整机 `q, dq` 和右臂 `ddq_des` 计算 `tau_ff`
- 反馈路径：对 `ddq_des` 积分一拍得到 `q_ref, dq_ref`，计算小幅 PD 修正 `tau_pd`
- 执行器命令：`tau = clip(tau_ff + tau_pd, tau_min, tau_max)`

这样 `tau_ff` 负责实现 LQR 要求的加速度并补偿重力、科氏力等动力学项，PD 只修正模型误差、离散延迟和扰动。

---

## 10. 与底层力矩接口的对应

LQR 输出为关节加速度：

$
u_0^\star = \ddot q^\star
$

理想的逆动力学前馈为：

$
\tau_{ff} = M(q)\ddot q^\star + h(q,\dot q)
$

当前 MuJoCo 实现构造完整的整机期望加速度向量，暂时令非右臂自由度的期望加速度为零：

```python
scratch.qpos[:] = data.qpos
scratch.qvel[:] = data.qvel
scratch.qacc[:] = 0.0
scratch.qacc[right_arm_qvel_indices] = ddq_des
mujoco.mj_inverse(model, scratch)
tau_ff = scratch.qfrc_inverse[right_arm_qvel_indices]
```

同时将 `ddq_des` 转换为反馈项所需的短时参考：

$
\dot q_{ref} = \dot q + \ddot q^\star \Delta t
$

$
q_{ref} = q + \dot q \Delta t + \frac{1}{2}\ddot q^\star \Delta t^2
$

最终命令为：

$
\tau = \tau_{ff} + K_p(q_{ref}-q) + K_d(\dot q_{ref}-\dot q)
$

`main_sim.py` 中 LQR 每 `0.006 s` 更新一次，`ddq_des/q_ref/dq_ref` 在更新间隔内保持；逆动力学使用最新整机状态，每个 `0.002 s` 仿真步重算一次。当前 XML 的右臂 motor 为 `gear=1` 的直接驱动，因此右臂 `qfrc_inverse` 可直接对应 `ctrl[18:23]`，最后按关节 `actuatorfrcrange` 限幅。

当前实现里，力矩上下限的来源也需要明确写死：

- 模型根来源：`resources/g1_description/g1_29dof_with_hand.xml` 中右臂 5 个被控 joint 的 `actuatorfrcrange`
- MuJoCo 读取结果：`sim_support.py` 在 `resolve_direct_drive_joint_group(...)` 中通过 `model.jnt_actfrcrange[joint_ids].copy()` 读取为 `torque_limits`
- 实际执行位置：`apply_computed_torque_control(...)` 中
  `tau_cmd = np.clip(tau_ff + tau_pd, torque_limits[:, 0], torque_limits[:, 1])`
- 当前右臂 5 个受控关节的上下限均为 `[-25, 25] N·m`：`right_shoulder_pitch_joint`、`right_shoulder_roll_joint`、`right_shoulder_yaw_joint`、`right_elbow_joint`、`right_wrist_roll_joint`
- 记录用途：这个限幅属于“底层执行器/模型层硬限幅”，用于保证最终写入 `d.ctrl` 的力矩不超过 XML 定义的可用力矩范围

---

## 11. 与 MPC 的关系和区别

LQR 与 `MPC_DESIGN.md` 的关系：

- 使用相同的状态定义 `x=[q; dq]`
- 使用相同的物理目标：末端线加速度、角加速度、重力方向误差、姿态正则、速度正则
- 使用相同的 base 扰动前馈建模方式
- 使用相同的输出接口：先求 `ddq`，再由逆动力学转为前馈力矩

LQR 相比 MPC 的简化点：

- 不显式处理硬约束
- 不需要 QP 求解器
- 每步只做一次 Riccati 递推
- 计算更轻，更适合作为 MPC 前的过渡版本

LQR 相比 MPC 的缺点：

- 无法严格保证关节位置、速度、加速度约束
- 需要依赖权重和输出裁剪来避免过激控制

---

## 12. 工程实现约定

初版实现建议：

- 控制维数：`n = 5`
- 控制量：`u = ddq`
- horizon：`N = 10 ~ 20`
- 每个控制周期重新线性化
- 对 $H_k$ 做正定性保护：

$
H_k \leftarrow \frac{1}{2}(H_k + H_k^T) + \lambda I
$

其中 $\lambda > 0$ 为很小正数，例如 `1e-6 ~ 1e-4`

对于有界执行器，通常需要在执行前处理输出约束：

$
u_0^\star \leftarrow \text{clip}(u_0^\star, u_{min}, u_{max})
$

当前实验暂时旁路 `u = ddq_des` 的后处理，以便直接验证 LQR 原始输出：

- 配置来源：`configs/g1.yaml` 中的 `lqr_max_ddq`、`lqr_max_dq`、`lqr_r_ddq`、`lqr_ddq_rate_limit`、`lqr_ddq_smoothing_alpha`
- 代码根来源：`arm_lqr.py` 的 `ArmLQRPolicy.__init__(...)`
- 当前执行路径：Riccati 得到的 `u_raw` 直接作为 `ddq_des`，不调用 `_post_process_ddq(...)`
- `max_ddq`、ddq 变化率限制、ddq 平滑和 ddq joint-limit guard 当前均不生效
- `max_dq = 1.0 rad/s` 仍用于一拍积分后的 `dq_ref` 限幅；`r_ddq = 0.25` 仍属于 LQR 代价函数并正常生效

此外，当前代码仍读取 MuJoCo 的右臂 `jnt_range` 并传给 `ArmLQRPolicy.set_joint_limits(...)`，用于裁剪一拍积分得到的 `q_ref`。由于 `_post_process_ddq(...)` 已旁路，基于 `ddq_des` 的轻量 joint-limit guard 当前不生效。

---

## 13. 程序接口建议

当前代码接口为：

```python
target_q, target_dq, ddq_des = arm_policy.compute_action(arm_obs, helpers)
```

其中：

- `target_q, target_dq`：用于小/中等增益 PD 反馈的短时参考
- `ddq_des`：送入逆动力学的右臂最优关节加速度

轨迹文件额外保存 `qacc`、`right_arm_ddq_des`、`right_arm_tau_ff` 和 `right_arm_tau_pd`，用于检查实际加速度跟踪和两部分力矩占比。

---

## 14. 最终一句话

本项目中的 LQR 本质上是：

**基于当前工作点局部线性化、带扰动前馈仿射项、有限时域时变 Riccati 递推的单臂稳定控制器。**

它与 MPC 使用同一套物理目标和建模方式，只是把“带约束 QP 优化”替换成了“无约束二次型动态规划求解”。
