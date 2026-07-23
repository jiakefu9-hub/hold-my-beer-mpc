# 单臂 LQR 稳定控制系统设计文档

## 1. 目标

LQR 作为当前项目中 MPC 的简化版，用于在**不显式处理硬约束**的前提下，完成右臂持杯稳定控制。其建模、符号和物理目标尽量与 `MPC_DESIGN.md` 保持一致，便于后续从 LQR 平滑切换到 MPC。

系统分工保持不变：

- 下肢：沿用现有 RL locomotion
- 上肢：右臂用有限时域、时变 LQR 控制
- 扰动前馈：使用预测得到的 base 扰动进入每个时刻的局部模型
- 位置目标：抓持点跟踪相对于 torso/IMU 的名义位置，避免机器人行走时把手固定在世界中的某一点
- 输出：LQR 优化关节加速度 `u_k = ddq_k`，由 MuJoCo contact-aware 逆动力学转换为前馈力矩，并叠加 `q_ref, dq_ref` 的 PD 修正

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

### 4.3 Torso-relative 末端位置误差

末端位置不固定在世界坐标系中，而定义在 torso/IMU 坐标系 `{B}` 中：

$
{}^B p_E(q) = ({}^W R_B)^T\left({}^W p_E(q)-{}^W p_B\right)
$

参考位置只由名义右臂姿态 $q_{nom}$ 决定，并在 `KinematicsHelper` 初始化时计算一次、随后缓存：

$
{}^B p_{E,ref} = {}^B p_E(q_{nom})
$

因此位置误差为：

$
r_p(q) = {}^B p_E(q)-{}^B p_{E,ref} \in \mathbb{R}^3
$

因为该参考量表达在 torso/IMU 局部坐标系中，torso 在世界系中的平移和旋转都会被相对变换消去，所以无需每个控制周期重算。机器人向前行走时，位置参考会随 torso 一起移动，不会要求手臂反向伸展以维持一个固定的世界坐标点。当前实现使用完整 IMU 坐标系；如果以后发现 torso roll/pitch 使位置目标摆动过强，可以再单独评估“torso 原点 + yaw-only 朝向”的参考系，但本版不额外引入该处理。

在工作点 $\bar q_k$ 附近线性化：

$
r_p(q) \approx r_p(\bar q_k)+J_{p,k}(q-\bar q_k)=d_{p,k}+G_{p,k}x_k
$

其中：

$
J_{p,k}=({}^W R_B)^T{}^W J_v(\bar q_k),\qquad
G_{p,k}=J_{p,k}S_q,\qquad
d_{p,k}=r_p(\bar q_k)-J_{p,k}\bar q_k
$

右臂关节不会改变上游 IMU site，因此上式中 torso 位姿对右臂 $q$ 的导数为零。torso-relative 末端位置只由右臂关节角决定，其速度可写为：

$
v_{p,k}=J_{p,k}\dot q_k=G_{pv,k}x_k,
\qquad G_{pv,k}=J_{p,k}S_v
$

加速度代价不能惩罚近似恒定的漂移速度，因此当前版本单独加入 $v_p$ 代价，抑制末端持续向后或横向漂移。

### 4.4 有方向的三维重力误差

定义末端防洒水误差：

$
r_g(q) = {}^E R_W(q)g^W-g^E_{ref} \in \mathbb{R}^3,
\qquad g^E_{ref}=\begin{bmatrix}0&0&-\|g^W\|\end{bmatrix}^T
$

旧版只取 $g_E$ 的前两个分量，正立和精确倒立时都可能得到零误差。新的三维差值保留方向：正立时 $r_g=0$，倒立时第三维约为 $2\|g\|$，因此评估指标能够明确区分正立和倒立。

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

### 4.5 程序实现对应关系

后续程序里，每个预测步 `k` 的这些量应按下面方式构造：

- base 预测器提供 `{}^W a_{B,k}, {}^W \omega_{B,k}, {}^W \alpha_{B,k}, {}^W R_{B,k}`
- 运动学 helper 提供 `{}^B p_E(\bar q_k), {}^B J_v(\bar q_k), {}^B J_\omega(\bar q_k)`
- `{}^B \dot J_v, {}^B \dot J_\omega, J_{g,k}` 可由解析法或有限差分法在工作点处计算
- 然后严格按本节公式生成 `D_acc,k, C_acc,k, B_acc,k, D_alpha,k, C_alpha,k, B_alpha,k, d_p,k, G_p,k, G_pv,k, d_g,k, G_g,k`

后文第 5 节到第 9 节中的所有 `Q_xx,k, Q_xu,k, Q_uu,k, f_x,k, f_u,k, P_k, p_k` 都建立在这里这些系数之上。因此本节就是后续程序实现时最直接的“物理建模说明书”。

---

## 5. 单步代价函数

每一步代价定义为：

$
\ell_k(x_k, u_k) = \|{}^W a_{E,k}\|_{Q_a}^2 + \|{}^W \alpha_{E,k}\|_{Q_\alpha}^2 + \|r_{p,k}\|_{Q_p}^2 + \|v_{p,k}\|_{Q_{pv}}^2 + \|r_{g,k}\|_{Q_g}^2 + \|q_k - q_{nom}\|_{Q_q}^2 + \|\dot q_k\|_{Q_v}^2 + \|u_k\|_{R}^2
$

其中 $Q_p\succeq0$ 是 torso-relative 末端位置权重，$Q_{pv}\succeq0$ 是同一坐标系下的末端速度权重。当前诊断文件中的 `velocity` 代价等于关节速度代价与末端位置速度代价之和。

`20260721_224416` 曾将三维重力方向权重从 $Q_g=30I_3$ 提高为 $Q_g=45I_3$，但右手重力误差、线加速度和角加速度均变差，并出现 elbow 下限冲击。单变量对照 `20260722_204009` 已恢复 $Q_g=30I_3$，其余执行层保持不变；当前代码继续使用 $Q_g=30I_3$。

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
Q_{xx,k} = S_v^T C_{acc,k}^T Q_a C_{acc,k} S_v + S_v^T C_{\alpha,k}^T Q_\alpha C_{\alpha,k} S_v + G_{p,k}^T Q_p G_{p,k} + G_{pv,k}^T Q_{pv} G_{pv,k} + G_{g,k}^T Q_g G_{g,k} + S_q^T Q_q S_q + S_v^T Q_v S_v
$

$
Q_{xu,k} = S_v^T C_{acc,k}^T Q_a B_{acc,k} + S_v^T C_{\alpha,k}^T Q_\alpha B_{\alpha,k}
$

$
Q_{uu,k} = B_{acc,k}^T Q_a B_{acc,k} + B_{\alpha,k}^T Q_\alpha B_{\alpha,k} + R
$

### 6.2 一次项向量

$
f_{x,k} = S_v^T C_{acc,k}^T Q_a D_{acc,k} + S_v^T C_{\alpha,k}^T Q_\alpha D_{\alpha,k} + G_{p,k}^T Q_p d_{p,k} + G_{g,k}^T Q_g d_{g,k} - S_q^T Q_q q_{nom}
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
    compute Jvdot(qbar_k, dqbar_k), Jwdot(qbar_k, dqbar_k), Jp_k, Jg_k
    construct D_acc,k, C_acc,k, B_acc,k from Section 4.1
    construct D_alpha,k, C_alpha,k, B_alpha,k from Section 4.2
    construct d_p,k, G_p,k from Section 4.3
    construct d_g,k, G_g,k from Section 4.4
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

- 名义力矩路径：用当前整机 `q, dq` 和右臂 `ddq_des` 计算 `tau_ff`，同时对 `ddq_des` 积分一拍得到 `q_ref, dq_ref` 并计算 `tau_pd`，两者组成 `tau_nom`
- 前向动力学修正路径：固定其他执行器当前力矩，数值计算 `G_tau = d(ddq_right)/d(tau_right)`，求出右臂力矩修正 `delta_tau`
- 完整验收路径：用完整 `mj_forward` 检查候选力矩是否真的让右臂加速度更接近 `ddq_des`，必要时缩小修正或退回 `tau_nom`
- 执行器命令：只有通过验收的 `tau_cmd` 才写入 `ctrl[18:23]`

因此当前的 `tau_ff + tau_pd` 只负责提供一个合理的起点，不再被当作最终力矩直接执行。位置项采用 torso-relative 误差，因此它约束手相对躯干的工作位置，不会阻止整机向前行走。

---

## 10. 与底层力矩接口的对应

### 10.1 当前代码从观测到真实执行的完整流程

下面描述的是 `main_sim.py`、`arm_lqr.py` 和 `sim_support.py` 当前实际运行的版本，不是早期的一拍积分 PD 版本。

#### 第 1 步：读取当前状态和 torso 扰动

每个 MuJoCo 仿真步长为 `0.002 s`。主程序读取：

- 右臂五个关节的当前角度 `q` 和速度 `dq`
- torso IMU 姿态和角速度
- MuJoCo accelerometer 给出的局部比力，转换为世界系线加速度
- 世界系角速度有限差分得到的 torso 角加速度

torso 线加速度和角加速度经过当前配置的限幅及低通滤波，再交给运动学线性化。这里的 torso 信号用于预测躯干运动对手部线加速度和角加速度的影响。

#### 第 2 步：LQR 每 0.006 s 计算一次 `ddq_des`

右臂 LQR 控制周期为：

$
\Delta t_{arm}=3\times0.002=0.006\ \mathrm{s}
$

LQR 使用状态 $x=[q;\dot q]$，根据末端线加速度、角加速度、torso-relative 位置、三维重力方向、关节姿态、速度和控制量代价，求有限时域第一拍控制：

$
u_0^\star=\ddot q_{des}
$

当前 DDQ 后处理完全旁路，因此 `ddq_raw` 直接成为 `ddq_des`，不经过 DDQ 硬限幅、rate limit、smoothing 或 joint-limit guard。`ddq_des` 在接下来的三个 `0.002 s` 仿真步内保持不变。

#### 第 3 步：一拍积分只为 PD 生成短时参考

当前状态积分一拍：

$
\dot q_{ref}=\dot q+\ddot q_{des}\Delta t_{arm}
$

$
q_{ref}=q+\dot q\Delta t_{arm}+\frac{1}{2}\ddot q_{des}\Delta t_{arm}^2
$

再计算：

$
\tau_{pd}=K_p(q_{ref}-q)+K_d(\dot q_{ref}-\dot q)
$

这条一拍积分路径只提供反馈修正，不再单独承担实现 `ddq_des` 的任务。

#### 第 4 步：逆动力学生成名义前馈力矩

在逆动力学临时数据 `inverse_dynamics_data` 中复制当前整机 `q, dq`，暂时设置：

```text
非右臂自由度期望加速度 = 0
右臂五关节期望加速度 = ddq_des
```

调用 `mj_inverse` 得到右臂 `qfrc_inverse`。随后加回 contact、joint/tendon limit 和 equality 等非摩擦约束，只让前馈补偿 `FRICTION_DOF/FRICTION_TENDON` 摩擦，得到 `tau_ff`。

最后形成并按执行器力矩范围裁剪：

$
\tau_{nom}=\operatorname{clip}(\tau_{ff}+\tau_{pd})
$

这里“其他自由度加速度为零”的近似只用于生成初始 `tau_nom`。最终加速度并不由这次逆动力学决定，后面的前向动力学映射允许浮动基和腿部根据当前力矩与接触自然运动。

#### 第 5 步：固定其他执行器力矩，计算名义加速度

构造完整控制向量：

- 腿部保持当前 RL 控制力矩
- 腰和左臂使用当前步 PD 力矩
- 右臂使用 `tau_nom`

将当前 `q, dq`、外力、接触状态和完整控制向量复制到另一个临时数据 `forward_dynamics_data`，调用完整 `mj_forward`：

```text
输入：当前整机 q、dq、全部执行器力矩、外力和当前约束
输出：在这一状态和力矩下，MuJoCo 求得的整机 qacc
```

从整机 `qacc` 中取出右臂五维，记为 `ddq_baseline`。这一步是前向动力学，即“给定状态和力矩求加速度”，不是前向运动学。

#### 第 6 步：数值计算力矩到右臂加速度的局部映射

分别只给五个右臂执行器增加 `0.1 Nm`，每次重新计算前向动力学：

$
G_\tau[:,j]\approx
\frac{\ddot q_{right}(\tau_{nom}+0.1e_j)-\ddot q_{baseline}}{0.1}
$

因此 $G_\tau$ 是 `5 x 5` 矩阵：第 `j` 列表示第 `j` 个右臂力矩变化会怎样同时影响五个右臂关节加速度。每次扰动前都会恢复相同的 `qacc_warmstart`，避免前一次 MuJoCo 约束求解污染下一列。

#### 第 7 步：阻尼最小二乘求力矩修正

在当前工作点附近使用局部模型：

$
\ddot q_{right}\approx\ddot q_{baseline}+G_\tau\Delta\tau
$

通过 SVD 求解：

$
\Delta\tau=\arg\min_{\Delta\tau}
\|G_\tau\Delta\tau-(\ddot q_{des}-\ddot q_{baseline})\|_2^2
+\lambda\|\Delta\tau\|_2^2
$

当前 `lambda=5.0`。此时得到的是局部线性模型认为合适的 `delta_tau_raw`，还不能直接认为它在完整 MuJoCo 约束动力学中一定有效。

#### 第 8 步：完整前向动力学验收与回退

依次构造候选力矩：

```text
tau_candidate = clip(tau_nom + scale * delta_tau_raw)
scale = 1.0, 0.5, 0.25, 0.125
```

对每个候选都调用一次完整 `mj_forward`，直接得到该候选在当前状态下的 `ddq_candidate`，并比较：

$
e_{candidate}=\|\ddot q_{candidate}-\ddot q_{des}\|_2
$

$
e_{baseline}=\|\ddot q_{baseline}-\ddot q_{des}\|_2
$

程序会完整计算四个候选，不提前停止。每个候选必须同时满足：

1. 总误差下降：$e_{candidate}<e_{baseline}$；
2. 最大单关节误差：$\max_j|\ddot q_{candidate,j}-\ddot q_{des,j}|\leq4\ \mathrm{rad/s^2}$；
3. 瞬时加速度安全上限：$\max_j|\ddot q_{candidate,j}|\leq8\ \mathrm{rad/s^2}$。

在全部严格合格候选中选择 $e_{candidate}$ 最小者，而不是选择第一个改善者。如果没有严格合格项，但存在总误差下降且满足加速度上限的候选，则优先选择最大单关节误差最小者作为临时工作点；如果连这样的候选也没有，则使用总误差下降最多的候选作为临时工作点。临时工作点不会因为未达到单关节阈值而阻止第二轮重线性化。`4` 和 `8` 是本轮仿真实验阈值，不代表未经验证即可用于真机的硬件安全规格。

这里的“完整验收”是完整求解当前 MuJoCo 接触、摩擦、限位等约束，而不是继续使用 $G_\tau$ 的线性预测。所有候选都只在临时 `MjData` 中计算，不改变真实 `data.qpos/qvel`，不推进仿真时间，也不会让机器人依次执行四个力矩。

#### 第 9 步：高残差时在已接受力矩处重线性化一次

设第一轮通过验收的力矩和完整前向动力学加速度为 $\tau_1,\ddot q_1$，第一轮残差为：

$
r_1=\ddot q_{des}-\ddot q_1
$

当第一轮取得总误差下降，并且残差超过阈值或尚未满足单关节误差条件时，触发第二轮。残差阈值为：

$
\|r_1\|_2>\rho,qquad \rho=5\ \mathrm{rad/s^2}
$

程序才触发第二轮。第二轮不是重复使用原来的 $G_\tau$，而是在 $\tau_1$ 处重新对五个右臂力矩做 `0.1 Nm` 扰动：

$
G_{\tau,1}[:,j]\approx
\frac{\ddot q_{right}(\tau_1+0.1e_j)-\ddot q_1}{0.1}
$

然后求剩余残差对应的第二次修正：

$
\Delta\tau_1=
\arg\min_{\Delta\tau}
\|G_{\tau,1}\Delta\tau-r_1\|_2^2
+\lambda\|\Delta\tau\|_2^2
$

第二轮候选为 $\operatorname{clip}(\tau_1+s\Delta\tau_1)$，仍对 `1/0.5/0.25/0.125` 四项全部执行完整 `mj_forward`，应用相同的总误差、单关节误差和瞬时加速度判据，再选择严格合格项中总误差最小者。

如果两轮结束后的最终瞬时加速度仍超过 $8\ \mathrm{rad/s^2}$，程序不会切换到未经验证的零力矩或旧力矩，而是在当前最佳力矩处最多再重线性化两次。每次安全救援仍要求总误差继续下降，并对四个候选执行完整验收。若接触冲击使当前右臂力矩空间内不存在满足上限的候选，程序保留误差最小的结果并将 `safety_fallback_satisfied=false` 写入记录；因此该上限是候选筛选与有限次安全救援规则，不应误称为任何接触状态下都能保证的真机硬约束。

这样做的原因是：第一轮候选可能已经让系统进入新的 friction/contact/limit 约束模式，第一轮在 $\tau_{nom}$ 处计算的 $G_{\tau,0}$ 已经过时；在 $\tau_1$ 处重新计算 $G_{\tau,1}$ 才能描述新工作点附近的力矩到加速度关系。

#### 第 10 步：只执行最终通过验收的力矩

最终选择的右臂力矩写入：

```python
d.ctrl[18:23] = tau_cmd
```

然后整个程序只调用一次真实的 `mj_step`，让机器人向前推进 `0.002 s`。因此真实机器人在这一拍只看到一个最终力矩，不会看到验收过程中被拒绝的候选。

#### 第 11 步：记录加速度、验收分布并评估执行效果

当前轨迹同时记录：

- `right_arm_ddq_des`：LQR 期望加速度
- `right_arm_qacc_mapping_predicted`：局部线性模型 $\ddot q_b+G_\tau\Delta\tau$ 的预测
- `right_arm_qacc_mapping_validated`：完整 `mj_forward` 验收得到的瞬时加速度
- `right_arm_qacc`：真实 `mj_step` 对应的 MuJoCo 瞬时加速度
- `right_arm_ddq_real`：相邻两个 6 ms 手臂更新时刻之间，由速度差分得到的实际平均加速度
- `right_arm_forward_dynamics_validation_scale`：第一轮采用的 `1/0.5/0.25/0.125/0` 比例
- `right_arm_forward_dynamics_second_pass_triggered/accepted`：第二轮是否触发、是否找到更优候选
- `right_arm_second_pass_validation_scale`：第二轮最终采用的候选比例
- `right_arm_forward_dynamics_*_rejections`：因总误差、单关节误差和加速度上限被拒绝的候选数量
- `right_arm_forward_dynamics_safety_fallback_used/satisfied/attempts`：是否进入额外安全救援、最终是否满足加速度上限及救援轮数

最终打印的 DDQ `correlation/gain/RMSE` 比较的是同一 6 ms 区间内的 `ddq_des` 与 `right_arm_ddq_real`。每次实验还生成 `ddq_tracking.png`，在五个关节的时序图中叠加 `ddq_des/ddq_real` 并标注 correlation、gain 和 RMSE。`lqr_tracking_diagnostics.json` 记录两轮候选比例、触发率、接受率、安全候选数，以及因总误差、单关节误差和加速度上限被拒绝的次数。

完整验收主要负责避免局部线性映射在 friction/contact/limit 切换处生成错误力矩。它会选出当前四个比例中误差最小的安全候选，但仍不能保证 6 ms 区间平均加速度没有接触瞬态，也不能在没有合格候选时凭空构造可实现的安全力矩。

当前完整数据流可以概括为：

```text
q, dq, torso motion
        -> finite-horizon LQR (每 6 ms)
        -> ddq_des
        -> 一拍积分 -> q_ref, dq_ref -> tau_pd
        -> inverse dynamics -> tau_ff
        -> tau_nom = clip(tau_ff + tau_pd)
        -> baseline forward dynamics
        -> finite-difference G_tau
        -> damped least-squares delta_tau
        -> full forward-dynamics validation/backtracking
        -> if residual > 5 rad/s^2: re-linearize and correct once more
        -> tau_cmd
        -> d.ctrl[18:23]
        -> one real mj_step (每 2 ms)
```

### 10.2 逆动力学、约束力与局部映射的数学细节

LQR 输出为关节加速度：

$
u_0^\star = \ddot q^\star
$

无接触时，理想的逆动力学前馈为：

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
tau_inverse = scratch.qfrc_inverse[right_arm_qvel_indices]
efc_J = np.asarray(scratch.efc_J).reshape(-1, model.nv)[:scratch.nefc]
efc_type = scratch.efc_type[:scratch.nefc]
friction_rows = np.isin(efc_type, FRICTION_CONSTRAINT_TYPES)
qfrc_friction = efc_J[friction_rows].T @ scratch.efc_force[:scratch.nefc][friction_rows]
qfrc_nonfriction = scratch.qfrc_constraint - qfrc_friction
tau_nonfriction = qfrc_nonfriction[right_arm_qvel_indices]
tau_ff = tau_inverse + tau_nonfriction
```

MuJoCo 在当前符号约定下满足：

$
qfrc_{inverse}=M(q)\ddot q+h(q,\dot q)-qfrc_{passive}-qfrc_{constraint}
$

`qfrc_constraint` 同时包含 contact、`FRICTION_DOF`、joint/tendon limit 和 equality 等约束。当前实现只从加回量中排除 `FRICTION_DOF` 和 `FRICTION_TENDON`：

$
\tau_{ff}=qfrc_{inverse}+qfrc_{constraint}-qfrc_{friction}
$

这样 contact、joint/tendon limit 和 equality 等约束不会被执行器主动抵消，而 `frictionloss` 仍保留在 `qfrc_inverse` 中，由执行器正常补偿。这称为本项目中的 **non-friction-constraint-aware inverse dynamics**。当瓶子撞到 torso 或关节进入限位时，控制器不再为强行实现不可达的 `ddq_des` 而对抗这些约束力。该处理不等于完整的接触优化；若以后需要主动推压环境，仍需显式建模接触力与接触任务。

但上述逆动力学仍把非右臂加速度假设为零，与浮动基行走不一致。因此当前实现只把它作为名义力矩：

$
\tau_{nom}=\tau_{ff}+K_p(q_{ref}-q)+K_d(\dot q_{ref}-\dot q)
$

保持腿、腰和左臂当前力矩不变，先用 MuJoCo 前向动力学计算 $\tau_{nom}$ 对应的右臂基准加速度 $\ddot q_b$。然后对右臂五个力矩分别增加小扰动 $\epsilon$，用有限差分构建：

$
G_{\tau}[:,j]\approx
\frac{\ddot q_{right}(\tau_{nom}+\epsilon e_j)-\ddot q_b}{\epsilon}
$

局部映射为：

$
\ddot q_{right}\approx\ddot q_b+G_{\tau}\Delta\tau
$

一次阻尼最小二乘求解：

$
\Delta\tau=\arg\min_{\Delta\tau}
\|G_{\tau}\Delta\tau-(\ddot q_{des}-\ddot q_b)\|^2
+\lambda\|\Delta\tau\|^2
$

局部线性模型不能保证跨越 friction/contact/limit 约束模式切换后仍然成立，因此最终力矩不再直接采用完整修正。对

$
\tau(s)=\operatorname{clip}(\tau_{nom}+s\Delta\tau),
\qquad s\in\{1,0.5,0.25,0.125\}
$

对四个候选全部执行完整 `mj_forward`，计算真实的瞬时右臂加速度。候选必须满足

$
\|\ddot q_{right}(\tau(s))-\ddot q_{des}\|_2
<
\|\ddot q_b-\ddot q_{des}\|_2
$

并同时满足最大单关节误差不超过 $4\ \mathrm{rad/s^2}$、加速度绝对值不超过 $8\ \mathrm{rad/s^2}$。程序从严格合格候选中选择总误差最小者；严格候选不可行时允许误差下降的候选作为下一轮临时工作点。这里的“验收”只在临时 `MjData` 中求值，不推进真实仿真时间，也不修改 LQR 的 `ddq_des`。

若第一轮残差范数超过 $5\ \mathrm{rad/s^2}$ 或尚未满足单关节误差条件，则在该候选力矩处重新构建 $G_{\tau,1}$，对剩余残差再求一次阻尼最小二乘，并执行同样的完整候选验收。因为四个候选现在全部求值，第一轮固定需要1次基准、5次扰动和4次验收，即10次前向动力学；第二轮触发时额外需要5次扰动和4次验收，即9次。两轮后仍超过加速度上限时，最多增加两轮同规模安全救援。浮动基、腿部和接触在每次 `mj_forward` 中自然响应，因此不需要人为指定它们的加速度。

同时将 `ddq_des` 转换为反馈项所需的短时参考：

$
\dot q_{ref} = \dot q + \ddot q^\star \Delta t
$

$
q_{ref} = q + \dot q \Delta t + \frac{1}{2}\ddot q^\star \Delta t^2
$

逆动力学和 PD 生成局部前向动力学映射的初始名义命令：

$
\tau_{nom} = \tau_{ff} + K_p(q_{ref}-q) + K_d(\dot q_{ref}-\dot q)
$

`main_sim.py` 中 LQR 每 `0.006 s` 更新一次，`ddq_des/q_ref/dq_ref` 在更新间隔内保持；逆动力学使用最新整机状态，每个 `0.002 s` 仿真步重算一次。当前 XML 的右臂 motor 为 `gear=1` 的直接驱动，因此右臂广义力可直接对应 `ctrl[18:23]`，最后按关节 `actuatorfrcrange` 限幅。

当前实现里，力矩上下限的来源也需要明确写死：

- 模型根来源：`resources/g1_description/g1_29dof_with_hand.xml` 中右臂 5 个被控 joint 的 `actuatorfrcrange`
- MuJoCo 读取结果：`sim_support.py` 在 `resolve_direct_drive_joint_group(...)` 中通过 `model.jnt_actfrcrange[joint_ids].copy()` 读取为 `torque_limits`
- 实际执行位置：`apply_computed_torque_control(...)` 先计算 `tau_nominal=tau_ff+tau_pd`；`local_forward_dynamics_torque_mapping(...)` 再计算 `delta_tau`，并通过完整前向动力学验收 `1/0.5/0.25/0.125` 倍候选，最后只执行通过验收的 `tau_cmd`
- 当前右臂 5 个受控关节的上下限均为 `[-25, 25] N·m`：`right_shoulder_pitch_joint`、`right_shoulder_roll_joint`、`right_shoulder_yaw_joint`、`right_elbow_joint`、`right_wrist_roll_joint`
- 记录用途：这个限幅属于“底层执行器/模型层硬限幅”，用于保证最终写入 `d.ctrl` 的力矩不超过 XML 定义的可用力矩范围

### 10.3 DDQ 跟踪何时可以进入下一阶段

correlation 和 gain 没有适用于所有机器人的唯一标准，还必须同时检查 RMSE、P95/最大误差、力矩饱和与时序图。当前项目采用两级标准：

- 工程放行门槛：每个关节 `correlation >= 0.8`，`gain` 位于 `[0.8, 1.2]`，同时不存在少量巨大尖峰或持续力矩饱和
- 更严格目标：每个关节 `correlation >= 0.9`，`gain` 位于 `[0.9, 1.1]`，并继续降低 normalized RMSE

达到工程门槛后，加速度到力矩映射不再是首要故障，应冻结该层并进入末端姿态、位置、碰撞和实时性等控制目标调整；严格目标可以在后续独立优化中继续追求。

实验 `20260721_204908` 的五关节 correlation 为 `[0.838, 0.897, 0.881, 0.923, 0.878]`，gain 为 `[1.005, 1.011, 0.995, 1.014, 0.873]`，误差范数 RMS/P95/最大值为 `1.191/2.517/8.652 rad/s^2`。该结果已通过工程放行门槛，但 shoulder pitch/yaw 和 wrist roll 尚未全部达到严格 correlation/gain 目标。

---

## 11. 与 MPC 的关系和区别

LQR 与 `MPC_DESIGN.md` 的关系：

- 使用相同的状态定义 `x=[q; dq]`
- 使用相同的物理目标：末端线加速度、角加速度、torso-relative 末端位置、有方向三维重力误差、姿态正则、速度正则
- 使用相同的 base 扰动前馈建模方式
- 使用相同的输出接口：先求 `ddq`，再由逆动力学转为前馈力矩

LQR 相比 MPC 的简化点：

- 不显式处理硬约束
- 不需要 QP 求解器
- 每步只做一次 Riccati 递推
- 计算更轻，更适合作为 MPC 前的过渡版本

LQR 相比 MPC 的缺点：

- 无法严格保证关节位置、速度、加速度约束
- 当前无 ddq 约束，需要依赖权重与最终执行器力矩上限避免过激控制；这也是后续切换到带硬约束 MPC 的主要原因

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

当前实验完全旁路 `u = ddq_des` 后处理：

- torso-relative 位置权重由 `configs/g1.yaml` 的 `lqr_q_position` 配置，当前采用 $Q_p=60I_3$
- torso-relative 末端速度权重由 `lqr_q_ee_velocity` 配置，当前采用 $Q_{pv}=10I_3$；它用于抑制近似恒速的位置漂移，不修改测量值
- 当前关节姿态权重采用 $Q_q=\operatorname{diag}(40,50,30,10,1)$。shoulder pitch/roll 保持上臂姿态，shoulder yaw 防止向 torso 中线内旋，elbow 防止过度屈肘把手收回胸前，wrist roll 仍保留较大调姿自由度
- Riccati 得到的 `u_raw` 直接作为 `ddq_des`
- 不启用 `max_ddq` 硬限幅、joint-limit guard、ddq 变化率限制或 ddq 平滑
- `_apply_ddq_safety(...)` 和 `_post_process_ddq(...)` 暂时保留，只用于以后受控对比实验
- `max_dq = 1.0 rad/s` 仍用于一拍积分后的 `dq_ref` 限幅；物理 `q_ref` 仍裁剪到 MuJoCo 关节范围；它们不修改送入逆动力学的 `ddq_des`
- 最终力矩仍受 XML `actuatorfrcrange` 硬限幅，这是执行器层保护，不属于 ddq 后处理

---

## 13. 程序接口建议

当前代码接口为：

```python
target_q, target_dq, ddq_des = arm_policy.compute_action(arm_obs, helpers)
```

其中：

- `target_q, target_dq`：用于小/中等增益 PD 反馈的短时参考
- `ddq_des`：送入逆动力学的右臂最优关节加速度

完整高频数据保存在 `trajectory.npz`，至少包括：

- LQR 原始/最终 `ddq`、一拍 `q_ref/dq_ref`、实际 `q/dq/qacc`
- `qfrc_inverse`、contact/total/non-contact 约束分量、contact-aware `tau_ff`、`tau_pd`、限幅前后总力矩
- torso 线加速度与角加速度的 raw/filtered 值、torso 角速度
- torso-relative 末端位置/参考/误差、三维有方向重力误差、upright alignment 和接触数量
- 以相邻两次 `0.006 s` 手臂更新严格对齐的 `ddq_des/ddq_real/error`，以及一步模型/真实七项代价/error
- 第一轮 `1/0.5/0.25/0.125/0` 验收比例、第一轮残差、第二轮触发/接受状态、第二轮修正与最终残差

同时生成 `control_preview.csv` 供直接打开查看，并在 `right_arm_diagnostics.json` 中保存各信号的 RMS、最大值、饱和率和倒立比例等汇总。专用的 `lqr_tracking_preview.csv` 保存逐控制区间跟踪值，`lqr_tracking_diagnostics.json` 保存每个关节和每个代价项的 RMSE、MAE、最大误差、归一化 RMSE、相关系数、增益以及两轮候选分布。`ddq_tracking.png` 用五个时序子图叠加 `ddq_des/ddq_real`，并直接标注各关节 correlation、gain 和 RMSE。NPZ 是无损分析源，CSV/JSON/PNG 用于快速人工检查。

---

## 14. 最终一句话

本项目中的 LQR 本质上是：

**基于当前工作点局部线性化、带扰动前馈仿射项、有限时域时变 Riccati 递推的单臂稳定控制器。**

它与 MPC 使用同一套物理目标和建模方式，只是把“带约束 QP 优化”替换成了“无约束二次型动态规划求解”。
