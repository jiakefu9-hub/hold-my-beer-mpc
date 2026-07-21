# 单臂 MPC 稳定控制系统设计文档

## 🎯 项目目标 (Project Objective)

本项目旨在实现人形机器人在移动过程中的单臂末端稳定控制。具体设定如下：
- **下肢控制**：采用现有的强化学习 (RL) 策略进行运动控制。
- **上肢控制**：采用模型预测控制 (MPC) 算法控制单侧手臂，确保末端执行器 (End-Effector, EE) 保持稳定。
- **核心创新点（前馈补偿）**：基于死锁状态下采集的躯干震动数据，拟合/训练出一个机座（Base）的周期性扰动模型。将该扰动模型作为前馈补偿项（Feedforward Compensation）引入到 MPC 的预测时域中，从而在震动传导至末端前提前主动抵消，实现极致的稳定性。
- **验证与评估**：在末端绑带装有水的瓶子作为直观的稳定性参考。然而，实际算法表现和性能评估将严格依赖末端的真实数据（线加速度、角加速度和基于重力方向的倾斜误差）。
- **位置参考的工程定义**：末端抓持点跟踪 torso/IMU 坐标系中的名义位置，而不是固定的世界坐标点，使手臂能随行走中的躯干一起平移。
- **姿态参考的工程定义**：采用有方向的三维重力误差作为“防洒水”指标，使正立与倒立不再得到相同误差；同时保留关节名义姿态正则项，避免手臂漂移到不自然构型。

---

## 🌍 物理建模与状态空间构建 (Physics & State-Space Modeling)

### 零、 坐标系与符号定义（严格基准）

我们定义以下三个核心坐标系：
- $\{W\}$ (World)：世界惯性坐标系（静止参考系）。
- $\{B\}$ (Base)：基座坐标系（绑定在躯干 IMU 上）。
- $\{E\}$ (End-Effector)：末端执行器坐标系（水杯）。

**符号约定：**
- ${}^W R_B \in SO(3)$：表示从 $\{B\}$ 到 $\{W\}$ 的旋转矩阵。
- ${}^B p_E(q) \in \mathbb{R}^3$：通过正向运动学（FK）求得的，末端在 $\{B\}$ 系下的位置。
- ${}^B J(q) = \begin{bmatrix} {}^B J_v \\ {}^B J_\omega \end{bmatrix} \in \mathbb{R}^{6 \times n}$：在 $\{B\}$ 系下表达的几何雅可比矩阵。
- $q, \dot{q}, \ddot{q} \in \mathbb{R}^n$：手臂的关节位置、速度、加速度。

**任务末端（抓持点）工程定义：**
- 当前项目中，真正用于稳定控制与评估的末端点不是 `left/right_wrist_roll_rubber_hand` 的 body 原点，而是定义在瓶身大圆柱中心的 `left/right_grasp_site`。
- 这两个 `grasp site` 与 `wrist_roll_rubber_hand` 坐标系保持平行、刚性固连，因此其姿态与手腕末端 body 姿态一致，只是在位置上做了固定平移。
- 相对于原 `wrist_roll_rubber_hand` 原点，左手抓持点的局部平移为 $[0.2, -0.035, 0]^T$ m，右手抓持点的局部平移为 $[0.2, 0.035, 0]^T$ m。
- 后续仿真中的末端位置、姿态、可视化以及需要迁移到真机上的“手心/抓持点”参数，均应以这两个 `grasp site` 为准。

**真实系统姿态参考的工程约定：**
- 在真实机器人上，IMU 输出的初始四元数**不应默认**等于 $[1, 0, 0, 0]$。因为这取决于机器人启动瞬间的真实姿态以及底层状态估计器对世界系的定义。
- 因此，在工程实现中，应将**第一帧四元数**记为参考姿态 $q_{ref}$（或对应旋转矩阵 $R_{ref}$）。
- 如果后续关注的是相对扰动而非绝对朝向，则应统一使用“当前姿态相对于 $q_{ref}$ 的变化”来定义姿态误差与姿态漂移。

### 一、 绝对加速度的严格物理推导（观测方程核心）

**目标：**求出末端在 $\{W\}$ 系下的线加速度 ${}^W a_E$ 和角加速度 ${}^W \alpha_E$。

**1. 末端线加速度 ${}^W a_E$ 的严密推导**

末端在 $\{W\}$ 系下的绝对位置为：
${}^W p_E = {}^W p_B + {}^W R_B \cdot {}^B p_E(q)$

求一阶导数（速度），根据 $\frac{d}{dt}({}^W R_B) = [{}^W \omega_B]_{\times} {}^W R_B$ （其中 $[\cdot]_{\times}$ 是向量的反对称矩阵），以及 $\frac{d}{dt}({}^B p_E) = {}^B J_v \dot{q}$：
${}^W v_E = {}^W v_B + {}^W \omega_B \times ({}^W R_B {}^B p_E) + {}^W R_B ({}^B J_v \dot{q})$

求二阶导数（加速度）【全项保留，无一遗漏】：
${}^W a_E = {}^W a_B + \underbrace{{}^W \alpha_B \times ({}^W R_B {}^B p_E)}_{\text{欧拉加速度}} + \underbrace{{}^W \omega_B \times ({}^W \omega_B \times ({}^W R_B {}^B p_E))}_{\text{向心加速度}} + \underbrace{2 {}^W \omega_B \times ({}^W R_B {}^B J_v \dot{q})}_{\text{科氏加速度}} + {}^W R_B \underbrace{({}^B \dot{J}_v \dot{q} + {}^B J_v \ddot{q})}_{\text{相对加速度}}$

整理为关于未知变量的仿射线性形式 $a = D + C\dot{q} + Bu$（令 $u = \ddot{q}$）：
- **$D_{acc}$ (已知常数项)**： ${}^W a_B + {}^W \alpha_B \times ({}^W R_B {}^B p_E) + {}^W \omega_B \times ({}^W \omega_B \times ({}^W R_B {}^B p_E))$
- **$C_{acc} \dot{q}$ (速度耦合项)**： $\left( 2 [{}^W \omega_B]_{\times} {}^W R_B {}^B J_v + {}^W R_B {}^B \dot{J}_v \right) \dot{q}$
- **$B_{acc} u$ (控制映射项)**： $({}^W R_B {}^B J_v) u$

这里需要特别说明：严格来说，${}^B J_v = {}^B J_v(q)$ 是关节构型 $q$ 的函数，而 ${}^B \dot{J}_v = {}^B \dot{J}_v(q, \dot q)$ 同时依赖 $q$ 和 $\dot q$。因此，诸如 ${}^B \dot{J}_v \dot q$ 这样的项在严格数学上通常会包含更高阶的非线性（例如三角函数乘以速度平方），并不是全局精确的仿射形式。本文后续将其写成 $D_{acc} + C_{acc}\dot q + B_{acc}u$，其含义是：在每个 MPC 控制步长，围绕当前工作点 $(q^\star, \dot q^\star)$ 对这些非线性项做局部线性化，并将 ${}^B J_v(q^\star)$、${}^B \dot{J}_v(q^\star, \dot q^\star)$ 以及 ${}^W R_B, {}^W \omega_B, {}^W \alpha_B$ 等量冻结为本轮 QP 中的已知系数。换言之，这里的 $C_{acc}, B_{acc}, D_{acc}$ 应理解为“当前工作点附近的局部仿射近似系数”，而不是对所有状态都严格成立的全局常数矩阵。

**2. 末端角加速度 ${}^W \alpha_E$ 的严密推导**

绝对角速度：
${}^W \omega_E = {}^W \omega_B + {}^W R_B ({}^B J_\omega \dot{q})$

求导得绝对角加速度：
${}^W \alpha_E = {}^W \alpha_B + {}^W \omega_B \times ({}^W R_B {}^B J_\omega \dot{q}) + {}^W R_B ({}^B \dot{J}_\omega \dot{q} + {}^B J_\omega \ddot{q})$

整理为线性形式：
- **$D_{\alpha}$ (已知常数项)**： ${}^W \alpha_B$
- **$C_{\alpha} \dot{q}$ (速度耦合项)**： $\left( [{}^W \omega_B]_{\times} {}^W R_B {}^B J_\omega + {}^W R_B {}^B \dot{J}_\omega \right) \dot{q}$
- **$B_{\alpha} u$ (控制映射项)**： $({}^W R_B {}^B J_\omega) u$

同理，${}^B J_\omega = {}^B J_\omega(q)$、${}^B \dot{J}_\omega = {}^B \dot{J}_\omega(q, \dot q)$ 本身并不是常数。这里将它们并入 $C_{\alpha}$ 和 $B_{\alpha}$，同样表示在当前工作点处冻结系数后的局部线性化模型。后续 QP 推导默认建立在这一“逐时刻重线性化 / 逐时刻冻结系数”的工程近似基础上。

### 二、 状态空间模型的严格构建 (State-Space)

由于 torso-relative 末端位置和三维重力误差都可以作为 $q_k$ 的非线性观测量进行局部线性化，不需要把它们单独加入状态向量。因此这里采用更简洁的二阶关节空间状态：

定义状态向量 $x_k \in \mathbb{R}^{2n}$ 以及控制输入 $u_k \in \mathbb{R}^n$：
$x_k = \begin{bmatrix} q_k \\ \dot{q}_k \end{bmatrix}, \quad u_k = \ddot{q}_k$

定义选择矩阵 $S_q = \begin{bmatrix} I_{n\times n} & 0_{n\times n} \end{bmatrix}$、$S_v = \begin{bmatrix} 0_{n\times n} & I_{n\times n} \end{bmatrix}$，使得 $q_k = S_q x_k$、$\dot{q}_k = S_v x_k$。

离散化状态转移方程采用标准二阶积分器形式：
$A_k = \begin{bmatrix} I_n & \Delta t \cdot I_n \\ 0 & I_n \end{bmatrix}, \quad B_k = \begin{bmatrix} 0.5 \Delta t^2 \cdot I_n \\ \Delta t \cdot I_n \end{bmatrix}$

即：
$x_{k+1} = A_k x_k + B_k u_k$

这里与下肢步态相关的基座扰动仍然通过 ${}^W a_{B,k}, {}^W \omega_{B,k}, {}^W \alpha_{B,k}, {}^W R_{B,k}$ 进入末端线加速度和角加速度观测方程，而不再额外构造 $e_{ori}$ 的状态演化方程。

**预测扰动 $d_k$ (由外部预测器提供)：**
系统在未来每个时刻 $k$ 都受到来自下肢移动导致的基座扰动，其包含了基座在世界系 $\{W\}$ 下的线加速度、角速度、角加速度以及旋转矩阵：
$d_k = \begin{bmatrix} {}^W a_{B, k} \\ {}^W \omega_{B, k} \\ {}^W \alpha_{B, k} \\ {}^W R_{B, k} \end{bmatrix}$
*(这些变量构成了前文常数项 $D_{acc}, D_{\alpha}$ 和矩阵 $A_k, B_k$ 中的时变已知成分)*

---

## 📐 MPC 算法推导：从物理代价函数映射到 OSQP 标准型

这是一种做硬核运控算法必须具备的“打破砂锅问到底”的精神。你说得对，从原始的物理代价函数跳跃到 OSQP 的标准二次型矩阵，中间跨越了极其繁琐的多项式展开和矩阵重新组合。之前直接给出结论确实略去了最关键的代数合并过程。

对于底层 C++ 落地来说，如果不知道这其中的每一项是怎么乘出来的，一旦矩阵维度报错，根本无从查起。

现在，我为你**逐项展开、毫无保留地重写这段严密的代数推导过程**。

---

### 目标：从物理代价函数映射到 OSQP 标准型

我们在每一个控制步长 $k$ 下的局部代价函数为：
$J_k = ||{}^W a_{EE, k}||_{Q_a}^2 + ||{}^W \alpha_{EE, k}||_{Q_\alpha}^2 + ||r_{p,k}||_{Q_p}^2 + ||r_{g,k}||_{Q_g}^2 + ||q_k||_{Q_q}^2 + ||\dot{q}_k||_{Q_v}^2 + ||u_k||_R^2$

其中：

$
r_p(q)={}^B p_E(q)-{}^B p_E(q_{nom}) \in \mathbb{R}^3
$

是 torso/IMU 相对末端位置误差。对于当前 IMU 与右肩固连的模型，${}^B p_E(q_{nom})$ 是只由名义右臂角决定的常量，可在控制器初始化时计算一次并缓存；它随 torso 的世界位置一起移动，因此不会妨碍整机行走。本版先不加入末端速度代价。

$
r_g(q)={}^E R_W(q)g^W-g^E_{ref}\in\mathbb{R}^3,
\qquad g^E_{ref}=\begin{bmatrix}0&0&-\|g^W\|\end{bmatrix}^T
$

是有方向的三维重力误差。正立时为零，倒立时第三维约为 $2\|g\|$，修复旧二维 XY 投影无法区分正立与精确倒立的问题。在当前实验设定中 $q_{nom}=0$。

这里我们将 MPC 的决策变量取为关节加速度 $u_k = \ddot q_k$，而不是直接取 $q$、$\dot q$ 或 $\tau$，主要有以下原因：

- 末端线加速度和角加速度与 $\ddot q$ 的关系最直接，因此当前代价函数中的核心目标项可以更自然地写成二次型。
- 在离散时间下，$q$ 和 $\dot q$ 可以由 $\ddot q$ 通过二阶积分器直接预测，状态更新形式规整，便于保持标准 QP / OSQP 结构。
- 当前方案主要依赖运动学映射和离散积分模型，不必显式引入完整刚体动力学，因此建模和调试复杂度更低，更适合在线快速求解。
- 工程实现上也更清晰：上层 MPC 优化 $\ddot q$，下层通过逆动力学生成前馈力矩，再叠加短时 $q_{ref}$、$\dot q_{ref}$ 的 PD 修正。

如果直接把 $q$ 或 $\dot q$ 作为决策变量，则末端线加速度和角加速度无法由决策变量自然表示，为了把这些项写进代价函数，通常还需要间接引入 $\ddot q$，模型表达会更绕，约束设计也不如当前方案直观。

如果直接把 $\tau$ 作为决策变量，则通常需要显式引入机械臂动力学模型
$M(q)\ddot q + h(q, \dot q) = \tau + J^T F_{ext}$
这会额外带来惯量矩阵、重力项、科氏力项和外力项等建模负担，使系统非线性更强，在线线性化、求解和调试复杂度都明显上升。因此，对当前阶段而言，直接以 $\ddot q$ 作为决策变量是在控制目标表达自然性、QP 建模简洁性、在线计算效率和工程实现可控性之间的更合适折中。

OSQP 求解器**唯一认识**的数学形式是（注意前面的 $\frac{1}{2}$）：
$$J_{OSQP} = \frac{1}{2} z_k^T H_k z_k + f_k^T z_k$$
其中超级变量 $z_k = \begin{bmatrix} x_k \\ u_k \end{bmatrix}$。

将其展开，OSQP 实际上在优化的形式是：
$$\frac{1}{2} \begin{bmatrix} x_k^T & u_k^T \end{bmatrix} \begin{bmatrix} H_{xx} & H_{xu} \\ H_{ux} & H_{uu} \end{bmatrix} \begin{bmatrix} x_k \\ u_k \end{bmatrix} + \begin{bmatrix} f_x^T & f_u^T \end{bmatrix} \begin{bmatrix} x_k \\ u_k \end{bmatrix}$$
$= \frac{1}{2} x_k^T H_{xx} x_k + \frac{1}{2} u_k^T H_{uu} u_k + x_k^T H_{xu} u_k + f_x^T x_k + f_u^T u_k$

*(注：由于 $H_k$ 是对称矩阵，所以 $H_{ux} = H_{xu}^T$，且交叉项 $\frac{1}{2} x_k^T H_{xu} u_k + \frac{1}{2} u_k^T H_{ux} x_k = x_k^T H_{xu} u_k$)*

**核心任务：** 我们必须把原始的 $J_k$ 完全展开，然后把对应 $x_k$ 和 $u_k$ 的系数“对号入座”到上面的 OSQP 公式中，以此反推 $H_{xx}, H_{uu}, H_{xu}, f_x, f_u$。

---

### 步骤一：提取状态自身的惩罚项 (State & Control Penalty)

首先处理最简单的三项：关节名义姿态正则、关节速度、关节加速度控制量。
定义一个组合状态权重矩阵 $Q_{state}$：
$Q_{state} = \text{block\_diag}(Q_q, \ Q_v)$
*(因为状态 $x_k = [q_k^T, \dot{q}_k^T]^T$，这里直接对关节位置与关节速度进行惩罚；在当前实验设定中 $q_{\text{nom}} = 0$)*

则关节位置与速度两项可以直接写为：
$||q_k||_{Q_q}^2 + ||\dot{q}_k||_{Q_v}^2 = x_k^T Q_{state} x_k$
因此在当前设定下，该部分只贡献二次项，不再产生由 $q_{\text{nom}}$ 引入的一次项。同时，控制输入项为 $||u_k||_R^2 = u_k^T R u_k$。

torso-relative 位置误差与三维重力误差都不是独立状态，而是由 $q_k$ 决定的非线性观测项。二者在当前工作点 $q^\star$ 附近分别做一阶泰勒展开：

$
r_p(q_k)\approx r_p(q^\star)+J_p(q^\star)(q_k-q^\star)=d_p+G_p x_k
$

$
r_g(q_k)\approx r_g(q^\star)+J_g(q^\star)(q_k-q^\star)=d_g+G_g x_k
$

其中：

$
J_p(q^\star)=({}^W R_B)^T{}^W J_v(q^\star)\in\mathbb{R}^{3\times n},
\quad G_p=J_pS_q,
\quad d_p=r_p(q^\star)-J_pq^\star
$

$
J_g(q^\star)=\left.\frac{\partial r_g}{\partial q}\right|_{q=q^\star}\in\mathbb{R}^{3\times n},
\quad G_g=J_gS_q,
\quad d_g=r_g(q^\star)-J_gq^\star
$

因此两个平方代价都保持标准凸二次型，例如：

$
\|r_p(x_k)\|_{Q_p}^2\approx x_k^TG_p^TQ_pG_px_k+2d_p^TQ_pG_px_k+\text{const}
$

重力项同理得到 $G_g^TQ_gG_g$ 和 $G_g^TQ_gd_g$。只要每轮 QP 冻结 $J_p,J_g,d_p,d_g$，加入末端位置并不会把本轮 OSQP 问题变成非线性规划；下一控制周期再重新线性化即可。

---

### 步骤二：通用二次型展开引理 (The General Expansion Lemma)

末端线加速度和角加速度的观测方程都具有相同的仿射结构：
$$y = D + C S_v x_k + B u_k$$
*(其中 $y$ 代表加速度，$D$ 是常数扰动，$C$ 是速度耦合矩阵，$S_v$ 是提取 $\dot{q}_k$ 的选择矩阵，$B$ 是控制映射矩阵)*

我们需要展开 $||y||_Q^2 = y^T Q y$：
$$y^T Q y = (D + C S_v x_k + B u_k)^T Q (D + C S_v x_k + B u_k)$$

利用矩阵乘法分配律展开（共 9 项），合并标量转置相同的项（例如 $x_k^T M^T Q N u_k = u_k^T N^T Q M x_k$）：

1.  **关于 $x_k$ 的纯二次项：** $(C S_v x_k)^T Q (C S_v x_k) = x_k^T (S_v^T C^T Q C S_v) x_k$
2.  **关于 $u_k$ 的纯二次项：** $(B u_k)^T Q (B u_k) = u_k^T (B^T Q B) u_k$
3.  **关于 $x_k$ 和 $u_k$ 的交叉项：** $2 (C S_v x_k)^T Q (B u_k) = 2 x_k^T (S_v^T C^T Q B) u_k$
4.  **关于 $x_k$ 的一次项（线性项）：** $2 D^T Q (C S_v x_k) = x_k^T (2 S_v^T C^T Q D)$
5.  **关于 $u_k$ 的一次项（线性项）：** $2 D^T Q (B u_k) = u_k^T (2 B^T Q D)$
6.  **常数项：** $D^T Q D$（对优化无影响，直接丢弃）

---

### 步骤三：合并所有项并“对号入座” (Assembling the OSQP Matrices)

现在，我们将线加速度（下标 $acc$）和角加速度（下标 $\alpha$）代入上述展开引理，并加上【步骤一】中的 $Q_{state}$ 和 $R$。

合并后，原始物理代价函数 $J_k$ 中各部分的系数如下：

**1. 对应 $x_k^T [\dots] x_k$ 的系数总和：**
$\text{Coef}_{xx} = \underbrace{S_v^T C_{acc}^T Q_a C_{acc} S_v}_{\text{线加速度的 } x \text{ 二次项}} + \underbrace{S_v^T C_{\alpha}^T Q_\alpha C_{\alpha} S_v}_{\text{角加速度的 } x \text{ 二次项}} + \underbrace{G_p^T Q_p G_p}_{\text{torso-relative 位置误差}} + \underbrace{G_g^T Q_g G_g}_{\text{三维重力误差}} + \underbrace{Q_{state}}_{\text{状态自身惩罚}}$
**2. 对应 $u_k^T [\dots] u_k$ 的系数总和：**
$\text{Coef}_{uu} = \underbrace{B_{acc}^T Q_a B_{acc}}_{\text{线加速度的 } u \text{ 二次项}} + \underbrace{B_{\alpha}^T Q_\alpha B_{\alpha}}_{\text{角加速度的 } u \text{ 二次项}} + \underbrace{R}_{\text{控制量自身惩罚}}$
**3. 对应 $x_k^T [\dots] u_k$ 的交叉系数总和：**
$\text{Coef}_{xu} = \underbrace{2 S_v^T C_{acc}^T Q_a B_{acc}}_{\text{线加速度的 } x,u \text{ 交叉项}} + \underbrace{2 S_v^T C_{\alpha}^T Q_\alpha B_{\alpha}}_{\text{角加速度的 } x,u \text{ 交叉项}}$
**4. 对应 $x_k$ 一次项 $f_x$ 的系数总和：**
$f_x = \underbrace{2 S_v^T C_{acc}^T Q_a D_{acc}}_{\text{线加速度的 } x \text{ 线性项}} + \underbrace{2 S_v^T C_{\alpha}^T Q_\alpha D_{\alpha}}_{\text{角加速度的 } x \text{ 线性项}} + \underbrace{2 G_p^T Q_p d_p}_{\text{torso-relative 位置一次项}} + \underbrace{2 G_g^T Q_g d_g}_{\text{三维重力误差一次项}}$
**5. 对应 $u_k$ 一次项 $f_u$ 的系数总和：**
$f_u = \underbrace{2 B_{acc}^T Q_a D_{acc}}_{\text{线加速度的 } u \text{ 线性项}} + \underbrace{2 B_{\alpha}^T Q_\alpha D_{\alpha}}_{\text{角加速度的 } u \text{ 线性项}}$

---

### 步骤四：推导最终的 Hessian ($H_k$) 和 Gradient ($f_k$)

最后一步，我们必须将推导出的系数与 OSQP 标准型严格对齐。

回忆 OSQP 的公式：$\frac{1}{2} x_k^T H_{xx} x_k + \frac{1}{2} u_k^T H_{uu} u_k + x_k^T H_{xu} u_k + f_x^T x_k + f_u^T u_k$

* **推导 $H_{xx}$：** 因为 OSQP 公式里有一个 $\frac{1}{2}$，所以 $H_{xx} = 2 \times \text{Coef}_{xx}$。
    $H_{xx} = 2 \left( S_v^T C_{acc}^T Q_a C_{acc} S_v + S_v^T C_{\alpha}^T Q_\alpha C_{\alpha} S_v + G_p^T Q_p G_p + G_g^T Q_g G_g + Q_{state} \right)$
* **推导 $H_{uu}$：** 同理，因为有 $\frac{1}{2}$，所以 $H_{uu} = 2 \times \text{Coef}_{uu}$。
    $$H_{uu} = 2 \left( B_{acc}^T Q_a B_{acc} + B_{\alpha}^T Q_\alpha B_{\alpha} + R \right)$$
* **推导 $H_{xu}$：** OSQP 交叉项公式里**没有** $\frac{1}{2}$，所以 $H_{xu}$ 直接等于 $\text{Coef}_{xu}$。
    $$H_{xu} = 2 \left( S_v^T C_{acc}^T Q_a B_{acc} + S_v^T C_{\alpha}^T Q_\alpha B_{\alpha} \right)$$
* **推导 $H_{ux}$：** 为保证 Hessian 对称矩阵的性质：
    $$H_{ux} = H_{xu}^T = 2 \left( B_{acc}^T Q_a C_{acc} S_v + B_{\alpha}^T Q_\alpha C_{\alpha} S_v \right)$$
* **推导 $f_x$ 和 $f_u$：** 一次项完全一一对应，无需乘以 2（因为前面的展开过程中已经产生了 2）。
    $f_x = 2 \left( S_v^T C_{acc}^T Q_a D_{acc} + S_v^T C_{\alpha}^T Q_\alpha D_{\alpha} + G_p^T Q_p d_p + G_g^T Q_g d_g \right)$
    $f_u = 2 \left( B_{acc}^T Q_a D_{acc} + B_{\alpha}^T Q_\alpha D_{\alpha} \right)$

---

**最终结论汇总（写入 C++/Eigen 的直接依据）：**

$H_k = \begin{bmatrix} H_{xx} & H_{xu} \\ H_{ux} & H_{uu} \end{bmatrix}$
$f_k = \begin{bmatrix} f_x \\ f_u \end{bmatrix}$

---

## 🧮 零、 维度与超级变量定义 (The Super-Variable)

假设你的预测时域为 $N$ 步。状态变量 $x_k \in \mathbb{R}^{n_x}$ （在我们的例子中，$n_x = 2n$）。控制输入 $u_k \in \mathbb{R}^{n_u}$ （在我们的例子中，$n_u = n$）。

为了让求解器一次性看到所有的未来，我们把从 $k=0$ 到 $k=N$ 的所有状态，以及从 $k=0$ 到 $k=N-1$ 的所有控制量，垂直拼接成一个超级变量向量 $Y$。注意：包含了终端状态 $x_N$，但没有 $u_N$（因为第 $N$ 步不需要再输出了）。

超级变量 $Y$ 的全貌如下（总维度 $M = N(n_x + n_u) + n_x$）：
$Y = \begin{bmatrix} x_0 \\ u_0 \\ x_1 \\ u_1 \\ \vdots \\ x_{N-1} \\ u_{N-1} \\ x_N \end{bmatrix} \in \mathbb{R}^M$

---

## 🏗️ 一、 全局代价矩阵构建 (Global Cost Matrices)

OSQP 的目标函数为 $\min_Y \frac{1}{2} Y^T P Y + q_{vec}^T Y$。我们只需将你之前算出的单步 $H_k = \begin{bmatrix} H_{xx, k} & H_{xu, k} \\ H_{ux, k} & H_{uu, k} \end{bmatrix}$ 和 $f_k = \begin{bmatrix} f_{x, k} \\ f_{u, k} \end{bmatrix}$ 沿着对角线排布。对于最后的终端状态 $x_N$，我们只有状态惩罚 $H_{xx, N}$ 和 $f_{x, N}$（在当前 $q_{\text{nom}}=0$ 的设定下，终端项只保留关节构型与关节速度的二次惩罚，因此 $f_{x,N}=0$；终端代价可以加重力姿态误差项，但初版为了更简单更稳不加更合适）。

在当前定义下，由于 $q_{\text{nom}} = 0$，终端代价若仅保留 $||q_N||_{Q_q}^2 + ||\dot q_N||_{Q_v}^2$，则有 $Q_{\text{terminal}} = \text{block\_diag}(Q_q, Q_v)$，并且终端状态项直接写为
$x_N^T Q_{\text{terminal}} x_N$
与 OSQP 标准型对齐可得：
$H_{xx, N} = 2 Q_{\text{terminal}} = 2\,\text{block\_diag}(Q_q, Q_v), \qquad f_{x, N} = 0$

**1. 全局 Hessian 矩阵 $P$ (Block-Diagonal, 维度 $M \times M$)：**

$P = \begin{bmatrix}
\begin{array}{cc|}
H_{xx, 0} & H_{xu, 0} \\
H_{ux, 0} & H_{uu, 0} \\
\hline
\end{array} & \mathbf{0} & \dots & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \begin{array}{|cc|}
H_{xx, 1} & H_{xu, 1} \\
H_{ux, 1} & H_{uu, 1} \\
\hline
\end{array} & \dots & \mathbf{0} & \mathbf{0} \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
\mathbf{0} & \mathbf{0} & \dots & \begin{array}{|cc|}
H_{xx, N-1} & H_{xu, N-1} \\
H_{ux, N-1} & H_{uu, N-1} \\
\hline
\end{array} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \dots & \mathbf{0} & \begin{array}{|c|}
H_{xx, N} \\
\hline
\end{array}
\end{bmatrix}$

*(注：这是一个极其稀疏的分块对角矩阵，除了对角线上的方块外，全为 0)。*

**2. 全局一次项向量 $q_{vec}$ (维度 $M \times 1$)：**

$q_{vec} = \begin{bmatrix} f_{x, 0} \\ f_{u, 0} \\ f_{x, 1} \\ f_{u, 1} \\ \vdots \\ f_{x, N-1} \\ f_{u, N-1} \\ f_{x, N} \end{bmatrix}$

---

## 🔗 二、 全局动力学等式约束 (Global Dynamics Constraints)

这是极其考验逻辑的一步。在物理上，系统必须符合状态转移：
- **初始约束**：预测的第 0 步状态必须等于当前传感器的真实读数，即 $x_0 = x_{curr}$。
- **转移约束**：每一个未来状态必须由前一个状态和输入积分而来，即 $x_{k+1} = A_k x_k + B_k u_k + d_{ext, k}$。

我们把这些方程移项，让所有包含未知变量（$Y$ 里面的东西）留在左边，常数留在右边：
- $I \cdot x_0 = x_{curr}$
- $-A_0 x_0 - B_0 u_0 + I \cdot x_1 = d_{ext, 0}$
- $-A_1 x_1 - B_1 u_1 + I \cdot x_2 = d_{ext, 1}$
- ...

将其写成庞大的矩阵形式 $A_{dyn} Y = b_{dyn}$：

**1. 动力学矩阵 $A_{dyn}$ (带状稀疏矩阵，维度 $(N+1)n_x \times M$)：**

请仔细观察它的阶梯状（Banded）结构，这是 C++ 中使用 `SparseMatrix` 插入元素（Triplet）时的核心依据：

$A_{dyn} = \begin{bmatrix}
I_{n_x} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \dots & \mathbf{0} & \mathbf{0} \\
-A_0 & -B_0 & I_{n_x} & \mathbf{0} & \mathbf{0} & \dots & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & -A_1 & -B_1 & I_{n_x} & \dots & \mathbf{0} & \mathbf{0} \\
\vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \dots & I_{n_x} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \dots & -A_{N-1} & -B_{N-1} & I_{n_x}
\end{bmatrix}$

*(注：第一行的 $I_{n_x}$ 负责锁死 $x_0$。之后的每一行中的 $[-A_k, -B_k, I_{n_x}]$ 块都向右平移了 $(n_x + n_u)$ 个单位，完美对齐向量 $Y$ 中的 $[x_k, u_k, x_{k+1}]$)。*

**2. 动力学常数向量 $b_{dyn}$ (维度 $(N+1)n_x \times 1$)：**

$b_{dyn} = \begin{bmatrix} x_{curr} \\ d_{ext, 0} \\ d_{ext, 1} \\ \vdots \\ d_{ext, N-1} \end{bmatrix}$

---

## 🚧 三、 全局边界不等式约束 (Global Bound Constraints)

我们需要限制硬件的极限：关节位姿 $q$、关节速度 $\dot{q}$ 和关节加速度 $u$。因为这三者全部都在超级向量 $Y$ 里面，所以提取它们的矩阵仅仅是一个巨大的单位阵！设我们要约束 $Y_{min} \le Y \le Y_{max}$，则：

**1. 不等式矩阵 $A_{ineq}$ (全维度单位阵，维度 $M \times M$)：**

$A_{ineq} = \begin{bmatrix}
I_{n_x} & 0 & 0 & \dots & 0 \\
0 & I_{n_u} & 0 & \dots & 0 \\
0 & 0 & I_{n_x} & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \dots & I_{n_x}
\end{bmatrix} = I_{M \times M}$

**2. 不等式上下界 $l_{ineq}$ 和 $u_{ineq}$ (维度 $M \times 1$)：**

如果某些状态（比如关节位置或关节速度中的某些分量）不需要硬约束，就填入 $-\infty$ 和 $\infty$。

$l_{ineq} = \begin{bmatrix} x_{min} \\ u_{min} \\ x_{min} \\ u_{min} \\ \vdots \\ x_{min} \end{bmatrix}, \quad
u_{ineq} = \begin{bmatrix} x_{max} \\ u_{max} \\ x_{max} \\ u_{max} \\ \vdots \\ x_{max} \end{bmatrix}$

---

## 🧩 四、 最终 OSQP 标准型大一统组装

OSQP 求解器的接口要求把“等式约束”和“不等式约束”垂直叠放在一起，统一写成：
$l \le A_{cons} Y \le u$

对于等式约束，只需令其上下界相等（$l = u$）即可被 OSQP 自动识别为等式。

**1. 终极约束矩阵 $A_{cons}$ (维度 $((N+1)n_x + M) \times M$)：**

$A_{cons} = \begin{bmatrix} A_{dyn} \\ A_{ineq} \end{bmatrix}$

**2. 终极约束下界 $l$ 和 上界 $u$：**

$l = \begin{bmatrix} b_{dyn} \\ l_{ineq} \end{bmatrix}, \quad
u = \begin{bmatrix} b_{dyn} \\ u_{ineq} \end{bmatrix}$

---

## ⚙️ 从 MPC 的 ddq 到执行力矩

MPC 仍以 $u_k=\ddot q_k$ 为决策变量，不把完整刚体动力学放入预测状态方程。在线只执行第一拍 $\ddot q_0^\star$，下层执行器使用 inverse dynamics 转成力矩：

$
\tau=\tau_{ff}+K_p(q_{ref}-q)+K_d(\dot q_{ref}-\dot q)
$

MuJoCo 仿真先采用与 LQR 相同的非摩擦约束感知逆动力学生成名义力矩：

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
tau_ff = tau_inverse + qfrc_nonfriction[right_arm_qvel_indices]
```

`qfrc_inverse` 加回 contact、joint/tendon limit 和 equality 等非摩擦约束，仅保留 `FRICTION_DOF` 与 `FRICTION_TENDON` 的摩擦补偿。然后固定其他执行器力矩，使用一次基准和五次右臂力矩扰动的 MuJoCo 前向动力学构建 $G_\tau=\partial\ddot q_{right}/\partial\tau_{right}$，再用阻尼最小二乘一次求出力矩修正。

由于摩擦、接触和限位约束会发生离散切换，下层还必须用完整前向动力学依次验收 $1/0.5/0.25/0.125$ 倍力矩修正；只接受比当前工作点更接近 `ddq_des` 的候选。若第一轮验收后的五关节残差范数仍超过 `5 rad/s²`，则在已接受力矩处重新构建一次 $G_\tau$，对剩余残差再求一次受完整前向动力学验收的修正。该过程最多两轮，不修改 MPC 输出，只负责防止局部力矩映射跨过约束切换点后产生错误加速度。该下层映射允许浮动基和腿部自然响应，不需要把完整刚体动力学放进 MPC 的 QP。它不是完整的接触力优化；主动接触任务仍需把接触力和摩擦锥纳入 MPC。

真机可用 Pinocchio/RBDL/厂商动力学库计算相同的无接触项：

$
\tau_{ff}=M(q)\ddot q_{des}+h(q,\dot q)
$

如果后续希望 MPC 同时考虑力矩上限，又不想把力矩作为决策变量，可以在每轮工作点冻结动力学：

$
\tau_k\approx M(\bar q_k)u_k+h(\bar q_k,\dot{\bar q}_k)
$

这样 $\tau_{min}\le\tau_k\le\tau_{max}$ 仍是关于 $u_k$ 的线性不等式，可继续由 OSQP 求解。

## 💡 工程师落地 Check-List (代码实现防坑指南)

当你把这些宏伟的矩阵写进 C++ 时，请严格遵守以下法则：

1. **绝对不要用稠密矩阵类型**： 你的 $A_{cons}$ 矩阵非常庞大（例如对于 14 自由度，预测 20 步，矩阵维度将达到约 $1200 \times 900$），但里面 95% 以上的元素都是 0。如果你用 `Eigen::MatrixXd`，程序瞬间卡死。
2. **必须使用 Triplet 插入法**： 在 Eigen 中构建这样的稀疏矩阵，必须声明 `std::vector<Eigen::Triplet<double>>`。然后在你的 `for (k=0...N)` 循环中，把每个单步小矩阵（比如 $-A_k$）的每一个非零元素，以 `(row, col, value)` 的形式 `push_back` 到 Triplet 列表里。
3. **坐标索引要精准**：
   - 在构建 $A_{dyn}$ 时，$-A_k$ 块的左上角坐标永远是： `row = (k+1)*n_x`，`col = k*(n_x + n_u)`。
   - $I_{n_x}$（转移到的下一个状态）的左上角坐标是：`row = (k+1)*n_x`，`col = (k+1)*(n_x + n_u)`。

只要你的下标索引严格遵守这张矩阵图纸，你的 C++ 代码就能在 1 毫秒内，极其优雅地解出抵抗外部颠簸和冲撞的最优前馈加速度序列。
