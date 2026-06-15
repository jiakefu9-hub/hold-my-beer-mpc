# disturbance_model_new 数据流说明

## 1. 这个文件夹是干什么的

`disturbance_model_new/` 用来重新收集一版**近似直线行走**时的 torso 扰动数据，并基于这些数据直接在 `W` 系下构建新的扰动模板。

这套流程和旧的 `disturbance_model/` 最大的区别是：

- 旧流程里，机器人原本会慢慢偏航，所以先采原始数据，再经过 `W -> H` 的变换，最后在 `H` 系下建模板。
- 新流程里，通过调节 `cmd_init = [0.5, 0, 0.01322]`，机器人已经可以近似直线行走，因此重新采一版数据，直接在 `W` 系下建模板。
- 这样做的目的不是推翻旧流程，而是得到一版**更直观、更容易解释**的直线行走扰动模板。

---

## 2. 这个文件夹目前包含什么流程

当前 `disturbance_model_new/` 可以分成两个阶段：

1. **重新采集 torso 扰动数据，并检查 yaw 是否还在缓慢漂移**
2. **基于新的 world 数据构建 3 种扰动模板，并保存可视化图片**

对应脚本为：

- `collect_torso_disturbance_and_check_yaw.py`
- `build_world_disturbance_templates.py`

---

## 3. 第一步：重新采集 torso 扰动数据

### 3.1 脚本

```bash
python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model_new/collect_torso_disturbance_and_check_yaw.py g1.yaml
```

### 3.2 这个脚本做了什么

这个脚本本质上是在模仿旧的 `disturbance_model/get_disturbance.py`，但加入了以下增强：

- 在采集原始 torso 扰动数据的同时，直接做 `yaw` 直线性检查
- 同时保存 `local(IMU)` 和 `world(W)` 两套数据
- 自动保存 `npz + csv + 图像`
- 自动计算稳定段的：
  - 平均 yaw
  - yaw 标准差
  - yaw 线性拟合斜率
- 最后根据 yaw 斜率给出 `cmd[2]` 是否还需要调大/调小的建议

### 3.3 运行时的控制设定

- 下肢继续使用 RL 步态策略
- 上肢锁定在 `arm_waist_target`
- 采集期间使用 `g1.yaml` 中的 `cmd_init`
- 当前目标是验证：在这组 `cmd_init` 下，机器人是不是已经近似直线走

---

## 4. 第一步里到底采了哪些数据

这里要区分三类：

### 4.1 直接“测出来”的原始量

这些量是直接从 MuJoCo 当前状态或传感器里拿到的：

#### 1. 躯干四元数
- 字段：`torso_quaternion`
- 来源：`d.xquat[torso_id]`
- 含义：躯干姿态四元数

#### 2. torso IMU 局部线加速度
- 字段：`torso_linear_acceleration_local`
- 来源：`imu-torso-linear-acceleration`
- 含义：IMU 局部坐标系下的线加速度

#### 3. torso IMU 局部角速度
- 字段：`torso_angular_velocity_local`
- 来源：`imu-torso-angular-velocity`
- 含义：IMU 局部坐标系下的角速度

#### 4. 左右脚高度
- 字段：
  - `left_foot_z`
  - `right_foot_z`
- 来源：`d.xpos[body_id][2]`
- 含义：左右脚在世界系下的高度

#### 5. 仿真时间与相位
- 字段：
  - `count`
  - `phase`
- 其中：
  - `count = counter * simulation_dt`
  - `phase = (count % period) / period`

---

### 4.2 在采集循环里即时算出来的量

这些量不是传感器直接给的，而是在采集脚本里根据原始量算出来的。

#### 1. yaw
由四元数提取：

$$
\text{yaw} = \text{atan2}\big(2(wz + xy), 1 - 2(y^2 + z^2)\big)
$$

字段：
- `yaw`

#### 2. yaw_unwrapped
为了消除 `[-\pi, \pi]` 跳变，采集结束后做：

$$
\text{yaw\_unwrapped} = \text{unwrap}(\text{yaw})
$$

字段：
- `yaw_unwrapped`

#### 3. 旋转矩阵 ${}^W R_{IMU}$
由四元数转换得到：

$$
{}^W R_{IMU}
$$

字段：
- `R_world_from_imu`

它的含义是：把 IMU 局部系向量转到世界系。

---

## 5. 为什么同时保留 local 和 world 两套数据

因为这两套数据用途不同：

- `local(IMU)`：
  - 更接近真实传感器原始输出
  - 更方便和真机 IMU 语义对齐
- `world(W)`：
  - 更方便做平均、做模板、看整体扰动方向
  - 更适合后续直接构建 `W` 系模板

所以在这个新流程里，两套都保留。

---

## 6. 第一步里 local 数据怎么转成 world 数据

### 6.1 世界系角速度

由局部角速度直接旋转：

$$
{}^W \omega = {}^W R_{IMU}\, {}^{IMU}\omega
$$

字段：
- `torso_angular_velocity_world`

### 6.2 世界系线加速度

先把局部加速度旋到世界系，再减去重力反应项：

$$
{}^W a = {}^W R_{IMU}\, {}^{IMU} a - \begin{bmatrix}0\\0\\9.81\end{bmatrix}
$$

字段：
- `torso_linear_acceleration_world`

### 6.3 局部角加速度

由局部角速度差分得到：

$$
{}^{IMU}\alpha_k \approx \frac{{}^{IMU}\omega_k - {}^{IMU}\omega_{k-1}}{\Delta t}
$$

字段：
- `torso_angular_acceleration_local`

### 6.4 世界系角加速度

先得到世界系角速度，再差分：

$$
{}^W \alpha_k \approx \frac{{}^W \omega_k - {}^W \omega_{k-1}}{\Delta t}
$$

字段：
- `torso_angular_acceleration_world`

---

## 7. 为什么没有直接从 MuJoCo 后台“直接读取 IMU 点在 W 系下的量”

因为这里要的是：

- **IMU 这个具体传感器点**
- 并且希望 `local` 和 `world` 之间语义完全一致

MuJoCo 对此没有一个一步到位、比“原始 IMU 数据 + 坐标变换”更干净的直接接口。

因此当前策略是：

- `local`：保留 IMU 原始测量
- `world`：由同一批原始 IMU 数据统一变换得到

这样做的优点是：

- 前后语义一致
- 更接近真机处理链路
- 更不容易把 sensor 量和 body 量混淆

---

## 8. 第一步的输出文件

运行 `collect_torso_disturbance_and_check_yaw.py` 后，默认生成：

- `torso_disturbance_straight.npz`
- `torso_disturbance_straight_preview.csv`
- `torso_disturbance_straight_local.png`
- `torso_disturbance_straight_world.png`

### 8.1 `npz`
最完整，保存全部字段。

### 8.2 `csv`
便于直接查看关键数据，包括：

- 四元数
- yaw
- yaw_unwrapped
- yaw 拟合线
- local 加速度/角速度/角加速度
- world 加速度/角速度/角加速度
- 左右脚高度

### 8.3 `local.png`
用于检查 IMU 局部坐标系下的数据变化。

### 8.4 `world.png`
用于检查转换到世界系后的数据变化，并观察是否适合直接做 `W` 系模板。

---

## 9. 第一步结束后如何判断“是不是已经够直了”

这个脚本重点输出 3 个 yaw 相关量：

### 9.1 稳定段平均 yaw
- 表示稳定行走时，躯干整体有无固定偏置

### 9.2 稳定段 yaw 标准差
- 表示稳定段里的周期性波动大小

### 9.3 稳定段 yaw 线性拟合斜率
- 这是**最关键指标**
- 它表示：机器人是不是还在持续慢慢偏航

当前你得到的是大概：

- `稳定段平均 yaw ≈ -0.0226 rad`
- `稳定段标准差 ≈ 0.0041 rad`
- `稳定段线性拟合斜率 ≈ 0.000018 rad/s`

这说明：

- 机器人存在一个固定的小姿态偏置
- 但已经没有明显的持续偏航漂移
- 也就是说，现在是“基本直线走”

这也是为什么现在可以考虑重新用 `W` 系数据建模板。

---

## 10. 第二步：基于新的 world 数据构建模板

### 10.1 脚本

```bash
python /home/fjk/g1_ws/hold-my-beer-mpc/disturbance_model_new/build_world_disturbance_templates.py
```

### 10.2 输入文件

默认输入：

- `disturbance_model_new/torso_disturbance_straight.npz`

这个脚本只使用其中的：

- `count`
- `phase`
- `torso_linear_acceleration_world`
- `torso_angular_velocity_world`
- `left_foot_z`
- `right_foot_z`
- `gait_period`

也就是说，这一步是**只基于 world 数据**建模板。

---

## 11. 第二步的数据是如何变成模板的

### 11.1 先丢弃启动段

默认丢弃前 `4.0s`：

$$
\text{mask} = (count \ge 4.0)
$$

这样做的目的是：

- 去掉起步过渡段
- 只保留稳定的周期性行走数据

### 11.2 按 phase 分 bin

默认：
- `period = 0.8s`
- `num_bins = 100`

每个样本根据当前 `phase` 分配到某个 bin：

$$
\text{bin\_id} = \lfloor \text{phase} \cdot N \rfloor
$$

### 11.3 在每个 bin 内做平均和标准差

对每个 bin 内的数据分别求：

- `acc_W` 均值和标准差
- `omega_W` 均值和标准差
- 左右脚高度均值和标准差

得到：

- `torso_linear_acceleration_template`
- `torso_linear_acceleration_std`
- `torso_angular_velocity_template`
- `torso_angular_velocity_std`

### 11.4 角加速度模板不是直接平均出来的

这里沿用了旧流程里的原则：

- **不直接对原始角加速度平均**
- 而是先得到 `omega_W` 的均值模板
- 再对模板做环形中心差分

公式：

$$
\alpha_i \approx \frac{\omega_{i+1} - \omega_{i-1}}{2 \Delta t_{bin}}
$$

得到：

- `torso_angular_acceleration_template`

这样做的目的是：

- 避免直接平均 noisy 的原始角加速度
- 让最终模板更平滑、更稳定

---

## 12. 三种模板分别是什么

### 12.1 Raw Template
- `acc_W`：直接用原始均值模板
- `omega_W`：直接用原始均值模板
- `alpha_W`：由原始 `omega_W` 模板差分得到

输出：
- `world_disturbance_template.npz`
- `world_disturbance_template_preview.csv`

---

### 12.2 Half Smoothed Template
- `acc_W`：保留 raw
- `omega_W`：做轻度环形滑动平均
- `alpha_W`：由平滑后的 `omega_W` 求导得到

输出：
- `world_disturbance_template_half_smoothed.npz`
- `world_disturbance_template_half_smoothed_preview.csv`

---

### 12.3 Fully Smoothed Template
- `acc_W`：轻度平滑
- `omega_W`：轻度平滑
- `alpha_W`：由平滑后的 `omega_W` 求导得到

输出：
- `world_disturbance_template_fully_smoothed.npz`
- `world_disturbance_template_fully_smoothed_preview.csv`

---

## 13. 第二步为什么还要保留左右脚高度

模板本身主要给 torso 扰动用，但左右脚高度保留下来是为了做一个辅助验证：

- 如果某个 phase bin 对应的左右脚高度分布很集中
- 说明这个 phase 的物理状态比较一致
- 那么这个模板更可信

所以这里的：

- `left_foot_z_mean/std`
- `right_foot_z_mean/std`

主要是拿来做周期唯一性和模板稳定性检查的。

---

## 14. 第二步的图片是怎么来的

`build_world_disturbance_templates.py` 会直接模仿旧的 `inspect_disturbance_template.py`，生成一个 `3 x 3` 的对比图：

- 列：
  - Raw
  - Half Smoothed
  - Fully Smoothed
- 行：
  - 世界系线加速度
  - 世界系角速度
  - 世界系角加速度

图片文件：
- `World_Disturbance_Template_Comparison.png`

这张图的作用是：
- 直接肉眼比较三种模板差异
- 判断是不是需要平滑
- 评估哪一版更适合后续用于前馈

---

## 15. 为什么新的 W 系模板和旧的 H 系模板看起来很像

你后来发现：

- 新的 `W` 系模板图片
- 和旧的 `H` 系模板图片
- 肉眼看起来几乎一样

这不是程序用错数据，而是因为：

### 15.1 新模板确实用的是新数据
- 新脚本默认输入是：
  - `disturbance_model_new/torso_disturbance_straight.npz`

### 15.2 现在机器人已经近似直线走
- 旧流程里之所以要 `W -> H`
- 是因为以前会慢偏航
- 现在经过 `cmd[2]` 调整后，慢偏航几乎消失了
- 所以 `W` 和“去偏航后的 H”本来就会非常接近

### 15.3 描述的还是同一个物理扰动
- 都是在看 torso 走路时的周期扰动
- 因此波形本来就会相似

也就是说：

- **新模板确实是新的**
- 只是因为你把机器人调直了，所以它和旧 H 模板越来越像，这是合理现象

---

## 16. 当前这套新流程的整体逻辑

把整个 `disturbance_model_new/` 压缩成一句话，就是：

1. 先用 `collect_torso_disturbance_and_check_yaw.py`
   - 重新采一版近似直线走的 torso 数据
   - 同时检查 yaw 是否仍在慢漂移
2. 如果 yaw 斜率已经接近 0
   - 说明可以接受为“基本直线走”
3. 再用 `build_world_disturbance_templates.py`
   - 直接基于 `W` 系数据构建 raw / half / fully 三种模板
4. 最终得到一套新的、在直线行走条件下的 `W` 系 torso 扰动模板

---

## 17. 以后回顾这个项目时，最应该记住的几点

### 1. 旧流程的意义
旧的 `H` 系流程不是白做的，它解决的是：

- 机器人会慢偏航
- 直接在固定 `W` 系下做模板会失真

这是一个很重要的 challenge 和工程亮点。

### 2. 新流程的意义
新的 `disturbance_model_new/` 不是否定旧流程，而是：

- 在你已经把机器人调到近似直线走之后
- 重新采一版更干净、更容易解释的 `W` 系数据

### 3. 现在最关键的判断标准
- 是否继续调 `cmd[2]`
- 主要看：
  - **稳定段 yaw 线性拟合斜率**
- 不是主要看平均 yaw

### 4. 新模板为什么和旧模板相似
- 因为现在已经近似直线走
- 所以 `W` 和旧的 `H` 已经很接近

---

## 18. 一句话总结

`disturbance_model_new/` 这一整套流程做的事情，就是：

**在已经调到近似直线行走的条件下，重新采集 torso 原始扰动数据，保留 local 与 world 两套表达，检查 yaw 是否仍有持续漂移，然后直接基于稳定段 world 数据构建 raw / half / full 三种新的周期扰动模板。**

这套流程的目标是让以后用于前馈的扰动模板更干净、更直观、更容易解释，也更方便和后续真机实验对齐。