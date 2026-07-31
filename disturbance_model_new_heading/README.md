# H-frame 节点与 6 ms 区间扰动模板

该目录生成与绝对世界航向无关的 heading-frame 模板。当前 MPC 使用 `templates_heading_interval/`：每个文件同时包含瞬时节点量和与一个 6 ms 控制输入对应的未来区间平均量。

## H 系定义

对当前步态周期 $C_j$，使用上一完整周期 $C_{j-1}$ 的 torso yaw 圆周平均：

$$
\psi_{H,j}
=\operatorname{atan2}
\left(
\sum_{i\in C_{j-1}}\sin\psi_i,
\sum_{i\in C_{j-1}}\cos\psi_i
\right).
$$

整个 $C_j$ 内保持该方向不变：

$$
{}^HR_W=R_z(-\psi_{H,j}).
$$

- $z_H$ 与世界系重力向上方向一致；
- $x_H$ 指向上一完整周期的平均 torso 朝向；
- $y_H$ 由右手系确定；
- 原点不参与当前向量和姿态模板。

节点向量和姿态转换为

$$
{}^Hz={}^HR_W{}^Wz,
\qquad z\in\{a_B,\omega_B,\alpha_B\},
$$

$$
{}^HR_B={}^HR_W{}^WR_B.
$$

姿态按 bin 使用 Markley 四元数均值，保存 `wxyz` 四元数及其对应旋转矩阵。区间线加速度和角加速度先由同一世界系中的速度端点差生成，再使用区间起点 H 系旋转，避免跨周期时混用两个不同 H 系。

## 程序

1. `disturbance_model_new/collect_torso_disturbance_and_check_yaw.py`
   - 在 `mj_step` 前按 2 ms 采样，保证相位标签与物理状态同拍；
   - 额外记录 IMU site 的世界系线速度，用于严格生成未来 6 ms 平均线加速度。
2. `convert_world_to_heading.py`
   - 按 phase 回绕识别步态周期；
   - 使用上一完整周期的 yaw 圆周均值建立 H；
   - 生成可检查的 H-frame 中间 NPZ、CSV 和诊断图。
3. `build_heading_disturbance_templates.py`
   - 在 400 个、间隔 2 ms 的相位起点上生成节点模板；
   - 对每个起点生成随后 6 ms 的滑动区间模板；
   - 输出 raw、half-smoothed、fully-smoothed 三种版本。
4. `run_interval_all.sh`
   - 在前台依次重新采集、转换并生成当前 MPC 模板。
5. `run_all.sh`
   - 为兼容旧命令保留，直接转入 `run_interval_all.sh`。

## 运行

```bash
./disturbance_model_new_heading/run_interval_all.sh
```

脚本不启动后台任务，任何一步失败都会立即退出。当前默认会打开采集 viewer；整个流程会留下原始 W 数据、H 中间数据和最终模板，便于复现。

## 主要输出

- `../disturbance_model_new/torso_disturbance_straight_interval.npz`
- `torso_disturbance_heading_interval.npz`
- `torso_disturbance_heading_interval_preview.csv`
- `torso_disturbance_heading_interval.png`
- `templates_heading_interval/heading_disturbance_template.npz`
- `templates_heading_interval/heading_disturbance_template_half_smoothed.npz`
- `templates_heading_interval/heading_disturbance_template_fully_smoothed.npz`

每个模板的 `template_schema_version=2`，`phase_reference=interval_start`，`interval_dt=0.006`。运行时还会校验模板 SHA-256、H 系定义、旋转矩阵合法性和全部 bin 是否有效。
