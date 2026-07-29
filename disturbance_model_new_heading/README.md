# 新版 H-frame 扰动模板

该目录使用 `disturbance_model_new/torso_disturbance_straight.npz` 中的同一批原始数据，重新生成与世界航向无关的 heading-frame 模板，并与原有 W-frame 模板做数值和趋势对比。

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

向量和姿态转换为

$$
{}^Hz={}^HR_W{}^Wz,
\qquad z\in\{a_B,\omega_B,\alpha_B\},
$$

$$
{}^HR_B={}^HR_W{}^WR_B.
$$

姿态按 bin 使用 Markley 四元数均值，保存 `wxyz` 四元数及其对应旋转矩阵。

## 程序

1. `convert_world_to_heading.py`
   - 按 phase 回绕识别步态周期；
   - 使用上一完整周期的 yaw 圆周均值建立 H；
   - 生成可检查的 H-frame 中间 NPZ、CSV 和诊断图。
2. `build_heading_disturbance_templates.py`
   - 生成 raw、half-smoothed、fully-smoothed 三种 H 模板。
3. `compare_heading_world_templates.py`
   - 直接比较 H/W 存储值；
   - 再把原 W 模板旋转到代表性 H 系做严格比较；
   - 输出 RMSE、相关系数、循环相位偏移、姿态 SO(3) 角距离和通过判断。
4. `run_all.sh`
   - 在前台顺序运行以上三个程序。

## 运行

```bash
./disturbance_model_new_heading/run_all.sh
```

脚本不启动后台任务，任何一步失败都会立即退出。

## 主要输出

- `torso_disturbance_heading.npz`
- `torso_disturbance_heading_preview.csv`
- `torso_disturbance_heading.png`
- `templates_heading/`
- `comparison_world_heading/comparison_metrics.json`
- `comparison_world_heading/comparison_summary.csv`
- `comparison_world_heading/Heading_vs_World_Template_Comparison.png`
- `comparison_world_heading/Heading_vs_World_Template_Direct_Comparison.png`
