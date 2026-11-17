# G1 arm stabilization during walking

Unitree G1 23DoF 行走及起停时的右臂末端姿态稳定：身体扰动预测、6 ms 加速度 MPC、
耦合五关节逆动力学执行。左臂为固定关节 PD 对照，腿部由独立行走控制器负责。

真机实验与主对照录像已完成。当前交接版本保留三组展示仿真、可复算的真机精选数据，
以及位置 PID、前馈加速度 PID（C）等对照。姿态收益较明确；线加速度与落脚冲击没有全面优势。

## 从这里开始

| 目标 | 入口 |
| --- | --- |
| 安装、打开三组回放、重跑仿真 | [快速开始](docs/QUICKSTART.md) |
| 查最终配置、数据、指标和录像对应关系 | [最终实验清单](docs/FINAL_EXPERIMENTS.md) |
| 看懂仿真与真机的控制链 | [架构](ARCHITECTURE.md) |
| 找算法和实验文档 | [文档导航](docs/README.md) |
| 研究真机实验及使用条件 | [真机入口](docs/g1_field_validation/README.md) |

## 三组展示仿真

在已安装 MuJoCo 和 NumPy 的环境中，从仓库根目录执行：

```bash
python tools/experiments/thesis_walk13/replay.py D01
python tools/experiments/thesis_walk13/replay.py D02
python tools/experiments/thesis_walk13/replay.py D03
```

- D01：右臂位置 PID。
- D02：右臂 No-Preview MPC，保持当前运动向量，并按当前角速度外推未来姿态。
- D03：右臂 Disturbance-Preview MPC，使用固定任务模板预览。

三组左臂均为名义 Kp=20、Kd=1，双臂零中立位，75 mm／250 g 圆环负载。
回放直接读取保存的物理轨迹，执行正运动学与绘图，不重算控制、不连接机器人。
鼠标旋转／缩放，空格暂停，R 重播，G 切换竖直参考线。
仿真正面放大与斜前方全身画面应使用同一条轨迹、同一倍速；参考线不放大实际角度。
当前三组均为 10 s 前进＋3 s 停车观察；完整指标见[结果报告](docs/simulation/experiments/20261117_walk10_stop3/README.md)。旧仿真数据和专用试验脚本已按要求清理。

## 仿真和真机的边界

| 内容 | 仿真 | 已完成的真机实验 |
| --- | --- | --- |
| 下肢 | 仓库 Torch 策略 `policy/motion.pt`；名义周期 0.8 s | PC1 内置运控；策略不可获取 |
| 扰动预测 | 固定 0–13 s 任务模板；已知 10 s 停车 | 固定 H0 坐标系中的学习预测器 |
| MPC | 6 ms、9 区间，q/dq 状态、ddq 输入 | 同类结构，独立的现场实现与配置 |
| 执行 | MuJoCo 正动力学候选检查与 C++ worker | 实测移动基座输入的耦合右臂五自由度模型，M*a+b |
| 评价 | 仿真状态和模型量 | 关节／IMU重建、统一离线滤波；不是独立末端传感器真值 |

D02 仍有状态反馈、当前 IMU 扰动和逆动力学补偿，不能称为开环或“全部前馈关闭”。
仿真数字不能直接当作真机性能预报，同名权重和 PD 增益也不保证相同闭环响应。
同步仿真中的计算长尾不会自动注入物理时间；真机存在少量 6 ms 超期，不宣称硬实时认证。

## 仓库布局

```text
main_sim.py / arm_*.py       仿真及控制器
configs/                    有效配置与历史复现实验配置
resources/ / policy/        物理模型、网格、下肢策略
assets/                     已冻结的真机扰动预测器
cpp/ / robot_model_backend/ 原生动力学、模型接口
right_arm_runtime/          仿真执行及独立 shadow/HIL 边界
tools/experiments/           仿真研究与展示入口
tools/g1_commissioning/      实际真机实验、离线分析、测试
docs/                       使用、算法、结果与精选真机归档
evaluation/                 本机生成数据，Git 忽略
```

完整本机数据、手机录像、SDK 安装和编译产物不随克隆提供；精选原始日志和展示回放已提交。
[清理记录](docs/history/REPOSITORY_CLEANUP_20261103.md)给出了删除项和恢复版本。
