# 五条行走轨迹：关键状态图集

2026-09-18。只使用本地保存的五条有效轨迹，不连接机器人、不输出控制命令。

## 从哪里打开

生成后的本地入口：

- [浏览器图集：22 张图](../../evaluation/hardware_shadow/commissioning/walk_predictor_study_20260918/figures/index.html)
- [全部图片 PDF：22 页](../../evaluation/hardware_shadow/commissioning/walk_predictor_study_20260918/figures/walk_signals.pdf)
- [来源哈希、图名及处理清单](../../evaluation/hardware_shadow/commissioning/walk_predictor_study_20260918/figures/manifest.json)

这些生成物在被 Git 忽略的 `evaluation/` 内，不随代码自动同步；需要分享时，把 `figures/` 整个文件夹或 PDF 发给对方即可。

重画命令（仓库根目录；需要 NumPy、SciPy、Matplotlib，不需要 Unitree SDK）：

```bash
MPLCONFIGDIR=/tmp/g1-walk-gallery-mpl /home/fjk/miniforge3/envs/g1_mpc/bin/python tools/g1_commissioning/plot_walk_signals.py
```

程序读取此前[五条轨迹审计](WALK_DATASET_AUDIT.md)生成的 `trial01.npz` 至 `trial05.npz`，不改原始日志或提取文件。

## 图里有什么

每张图用相同五种颜色表示五条完整轨迹。左列看整个 0–21 秒任务，右列放大同一段 9–11.2 秒。
左右腿在同一张图中上下排列，便于看同一关节的左右交替关系。

| 组别 | 数量 | 内容 |
| --- | ---: | --- |
| 下肢关节角 | 6 | 双侧髋 pitch/roll/yaw、膝、踝 pitch/roll；单位为度 |
| 主要下肢关节速度 | 3 | 双侧髋 pitch、膝、踝 pitch；单位为度/秒 |
| 世界系身体线加速度 | 3 | X/Y/Z；单位为 m/s² |
| 世界系身体角速度 | 3 | X/Y/Z；单位为 rad/s |
| 世界系身体角加速度估计 | 3 | X/Y/Z；单位为 rad/s² |
| 身体世界系姿态 | 3 | roll/pitch/yaw；单位为度 |
| 现有周期标记输入 | 1 | 左髋 pitch − 右髋 pitch |

编号对应 `WALK_DATASET_AUDIT.md` 的第 1–5 条有效轨迹。
**第 1 条与其余几条约差半个步态周期，不能因为它在同一秒处方向相反就判为异常。**

## 怎么看，别误读

1. 横轴是以任务开始为零点的**电脑接收时间**，不是源端采样时间，也不是步态相位。
   图没有为了重合而移动、拉伸或对齐某条轨迹。
2. 竖线 3/5/15/18 秒对应升权完成、前进请求、停车请求、开始退权。灰区 5–15 秒是请求前进阶段，
   不是定位系统测得的移动区间；此前没有记录绝对行走距离。
3. 关节角不低通。关节速度、世界系线加速度和角速度的浓线是**因果 15 Hz 低通**，淡线是未低通值；
   双髋角度差用因果 10 Hz 低通。滤波有延迟，没有人为前移曲线。
4. 角加速度不是直接读取的独立传感器量；这里从滤波角速度后向差分，再低通。
   “姿态、角速度、线加速度、角加速度”是四组量，不是四个标量。
5. 世界系采用身体 IMU 的四元数；线加速度由加速度计读数旋转后扣除重力，未扣静态偏置。
   姿态曲线是**身体**的，不是水瓶末端的；最终姿态预测应使用旋转表示，不能把欧拉角任意当普通向量相加。
6. 曲线相似只是线索：同一个膝角，在抬腿和放腿时可能各出现一次，需要结合速度方向或短时历史。
   只有使用当前和过去数据预测未来、再到独立整条轨迹上检验，才能判断能否作为前馈输入。

五条水瓶条件一致；操作者报告吊绳可能有轻微拖拽。图只能反映这批实验条件，不能证明拖拽没有影响。
