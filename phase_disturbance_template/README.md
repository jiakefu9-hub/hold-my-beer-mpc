# Legacy phase disturbance template

本目录只保留旧的 H 系周期模板运行资产，用于当前只读 hardware shadow 的
`template` 兼容模式。正式 MuJoCo 冻结方案不使用这里的周期模板，而是使用
[`../disturbance_template/`](../disturbance_template/) 中按绝对任务时间查询的
Full-Task Template v2。

## 保留内容

`templates_heading_interval/` 是 0.8 s 步态周期上的 H 系模板：

- 400 个相位起点，间隔 2 ms；
- 每个起点同时保存瞬时节点量和随后 6 ms 的区间平均量；
- H 方向来自上一完整步态周期的 torso yaw 圆周平均；
- 保留 `raw`、`half_smoothed`、`fully_smoothed` 三种历史版本；
- 当前兼容配置选择 `raw`。

这套模板只描述稳定周期并按 phase 循环回绕，第一周期需要等待 H 系建立。它不包含
Full-Task Template 的启动、绝对任务时间、6.4 s 直接停车或 continuous-H 语义。

## 为什么仍然保留

hardware shadow 尚未迁移到 Full-Task Template v2，因此仍通过
[`../configs/g1.yaml`](../configs/g1.yaml) 的 `mpc_disturbance_template_dir` 读取此处。
旧的 W 系采集数据、转换脚本、2 ms MPC 对照模板和重复图表已经从当前工作树删除；
需要追溯时可从 Git 历史或 `checkpoint/full-task-v2-24ms-20260815` 恢复。
