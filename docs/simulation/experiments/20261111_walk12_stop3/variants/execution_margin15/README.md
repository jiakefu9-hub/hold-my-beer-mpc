# 当前选择：execution_margin15

三组均完整录制 15 s。执行候选验收阈值由 10 改为 15 rad/s²，MPC 规划加速度上限仍为 8 rad/s²；两种 MPC 共同使用该设置。原配置与失败记录保留在上两级目录。

详情、回退记录及三条回放命令见[总说明](../../README.md)。`manifest.json` 的 `complete=true` 表示三组录满；`all_acceptance_gates_pass=false` 保留模板预览组 2 次 QP 回退造成的严格验收失败。
