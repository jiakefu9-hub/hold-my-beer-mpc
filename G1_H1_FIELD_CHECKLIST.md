# Unitree G1 H1 现场速查表

范围：**只收状态；不运行 MPC；不切模式；不取得 arm ownership；不发布命令。**
完整解释和故障树见 [G1_H1_FIELD_RUNBOOK.md](G1_H1_FIELD_RUNBOOK.md)。

## 任何一项不满足就 NO-GO

- [ ] 两人到场：一人持遥控/看机器人，一人操作电脑；STOP 口令已约定。
- [ ] 已按目标固件/遥控器实际说明确认阻尼、独立急停（若有）与关机操作；知道阻尼会
  让机器人下落，不把它当成硬件 E-stop。
- [ ] 实物/订单确认为支持二次开发的 G1 EDU/EDU+；普通 G1 直接 NO-GO。
- [ ] 已记录 DOF、Arm、revision、序列号、固件和 App Machine Type；与 23-DOF Rev
  1.0/Arm5 不一致时不把 capture 判为本项目契约通过。
- [ ] G1 无损伤/异物/报警，原装电池与遥控器有电；合格吊架脚轮锁止、两肩吊点牢固、
  双脚完全离地，四肢不缠绕，周围至少 2 m 净空。
- [ ] 一台开发机直连一台 G1；无其他开发机、控制例程、teleop 或 domain-0 仿真。
- [ ] 不进入 debug/develop mode，不关闭 factory locomotion/sport service。
- [ ] 只运行 `run_hardware_state_inspection.sh`；不运行 `run_hardware_shadow.sh` 或任何
  Unitree motion/arm/low-level 示例。

## 1. 主机预检

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
git status --short --branch
git rev-parse HEAD
git -C /home/fjk/g1_ws/unitree_sdk2 rev-parse HEAD

PYTHONPYCACHEPREFIX=/tmp/hold-my-beer-mpc-pycache \
  /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  run_hardware_shadow.py \
  --controller-config configs/g1.yaml \
  --hardware-config configs/g1_hardware_shadow.yaml \
  --check-config-discovery

taskset -pc $$
taskset -c 5 true
taskset -c 7 true
pgrep -af 'unitree|ros2|cyclonedds|teleop|lowcmd|arm_sdk|mujoco' || true
env | rg '^(G1_MPC_PYTHON|UNITREE_SDK2_DIR|UNITREE_STATE_BRIDGE_BUILD_DIR|UNITREE_INGRESS_SESSION_NONCE|CYCLONEDDS_URI)=' || true
```

- [ ] discovery config 输出 `PASS`。
- [ ] SDK2 为已审计的 `fa925bf...`，runtime/config 没有未复核修改。
- [ ] CPU 5/7 可用；否则记下两个 allowed CPU，稍后显式传参。
- [ ] 已人工确认没有其他可能发布机器人命令的进程。

## 2. 专用网络

插 G1 专用线前后对照：

```bash
ip -br link
ip -br -4 addr
nmcli -t -f DEVICE,TYPE,STATE,CONNECTION device status
```

在 Ubuntu Network 设置中把**人工追线确认的专用 NIC**设为：

```text
IPv4 manual: 192.168.123.99/24
Gateway: blank
DNS: blank
No default route / no sharing / no bridge
```

```bash
IFACE=replace_with_verified_g1_interface
cat "/sys/class/net/$IFACE/carrier"
ip -br -4 addr show dev "$IFACE"
ip route get 192.168.123.161
ping -I "$IFACE" -c 3 -W 1 192.168.123.161
```

- [ ] carrier=`1`。
- [ ] 本机为 `192.168.123.99/24`。
- [ ] `.161` route 直接走 `$IFACE`，无普通网关。
- [ ] PC1 ping 成功。失败时不启动 launcher。

## 3. 安全上电

- [ ] 机器人由合格吊架完全吊起，肩部吊绳受力、双脚离地；人员远离夹点/下落区。
- [ ] 先开启已绑定遥控器并确认 DL/连接，再按实物当前官方说明装电池、短按一次后
  长按至少 2 秒，等待约 1 分钟进入零力矩。
- [ ] 无红色异常/报警；不按 `L2+Up`、`R2+A`、`L2+R2`、`L2+A` 或其他模式键。

## 4. 唯一运行命令

替换 4 个变量；占位值不应直接运行：

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
IFACE=replace_with_verified_g1_interface
BRIDGE_CPU=5
INSPECT_CPU=7
GROUP=h1_g1_serial4_yyyymmdd_hhmm

env -u UNITREE_INGRESS_SESSION_NONCE -u CYCLONEDDS_URI \
  G1_MPC_PYTHON=/home/fjk/miniforge3/envs/g1_mpc/bin/python \
  UNITREE_SDK2_DIR=/home/fjk/g1_ws/unitree_sdk2 \
  UNITREE_STATE_BRIDGE_BUILD_DIR="/tmp/hold-my-beer-mpc-h1-state-only-$GROUP" \
  ./tools/realtime/run_hardware_state_inspection.sh "$IFACE" \
  --bridge-cpu "$BRIDGE_CPU" \
  --inspect-cpu "$INSPECT_CPU" \
  --duration-s 10 \
  --inspect-samples 500 \
  --group "$GROUP"
```

机器人有任何非预期动作/报警：现场操作员执行已验证的 emergency/damping；电脑操作员
Ctrl-C。先确保物理安全，不在站立时直接断电。

## 5. PASS 后人工验收

```bash
SESSION="evaluation/hardware_shadow/state_inspection/$GROUP"
find "$SESSION" -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort
sed -n '1,240p' "$SESSION/summary.json"
sed -n '1,200p' "$SESSION/state_bridge_summary.json"
wc -l "$SESSION/raw_state_trace.jsonl"
```

- [ ] 终端显示 `read-only inspection: PASS`。
- [ ] raw trace 为 500 行，5 个 evidence 文件完整。
- [ ] `output_capability=absent`，controller/predictor 均 false，publish count=0。
- [ ] protocol=3；observed/expected/bridge-summary 三处 nonce 是同一非零值；required
  flags=true；max skew <=5 ms。
- [ ] LowState valid、torso IMU、paired count 均覆盖 500；CRC/skew rejection 已记录。
- [ ] bridge 记录了最后一个 version；整段 mode 稳定；timestamp/tick 单调；state age
  <=20 ms（当前 trace 不能证明 version 在整段稳定）。
- [ ] 不用 `mode_machine_matches_reference` 单独判 PASS；以 App Machine Type、实物和
  对应固件/模型资料为准。
- [ ] quaternion norm 接近 1；q/dq/tau/temperature 无明显异常；22..26 与右臂姿态大致相符。
- [ ] 机器人/Explore 无报警，全程无模式、ownership 或动作变化。

脚本 PASS **不等于** model/index/sign/mode/IMU contract verified。任何 CRC rejection、
异常 mode、跳变、IMU/温度疑点都标为 REVIEW，用新 group 重跑；不放宽 gate。

## 6. 失败快速定位

| bridge counters | 首查方向 |
| --- | --- |
| LowState=0，torso=0 | NIC/IP/route、PC1 ping、DDS domain 0、multicast/firewall、目标 SDK 支持 |
| received>0，CRC-valid=0 | SDK/IDL/固件/type 不匹配；禁止关闭 CRC |
| CRC-valid>0，torso=0 | 目标是否发布 `rt/secondary_imu`；禁止用 pelvis IMU 替代 |
| 两路>0，paired=0 | host-arrival skew、CPU/负载、丢包；禁止放宽 5 ms |
| paired>0，collector incomplete | state age/rate、CPU；可增 duration，不减 500 samples |
| taskset fail | 从 `taskset -pc $$` 选两个 allowed CPU 并传参 |

失败目录可能只有 inspection/bridge logs 和 bridge summary；没有 raw trace/summary 就不是
有效 capture。保留旧目录，换唯一 group，不覆盖证据，不进 debug mode、不换 topic 猜测。

## 7. 收尾并停止

```bash
pgrep -af unitree_arm_state_bridge || true
ls /dev/shm/g1_state_inspection_* 2>/dev/null || true
```

- [ ] 复制整个 `$SESSION` 到独立存储并保存 hashes；`evaluation/` 被 Git 忽略。
- [ ] 保持完全吊装和绳索受力，按当前官方说明进入所需阻尼/关机前状态并正常关机。
- [ ] 断开 G1 网线、停用专用 profile、恢复普通 LAN，检查默认路由。
- [ ] 写下结论：`ACCEPTED CAPTURE / REVIEW / FAIL CLOSED / INCIDENT`。
- [ ] **到此停止：不改 YAML，不运行 H2/H3/MPC，不启用任何输出。**
