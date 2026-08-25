# Unitree G1 首次 H1 只读现场操作手册

状态基线：**2026-08-24；仅适用于 H1 state-only/read-only inspection**。

审计基线为仓库 `main` 的 `a065575e7a4799c8509b97f7a99dded12529bc46`，以及
`/home/fjk/g1_ws/unitree_sdk2` 的
`fa925bf6bb3fff439000266d70bde32eb5cd3597`。若代码、SDK、机器人固件或型号发生变化，
必须重新核对本文中的事实。现场一页版见
[G1_H1_FIELD_CHECKLIST.md](G1_H1_FIELD_CHECKLIST.md)。

## 1. 本次允许做什么

唯一目标是从目标 G1 收到真实 `rt/lowstate` 与 `rt/secondary_imu`，通过当前仓库的
state-only bridge 形成严格配对的 protocol-v3 状态，并保存只读证据。

本次明确不做：

- 不运行 MPC、predictor、完整 hardware shadow 或 `run_hardware_shadow.sh`；
- 不进入 debug/develop mode，不调用 `ReleaseMode`、`SelectMode` 或 motion switcher；
- 不关闭厂商 locomotion/sport service，不取得 arm ownership；
- 不运行 Unitree 的 arm、low-level control、teleoperation 或 sport-control 示例；
- 不创建、构建或运行 `rt/lowcmd`、`rt/arm_sdk` 或任何其他 command publisher；
- 不修改 `configs/g1_hardware_shadow.yaml` 中的 verification flags 或 mode whitelist；
- 不以本次结果开始 H2/H3、吊架控制或任何硬件输出。

如果现场需要上述任一步才能“让数据出现”，本次立即停在 H1，不把它当作排障捷径。

## 2. 先给结论：当前是否可以开始 H1

结论是：**软件侧已具备有条件开始 H1 的能力；整体现阶段仍是 NO-GO，直到现场四个
前置门全部确认。**

| 门 | 当前判断 | 开始前要求 |
| --- | --- | --- |
| 仓库输出隔离 | READY | 使用本文唯一入口；现场再次通过构建和 binary scan |
| 主机软件 | READY WITH CAVEAT | SDK/环境可构建；现场终端需确认可用 CPU affinity |
| G1 专网 | NOT READY | 当前有线接口仍为普通 `192.168.31.159/24`；必须确认 G1 专用 NIC 并设为 `192.168.123.99/24`、无网关 |
| 机器人身份与安全 | UNKNOWN | 确认 G1 EDU/EDU+、型号/固件/遥控器版本、实际紧急处置方法，以及合格吊架完全吊起 |
| 真实状态证据 | PARTIAL | 目前没有任何当前 protocol-v3 的有效真实样本 |

2026-08-23 00:39 的连接尝试在普通 `192.168.31.0/24` LAN 上得到 LowState/torso
IMU/paired state 全部为 0。它发生在当前 protocol-v3 commit 之前，目录中只有旧的
bridge log/summary，没有 session nonce、inspection log、raw trace 或 Python summary。
因此它只能证明那次连接失败，不能证明当前 H1 路径已在真机上运行。

另外，当前 Codex 执行环境的 cpuset 不包含默认 inspector CPU 7；这不代表用户自己的
现场终端也缺 CPU 7。必须在实际运行终端执行 `taskset` 检查，必要时通过 launcher
参数选择两个允许的 CPU。

## 3. 当前架构和安全边界

### 3.1 Simulation、H1、H2、H3 不是同一条已完成路径

```text
Simulation（已在冻结仿真条件下验证）
  main_sim.py / MuJoCo
    -> shared full-task control core
    -> RightArmSimProcess + C++ RNEA/MuJoCo mapper/executor
    -> final_tau -> d.ctrl -> mj_step

H1（本手册范围，真实状态只读）
  G1 domain 0
    -> rt/lowstate + rt/secondary_imu
    -> unitree_arm_state_bridge（subscriber only）
    -> CRC + 两路 host-arrival skew 配对 + session nonce
    -> protocol-v3 shared-memory state slot
    -> Python O_RDONLY + MAP_PRIVATE collector
    -> raw_state_trace.jsonl + summaries/logs

H2（尚未完成）
  H1 trace -> offline auditor + 人工型号/index/sign/tick/mode/IMU 契约冻结

H3（真实路径不存在）
  validated real state + authoritative TaskClockEvent
    -> full-task v2/continuous-H/MPC proposal only
```

仿真的 MuJoCo state、DDQ-to-torque mapper、`d.ctrl` 和 `mj_step()` 都不会进入 H1。
H1 Python collector 不实例化 predictor、MPC 或 command builder。

### 3.2 代码层的两道输出屏障

第一道屏障在 C++：

- `cpp/unitree_arm_adapter/src/state_bridge_main.cpp:16-20` 只包含 G1 state IDL、
  `ChannelFactory` 和 `ChannelSubscriber`；
- `state_bridge_main.cpp:30-31` 只使用 `rt/lowstate` 与 `rt/secondary_imu`；
- `state_bridge_main.cpp:385-401` 只创建两个 subscriber；
- `cpp/unitree_arm_adapter/CMakeLists.txt:50-53` 对
  `UNITREE_ARM_ADAPTER_BUILD_DDS=ON` 直接报错；当前仓库没有 command publisher target。

第二道屏障在 Python：

- `tools/realtime/run_hardware_state_inspection.sh:139-162` 强制 `BUILD_DDS=OFF`，并检查
  旧 DDS binary、command topic 和 `ChannelPublisher` 符号；
- `right_arm_runtime/unitree_shm.py:698-728` 用 `O_RDONLY` 打开 shared memory，再使用
  private copy-on-write mapping；意外 Python 写入不能到达共享对象；
- `run_hardware_shadow.py:133-297` 的 inspection 只检查/记录状态；summary 固定报告
  `controller_executed=false`、`predictor_executed=false`、`command_publish_count=0`。

注意：bridge 静态链接 `libunitree_sdk2.a`，并动态链接 CycloneDDS 等接收依赖。SDK 库
本身含有通用 DDS publisher machinery 符号，不等于该应用存在可达的
`ChannelPublisher`、`LowCmd`、`rt/arm_sdk` writer 或 command sink/topic；现场验收依据是
当前 target、应用源码、针对性的符号/topic scan 以及运行入口的组合，不能声称整个 ELF
字面上没有任何泛化的 publisher 相关符号。

launcher 的 binary scan 也只是额外防线：它只拒绝两个精确 command topic 字符串和
`ChannelPublisher` 标识，不能单独证明任意第三方库/进程没有 writer；因此源码/target
审计、唯一 launcher、进程排查和隔离专网四项缺一不可。

### 3.3 当前 H1/H2/H3 的精确状态

| 阶段 | 已完成 | 未完成/禁止声明 |
| --- | --- | --- |
| H0 / H1 软件准备 | protocol-v3、nonce、CRC、paired ingress、只读 collector、离线测试和 state-only build | 当前 protocol-v3 尚无真机运行证据 |
| H1 现场验收 | 仅有一次 0-sample 旧路径连接失败记录 | 没有 accepted real capture；仍为 `PARTIAL`、`hardware-unverified` |
| H2-prep | trace auditor 已实现并有 synthetic tests | 没有有效 H1 trace；型号/index/sign/tick/mode/IMU 均未 verified |
| H3-offline | synthetic/replay 的 full-task proposal path 有测试 | live H3 无权威 `TaskClockEvent`；现场 launcher 仍是 legacy template；不得运行 |
| Publisher-absent HIL | 2 ms supervisor、fake recording sink、receipt 已离线测试 | 不是 H1/H2/H3 实机证据；DDS/hardware write 始终为 0 |
| Hardware output | 无 | target/launcher 不存在且未授权 |

## 4. 资料可信度和版本处理

本文按以下优先级处理证据：

1. **当前仓库事实**：实际 launcher、编译目标、状态字段、失败条件和测试；
2. **目标机器人随附资料与 Unitree 当前官方资料**：型号、固件、物理操作和遥控器；
3. **Unitree 官方 GitHub/SDK 实现**：topic、IDL、示例中的 index 和网络接口语义；
4. **第三方真实经验**：只用于发现常见网络/DDS问题，不作为安全授权或型号契约。

本地 `local_reference/unitree_g1/` 的 10 份资料都是网页截图/扫描型 PDF，文本抽取几乎
为空。本次已逐页/逐图片块视觉阅读，覆盖全部 61 个 PDF 页及长页中拆出的截图块。
本地中文手册是 V1.4（2025-08-07），而在线当前英文手册已是 V1.6
（2026-08-12），必须把本地快照当作有版本的资料，而不是永远最新的说明。
特别要注意：手册和遥控说明存在多个固件/遥控版本，紧急/阻尼组合键历史上发生过变化。
现场只使用**目标机器人当前遥控器贴纸、Unitree Explore 中与该固件匹配的说明或随附
手册**；如果三者不一致，不上电，并联系 Unitree/供货方。Explore 在本次仅查看
Device/Data、温度和报警；不做校准、关节微调、解绑或 OTA。

## 5. 人员、场地和物品

第一次 session 至少安排两人：

- 现场操作员：只负责机器人、遥控器和官方上/下电流程；
- 计算机操作员：只负责网卡、终端、日志和停止程序。

两人开始前口头约定“STOP”口令；任何一人都可无条件停止。现场操作员全程持有已配对、
有电且已确认当前 emergency/damping 操作的遥控器，不兼任敲命令。

准备：

- 目标 G1、原装且状态正常的电池/充电器、已配对遥控器；
- 额定载荷足够的官方认可保护架/吊架、肩部专用吊点和合格绳具；第一次 H1 必须让
  双脚完全离地，脚轮锁止；
- 一根确认良好的网线和一个专用于 G1 的 USB Ethernet adapter；
- 当前 Ubuntu 主机、电源、离线可用的仓库和 SDK2；
- 记录机器人型号、序列号、固件、遥控器版本的相机或纸笔；
- 室内 0--40°C、平整、防滑、无水/油/砂石的区域；机器人四周至少 2 m 净空，无
  无关人员、动物、障碍或台阶。

遵循 Unitree 随机手册检查运输损伤、螺钉/连接器/风扇、异物、关节/绳索缠绕、电池
卡扣和电量。机器人是约 35 kg 的大功率设备；身体不得处在关节夹点、脚下或可能跌落
方向。不要把断电当作无承托机器人的第一急停方式，因为失去支撑力本身会造成跌倒。

## 6. 到现场前：主机离线预检

### 6.1 冻结并记录代码

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
git status --short --branch
git rev-parse HEAD
git -C /home/fjk/g1_ws/unitree_sdk2 status --short --branch
git -C /home/fjk/g1_ws/unitree_sdk2 rev-parse HEAD
```

本次审计值分别是 `a065575...` 和 `fa925bf...`。文档改动可以存在，但如果 `.py`、
`.cpp`、`.hpp`、`.sh`、`.yaml` 或 CMake 文件与审计基线不同，先停止并复核，不在机器人
旁临时修代码。

确认 discovery config 能加载：

```bash
PYTHONPYCACHEPREFIX=/tmp/hold-my-beer-mpc-pycache \
  /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  run_hardware_shadow.py \
  --controller-config configs/g1.yaml \
  --hardware-config configs/g1_hardware_shadow.yaml \
  --check-config-discovery
```

预期为 `HARDWARE_SHADOW_DISCOVERY_CONFIG: PASS`。不要改用 `--check-config`；后者要求
H2 才能确认的 flags，当前按设计应 fail closed。

### 6.2 确认 CPU 可用

```bash
taskset -pc $$
taskset -c 5 true
taskset -c 7 true
```

两条 `taskset -c` 都成功时可用默认 CPU 5/7。若任一失败，从第一条输出中选两个允许的
CPU，现场命令显式传 `--bridge-cpu` 和 `--inspect-cpu`。H1 不要求 PREEMPT_RT、
`SCHED_RR` 或 CPU 7；CPU affinity 只需可执行且避免明显重负载。

### 6.3 排除其他 DDS/机器人程序

关闭本机的 Unitree 官方控制例程、ROS2/Isaac/MuJoCo DDS 仿真、teleop 和其他机器人
进程，并断开其他外部开发机。只读检查器不能阻止网络上另一个进程向相同 domain/topic
发布命令。

```bash
pgrep -af 'unitree|ros2|cyclonedds|teleop|lowcmd|arm_sdk|mujoco' || true
env | rg '^(G1_MPC_PYTHON|UNITREE_SDK2_DIR|UNITREE_STATE_BRIDGE_BUILD_DIR|UNITREE_INGRESS_SESSION_NONCE|CYCLONEDDS_URI)=' || true
```

逐项人工判断输出。不要因为进程名“不像本项目”就忽略它。最安全的拓扑是一台开发机
通过专用线连接一台 G1，不把 G1 的 `192.168.123.0/24` 接入普通 LAN 或互联网。
记录任何环境覆盖；下面批准的现场命令会固定前三个项目路径，并清除遗留 nonce 与
ROS/CycloneDDS 配置，避免复用旧 session identity 或选错网卡。

## 7. 机器人身份与物理 GO/NO-GO

在上电前记录：

| 字段 | 现场填写 |
| --- | --- |
| 产品/序列号标签 |  |
| G1 / G1 EDU / EDU+ |  |
| 23 DOF / 29 DOF / 其他 |  |
| Arm5 / Arm7 / 手型 |  |
| 硬件 revision |  |
| 系统/运动固件版本 |  |
| Unitree Explore 版本 |  |
| Explore Machine Type |  |
| 遥控器型号/说明版本 |  |
| 已验证的 emergency/damping 操作 |  |
| 使用的支撑方式 |  |

本仓库声明的是 `g1_23dof_rev_1_0`、Arm5、右臂 slots 22..26；Unitree 当前
`unitree_ros` 表中该特定旧 revision 的参考 `mode_machine` 为 4，但同一张当前表也列出
较新的 23-DOF mode 10，以及多个 29-DOF mode。本地官方截图之间还存在 1/2/9 与
4/5/6 两套说明。因此这里的“声明”和“参考”都不是目标机器人事实，现场必须从 Explore
的 Device -> Data -> Robot -> Machine Type 和目标版本资料原样记录。Python summary
中的 `mode_machine_matches_reference` 只是当前仓库对 4 的诊断比较，**不是 H1 PASS
条件**。

普通 G1 的官方规格标为不支持 secondary development，G1 EDU/EDU+ 才支持。若铭牌/
订单不是 EDU 系列，本次直接 NO-GO 并向 Unitree/供货方确认；不通过“试一下 DDS”推断。
若 EDU 实物明确不是仓库声明的 23-DOF Rev 1.0/Arm5，可保留另行标注的只读发现，但
不得把结果记为本项目契约通过；第一次操作建议先做型号评审。

物理 GO 条件：

- 无运输损伤、液体/异物或松动，电池正确锁止且电量足够；
- 遥控器已绑定、DL/连接指示正常，现场操作员已在该固件说明中确认
  emergency/damping 与正常关机；
- 合格吊架脚轮锁止，绳索穿过两侧肩部专用吊点并可靠固定，机器人完全吊起、双脚
  明确离地，四肢自然且无缠绕；
- 场地满足至少 2 m 净空，所有人远离夹点和下落区域；
- 不需要进入 debug/develop mode，也不需要关闭厂商运动服务。

任一项不满足即 NO-GO。

## 8. 建立专用 G1 网络

### 8.1 识别物理接口，绝不猜网卡名

上电前、插线前后各运行一次：

```bash
ip -br link
ip -br -4 addr
nmcli -t -f DEVICE,TYPE,STATE,CONNECTION device status
```

插入 G1 官方指定的交换机/开发网口后，找到 carrier 从 0 变 1 的专用 Ethernet
interface。不要选择 Wi-Fi、普通办公 LAN 或只因名字含 `enx` 就认定它是 G1。

截至 2026-08-24 的主机快照是：

- `enx6c1ff701509c` 有 carrier，但地址是普通 LAN 的 `192.168.31.159/24`；
- `enxc6524f63a02d` 无 carrier、无 IPv4。

下一次很可能使用后者，但只有插线后的 carrier、物理追线和地址检查才能确认。

### 8.2 配置静态地址

Unitree 官方 quick-development/ROS2 资料建议把外部电脑设在
`192.168.123.0/24`，常用 `192.168.123.99/24`；PC1/运动控制机参考地址为
`192.168.123.161`，G1 EDU PC2 参考地址为 `192.168.123.164`。机器人侧没有可依赖的
DHCP，因此不要等待自动地址。

推荐使用 Ubuntu Settings -> Network -> 对应专用有线连接 -> IPv4 -> Manual：

```text
Address: 192.168.123.99
Netmask: 255.255.255.0  （或 /24）
Gateway: 留空
DNS: 留空
```

启用 “Use this connection only for resources on its network”（或等价的
`never-default` 设置）。最终目标是该连接 **never-default、无 gateway、无 DNS、无
connection sharing/bridge**。
不要把 PC 地址设成 `.161`、`.164` 或任何已知机器人设备地址。

配置后令 `IFACE` 只表示人工核对过的专用网卡：

```bash
IFACE=replace_with_verified_g1_interface
cat "/sys/class/net/$IFACE/carrier"
ip -br -4 addr show dev "$IFACE"
ip route get 192.168.123.161
ping -I "$IFACE" -c 3 -W 1 192.168.123.161
```

GO 结果应是：carrier `1`；本机 source 为 `192.168.123.99`；到 `.161` 的 route 直接走
该 interface、没有 `via` 普通路由器；三次 ping 有回复。对 G1 EDU 可额外 ping `.164`，
但 H1 不需要 SSH 登录 PC2。

ping 只证明 L3 连通，不证明 CycloneDDS multicast discovery、topic/type/QoS 或消息内容
正确。ping 失败时禁止启动 H1 launcher。

## 9. 安全上电

物理操作必须以实物随附/当前 Unitree 说明为准。当前 V1.4/V1.6 手册对应的通用顺序是：

1. 合格吊架已锁止，肩部吊绳受力，机器人完全离地；现场操作员持遥控器；
2. 先按当前说明开启已绑定的遥控器，确认 DL/连接状态；
3. 按正确方向装入原装电池并确认卡扣；
4. 对机器人电池键短按一次、再长按至少 2 秒；
5. 等待约 1 分钟完成初始化，预期关节进入零力矩；不触碰关节，不进入夹点；
6. 确认机器人无红色异常/报警。H1 不要求站立、行走或 debug/develop，也不按任何
   准备、运动、debug 或诊断姿态组合键。

当前手册中的 `L2+Up`、`R2+A`、`L2+R2`、`L2+A` 分别会涉及准备/运动/debug/诊断
姿态，本次全部禁止。当前手册把长按 `L2+B` 至少 5 秒描述为进入阻尼且机器人会缓慢
倒下；它是会失去站立支撑的软件保护动作，**不是独立硬件 E-stop**。只能在吊架已可靠
承重且现场人员已按目标固件确认时，把它列入异常处置。若实物有独立物理急停，其位置、
作用和复位流程必须由 Unitree/现场负责人说明，不能猜测。

如果实际手册要求的按键、LED、蜂鸣或姿态与本文不同，以该机器人版本为准并记录差异。
如果无法确认安全状态，关机/保持支撑并联系 Unitree，不尝试 SDK 控制例程。

## 10. 运行唯一批准的 H1 入口

先在现场终端确定两个可用 CPU。下面的变量必须人工替换；保留占位值会安全失败：

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

不要使用 `sudo` 运行整个 launcher。网络地址需要授权时在 Ubuntu 网络设置中单独完成；
subscriber、shared memory 和 evidence 应以当前普通用户运行。

launcher 会依次：

1. 检查 interface 存在、carrier、CPU、Python、SDK 和唯一 evidence 目录；
2. 用未验证但结构固定的 YAML 做 discovery-only config check；
3. 强制 `BUILD_DDS=OFF`，只构建 `unitree_arm_state_bridge`；
4. 扫描旧 command binary、command topic、unresolved library 和应用层
   `ChannelPublisher` 标识；
5. 在没有环境覆盖时生成随机的非零 session nonce，启动 domain 0 的两个 state
   subscriber；本文用 `env -u` 明确排除了现场终端中遗留 nonce 和 DDS XML 的复用；
6. CRC 校验 LowState；只有两路都为新样本且 callback host-arrival skew 不超过 5 ms 时
   才写 paired state；
7. Python 以只读/private mapping 收集 500 个 fresh、unique、finite、nonce/flags/time/tick
   单调的 paired samples；
8. 无论成功或失败都终止 bridge；正常路径删除本次唯一名 shared memory。

运行期间计算机操作员只看终端；现场操作员只看机器人。机器人任何非预期动作、报警、
遥控掉线、支撑异常或人员进入区域，立即由现场操作员执行**已按实际版本确认**的
emergency/damping 操作，计算机操作员按 Ctrl-C。先保证机器人受支撑和人员安全，再处理
进程；不要在站立时直接拔电池。

## 11. 程序 PASS 与 H1 人工接受是两回事

### 11.1 程序级 PASS

预期终端末尾为：

```text
read-only inspection: PASS
evidence: .../evaluation/hardware_shadow/state_inspection/GROUP
```

检查文件：

```bash
SESSION="evaluation/hardware_shadow/state_inspection/$GROUP"
find "$SESSION" -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort
sed -n '1,240p' "$SESSION/summary.json"
sed -n '1,200p' "$SESSION/state_bridge_summary.json"
```

完整成功目录至少应有：

- `inspection.log`；
- `state_bridge.log`；
- `state_bridge_summary.json`；
- `raw_state_trace.jsonl`，恰有 500 行；
- `summary.json`。

`summary.json` 至少人工确认：

| 字段 | 要求 |
| --- | --- |
| `mode` | `unitree_lowstate_inspection_only` |
| `output_capability` | `absent` |
| `protocol_version` | `3` |
| `controller_executed` / `predictor_executed` | 都是 `false` |
| `command_publish_count` | `0` |
| `sample_count` / `complete_requested_sample_count` | `500` / `true` |
| nonce | observed 只有 expected 的一个非零值，且 bridge summary 的 nonce 与它相等 |
| `required_ingress_flags_present` | `true` |
| `source_skew_ns_max` | 不大于 `5000000` |
| state age | 每样本已受 20 ms freshness gate；summary max 应小于等于 20 ms |
| sample/timestamp/tick | collector 已拒绝重复或倒退 |

`state_bridge_summary.json` 至少确认：

- `output_capability` 为 `absent`；
- `network_interface` 正是人工核对的接口；
- `ingress_session_nonce` 等于 `summary.json` 的 `expected_ingress_session_nonce`；
- LowState received/CRC-valid、torso IMU 和 paired count 都大于 0 且覆盖 500 条 trace；
- CRC rejected、rejected skew 和 max accepted skew 的实际值已记录；
- bridge summary 记录 last version/last modes；Python summary 的
  `observed_mode_pr`/`observed_mode_machine` 记录整段不同观察值。

任何 CRC rejection 或大量 skew rejection 即使脚本最终 PASS，也标为 REVIEW，检查线缆、
SDK/固件和主机负载后用新 group 重复；不要关闭 CRC 或放宽 5 ms gate。

### 11.2 人工 H1 接受

程序 PASS 只证明当前软件收齐了一组满足窄数据完整性条件的样本。它**不证明**：

- 物理机器人就是 `g1_23dof_rev_1_0`；
- slots 22..26 的每一轴、正负号和单位已经验证；
- `mode_machine=4` 或观察到的 `mode_pr` 已可加入 whitelist；
- quaternion 是 wxyz/world-from-IMU，或加速度是 specific force；
- `rt/secondary_imu` 的 frame、原点、重力/杆臂语义等于 `imu_in_torso`；
- 机器人满足主动控制的安全条件。

人工审阅至少记录：

- `mode_pr`、`mode_machine` 是否在整段 capture 中稳定；
- source dt、robot tick delta、state age 和 pair skew 的 min/mean/max；
- quaternion norm 是否接近 1；静止时 gyro、accelerometer 的方向和模长仅作候选观察；
- 35-slot q/dq/tau/temperature 是否有限、无明显跳变；22..26 是否与已知右臂姿态大致一致；
- Unitree Explore/机器人本体是否有温度、电机、通信或电池报警；
- 全程是否确实无本项目 command publish、无模式/ownership 改变和无非预期动作。

当前 protocol-v3 trace 不保存每电机 `mode/motorstate/voltage/sensor/reserve`，也不逐样本
保存 `LowState.version`；bridge summary 只保存最后一个 version。因此 H1 不是完整的
机器人健康检查或固件稳定性证明。厂商诊断与 H1 evidence 必须分别保留。

满足所有项目后，可把结论写成：

> H1 real state capture accepted for offline review; model/index/sign/mode/IMU contracts remain
> unverified; no output capability or command publication was present.

不得写成“hardware verified”“shadow controller passed”或“ready for command output”。

## 12. 安全结束、备份和恢复网络

1. 确认 launcher 已退出；
2. 检查无残留 bridge：

   ```bash
   pgrep -af unitree_arm_state_bridge || true
   ls /dev/shm/g1_state_inspection_* 2>/dev/null || true
   ```

3. 若是本次 launcher 自己的进程仍在，先对准确 PID 使用 `kill -TERM PID`；不要 broad
   `pkill`，不要删除不明 shared-memory object；
4. 复制整个 `$SESSION` 到独立存储并计算 SHA256。`evaluation/` 被 Git 忽略，不能依赖
   `git status` 保存现场证据；
5. 保持机器人完全吊装、肩部吊绳持续受力。由现场操作员按**当前机器人官方说明**进入
   该版本要求的阻尼/关机前状态，再执行正常关机；断电后关节不保持力矩；
6. 断开 G1 网线，停用 G1 专用 NetworkManager profile，恢复原普通 LAN；确认默认路由
   没有残留在 `192.168.123.0/24` 接口；
7. 第一次 session 在这里停止。不要现场修改 YAML flags，不运行 H2/H3 或 MPC。

## 13. 故障树

### A. launcher 在 DDS 前就失败

| 现象 | 检查 | 处理 |
| --- | --- | --- |
| interface does not exist | `ip -br link` | 重新物理追线并填实际 interface；不猜名称 |
| no carrier | sysfs carrier、接口灯、网线/adapter/机器人端口 | 重插或更换已知良好部件；仍无 carrier 则停止 |
| requested CPU unavailable | `taskset -pc $$` | 选两个 allowed CPU 并显式传参；不需要 PREEMPT_RT |
| SDK source not found | SDK 路径和 HEAD | 恢复审计过的 SDK；不要现场临时换未知 fork |
| config discovery fail | YAML diff、Python 环境 | 停止并离线复核；不要把 verification flags 改 true |
| output-capable binary present | build directory 内容 | 不运行；使用新的 state-only build dir 并审计来源 |
| unresolved shared libraries / symbol scan fail | `ldd`、build log | 不绕过；修复离线环境后重做预检 |
| session directory exists | group 名和旧 evidence | 保留旧目录，换唯一 group；不覆盖/删除证据 |

### B. ping 失败

按顺序检查：

1. G1 已完成启动、线接在官方指定 switch/development port；
2. carrier 为 1；
3. 本机专用 NIC 是 `192.168.123.99/24`，不是 `.31`、DHCP 或 link-local；
4. `.161` route 直接走该 NIC，没有 gateway；
5. 没有 IP 冲突、VLAN/bridge/connection sharing；
6. 更换已知良好网线/adapter/端口。

ping 不通不启动 DDS，不尝试 debug mode、SSH 修改机器人或控制示例。

### C. ping 通，但 LowState 和 torso IMU 都为 0

这是 DDS discovery/选择接口层，不是 MPC 问题：

- 确认传给 launcher 的 interface 与 `.161` route 完全一致；
- bridge 固定 domain ID 0；检查机器人当前 SDK/DDS domain 与固件兼容性；
- 确认同一 L2、multicast 未被交换机/VLAN/firewall 阻断；
- 确认没有其他 Unitree/simulation DDS 程序污染 domain 0；
- 只读查看 `state_bridge.log` 和 summary；必要时在隔离网线环境用
  `sudo tcpdump -ni "$IFACE" udp -c 100` 确认是否有 UDP 流量；抓包不发布消息；
- 查看 `sudo ufw status verbose`，不要为了测试永久关闭 firewall；如需规则，单独设计
  限定到 G1 interface/subnet 的临时规则；
- 核对目标是否为支持二次开发的 G1 EDU，以及 Unitree SDK/固件兼容矩阵。

仍为 0 时保存 evidence，联系 Unitree 并提供型号、序列号、固件、接口/IP/route、
ping、bridge log/summary 和 SDK SHA。不要换 topic 猜测。

### D. LowState > 0，但 CRC-valid = 0

最可能是 IDL/SDK/固件不匹配或数据不是预期 G1 `unitree_hg::LowState_`。核对 SDK
`fa925bf...`、target firmware 和 topic/type；保存首轮 counters。**禁止关闭 CRC**。

### E. LowState valid > 0，但 torso IMU = 0

`rt/secondary_imu` 是否存在于目标型号/固件仍是现场未知项。当前仓库不允许用
`LowState.imu_state`（pelvis candidate）冒充 torso IMU，也不允许现场改 topic。保存证据并
向 Unitree 确认该型号的 torso IMU topic/type/发布条件；不要进入 debug mode试探。

### F. 两路都有数据，但 paired state 为 0 或 skew rejection 很多

5 ms 是两次 callback 的**本机到达时间差**，不是机器人源时钟同步或物理测量年龄。
检查主机负载、CPU affinity、线缆、DDS丢包和两 topic 的实际 rate；关闭无关进程并重跑。
不要把 5 ms 调大来获得“PASS”。

### G. 有 paired state，但 collector incomplete/stale

检查 `inspection.log` 的具体 fail-closed 原因、bridge paired count、CPU 可用性和 state age。
可以保持 500 samples、把 `--duration-s` 增至 20 后用新 group 重试；不要减少完整性要求、
增加 20 ms freshness 或跳过 tick/nonce/flags 检查。

当前 collector 只在完整成功后写 `raw_state_trace.jsonl` 和 `summary.json`。中途失败通常
只留下 `inspection.log`、`state_bridge.log` 和 graceful bridge summary；部分样本不会持久化。
这是当前实现限制，不要把缺失 raw trace 的目录当作有效 capture。

### H. 机器人有非预期动作

这不是“继续观察”的情况：现场操作员立即执行已验证的 emergency/damping 动作并保证
支撑，计算机操作员 Ctrl-C。记录当时所有运行进程和网络参与者，隔离机器人后再调查。
由于本项目 H1 应用层没有 command publisher path，必须重点排查厂商状态机、遥控输入、
其他主机/进程和 domain 0 上的 publisher；在原因闭环前不得重试。

### I. Explore/机器人显示故障码

本地官方错误释义中，首次 H1 尤其要原样记录：`0x2` 下层反馈超时、`0x4` IMU 反馈
超时、`0x8` 电机反馈超时、`0x10` 电池反馈超时、`0x20` 物理遥控反馈超时、`0x80`
软启动错误、`0x100` 电机状态错误、`0x1000` 软急停、`0x40000` 髋部 IMU 超时、
`0x80000` 控制器检测电池欠压、`0x100000` 控制器检测电机欠压。任一异常都保存截图/
原始值并停止；本次不清错、不校准、不升级固件。电机子状态还有过流、过/欠压、芯片/
MOS/绕组温度、编码器、校准、通信和驱动版本等位，当前 H1 trace 并未采集这些位，必须
从厂商状态页另行保留。

## 14. H1 后仍需现场确认的未知项

即使获得一次完整 capture，以下仍不能自动冻结：

- G1 EDU 的准确 SKU、23/29 DOF、Rev、Arm5/Arm7 和固件/SDK兼容性；
- PC1/PC2 实际地址、外部 port 拓扑和目标 DDS domain；
- `rt/lowstate`、`rt/secondary_imu` 的实际 type、rate、QoS 和不同 factory mode 下可用性；
- `LowState.version`、`mode_pr`、`mode_machine`、uint32 tick 的稳定/频率/wrap 语义；
- 35-slot 的有效/无效位置、右臂 22..26 每轴物理名称、正负号和单位；
- torso/pelvis IMU 的安装位置、quaternion 顺序/方向、gyro frame、加速度是否含重力及
  translational lever arm；
- motor temperature 两个字段的目标硬件含义和限值；
- 当前 trace 未携带的 motorstate/voltage/sensor/故障码；
- emergency/damping、遥控掉线、低电量和正常关机在该固件上的真实行为。

这些是 H1 evidence 交给后续人工评审的输入，不是本次现场通过后自动修改 YAML 的项目。

## 15. 本地官方资料视觉阅读索引

以下文件均位于 `local_reference/unitree_g1/`，不受 Git 跟踪。截图 PDF 的“页”可能是一
整张很长的网页，因此下面同时按文件和页面/页面内标题定位：

| 文件 | 页数 | 与 H1 相关的视觉核对 |
| --- | ---: | --- |
| `0_使用手册_中文_.pdf` | 27 | V1.4；p12--17 为吊装/上电，p14--17/p27 为零力矩、遥控、阻尼与关机；首次 H1 必须完全吊起 |
| `1_关于G1.pdf` | 1 个长页 | 12 个图块；标准版/EDU 开发差别、PC1/PC2、网络、关节索引 |
| `2_操作指南.pdf` | 4 | 27 个图块；p2 仍混有旧按键，p3 按固件/DOF 给多套贴纸，证明不能硬编码组合键 |
| `3_应用开发.pdf` | 5 | 13 个图块；SDK/DDS、Ubuntu、`.161/.164`、外部 `.99/.222` 和显式 NIC |
| `4_底层运动开发.pdf` | 4 | 16 个图块；LowState/LowCmd、CRC、index、IMU；debug 是 command 冲突处理，不是 state 订阅前提 |
| `5_软件服务接口.pdf` | 9 | 48 个图块；p5 定义 LowState 订阅，p6 说明 debug 退出内置服务；所有 mutating service 禁用 |
| `6_高层运动开发.pdf` | 4 | 19 个图块；locomotion/RPC/Arm Action/`arm_sdk` 均不属于 H1 |
| `7_更多例程.pdf` | 5 | 39 个图块；DDS subscriber/ROS2 可供理解，但 motion/RL/控制例程不运行 |
| `8_常见错误及释义.pdf` | 1 | 3 个图块；反馈超时、软启动、电机/欠压/软急停等原始故障码只记录，不现场清错 |
| `8_常见问题.pdf` | 1 个长页 | 17 个图块；网络、固件、校准和 NX FAQ；首次 H1 不 SSH、不校准、不恢复系统 |

## 16. 在线交叉验证来源

官方/官方组织：

- [Unitree G1 产品页与安全提示](https://www.unitree.com/g1/)
- [Unitree Explore 的 G1 教程与当前手册入口](https://www.unitree.com/app/g1/)
- [G1 User Manual 当前网页（审阅时 V1.6）](https://marketing.unitree.com/article/en/G1/User_Manual.html)
- [G1 Remote Control 当前网页（审阅时 V1.3）](https://marketing.unitree.com/article/en/G1/Remote_Control.html)
- [Unitree Developer Quick Start](https://support.unitree.com/home/en/developer/Quick_start)
- [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2)
- [SDK2 G1 Arm5 index/topic 示例](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/g1/high_level/g1_arm5_sdk_dds_example.cpp)
- [SDK2 G1 CRC、LowState 与 secondary IMU 参考](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/g1/low_level/g1_ankle_swing_example.cpp)
- [Unitree ROS2 的外部 PC `192.168.123.99/24` 与 NIC 配置](https://github.com/unitreerobotics/unitree_ros2/blob/master/README.md)
- [Unitree 当前 G1 model/revision 与 `mode_machine` 表](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/README.md)
- [Unitree 官方组织 xr_teleoperate Wiki 的 DDS/L2/L3/NIC 排障说明](https://github.com/unitreerobotics/xr_teleoperate/wiki/CycloneDDS-and-UnitreeSDK-(en))
- [CycloneDDS 网卡选择](https://cyclonedds.io/docs/cyclonedds/latest/config/network_interfaces.html)
- [CycloneDDS discovery 与 multicast](https://cyclonedds.io/docs/cyclonedds/latest/about_dds/discovery_participants.html)

第三方只作经验交叉验证：

- [Weston Robot G1 network/topology guide](https://docs.westonrobot.com/tutorial/unitree/g1_dev_guide/)
- [Weston Robot G1 diagnostics guide](https://docs.westonrobot.com/tutorial/unitree/g1_diag_guide/)
- [G1 首次连接记录：静态地址、无 DHCP 和网络排障](https://zeulewan.github.io/robot-docs/project/blog/2026-02-18-unitree-g1-first-session-networking/)
- [Unitree 官方仓库中的 G1 EDU+ 固件差异用户报告](https://github.com/unitreerobotics/unitree_sdk2_python/issues/170)
  与其他 `Waiting to subscribe dds` issues 只说明固件以及有线/NIC/domain 选择是常见
  故障面；未被维护者确认的回复不作为解决方案或安全依据。

## 17. 现场最终记录模板

```text
Date/time/timezone:
Operators:
Robot SKU / serial / DOF / arm / revision / App Machine Type:
Robot firmware / Unitree Explore / remote manual version:
Verified emergency/damping method:
Physical support and factory state:
Repo HEAD / git diff note:
SDK2 HEAD:
NIC / MAC / carrier / IPv4:
Route to 192.168.123.161:
Ping PC1 / optional PC2:
Bridge CPU / inspector CPU:
GROUP / evidence backup location / hashes:
Launcher exit status:
LowState received / valid / rejected:
Torso IMU / paired / skew rejected / max skew:
Observed version / mode_pr / mode_machine:
Sample count / dt / tick delta / state age:
Quaternion norm / gyro / acceleration observation:
Right-arm slots 22..26 plausibility:
Temperature / Unitree alarm observation:
Unexpected motion or incident:
H1 decision: ACCEPTED CAPTURE / REVIEW / FAIL CLOSED / INCIDENT
Explicit stop: no YAML change, no H2/H3/MPC/output started
```
