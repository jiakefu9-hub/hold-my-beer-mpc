# Publisher-absent HIL 输出链离线验证（2026-09-11）

结果：**现有离线输出链验证 PASS；真实设备传输/硬件输出均为 0。**
这是继 [H1 状态采集与审计](G1_H1_SESSION_20260911.md) 后的独立软件验证，全部输入
来自带 synthetic 标记的本地 fixture。未订阅 G1、未使用真实 H1 trace 生成命令，未
进入 debug、未获取真实 ownership、未运行 MPC，未发送任何 G1 控制命令。

本次复用现有实现，没有修改 HIL、supervisor、formatter、共享内存协议或 gate。
上一轮尚未提交的 H1 tick 修正与记录完整保留。本轮只新增本报告、本地测试 evidence
和文档索引，未提交或 push。

## 验证了什么

本地合成 state / Python proposal → protocol-v3 shared memory → exact-source cache /
dispatcher → C++ supervisor 与 13-slot formatter → RecordingCommandSink → receipt。

这里的 **would-write** 表示“通过校验后写入本地记录容器”，不是网络发送。fixture
的 `--offline-fixture-policy` / `--offline-ownership-confirmed` 只在无设备 transport
的测试进程中模拟策略与 ownership 条件；没有改变真实机器人或 production YAML。

| 验证层 | 本次结果 |
| --- | --- |
| C++ / CTest | 9/9 PASS，无跳过 |
| Python 输出合同与 C++ 互操作 | 17/17 PASS，无跳过 |
| 保留证据的既有端到端测试 | 2 个 case PASS，各 30 条 receipt |
| fixture case | 3 次本地 would-write，真实 transport/hardware output 为 0 |
| unverified case | 0 次 would-write，真实 transport/hardware output 为 0 |
| 2 ms 周期合同 | 非 2000 µs 参数被拒绝；两组回执的 scheduled time 步长均为 2 ms |
| 无 publisher 构建隔离 | 无 SDK/DDS 依赖，无 command publisher / state bridge 可执行文件 |
| 试图启用输出 | DDS=ON 的 CMake 配置、`--enable-output` CLI 均以 exit 1 拒绝 |
| 网络调用跟踪 | 三组 strace `trace=network` 日志均为 0 次网络系统调用 |

HIL 使用全新构建目录
`/tmp/g1-publisher-absent-hil-20260911.ggMIwo`，显式设置
`UNITREE_ARM_ADAPTER_BUILD_DDS=OFF`、`UNITREE_ARM_ADAPTER_BUILD_STATE_BRIDGE=OFF`。
SDK 路径故意设为不存在的 `/tmp/g1-hil-intentionally-no-sdk`，仍完整构建通过；未安装、
移动或加载 Unitree SDK。

对 HIL、fixture writer、dry-run 二进制执行了 symbols/strings/ldd 检查：无
`ChannelPublisher`、`ChannelSubscriber`、`LowCmd_`、`rt/arm_sdk`、`rt/lowcmd`，无
Unitree SDK/CycloneDDS 依赖及未解析动态库。对全部 CTest、Python 互操作和归档 E2E
运行使用 `strace -f --seccomp-bpf -e trace=network`，跟踪范围内无网络系统调用。
这描述的是本次测试进程，不是整台电脑或实验室网络的流量审计。

## 保留下来的端到端结果

fixture case 的首条命令携带完整 session/source/task/policy identity，active mask
为 992（13-slot 中的右臂五槽），测试 arm weight 为 0.1。Python
`hardware_output_authorized=false`、`hardware_safety_certified=false`、
`REQUEST_OUTPUT=0`。这些测试值没有发送给机器人。

| 相对首个接受拍 | Receipt | Supervisor reason | 本地 would-write |
| --- | --- | --- | --- |
| 0 ms | 2 | AcceptedArmingPd（1） | true |
| 2 ms | 3 | AcceptedHeldCommand（32） | true |
| 4 ms | 4 | AcceptedHeldCommand（32） | true |
| 6 ms | 5 | CommandHoldExceeded（33） | false |

第四拍没有新 anchor，因此拒绝并保留原因 33；之后所有拍也没有 would-write。
三次成功记录均再次验证了 sink 当下时间没有超过 deadline/expiry，且 13-slot 的
`q/dq/ddq/kp/kd/tau`、source binding、active mask、weight、policy identity 与
Python 提交值一致。所有 60 条 receipt 都落盘，真实 transport/hardware 字段均为 false。

unverified case 没有开启 synthetic 输入例外，实际首先得到 InvalidState（8），
随后保持 LatchedFault（27），所以 0 次 would-write。该 case 同时没有启用 fixture
policy，但不能把首次拒绝原因写成 SitePolicyUnverified；后者由
`TestUnverifiedPolicyCannotArm` 及 rejection matrix 的独立测试覆盖。

fixture writer 有意停止更新状态，proposal 也有有限 expiry，故末段计数包含
`state_stale_count=8`、`command_stale_count=4`；它们是本地测试的预期拒绝，非 G1
连接故障。fixture 与 unverified 两组 `deadline_miss_count` 均为 0。
单组仅 30 拍，这些结果不构成长时稳定性、操作系统实时调度或真机 watchdog 验证。

## 拒绝条件与协议检查

现有 C++ 测试验证了：

- 未验证 policy/ownership/输出授权不能 arming；状态机的 arming、active、soft
  release、latched fault/reset 行为；
- stale/future state、session/restart nonce、proposal replay、source identity 错配、
  task epoch/anchor 跳变或重放、policy identity 不匹配的拒绝；
- 以精确 source state 绑定 proposal，另以最新 actuation state 处理 inactive slots；
  不能用最新状态冒充缺失的 source，也不能把重写旧 command 当成合法 hold；
- 0/2/4 ms 三拍 hold、缺失下一 anchor、expiry、deadline miss、ownership 丢失时
  继续 fail closed；RecordingCommandSink 在调用点重新读时钟并复核；
- 13-slot finite/mask/limits、inactive slot 零动作、direct torque 下禁止重复 robot-side
  PD、arm weight step 与 release 规则；
- sink 决策、内存 receipt 和 JSONL 的一致性，以及拒绝拍不能宣称 sink write 成功。

Python 测试验证了 offline-only certification、完整 identity 和 source binding、
13-slot 合同、expiry/replay/watchdog 拒绝，以及 protocol-v3 大小/字段偏移的跨语言
一致性。上一轮因默认 dry-run 未构建而跳过的 6 项互操作测试，本次通过指定新构建的
`UNITREE_ARM_DRY_RUN` 全部执行通过；只读 shared-memory client 不能写 command slot
的测试也已通过。

## 复现与证据

主要命令（全部为本地合成测试）：

```bash
cmake -S cpp/unitree_arm_adapter \
  -B /tmp/g1-publisher-absent-hil-20260911.ggMIwo \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DUNITREE_ARM_ADAPTER_BUILD_DDS=OFF \
  -DUNITREE_ARM_ADAPTER_BUILD_STATE_BRIDGE=OFF \
  -DUNITREE_SDK2_DIR=/tmp/g1-hil-intentionally-no-sdk \
  -DPython3_EXECUTABLE=/home/fjk/miniforge3/envs/g1_mpc/bin/python
cmake --build /tmp/g1-publisher-absent-hil-20260911.ggMIwo --parallel 4
ctest --test-dir /tmp/g1-publisher-absent-hil-20260911.ggMIwo --output-on-failure

UNITREE_ARM_DRY_RUN=/tmp/g1-publisher-absent-hil-20260911.ggMIwo/unitree_arm_adapter_dry_run \
  /home/fjk/miniforge3/envs/g1_mpc/bin/python -m unittest -v \
  right_arm_runtime.tests.test_hardware_output_contract \
  right_arm_runtime.tests.test_unitree_shm
```

本次原始 evidence 位于
[`evaluation/hardware_shadow/publisher_absent_hil/g1_hil_offline_20260911_01/`](evaluation/hardware_shadow/publisher_absent_hil/g1_hil_offline_20260911_01/)：

- `summary.json`：交叉审计结果、源码/二进制/证据 SHA256；
- `ctest.log` / `ctest.xml` / `python_tests.log`：9 项与 17 项测试结果；
- `receipts_fixture.jsonl` / `receipts_unverified.jsonl`：完整 60 条回执；
- `producer_*.json` / `hil_*.log` / `retained_e2e_summary.json`：Python 提交合同与 HIL 结果；
- `*_network.log`：三个测试进程树的网络系统调用跟踪；
- `forbidden_dds_config.log` / `forbidden_enable_output.log`：预期拒绝记录；
- `used_CMakeCache.txt` / `hil_link.txt`：实际构建与链接路径；
- `capture_existing_hil_e2e.py`：只重定向原测试临时目录的生命周期，并归档返回值，
  原 E2E 的 producer、HIL 和全部 assertions 均未修改。

证据按现有规则在 Git 中忽略；本报告不等同于原始日志已上传。执行的 HIL/core 与 Python
输出合同源码经 `git diff --exit-code HEAD` 确认未修改，基底 HEAD 为
`92d38b697bf1a522296af931169159dfdc326392`。HIL binary SHA256：
`e517d29aa184181c1d13595fdbbc72485fca65ddec3b5274fe5fc8a218031e71`。

`configs/g1_hardware_shadow.yaml` 与 HEAD 内容相同，SHA256 仍为
`aeb04cc954d7abfd3a319e33d5626dfb576e502e8d889eabce798f3c618591b7`。
所有 verification flags、mode whitelist 和 output_enabled 保持原状。

结束后已核对 HIL、fixture writer、dry-run 进程全部退出，无本次 HIL/Python 测试共享
内存残留；源码及已保存 evidence 的 SHA256 复核通过，`git diff --check` 通过。

本轮证明的是软件输出链的格式、绑定、拒绝与回执合同。实际机器人上的 model/sign/
IMU/mode、ownership/release、断线/进程崩溃后的机器人行为及真实输出时序仍未验证；
本结果不授予 H2/H3、MPC 或真实 command publisher 的运行权限。
