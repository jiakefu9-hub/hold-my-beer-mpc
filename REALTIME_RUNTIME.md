# PREEMPT_RT target runtime

这套运行方式只准备和验证实时环境，不改变 MPC、neural predictor 或控制语义。
仓库不会自动安装内核、修改 GRUB、改 IRQ affinity 或永久启用 capability。

## 目标运行定义

- 使用目标系统支持的 `CONFIG_PREEMPT_RT=y` 内核；`PREEMPT_DYNAMIC` 不算通过。
- 选择一个性能核逻辑 CPU 运行控制主进程，并同时隔离该物理核的全部 SMT
  siblings。当前 XPro-16 的推荐 control CPU 是 7，对应完整物理核为 6–7。
- control Python 与锁步 C++ worker 固定在 control CPU，以同一低优先级
  `SCHED_RR/10` 运行。两者通过 pipe 阻塞交接，不持续忙等。
- 数值库和 Torch 固定单线程，control core 使用 `performance` governor。
- control core 使用 `isolcpus=domain,managed_irq`、`nohz_full`、`rcu_nocbs`；
  普通设备 IRQ 留在 housekeeping CPUs。保留有界 Linux RT throttling，避免
  runaway 进程长期占满 CPU。
- 不使用 `SCHED_FIFO`、高 RT priority、全局 capability 或永久 systemd unit。

## 只读检查

```bash
/home/fjk/miniforge3/envs/g1_mpc/bin/python realtime_environment.py \
  --control-cpu 7
```

checker 会检查 PREEMPT_RT、物理核 topology、boot isolation、活动 IRQ、
irqbalance、governor、RT throttling 和所需命令。任何目标条件缺失都返回非零。
它同时打印按当前机器 topology 生成的一次性 boot 参数建议。

对于 Linux managed MSI-X，effective affinity 可能仍显示隔离 CPU，即使
`isolcpus=managed_irq` 已把该 CPU 从设备 queue map 中移除。checker 会保留
这类 configured affinity 作为诊断，但只把观察窗口内计数继续增加的 IRQ
判为 active conflict；启动阶段的历史计数不会伪装成运行期 IRQ 活动。
target timing gate 还会在每次仿真的 evaluation 窗口两端读取
`/proc/interrupts` 的数字 IRQ 计数；
这两个快照都位于完整 6 ms 控制路径计时之外。任何一次 run 在物理核
`6-7` 上出现正 IRQ 增量，或快照不可用，整批 target gate 都会失败。

## 需要手动完成的系统步骤

### 1. PREEMPT_RT kernel

按照目标机 OS/vendor 的受支持方式安装并选择一个
`CONFIG_PREEMPT_RT=y` 内核。仓库不执行内核安装。

当前机器是 Ubuntu 22.04。Canonical 当前支持的、把仓库访问和安装分开的
手动方式是：

```bash
sudo pro status
sudo pro enable realtime-kernel --access-only
sudo apt install linux-realtime-hwe-22.04
```

这些命令需要 Ubuntu Pro，并会安装内核包；执行前检查 Pro/Livepatch 提示和
APT 待修改列表。官方说明见
<https://documentation.ubuntu.com/pro/pro-client/enable_realtime_kernel/>。
仓库不会代替用户运行这些命令。安装后重启，从 GRUB 的 Advanced options
明确选择 `-realtime` kernel，然后验证：

```bash
uname -a
grep '^CONFIG_PREEMPT_RT=y$' "/boot/config-$(uname -r)"
```

恢复方法：从 boot menu 重新选择原有 generic kernel。确认 generic kernel
已实际启动后，如需关闭 Pro 服务可手动执行：

```bash
sudo pro disable realtime-kernel
```

这不会自动移除已安装 kernel。确认新内核稳定前不要删除原 generic kernel，
也不要从仍在运行的 realtime kernel 中盲目删除 kernel 包。官方恢复说明见
<https://documentation.ubuntu.com/real-time/latest/how-to/switch-from-realtime-to-generic-kernel/>。

### 2. 一次性 CPU/IRQ isolation

在 GRUB boot menu 按 `e`，只对本次启动在 `linux` 行末追加 checker 打印的
参数。当前 XPro-16、control CPU 7 的参数是：

```text
isolcpus=domain,managed_irq,6-7 nohz_full=6-7 rcu_nocbs=6-7 irqaffinity=0-5,8-17
```

按 `Ctrl-X` 或 `F10` 进行一次性启动。不要由仓库脚本修改
`/etc/default/grub`。恢复方法：正常重启且不追加这些参数。

#### 可选：当前机器已验证的持久化 isolation

一次性 GRUB 编辑仍是第一次验证新机器或新 topology 时的安全默认。只有在
`realtime_environment.py` 已确认 control CPU 7 的完整物理核确为 6--7，并且
上述一次性参数已经完成 target gate 后，才考虑持久化。

当前 XPro-16 已验证的可选配置文件是
`/etc/default/grub.d/99-disturbance-rt.cfg`，内容必须保持为一行：

```bash
GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT isolcpus=domain,managed_irq,6-7 nohz_full=6-7 rcu_nocbs=6-7 irqaffinity=0-5,8-17"
```

这行保留发行版/其他 drop-in 已有的 `GRUB_CMDLINE_LINUX_DEFAULT`，只追加本机已
验证参数。不要把 `6-7` 复制到另一台 CPU topology 不同的机器。手动安装步骤：

```bash
sudo install -d -m 0755 /etc/default/grub.d
sudoedit /etc/default/grub.d/99-disturbance-rt.cfg
sudo update-grub
```

先审查 `update-grub` 输出，再重启进入 realtime kernel。重启后验证实际生效值，
不能只检查配置文件：

```bash
uname -a
cat /proc/cmdline
/home/fjk/miniforge3/envs/g1_mpc/bin/python realtime_environment.py \
  --control-cpu 7
```

**完整 rollback：** 先让 GRUB 不再读取该 `.cfg`，重建菜单并重启。使用 `mv` 保留
可恢复副本，不需要删除文件：

```bash
sudo mv /etc/default/grub.d/99-disturbance-rt.cfg \
  /etc/default/grub.d/99-disturbance-rt.cfg.disabled
sudo update-grub
sudo reboot
```

重启后确认 `/proc/cmdline` 不再包含 `isolcpus/nohz_full/rcu_nocbs/irqaffinity`。
如果要恢复同一份已验证配置：

```bash
sudo mv /etc/default/grub.d/99-disturbance-rt.cfg.disabled \
  /etc/default/grub.d/99-disturbance-rt.cfg
sudo update-grub
sudo reboot
```

该 drop-in **只**持久化 boot isolation，不会永久设置 governor，也不会禁用
`irqbalance`。两者仍按下一节记录原状态、临时设置和恢复。

### 3. 临时 governor 与 irqbalance

先记录原状态：

```bash
cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor
systemctl is-active irqbalance
```

仅在本次验证前执行：

```bash
sudo cpupower -c 6-7 frequency-set -g performance
sudo systemctl stop irqbalance
```

如果 `irqbalance` 未安装或原本不是 active，不需要 stop。验证后按之前记录的
governor 分别恢复；仅当 irqbalance 原本 active 时恢复它：

```bash
sudo cpupower -c 6 frequency-set -g <原cpu6-governor>
sudo cpupower -c 7 frequency-set -g <原cpu7-governor>
sudo systemctl start irqbalance
```

若 checker 仍报告 active IRQ conflict，停止验证并检查相应 IRQ 与设备；不要
盲目写 `/proc/irq/*/smp_affinity_list`。优先修正本次 boot 的 `irqaffinity` 或
使用目标机厂商提供的 IRQ routing 方法，重启即可撤销一次性 boot 配置。

## 一键 repeated timing gate

进入已激活的 `g1_mpc` 环境，在 checker PASS 后执行：

```bash
./tools/realtime/run_target_timing_gate.sh --control-cpu 7
```

脚本先再次运行只读 checker，然后创建最多 15 分钟的 transient systemd
service，仅临时授予 `RLIMIT_RTPRIO=20`。每个 control Python 和其 worker
由 `run.sh` 实际切换到 CPU 7 + `SCHED_RR/10`，并在运行时再次 fail-closed
检查。unit 在结束后自动回收，没有永久 service 或 capability。

中断后继续同一个 group：

```bash
./tools/realtime/run_target_timing_gate.sh \
  --control-cpu 7 --group <之前的group> --resume
```

原始运行保存在 gitignored `evaluation/`；轻量摘要写入
`evaluation_summary/realtime_timing_ablation/summary.json`。

## 最终 timing pass/fail

标准不因目标机而放宽。必须同时满足：

- target runtime checker PASS，且每次运行实测 main/worker 都是指定 CPU 的
  `SCHED_RR/10`；
- 3 个 unseen schedules × 4 seeds，共至少 12 runs 和 9588 个完整 evaluation
  区间；
- 完整 6 ms 路径 overrun 为 0，worst sample `< 6.0 ms`；
- 每个 run 的 p99 均 `<= 5.5 ms`；
- critical nonfinite 为 0；正常条件 QP success 每个 run 均 `>= 99%`；
- 每个 run 都成功采集 evaluation IRQ 快照，隔离物理核上的 IRQ 增量为 0。

`run_target_timing_gate.sh` 在任一条件失败时返回非零。通过的仿真 timing gate
仍不等于真机证据；DDS、驱动、总线和真实状态估计延迟必须在后续硬件阶段单独
测量。
