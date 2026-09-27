# G1 真机 MPC：程序、运行和验证

本文件对应 [`g1_walk_mpc.py`](../../tools/g1_commissioning/g1_walk_mpc.py)。
它是**左臂固定 PD、右臂 MPC、单腰 yaw 固定、腿部使用内置运控**的独立实验程序，
不是整机底层控制，不使用 `rt/lowcmd`，也不切模式或进入 debug。

这是首次上机候选，不是已经通过真机验证的控制器。先完成当前 6 ms PID 复验，再进行 MPC。
不需要先专门再采一批走路数据；新的 MPC 运行会同时保留预测输入、预测输出和真实反馈供检验。
但是 MPC 改变了手臂运动，不能事先保证旧固定手臂数据训练的预测器在新闭环下仍有同样精度。

## 1. 到现场最重要的几件事

1. 机器人已经双脚着地、在 FSM 500 自主平衡，当前没有其他用户程序控制手臂。
   网线和状态连接正常后，先按 [PID 文档](HARDWARE_PID.md) 完成一轮 6 ms PID 复验。
2. PID 平顺、停车和三秒退权正常，查看实际周期与状态年龄；不能只看到程序退出就宣布通过。
3. MPC 先运行 `--task stationary`：不发前进或转向目标，验证双臂接管、右臂控制和退权。
4. 这一轮正常，再单独运行 `--task walk`。首次不循环自动跑多轮。
   以现场实际剩余距离为准；0.5 m/s × 10 s 是速度请求，**不是限位器或保证行进距离**。
5. 正常主动结束用 Ctrl-C：请求停车并渐退 weight。异常使用本机已验证的现场停止方式。
   拔网线不等同于有保证的急停；丢包、断网或强杀进程时，软件无法保证把退权序列发完。

程序仍有原来的状态新鲜度、CRC、FSM、遥控器 L2+B、日志故障和 DDS 写失败检查。
**没有新增“实测手腕速度稍大就停”的规则。** 下文 0.07／0.20 限制的是生成的参考，不是传感器速度门槛。

## 2. 控制什么、何时控制

| 部位 | 名义角，单位度 | 控制方式 |
| --- | --- | --- |
| 左臂五轴 | `[-4, -1, 0, -8.1, 0]` | 固定 q，kp=20、kd=1 |
| 右臂五轴 | `[-4, +1, 0, -7.8, 0]` | 名义角附近的 MPC q/dq 参考，kp=20、kd=1 |
| 唯一的腰 yaw | 0 | 固定 q，kp=20、kd=1 |

五轴顺序：肩 pitch、肩 roll、肩 yaw、肘 pitch、腕 roll。
无效的腰 roll/pitch 槽保持零，不能给 23DoF 机器虚构两根腰轴。
所有有效 Arm SDK 槽的附加 `tau=0`；这**不等于电机零力矩**，固件 PD 仍产生驱动力矩。

| 时间 | 行走任务 | 静止任务 |
| --- | --- | --- |
| 0–3 s | 从实测初始姿态进入名义姿态，weight 0→1 | 相同 |
| 3–5 s | 右臂 MPC 开始，左臂固定；走前 yaw 圆周平均定义 H0 | 相同 |
| 5–15 s | 请求前进 0.5 m/s，航向保持 H0 +X | 前进和转向请求始终为零 |
| 15–18 s | 前进请求归零，航向保持继续，右臂 MPC 继续 | 保持静止控制 |
| 18 s 后 | 全零速度请求／回复等待，冻结参考，weight 至少三秒 1→0 | 相同退权 |

主指标仍是完整 **`[5,18)`**：起步、行走、减速和停车等待全部计入。
时间窗不能单独证明物理上已经完全静止，要同时看腿部反馈和现场观察。
H0 是走前平均 yaw 定义的**固定**坐标系；不是每周期随身体旋转的坐标系，Z 仍是导航系竖直方向。

## 3. 这是真正的 MPC，但不冒充仿真力矩执行器

复用 [`arm_mpc.py`](../../arm_mpc.py) 的九段、每段 6 ms、共 54 ms 的 MPC，
保留末端线加速度、角加速度、角速度、竖直、关节姿态、关节速度、控制量七项代价。
权重集中在 [`configs/hardware_mpc.yaml`](../../configs/hardware_mpc.yaml)，初值取自当前仿真配置。
瓶子中心仍是 XML 的 `right_grasp_site`，机体原点仍是 `imu_in_torso`。

**关键区别：仿真用逆动力学把期望关节加速度变成力矩；这里用 Arm SDK 固件 PD 跟踪位置／速度参考。**
直接把仿真那一点点 one-step PD 修正原样发布，不能证明执行了仿真的关节加速度。因此本程序明确采用
“参考轨迹 MPC”，优化状态是持续保存的 `q_ref、dq_ref`，并非假装真实电机是理想加速度源。

每周期用实测 q/dq 与参考之差，修正短时任务几何：

`预测实际 q(k) ≈ 参考 q(k) + 当前跟踪误差 + k×6ms×当前速度误差`

速度采用同样的当前误差修正。几何／重力项的仿射常数也做对应变换，当前真实瓶子姿态会影响优化。
这是未标定固件伺服响应时的首版近似，**不是已辨识的真实动力学模型**。
软件保存真实跟踪误差，后续根据 PID／MPC 实测决定是否需要标定伺服动态或调整参数。
不在本次开发中擅自改成力矩控制或放开全身输出。

参考约束：名义姿态各轴 ±5°、速度 ≤0.07 rad/s、加速度 ≤0.20 rad/s²。
额外参考 governor 按实际间隔与 6 ms 的较小者推进，预留离散制动距离；延迟不能放大单拍动作。
它沿用已验证 PID 的保守输出范围，但不保证此范围足以实现最优稳瓶；会记录参考边界和 governor 影响。
左臂、腰和全局 weight 不交给 MPC 求解器。

### 求解器与仿真程序怎样对应

`hardware_mpc_solver.py` 对原 QP 做**代数等价的状态消元**：
将每一步 `q/dq` 写成初始参考状态和未来加速度的函数，从 155 个变量减到 45 个。
按物理边界做数值缩放，批量计算相同代价，求解后还原完整轨迹并检查原约束。
随机代价矩阵、积分动态、完整约束和原仿真代价的对照测试保留在 `test_hardware_mpc.py`。
这不是换成 PID，也不是删掉 MPC 约束。

真机候选使用 **DAQP 0.9.1** 求解这个小型稠密凸 QP。实测轨迹回放中，OSQP 在参考接近
位置边界时出现过未收敛；换求解器没有改变目标函数或放宽动作范围。仿真的 OSQP 路径未改动。
原 OSQP 适配仅保留作显式离线比较，现场不自动切换求解器或使用失败后的替代动作。
DAQP 的算法和参数依据：[官方实现](https://github.com/darnstrom/daqp)、
[官方参数说明](https://darnstrom.github.io/daqp/parameters/)。

当前原生求解调用预算 3.5 ms、最多 1000 次迭代，缩放后 primal/dual 容差分别为 `1e-7/1e-10`；
还原后完整原约束残差必须 ≤`1e-6`。只接受最优成功状态，并检查调用的实际墙钟时间。
这不是硬实时中断：原生调用返回后才能拒绝超时结果，Python、DDS 和调度仍需另计。
预热阶段在创建 DDS 前给首次分配更多时间，进入现场循环前恢复上述上限。
未收敛、不可行或超时不会把无效解发给机器人，而是触发已有停车／受控退权路径。

## 4. 扰动预测怎样接入，为什么不是每次重新训练

发布的 [`assets/g1_hardware_mpc_predictor/`](../../assets/g1_hardware_mpc_predictor/)
包含约 6.7 MB 数值模型和 SHA256 清单。启动时只加载一次、建立一次搜索树。
不依赖被 Git 忽略的原始实验文件才能启动，不在控制循环里训练模型或读取文件。

每拍先从最近的十二个腿关节 q/dq，加上当前身体三轴线加速度、角速度、角加速度，组成 33 个数。
在 15,099 个开发样本里找 8 个相近状态，一次查询给出后续九段扰动，
再加上逐渐衰减的当前 IMU 偏差修正。模型只来自新数据 01–05／07／08；旧五条没有混入。
原方法与计算记录见 [预测改进报告](H0_WALK_PREDICTOR_REFINEMENT.md)。

读取缓存按 2 ms 的“只取该时刻之前最后一帧”推进，不能把新到的包回填到过去。
加速度处理为 `R_H0I × IMU比力 + [0,0,-9.81]`；角速度也转到 H0。
角加速度从滤波角速度差分再滤波，不是另一个直接读取的传感器通道。
固定 yaw 旋转与线性滤波可交换，内部先在导航系滤波再转 H0，不需要在冻结 yaw0 时伪造历史。

重要限制：

- 模型预测 **15 Hz 因果滤波后的值**。单层低频延迟约 9.64 ms，角加速度有两层滤波；
  没有把它宣称成无延迟的原始冲击，也没有凭猜测把时间轴向前挪。
- 九个 acc／alpha 标签是未来各 6 ms 区间的平均，omega 是节点值；共 10 个姿态节点、9 个区间。
  姿态从当前实测旋转出发，用预测角速度在 SO(3) 上积分，不直接平均欧拉角当旋转矩阵。
- 预测网格锚点最多比查询早 2 ms，接收年龄和锚点都记录；没有硬件源时间戳，不能伪称测到了单向网络延迟。
- 开始走前、H0 尚未冻结时，以及整个 `stationary` 任务，只使用当前滤波扰动延续。
  行走任务 `[5,18)` 才启用学得的预测；`--predictor hold_current` 可以做无历史模型前馈的对照。
- 最近邻距离只是诊断，不是经过验证的置信概率。旧离线 RMSE 改善不能当作 MPC 控制效果。

## 5. 文件与程序一一对应

| 文件 | 职责 |
| --- | --- |
| `g1_walk_mpc.py` | CLI、离线 preflight、加载／预热、接入现场流程 |
| `g1_walk_pid.py::run_device` | PID/MPC 共用状态门、SDK 订阅与输出、停车／退权；默认 PID 行为保留 |
| `hardware_mpc_predictor.py` | 因果接收缓存、H0 滤波、冻结模型检索、10 节点／9 区间 |
| `hardware_mpc_control.py` | 实测几何、参考状态 MPC、参考 governor |
| `hardware_mpc_solver.py` | 原 MPC QP 的等价消元和批量代价 |
| `configs/hardware_mpc.yaml` | 周期、预测长度、权重、求解预算、参考范围 |
| `benchmark_hardware_mpc.py` | 无 DDS 的完整循环回放与时间统计 |
| `audit_hardware_mpc_replays.py` | 从保存的回放日志重算参考边界审计和时间统计总数 |
| `analyze_hardware_mpc.py` | 与 PID 同口径的 H0 瓶子中心指标，输出标记为 MPC |

## 6. 电脑端命令

以下命令从仓库根目录运行；当前使用 `g1_mpc` 环境，不用 base Python。

```bash
cd /home/fjk/g1_ws/hold-my-beer-mpc
conda activate g1_mpc
python tools/g1_commissioning/g1_walk_mpc.py --preflight --cpu 2
```

preflight 只检查本地模型、库、实际 QP、SDK 数据包／CRC；不创建 DDS participant 或 publisher。
本机已有 SDK2、Python 数值环境；本次只补装了 `daqp==0.9.1`。换环境时如缺此依赖，执行
`python -m pip install --no-deps -r tools/g1_commissioning/requirements-hardware-mpc.txt`，
不要因此重装 SDK 或升级已有数值环境。
换电脑／重新克隆缺少本地 C++ 库时，先执行 `bash tools/g1_commissioning/build_hardware_mpc.sh`。
库在持久的 `build/right_arm_rnea/`，不再依赖重启会消失的 `/tmp`。不把本机 `.so` 提交到 Git。

复制并现场核对 [`mpc_walk_capture.template`](../../tools/g1_commissioning/profiles/mpc_walk_capture.template)，
保存为本地实验 profile；已有机型、映射、姿态信息可以引用之前记录，不需要重做 A0。
但不能把未知项目批量填 true。MPC 参数／现场确认应是本轮真实确认。

```bash
# 只有实际完成 6 ms PID 复验后，才使用 --pid-6ms-validated。
python tools/g1_commissioning/g1_walk_mpc.py enx6c1ff701509c \
  --execute --task stationary --cpu 2 \
  --profile /path/to/reviewed_mpc_profile.conf \
  --output-dir "evaluation/hardware_shadow/commissioning/mpc_static_$(date +%Y%m%d_%H%M%S)" \
  --pid-6ms-validated --permit-real-output MPC_WALK_H0_CAPTURE
```

成功后，把 `--task stationary` 改成 `--task walk`，换一个新的输出目录，才是行走实验。
`--pid-6ms-validated` 是操作者声明，不是程序伪造的验收证明。
默认不带 `--execute` 时始终只做离线检查；即使提供网卡也不会自动控制机器人。
真实运行仍在新鲜状态检查后要求键入 `EXECUTE <robot_id>`，再次复查后才创建 publisher。

结束后：

```bash
python tools/g1_commissioning/analyze_hardware_mpc.py /path/to/run/raw.jsonl \
  --output-dir /path/to/run/analysis
```

保存完整原始状态、约 2 ms 的预测输入、每拍预测和发送参考、求解状态、输入接收时间、
H0、FSM、速度 RPC、DDS Write 调用和完整周期时间。分析得到双侧瓶子姿态、竖直偏差、
线／角加速度、参考跟踪误差和耗时；这些是 **IMU＋实测关节＋XML 的估计**，不是瓶子上有直接传感器。
MPC 分析器拒绝主窗口内任一状态／命令流存在超过 100 ms 的缺口或不覆盖整个 `[5,18)`。
更小的缺口仍需结合状态年龄和逐周期时间审查；早停记录不能冒充完整行走结果。

## 7. 时间与当前验证状态

本机已确认 `6.8.1-1057-realtime`、`/sys/kernel/realtime=1`，PREEMPT_RT 已开启；
当前工具进程为 `SCHED_OTHER`，未改 FIFO/RR、IRQ、governor 或系统权限。
BLAS 单线程、控制线程绑定 CPU 2；其他线程不被整个进程的 taskset 一起挤到这个核。
绑定不等于独占，该 CPU 还有 SMT 同核线程。
模型／QP 预热后，一次短时实验期间暂停 Python 循环垃圾回收，结束恢复原设置，
减少长停顿；正常引用计数仍工作，接收缓存和日志队列都有大小上限。

模型查询本身不是主要成本；必须把预处理、几何、QP、CRC、日志入队和实际循环间隔一起统计。
离线回放还不包括真实 DDS 解包、网络和 RPC 并发，不能保证明天现场每次都满足 6 ms。
`DDS Write` 返回耗时也不是“电脑到电机执行再返回”的延迟。

最终六轮实测输入回放：完整工作 P99 **4.18–4.39 ms**，预测全流程 P99 **1.11–1.22 ms**；
主窗口共 12,987 周期，9 次 deadline miss（约 0.069%），最长循环间隔 12.014 ms。
全部正常结束／退权、无求解失败或日志丢失。查表开销已计入，但这仍不是现场实时性保证。

完整离线测量、发现的问题和最终版本结果见
[本次开发验证记录](sessions/20260927_HARDWARE_MPC_PREPARATION.md)。现场 PID/MPC 验证结果应另开记录，
不要把本次离线记录改写成已经完成真机实验。
