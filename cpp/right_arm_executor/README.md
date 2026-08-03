# Right-arm C++ safety executor

这是固定 5 关节的 C++17 安全执行核心和稳定 C ABI 共享库。它不连接
Unitree DDS，也不依赖 MuJoCo、Pinocchio 或 Python；Python、C 或其他进程
可以通过 `libright_arm_executor.so` 调用同一份执行逻辑。

## 两种互斥输出语义

输入 `tau_ff` 始终只表示“不含 PD 的前馈力矩”。输出模式必须在创建
handle 时确定，运行中不能把两种含义混用。

### `RAE_OUTPUT_HOST_FULL_TORQUE`

主机计算并限制完整力矩：

```text
predicted_pd_tau = Kp * (q_ref - q) + Kd * (dq_ref - dq)
actuator_tau_ff = clamp(tau_ff + predicted_pd_tau, tau_min, tau_max)
actuator_kp = actuator_kd = 0
```

适配器向设备发送 `actuator_*` 字段时，设备增益严格为零，所以 PD 只在
C++ 主机执行一次。当前 Python 仿真若已经得到最终力矩 `tau_final`，保持
等价的桥接输入是 `tau_ff = tau_final - tau_pd_python`，再由本核心加回一次
相同 PD。

### `RAE_OUTPUT_DEVICE_PD`

主机不把 PD 加入发送给设备的前馈：

```text
actuator_q_ref  = limited q_ref
actuator_dq_ref = limited dq_ref
actuator_kp/kd  = configured Kp/Kd
actuator_tau_ff = limited input tau_ff   # 仍不含 PD
```

`predicted_total_tau_*` 只是使用当前主机状态得到的诊断值。设备状态在传输
期间还会变化，因此输出总会带
`RAE_FLAG_DEVICE_TOTAL_TORQUE_LIMIT_REQUIRED`；最终的“前馈+PD”总力矩必须
由设备端再次执行硬限幅。这样既不重复 PD，也不虚假宣称主机能限制未来
设备状态下的精确总力矩。

## 时间戳和回退

`rae_input_v1` 同时携带：

- `command_timestamp_ns`：上层命令生成时刻；
- `state_timestamp_ns`：`q/dq` 实际采样时刻；
- `now_ns`：调用 `rae_step_v1()` 时刻。

三者必须来自同一个单调时钟。命令和状态分别使用
`command_timeout_ns`、`state_timeout_ns`，所以“新命令+旧状态”不会被误判
为有效控制拍。

命令非法或超时时进入阻尼回退。主机完整力矩模式使用尚且有效的状态计算
`-timeout_damping*dq`；状态已经无效/陈旧时只能输出零力矩。设备 PD 模式
则发送 `kp=0, kd=timeout_damping, tau_ff=0`，让设备使用自己的最新编码器
速度执行局部阻尼。上述策略仍必须经过真机验证：纯阻尼可能让持瓶手臂
下落，零力矩也不天然等于安全。

## C ABI v1

头文件：

```text
include/right_arm_executor/right_arm_executor_c.h
```

调用顺序：

```text
rae_get_default_config_v1()
    -> 修改并确认所有 5 维参数
rae_create_v1()
    -> 返回 opaque handle（初始化阶段可以分配内存）
rae_step_v1()
    -> 固定结构体输入/输出，Step 路径无动态分配
rae_destroy()
```

运行期超时、NaN 或 Inf 通过 `output.executor_mode` 报告；API 返回非零只
表示空指针、ABI 版本、结构体尺寸或配置错误。Python `ctypes.Structure`
必须逐字段复刻 `rae_config_v1`、`rae_input_v1` 和 `rae_output_v1` 的顺序，
并填写：

```text
struct_size = ctypes.sizeof(对应结构体)
abi_version = RAE_ABI_VERSION_V1 (=1)
```

适配器真正发送的是输出中的 `actuator_q_ref/dq_ref/kp/kd/tau_ff`。其余
数组是诊断。`core_elapsed_ns` 仅包围 `RightArmExecutor::Step()`，不包含
C 数组复制、ctypes、IPC 或 Python 墙钟开销。

## 一键构建、测试和微基准

```bash
./cpp/right_arm_executor/build_and_test.sh
```

默认构建目录：

```text
/tmp/hold-my-beer-mpc-right-arm-executor-build
```

主要输出：

```text
libright_arm_executor.so        # C ABI 共享库软链接
libright_arm_executor.so.1      # SONAME
right_arm_executor_example
right_arm_executor_test         # C++ 核心测试
right_arm_executor_c_test       # 由 C 编译器编译的 ABI 测试
right_arm_executor_benchmark    # 固定输入微基准
```

脚本依次执行 Release 构建、CTest、示例和两种输出模式的 C ABI 微基准。
可以用 `RAE_BENCHMARK_ITERATIONS` 调整基准循环次数。微基准只说明本机纯
计算开销，不包含 Python、DDS、传感器或机器人总线延迟。

## 文件职责与真机边界

- `right_arm_executor.hpp/.cpp`：无通信、无动态分配的计算核心；
- `right_arm_executor_c.h/.cpp`：版本化固定布局 C ABI；
- `right_arm_executor_c_test.c`：真实 C 调用与不重复 PD 验证；
- `right_arm_executor_benchmark.cpp`：共享库热路径微基准；
- `main.cpp`：本地示例，不发送机器人命令。

未来 Unitree 适配器负责从 `rt/lowstate` 读取带采样时间的状态、接收上层
命令并向设备发布 `actuator_*`。实时循环不应解析 YAML、格式化日志或分配
内存。本模块只使用局部 5 维顺序：shoulder pitch、shoulder roll、
shoulder yaw、elbow pitch、wrist roll；MuJoCo actuator 索引和真机 motor
索引必须由适配器单独映射，不能直接复用。
