# 右臂运行时边界

这个目录保存 Python 与手写 C++ 实时部件之间的窄接口。`unitree_shm.py` 是 `cpp/unitree_arm_adapter` protocol v2 的 Python 客户端：Python 写 13 维 arm SDK 命令，读取 35 电机状态和 C++ 2 ms 循环状态。它只打开已经由 C++ 创建的 POSIX 共享内存，**不会创建 DDS、不会发送真机命令**。

## 固定 ABI

| 项目 | protocol v2 |
| --- | ---: |
| magic | `0x473141524d504331` |
| 总大小 | 2304 B |
| command 槽偏移 / payload | 64 / 656 B |
| state 槽偏移 / payload | 768 / 1392 B |
| status 槽偏移 / payload | 2176 / 96 B |

Python 在打开时检查 magic、version、总大小、槽偏移和 64 字节对齐；字段偏移也在导入时逐项检查。跨进程读写使用 `libatomic` 的 acquire/release 加 seqlock，一次读取只会接受同一个偶数序号前后的完整快照。三个槽都是单写者：Python 只写 command，C++ DDS 回调写 state，C++ 2 ms 循环写 status。

## 两种命令不能混用

- `write_robot_pd_plus_feedforward(...)`：机器人底层执行 `q_ref/dq_ref/kp/kd` 的 PD，`tau_ff` 只能是不含 PD 的前馈力矩。
- `write_direct_torque(...)`：`tau_cmd` 已经包含反馈力矩；C++ 使用最新实测 q，并将发送给底层的 kp、kd 强制置零，防止重复 PD。

两种函数的 `request_output` 都默认是 `False`。即便将它显式设为 `True`，C++ 真机进程仍必须另外带 `--enable-output` 才可能建立发布器；本地 `unitree_arm_adapter_dry_run` 根本没有 DDS 代码和该选项。

简化示例：

```python
from right_arm_runtime.unitree_shm import UnitreeArmSharedMemoryClient

zeros = [0.0] * 13
with UnitreeArmSharedMemoryClient("/g1_arm_mpc", wait_timeout_s=1.0) as ipc:
    state = ipc.read_state()
    ipc.write_robot_pd_plus_feedforward(
        arm_weight=0.2,
        q_ref=zeros,
        dq_ref=zeros,
        kp=[20.0] * 13,
        kd=[1.0] * 13,
        tau_ff=zeros,
        request_output=False,
    )
    status = ipc.read_status()
```

## 本地一致性测试

先构建永不访问 DDS 的 C++ dry-run，再运行 Python 测试：

```bash
cpp/unitree_arm_adapter/build_and_test.sh
/home/fjk/miniforge3/bin/conda run -n g1_mpc \
  python -m unittest right_arm_runtime.tests.test_unitree_shm -v
```

测试会对比 C++ `--print-layout`，故障注入 magic/version/layout_size，检查默认输出关闭及两种 PD 语义，并分两个互斥单写者阶段验证 C++→Python 状态读取和 Python→C++ 命令确认。两阶段均使用 `unitree_arm_adapter_dry_run`，不会启用 DDS 或真机输出。
