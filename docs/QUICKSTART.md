# 安装、回放与复现

从仓库根目录执行下列命令。回放和离线指标复算不需要机器人、Unitree SDK 或网络。
本项目的实测开发环境为 Linux x86_64、Python 3.10；图形回放需要本地图形桌面和 OpenGL。

## 只看已保存回放

已有 Python 环境可以安装最小依赖：

```bash
python -m pip install numpy==2.2.6 mujoco==3.6.0 glfw==2.10.0 PyYAML==6.0.3
python tools/experiments/thesis_walk13/replay.py D03 --check
python tools/experiments/thesis_walk13/replay.py D03
```

将 D03 换成 D01 或 D02 查看其他控制器。`--check` 无需显示器；普通回放使用鼠标调视角，
空格暂停、R 重播、G 切换参考线。三组来自提交的轨迹，运行不依赖 `evaluation/`。
离屏截图还需 Pillow；不自动尝试修改显示服务器或主机配置。

## 完整仿真环境

已工作的 `g1_mpc` 环境可以直接复用；不要为了整理重复安装它。
新电脑的依赖配方为：

```bash
conda env create -f environment-simulation.yml
conda activate g1_mpc
export G1_MPC_PYTHON="$(command -v python)"
bash cpp/build_runtime.sh
```

该配方根据当前已安装版本整理，包含 Pinocchio 的 C++ 开发文件。
本次验收使用同一已安装环境在独立检出目录重新构建；没有另做联网全新环境安装，
因此不同系统上的包解析和图形驱动仍需核对。Torch 不需要 GPU。
受磁盘余量限制，独立目录使用共享 Git 对象，并省略大体积压缩原始日志；
源码、模型、仿真回放和真机复算 NPZ 均在独立目录中。压缩原始日志另在主目录逐文件校验。
原 `requirements-mpc.txt` 只包含历史新增 OSQP 依赖，不能当成完整环境清单。

启动器优先使用显式 `G1_MPC_PYTHON`，否则查找激活的 g1_mpc 或 Conda 环境。
`run.sh` 和 `cpp/build_runtime.sh` 使用同一个解释器，不再写死原作者的 home 目录。
默认构建在 `/tmp/hold-my-beer-mpc-*-build`；同时维护多个检出目录时请分别设置
`RIGHT_ARM_RNEA_BUILD_DIR`、`RIGHT_ARM_EXECUTOR_BUILD_DIR`、`DDQ_TORQUE_MAPPER_BUILD_DIR`、
`RIGHT_ARM_SIM_RUNTIME_BUILD_DIR`，避免 CMake 缓存串用。

## 重新运行三组展示仿真

```bash
python tools/experiments/thesis_walk13/run.py --variant my_reproduction_01
```

先采集 11 条建模＋4 条检查轨迹，再运行位置 PID、无未来预览 MPC 与模板预览 MPC。任务为 10 s 前进＋3 s 停车观察，记录额外 60 ms 预测尾部。使用新的 `--variant` 名称，避免覆盖现有记录；入口固定使用本机 CPU 2，运行前确认该 CPU 在当前进程亲和性中可用。

复现入口只依赖当前 `thesis_walk13/` 支持模块、当前归档配置及共享模型/控制代码。配置、模板与原生库使用运行清单记录的参数；原生库默认由上述构建脚本生成到 `/tmp/hold-my-beer-mpc-*-build`。新结果保存在 `evaluation/thesis_walk10_stop3_<variant>/` 和当前结果目录的 `variants/<variant>/` 中。

保存轨迹的回放不重算控制。重新进行数值仿真不保证逐值重现，完整记录与严格验收状态分别报告；当前三组的具体状态见[结果说明](simulation/experiments/20261117_walk10_stop3/README.md)。

## 验证数据及重新计算真机指标

```bash
python tools/verify_handoff.py
python tools/verify_handoff.py --hardware-archive
python docs/g1_field_validation/evidence/20261014_final_archive/recompute_phase.py
```

前两条校验配置、模型、回放以及可选的精选真机文件哈希；第三条从归档 NPZ 复算分阶段指标，
需要 NumPy/SciPy，不需要本机原始 evaluation 目录。计算定义见归档报告。

## 真机代码的离线检查

真机依赖另见 `tools/g1_commissioning/requirements-hardware-mpc.txt` 与[现场工具说明](../tools/g1_commissioning/README.md)。
先构建真机路径使用的本地运动学库并做离线 preflight（不连接机器人）：

```bash
bash tools/g1_commissioning/build_hardware_mpc.sh
```

它写入 `build/right_arm_rnea/`，与仿真默认的 `/tmp` 库位置不同；环境变量不能覆盖真机 YAML 中显式指定的库路径。
最终 M 系列程序的默认 profile 路径是历史本机路径。新检出目录要运行相关离线测试时，
可从已提交副本恢复到该路径：

```bash
mkdir -p evaluation/hardware_shadow/commissioning/g1_pid_6ms_20261005_132052
cp docs/g1_field_validation/evidence/20261014_final_archive/runs/g1_pid_6ms_20261005_132052/arm_profile.conf \
  evaluation/hardware_shadow/commissioning/g1_pid_6ms_20261005_132052/arm_profile.conf
PYTHONPATH=tools/g1_commissioning:. python -m unittest discover -s tools/g1_commissioning/tests -p test_morning_20261013.py
```

这个副本是旧机器的实验条件，离线检查通过不代表新机器已获现场验证。
真机使用先读 [现场入口](g1_field_validation/README.md)，明确 profile、网卡、SDK、模式和停止方法。
初学者无需运行真实输出命令即可完成整个展示和数据复算流程。
