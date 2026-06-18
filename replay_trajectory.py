"""Replay a saved MuJoCo trajectory from trajectory.npz.

用法示例：
1. 直接使用轨迹文件里保存的 xml_path：
   python /home/fjk/g1_ws/hold-my-beer-mpc/replay_trajectory.py \
       /home/fjk/g1_ws/hold-my-beer-mpc/evaluation/left_fixed_right_pid/20260618_175424/trajectory.npz

2. 指定回放速度，例如半速回放：
   python /home/fjk/g1_ws/hold-my-beer-mpc/replay_trajectory.py \
       /home/fjk/g1_ws/hold-my-beer-mpc/evaluation/left_fixed_right_pid/20260618_175424/trajectory.npz \
       --speed 0.5

3. 如果轨迹文件里没有 xml_path，手动指定模型：
   python /home/fjk/g1_ws/hold-my-beer-mpc/replay_trajectory.py \
       /path/to/trajectory.npz \
       --xml-path /home/fjk/g1_ws/hold-my-beer-mpc/resources/g1_description/g1_23dof_rev_1_0.xml

说明：
- 回放时可以在 MuJoCo viewer 里手动拖动视角。
- 这个脚本是“轨迹回放”，不是 mp4 播放器；它的优点是可以自由换视角、暂停观察。
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np


def _load_scalar(payload, key, default=None):
    """从 npz 里读取标量字段，兼容 numpy scalar / 0-d array。"""
    if key not in payload:
        return default
    value = payload[key]
    if np.isscalar(value):
        return value
    if getattr(value, "shape", ()) == ():
        return value.item()
    return value


def main():
    # trajectory_path: 由 main_sim.py 保存下来的 trajectory.npz
    # --xml-path: 可选；当 npz 里没有保存 xml_path 时手动指定模型文件
    # --speed: 回放倍率，0.5 表示半速，2.0 表示两倍速
    parser = argparse.ArgumentParser(description="Replay a saved MuJoCo trajectory.")
    parser.add_argument("trajectory_path", type=str, help="Absolute path to trajectory.npz")
    parser.add_argument("--xml-path", type=str, default=None, help="Override xml_path stored in trajectory.npz")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier, e.g. 0.5 or 2.0")
    args = parser.parse_args()

    # 读取轨迹数据。最核心的是 qpos；如果有 qvel / time，就一并用于更真实的回放节奏。
    payload = np.load(args.trajectory_path, allow_pickle=True)
    qpos_traj = np.asarray(payload["qpos"])
    qvel_traj = np.asarray(payload["qvel"]) if "qvel" in payload else None
    time_traj = np.asarray(payload["time"]) if "time" in payload else None

    xml_path = args.xml_path
    if xml_path is None:
        xml_path = _load_scalar(payload, "xml_path", None)
    if xml_path is None:
        raise ValueError("trajectory.npz 中没有 xml_path，请通过 --xml-path 手动提供。")

    simulation_dt = float(_load_scalar(payload, "simulation_dt", 0.002))
    playback_dt = simulation_dt / max(args.speed, 1e-6)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print(f"Replaying: {args.trajectory_path}")
    print(f"XML: {xml_path}")
    print(f"Frames: {len(qpos_traj)} | dt = {simulation_dt:.6f}s | speed = {args.speed:.3f}x")

    # launch_passive 会打开一个可交互 viewer。
    # 回放过程中你可以手动旋转、缩放、平移视角来看细节。
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(len(qpos_traj)):
            if not viewer.is_running():
                break

            frame_start = time.time()

            # 把第 i 帧保存的机器人状态写回 MuJoCo，然后做一次正运动学刷新显示。
            data.qpos[:] = qpos_traj[i]
            if qvel_traj is not None and qvel_traj.shape[0] > i:
                data.qvel[:] = qvel_traj[i]
            mujoco.mj_forward(model, data)
            viewer.sync()

            # 如果轨迹里保存了 time，就按原始时间间隔回放；否则退回到固定 dt。
            if time_traj is not None and i + 1 < len(time_traj):
                dt = float(time_traj[i + 1] - time_traj[i]) / max(args.speed, 1e-6)
            else:
                dt = playback_dt

            remain = dt - (time.time() - frame_start)
            if remain > 0:
                time.sleep(remain)


if __name__ == "__main__":
    main()