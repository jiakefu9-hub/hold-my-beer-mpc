#!/usr/bin/env python3
"""Offline five-episode signal gallery; no SDK, DDS, networking or robot output.

All time axes use task-relative host receive time. This is visualization, not
phase normalization, fitted model evaluation, or a deployable disturbance file.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np

from analyze_walk_dataset import TRIALS, asof, lowpass, prepare

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "evaluation/hardware_shadow/commissioning"
COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#A87900"]
FULL = (-.2, 21.)
ZOOM = (9., 11.2)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_font():
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10,
                         "axes.grid": True, "grid.alpha": .22})


def decorate(ax, limits, ylabel):
    ax.set_xlim(*limits)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("任务时间 / s（电脑收到消息的时刻）")
    if limits == FULL:
        ax.axvspan(5, 15, color="gray", alpha=.065, zorder=-10)
        for x in (3, 5, 15, 18):
            ax.axvline(x, color="gray", lw=.7, ls=":")


def create_plot(out, pdf, records, slug, title, panels, ylabel, caption):
    """panels: [(label, five arrays, optional five raw arrays)]."""
    fig, axes = plt.subplots(len(panels), 2, figsize=(14, 3.6*len(panels)),
                             squeeze=False, constrained_layout=True)
    for row, (label, signals, raw_signals) in enumerate(panels):
        for col, limits in enumerate((FULL, ZOOM)):
            ax = axes[row, col]
            for k, (r, y) in enumerate(zip(records, signals)):
                mask = (r["t"] >= limits[0]) & (r["t"] <= limits[1])
                if raw_signals is not None:
                    ax.plot(r["t"][mask], raw_signals[k][mask], color=COLORS[k],
                            alpha=.16, lw=.45, rasterized=True)
                ax.plot(r["t"][mask], y[mask], color=COLORS[k], lw=.9,
                        alpha=.85, label=f"轨迹 {k+1}", rasterized=True)
            decorate(ax, limits, ylabel)
            ax.set_title(f"{label} · {'全任务' if col == 0 else '同一时段放大'}")
    axes[0, 0].legend(loc="upper right", ncol=5, fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.savefig(out/f"{slug}.png", dpi=135)
    pdf.savefig(fig, dpi=120)
    plt.close(fig)
    return dict(slug=slug, title=title, caption=caption, image=f"{slug}.png")


def gallery_html(out, figures):
    links = "\n".join(f'<li><a href="#{f["slug"]}">{html.escape(f["title"])}</a></li>'
                      for f in figures)
    cards = "\n".join(
        f'<section id="{f["slug"]}"><h2>{i+1}. {html.escape(f["title"])}</h2>'
        f'<p>{html.escape(f["caption"])}</p><a href="{f["image"]}">'
        f'<img loading="lazy" src="{f["image"]}" alt="{html.escape(f["title"])}"></a>'
        '<p><a href="#top">回到目录</a></p></section>' for i, f in enumerate(figures))
    content = f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 五条轨迹关键数据图集</title>
<style>body{{max-width:1300px;margin:30px auto;padding:0 18px;font:17px/1.65 sans-serif;color:#222}}
img{{width:100%;height:auto}}section{{border-top:1px solid #bbb;margin-top:28px}}a{{color:#07578b}}
nav ul{{columns:2}}.note{{background:#edf5f8;padding:16px}}@media(max-width:650px){{nav ul{{columns:1}}}}</style>
<h1 id="top">G1 五条轨迹关键数据图集</h1>
<p>2026-09-18，仅离线画图；原始采集日期 2026-09-17。五种颜色始终对应同一条轨迹。</p>
<div class="note"><p>横轴是程序的任务时间，不是步态相位。左右列是同一数据的全任务和 9–11.2 秒放大。
灰色区间 5–15 秒表示请求前进，不等于已经准确测到了机器人实际运动距离。
竖线 3/5/15/18 秒分别表示升权完成、开始前进请求、停车请求、开始退权。</p>
<p>所有曲线先按各自消息的电脑接收时间放到 2 ms 网格，取当时已收到的最新值，未用未来插值。
关节角保持原值；关节速度和世界系 IMU 加速度/角速度采用因果 15 Hz 低通，淡线表示低通前值。
角加速度是角速度的后向差分再低通，不是额外测得的传感器值。滤波有延迟，图中未向前平移。</p>
<p>世界系来自机身 IMU 四元数，不是额外标定的实验室坐标系。
线加速度做了重力扣除：a_W = R_WI f_I + [0, 0, -9.81]，未扣掉静态偏置。
roll/pitch/yaw 仅用于看姿态波形；最终控制接口应使用旋转矩阵或四元数。</p>
<p>第 1 条与后四条约差半周期，所以时间重合不等于动作阶段相同。
不要只靠肉眼同时起伏就下结论“能预测”：还需只用过去数据的跨轨迹检验。</p></div>
<p><a href="walk_signals.pdf">全部图片 PDF（{len(figures)} 页）</a> · <a href="manifest.json">来源与处理清单</a></p>
<nav><h2>目录</h2><ul>{links}</ul></nav>{cards}
</html>'''
    (out/"index.html").write_text(content, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=BASE/"walk_dataset_audit_20260917")
    p.add_argument("--output-dir", type=Path, default=BASE/"walk_predictor_study_20260918/figures")
    args = p.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        p.error("output directory must differ from source extraction")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_font()
    raws = [dict(np.load(args.input_dir/f"trial{i:02d}.npz")) for i in range(1, 6)]
    records = [prepare(d) for d in raws]
    figures = []
    degree = 180/np.pi
    with PdfPages(args.output_dir/"walk_signals.pdf") as pdf:
        def add(*a, **kw):
            figures.append(create_plot(args.output_dir, pdf, records, *a, **kw))

        joints = [(0, "hip_pitch", "髋 pitch（前后摆）"),
                  (1, "hip_roll", "髋 roll（侧向摆）"),
                  (2, "hip_yaw", "髋 yaw（扭转）"),
                  (3, "knee", "膝关节"),
                  (4, "ankle_pitch", "踝 pitch"),
                  (5, "ankle_roll", "踝 roll")]
        for j, name, label in joints:
            panels = [(side, [r["q"][:, j+offset]*degree for r in records], None)
                      for side, offset in (("左腿", 0), ("右腿", 6))]
            add(f"q_{name}", f"{label}角度：五条轨迹", panels, "关节角 / °",
                f"左右腿分别绘制。读取 LowState motor {j} / {j+6}，无低通、无相位对齐。")
        for j, name, label in (joints[0], joints[3], joints[4]):
            panels = [(side, [lowpass(r["dq"][:, j+offset], 15)*degree for r in records],
                       [r["dq"][:, j+offset]*degree for r in records])
                      for side, offset in (("左腿", 0), ("右腿", 6))]
            add(f"dq_{name}", f"{label}速度：五条轨迹", panels, "关节速度 / (°/s)",
                "实测关节速度 dq，不是本图对角度求导；浓线因果 15 Hz 低通，淡线未低通。角度相同但速度方向相反，可能是不同动作阶段。")
        raw_omega = [asof(d["imu_t"], d["world_omega"], r["t"])[0]
                     for d, r in zip(raws, records)]
        for j, axis in enumerate("XYZ"):
            add(f"acc_world_{axis.lower()}", f"世界系 {axis} 线加速度：五条轨迹",
                [(f"a_W{axis}", [r["acc"][:, j] for r in records],
                  [r["acc_raw"][:, j] for r in records])], "线加速度 / (m/s²)",
                "由 IMU 四元数旋转加速度计读数并扣重力；浓线因果 15 Hz 低通，淡线是低通前的世界系值，不是原始 IMU 局部坐标值。未校准偏置。")
        for j, axis in enumerate("XYZ"):
            add(f"omega_world_{axis.lower()}", f"世界系 {axis} 角速度：五条轨迹",
                [(f"ω_W{axis}", [r["omega"][:, j] for r in records],
                  [w[:, j] for w in raw_omega])], "角速度 / (rad/s)",
                "陀螺仪读数由当前四元数旋转到世界系；浓线因果 15 Hz 低通，淡线低通前。")
        for j, axis in enumerate("XYZ"):
            add(f"alpha_world_{axis.lower()}", f"世界系 {axis} 角加速度估计：五条轨迹",
                [(f"α_W{axis}", [r["alpha"][:, j] for r in records], None)], "角加速度估计 / (rad/s²)",
                "先对世界系角速度做因果 15 Hz 低通，再按 2 ms 网格做后向差分，最后再因果 15 Hz 低通。它是派生估计，不是直接读出的第四种传感器通道。")
        for j, (axis, label) in enumerate((("roll", "roll 横滚"), ("pitch", "pitch 俯仰"), ("yaw", "yaw 航向"))):
            add(f"orientation_{axis}", f"身体世界系姿态 {label}：五条轨迹",
                [(label, [r["rpy"][:, j]*degree for r in records], None)], "姿态角 / °",
                "机身 IMU 同时记录的 RPY，沿各轨迹解除 ±π 跳变，未做低通。它表示相对 IMU 世界参考的身体方向；不是关节角，也不是瓶子末端姿态。")
        add("hip_pitch_difference", "双髋 pitch 差：已有周期标记的输入",
            [("左髋 pitch − 右髋 pitch", [r["hip"]*degree for r in records],
              [(r["q"][:, 0]-r["q"][:, 6])*degree for r in records])], "角度差 / °",
            "浓线因果 10 Hz 低通，淡线未低通。此前审计用这条信号的过零位置作为动作标记，不是官方步态相位或真实触地传感器。")
    gallery_html(args.output_dir, figures)
    manifest = dict(created_for="2026-09-18 offline predictor-method investigation",
                    source_trials=[dict(index=i, name=TRIALS[i-1],
                         npz_sha256=sha256(args.input_dir/f"trial{i:02d}.npz")) for i in range(1, 6)],
                    grid_dt_s=.002, full_window_s=FULL, detail_window_s=ZOOM,
                    joint_angles="degrees; q no lowpass", joint_velocity="degrees/s; causal 15 Hz plus unfiltered trace",
                    imu="torso rt/secondary_imu; world conversion from prior extraction; 15 Hz causal analysis filter",
                    alpha="backward derivative of 15 Hz world omega, then causal 15 Hz",
                    sample_policy="latest sample already received by each grid timestamp; no forward interpolation",
                    no_phase_alignment=True, no_robot_execution=True, figures=figures,
                    script_sha256=sha256(Path(__file__)),
                    extraction_script_sha256=sha256(Path(__file__).with_name("analyze_walk_dataset.py")))
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(figures=len(figures), output=str(args.output_dir),
                         html=str(args.output_dir/"index.html"), pdf=str(args.output_dir/"walk_signals.pdf")), ensure_ascii=False))


if __name__ == "__main__":
    main()
