#!/usr/bin/env python3
"""Reproducible, offline-only H0 signal gallery and development-set diagnostics.

This script never imports an SDK or sends messages. Associations are descriptive,
not held-out prediction scores. Phase-normalized plots deliberately use future
cycle boundaries and are labelled as retrospective visualization, never a
causal predictor. Run only after prepare.py has produced trialXX_prepared.npz.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_walk_dataset import leg_events

DEVELOPMENT = (1, 2, 3, 4, 5, 7, 8)
FULL = (0., 21.)
ZOOM = (9., 11.2)
JOINTS = ((0, "hip_pitch", "髋 pitch"), (1, "hip_roll", "髋 roll"),
          (2, "hip_yaw", "髋 yaw"), (3, "knee", "膝"),
          (4, "ankle_pitch", "踝 pitch"), (5, "ankle_roll", "踝 roll"))
COLORS = [plt.get_cmap("tab20")(i) for i in (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 5)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fonts():
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10,
                         "axes.grid": True, "grid.alpha": .2})


def decorate(ax, limits, ylabel):
    ax.set_xlim(*limits)
    ax.set_xlabel("任务时间 / s（电脑接收时刻）")
    ax.set_ylabel(ylabel)
    if limits == FULL:
        ax.axvspan(5, 15, color="gray", alpha=.06, zorder=-10)
        for x in (3, 5, 15, 18):
            ax.axvline(x, color="gray", lw=.65, ls=":")


def overlay(out, pdf, records, slug, title, panels, ylabel, caption):
    fig, axes = plt.subplots(len(panels), 2, figsize=(14, 3.65*len(panels)),
                             squeeze=False, constrained_layout=True)
    for row, (label, signals, raw) in enumerate(panels):
        for col, limits in enumerate((FULL, ZOOM)):
            ax = axes[row, col]
            for k, (r, signal) in enumerate(zip(records, signals)):
                mask = (r["t"] >= limits[0]) & (r["t"] <= limits[1])
                if raw is not None:
                    ax.plot(r["t"][mask], raw[k][mask], color=COLORS[k],
                            alpha=.085, lw=.4, rasterized=True)
                ax.plot(r["t"][mask], signal[mask], color=COLORS[k], alpha=.8,
                        lw=.8, label=f"轨迹 {k+1:02d}", rasterized=True)
            decorate(ax, limits, ylabel)
            ax.set_title(label + (" · 全任务" if col == 0 else " · 同一时段放大"))
    axes[0, 0].legend(ncol=4, fontsize=7, loc="upper right")
    fig.suptitle(title, fontsize=14)
    fig.savefig(out / (slug + ".png"), dpi=130)
    pdf.savefig(fig, dpi=110)
    plt.close(fig)
    return dict(slug=slug, title=title, caption=caption, image="figures/"+slug+".png")


def association_features(r):
    values, names = [], []
    for key in ("qf", "dqf"):
        for j, name, _ in JOINTS:
            left, right = r[key][:, j], r[key][:, j+6]
            for kind, value in (("left", left), ("right", right),
                                ("sum", left+right), ("difference", left-right)):
                values.append(value)
                names.append(f"{name}_{key}_{kind}")
    return np.column_stack(values), names


def associations(records, out):
    """No held-out 09--12 data in correlation selection or ranking."""
    targets = [f"{kind}_{axis}" for kind in ("acc", "omega", "alpha") for axis in "xyz"] + ["roll", "pitch", "yaw"]
    correlations = []
    for trial in DEVELOPMENT:
        r = records[trial-1]
        features, names = association_features(r)
        anchors = np.flatnonzero((r["t"] >= 7) & (r["t"] < 14.7))[::3]
        # Same LEFT convention as benchmark: acc/alpha mean of 18,20,22 ms;
        # omega/orientation are endpoint values, not a centered future filter.
        ys = r["y"][anchors+12].copy()
        for lo in (0, 6):
            ys[:, lo:lo+3] = np.mean(np.stack([r["y"][anchors+j, lo:lo+3]
                                                for j in (9, 10, 11)]), axis=0)
        x = features[anchors]
        x = x-x.mean(axis=0)
        ys = ys-ys.mean(axis=0)
        denominator = np.sqrt(np.sum(x*x, axis=0))[:, None] * np.sqrt(np.sum(ys*ys, axis=0))[None, :]
        correlation = (x.T @ ys) / np.maximum(denominator, 1e-12)
        correlations.append(correlation)
    matrix = np.stack(correlations)
    median = np.median(matrix, axis=0)
    candidates = []
    for i, name in enumerate(names):
        for j, target in enumerate(targets):
            rows = matrix[:, i, j]
            candidates.append(dict(feature=name, target=target,
                                   median_pearson=float(np.median(rows)),
                                   minimum=float(np.min(rows)), maximum=float(np.max(rows)),
                                   same_sign_trials=int(np.sum(np.sign(rows) == np.sign(np.median(rows)))),
                                   by_trial={str(k): float(v) for k, v in zip(DEVELOPMENT, rows)}))
    result = dict(development_trials=list(DEVELOPMENT), excluded_from_selection=[6, 9, 10, 11, 12],
                  future_horizon_ms=24, window_s=[7, 14.7], sample_stride_ms=6,
                  target_definition="LEFT acc/alpha mean at +18,+20,+22 ms; omega/RPY at +24 ms",
                  input_definition="15 Hz causal filtered joint angles and velocities; bilateral individual/sum/difference",
                  warning="Descriptive repeated-cycle correlations, not independence, causality, or held-out forecasting performance; yaw drift can correlate spuriously.",
                  top_by_target={target: sorted([x for x in candidates if x["target"] == target],
                                               key=lambda x: abs(x["median_pearson"]), reverse=True)[:5]
                                 for target in targets},
                  features=names, targets=targets, median_matrix=median.tolist(),
                  all_associations=candidates)
    dump(out / "associations.json", result)
    return result


def events_and_alignment(records, out):
    positive = [np.array([r["t"][i] for i, sign in leg_events(r["t"], r["hip"]) if sign == 1])
                for r in records]
    train_periods = [np.diff(e[(e >= 7) & (e < 14.8)])
                     for i, e in enumerate(positive, 1) if i in DEVELOPMENT]
    train_period = float(np.median(np.concatenate(train_periods)))
    result = []
    cycles = []
    phase = np.arange(128) / 128
    for k, (r, events) in enumerate(zip(records, positive), 1):
        active = events[(events >= 7) & (events < 14.8)]
        predictions = []
        for j in range(1, len(events)-1):
            if events[j] < 7 or events[j+1] >= 14.8:
                continue
            past = np.diff(events[max(0, j-3):j+1])
            predicted = events[j] + np.median(past)
            predictions.append(dict(anchor_s=float(events[j]), target_s=float(events[j+1]),
                                    causal_prediction_s=float(predicted),
                                    causal_error_ms=float(1000*(predicted-events[j+1])),
                                    fixed_training_period_error_ms=float(1000*(events[j]+train_period-events[j+1]))))
        percycles = []
        for a, b in zip(active[:-1], active[1:]):
            source = np.column_stack((r["acc"], r["hip"]))
            percycles.append(np.column_stack([np.interp(a+phase*(b-a), r["t"], source[:, j]) for j in range(4)]))
        cycles.append(np.stack(percycles) if percycles else np.empty((0, 128, 4)))
        errors = np.array([x["causal_error_ms"] for x in predictions])
        periods = np.diff(active)
        result.append(dict(trial=k, positive_landmarks_s=events.tolist(),
                           steady_period_count=len(periods),
                           median_period_s=float(np.median(periods)) if len(periods) else None,
                           period_std_s=float(np.std(periods)) if len(periods) else None,
                           prediction_count=len(errors),
                           causal_next_event_mae_ms=float(np.mean(np.abs(errors))) if len(errors) else None,
                           causal_next_event_p95_abs_ms=float(np.percentile(np.abs(errors), 95)) if len(errors) else None,
                           predictions=predictions))
    heldout = [x for x in result if x["trial"] in (9, 10, 11, 12)]
    held_errors = [p["causal_error_ms"] for x in heldout for p in x["predictions"]]
    late_errors = [p["causal_error_ms"] for x in heldout for p in x["predictions"] if p["anchor_s"] >= 8]
    temporal = np.stack([r["acc"][(r["t"] >= 7) & (r["t"] < 14.7)] for r in records])
    cycle_means = np.stack([c.mean(axis=0)[:, :3] for c in cycles])
    within_cycle = [float(np.sqrt(np.mean((c[:, :, :3]-c.mean(axis=0)[None, :, :3])**2))) for c in cycles]
    doc = dict(landmark_definition="Positive zero crossing of 10 Hz causal filtered left-minus-right hip pitch; armed after <= -0.03 rad; same-sign refractory 0.3 s.",
               disclaimer="Kinematic landmarks, NOT official policy phase, foot contact, or IMU impact timestamps. A good period does not alone guarantee impact prediction.",
               training_period_s=train_period, training_period_trials=list(DEVELOPMENT),
               steady_cycle_window_s=[7, 14.8],
               causal_period_estimate="median of up to three already completed same-sign cycles",
               heldout_trials=[9, 10, 11, 12],
               heldout_causal_next_event_mae_ms=float(np.mean(np.abs(held_errors))),
               heldout_causal_next_event_p95_abs_ms=float(np.percentile(np.abs(held_errors), 95)),
               heldout_after8s_causal_next_event_mae_ms=float(np.mean(np.abs(late_errors))),
               heldout_after8s_causal_next_event_p95_abs_ms=float(np.percentile(np.abs(late_errors), 95)),
               descriptive_repeatability=dict(
                   time_aligned_acc_between_trial_rms_std_xyz=np.sqrt(np.mean(np.var(temporal, axis=0), axis=0)).tolist(),
                   retrospectively_phase_aligned_cycle_means_between_trial_rms_std_xyz=np.sqrt(np.mean(np.var(cycle_means, axis=0), axis=0)).tolist(),
                   within_trial_cycle_acc_rms_residual=within_cycle,
                   unit="m/s^2",
                   warning="Not a prediction benchmark. Phase statistic both rephases retrospectively and averages cycles; it therefore also suppresses noise, unlike pointwise time statistic."),
               retrospective_visualization="Phase-normalized plots use both observed boundaries, so cannot be counted as a real-time prediction result.",
               trials=result)
    dump(out / "event_summary.json", doc)
    return cycles, phase, doc


def diagnostic_figures(out, pdf, association, cycles, phase):
    figures = []
    matrix = np.array(association["median_matrix"])
    # Rank inputs only on the development set; show all output channels.
    order = np.argsort(np.max(np.abs(matrix[:, :9]), axis=1))[::-1][:20]
    fig, ax = plt.subplots(figsize=(14, 9), constrained_layout=True)
    plot = ax.imshow(matrix[order], aspect="auto", vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_yticks(np.arange(len(order)), [association["features"][i] for i in order], fontsize=9)
    ax.set_xticks(np.arange(12), association["targets"], rotation=40, ha="right")
    ax.set_title("开发轨迹 01–05、07–08：当前腿状态与 24 ms 后 H0 扰动的相关性\n每格为各轨迹 Pearson 相关系数的中位数，不是预测精度")
    ax.grid(False)
    fig.colorbar(plot, ax=ax, label="Pearson r 中位数")
    slug = "development_association_heatmap"
    fig.savefig(out / (slug + ".png"), dpi=135)
    pdf.savefig(fig)
    plt.close(fig)
    figures.append(dict(slug=slug, title="开发集相关性（不使用保留轨迹选特征）", image="figures/"+slug+".png",
                        caption="qf / dqf 分别为因果滤波关节角 / 速度；sum 为左右之和，difference 为左减右。周期共同驱动会造成相关，不能将相关系数当成预测误差改善。完整逐轨迹数值见 associations.json。"))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    labels = ["H0 X 线加速度 / (m/s²)", "H0 Y 线加速度 / (m/s²)",
              "H0 Z 线加速度 / (m/s²)", "左髋−右髋 pitch / °"]
    for j, ax in enumerate(axes.flat):
        for k, c in enumerate(cycles):
            if not len(c):
                continue
            y = c[:, :, j] * (180/np.pi if j == 3 else 1)
            for trace in y:
                ax.plot(phase, trace, color=COLORS[k], lw=.4, alpha=.08, rasterized=True)
            ax.plot(phase, y.mean(axis=0), color=COLORS[k], lw=1.2, label=f"轨迹 {k+1:02d}")
        ax.set_xlabel("事后归一化周期（0–1）")
        ax.set_ylabel(labels[j])
        ax.set_xlim(0, 1)
    axes[0, 0].legend(ncol=4, fontsize=7)
    fig.suptitle("同一腿部动作对齐后，波形是否相似？\n事后相位对齐使用了未来周期边界，仅为观察；不代表在线预测", fontsize=13)
    slug = "retrospective_phase_alignment"
    fig.savefig(out / (slug + ".png"), dpi=135)
    pdf.savefig(fig)
    plt.close(fig)
    figures.append(dict(slug=slug, title="事后周期对齐后的重复性", image="figures/"+slug+".png",
                        caption="仅使用 7–14.8 秒内完整同向双髋差过零周期。淡线各周期，浓线每条轨迹周期平均。此处允许使用完整周期的两个边界，不能用于证明因果预测效果；没有把保留轨迹加入预测训练。"))
    return figures


def gallery(out, figures):
    toc = "\n".join(f'<li><a href="#{f["slug"]}">{html.escape(f["title"])}</a></li>' for f in figures)
    cards = "\n".join(f'<section id="{f["slug"]}"><h2>{i+1}. {html.escape(f["title"])}</h2>'
                      f'<p>{html.escape(f["caption"])}</p><a href="{f["image"]}">'
                      f'<img loading="lazy" src="{f["image"]}" alt="{html.escape(f["title"])}"></a></section>'
                      for i, f in enumerate(figures))
    doc = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 十二条 H0 轨迹：离线数据图集</title><style>body{{max-width:1300px;margin:30px auto;padding:0 18px;font:16px/1.65 sans-serif;color:#222}}
img{{width:100%;height:auto}}section{{border-top:1px solid #bbb;margin-top:28px}}a{{color:#07578b}}nav ul{{columns:2}}.note{{background:#edf5f8;padding:16px}}</style>
<h1>G1 十二条 H0 轨迹：离线数据图集</h1>
<p>仅使用 2026-09-18 新十二条轨迹；旧五条牵拉采集数据不参与任何计算。十二种颜色始终对应相同编号。</p>
<div class="note"><p>H0 是本次行走前 3–5 秒平均朝向确定的固定参考系，不会跟着身体转动。参考在约 5.00006 秒冻结；此前图段仅为事后坐标重表达，不能作为当时已知的 H0 特征。+Z 继承 IMU 导航系。
身体、关节和末端坐标系不是 H0。图中身体姿态是身体 IMU 相对 H0，不是瓶子的末端姿态。</p>
<p>任务阶段：0–3 秒升权，3–5 秒等待，5–15 秒前进请求，15–18 秒停车请求后保持航向/平衡（不等于已经完全静止），18–21 秒退权。
主评估窗口为 5–18 秒。轨迹 06 缺少退出标记和最后一次应答，主模型比较排除；轨迹 12 的遥控器干预在退权阶段，单独标注。
全十二条仍画出供检查，画出不等于全部用于模型训练。</p>
<p>2 ms 网格取当时已经收到的最新消息，没有使用未来插值。IMU 线加速度由旋转后的比力扣除重力；
加速度/角速度浓线为因果 15 Hz 低通，淡线为滤波前数据。角加速度由滤波角速度后向差分再低通得到，非直接传感器读数。
所有因果滤波均保留自身延迟，没有向前平移。</p>
<p>前 22 张时间图没有做相位对齐。最后的周期对齐图是事后观察，需要完整周期，不能当作实时预测成绩。
相关性图仅用开发轨迹 01–05、07–08；测试轨迹 09–12 不参与关联筛选。</p></div>
<p><a href="gallery.pdf">全部 {len(figures)} 页 PDF</a> · <a href="gallery_manifest.json">来源与处理记录</a> ·
<a href="associations.json">逐轨迹相关系数</a> · <a href="event_summary.json">腿部周期及事件误差</a></p>
<nav><ul>{toc}</ul></nav>{cards}</html>'''
    (out / "gallery.html").write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figdir = args.output_dir / "figures"
    figdir.mkdir(exist_ok=True)
    fonts()
    paths = [args.input_dir / f"trial{i:02d}_prepared.npz" for i in range(1, 13)]
    records = [dict(np.load(p)) for p in paths]
    association = associations(records, args.output_dir)
    cycles, phase, event_summary = events_and_alignment(records, args.output_dir)
    figures = []
    degree = 180/np.pi
    with PdfPages(args.output_dir / "gallery.pdf") as pdf:
        def add(*a):
            figures.append(overlay(figdir, pdf, records, *a))
        for j, name, label in JOINTS:
            panels = [(side, [r["q"][:, j+offset]*degree for r in records], None)
                      for side, offset in (("左腿", 0), ("右腿", 6))]
            add("q_"+name, label+"角度：十二条轨迹", panels, "关节角 / °",
                f"左右腿分开；LowState motor {j}/{j+6}，角度保持原值、未低通；所有图的轨迹颜色一致。")
        for j, name, label in (JOINTS[0], JOINTS[3], JOINTS[4]):
            panels = [(side, [r["dqf"][:, j+offset]*degree for r in records],
                       [r["dq"][:, j+offset]*degree for r in records])
                      for side, offset in (("左腿", 0), ("右腿", 6))]
            add("dq_"+name, label+"速度：十二条轨迹", panels, "关节速度 / (°/s)",
                "实测 dq，不是对本图角度求导。浓线因果 15 Hz 低通，淡线未低通。角度相同但速度方向相反可能处于不同阶段。")
        for key, rawkey, title, units in (("acc", "acc_raw", "线加速度", "m/s²"),
                                          ("omega", "omega_raw", "角速度", "rad/s"),
                                          ("alpha", None, "角加速度估计", "rad/s²")):
            for j, axis in enumerate("XYZ"):
                add(f"{key}_h0_{axis.lower()}", f"H0 {axis} {title}：十二条轨迹",
                    [(f"{title} {axis}", [r[key][:, j] for r in records],
                      [r[rawkey][:, j] for r in records] if rawkey else None)],
                    title+" / ("+units+")", "固定 H0 参考系；使用身体 rt/secondary_imu。" +
                    ("角速度先因果 15 Hz 低通，后向差分再因果 15 Hz 低通；非独立传感器值。" if key == "alpha" else
                     "浓线因果 15 Hz 低通，淡线未低通。线加速度已扣除重力，未去除静态偏置。"))
        for j, (name, label) in enumerate((("roll", "横滚"), ("pitch", "俯仰"), ("yaw", "航向"))):
            add("orientation_"+name, f"身体 H0 系 {name} {label}：十二条轨迹",
                [(label, [r["rpy"][:, j]*degree for r in records], None)], "身体姿态 / °",
                "由固定 H0 与身体 IMU 的旋转关系计算，解除 ±π 跳变。不是 IMU 局部坐标系内的姿态，也不是瓶子末端姿态。")
        add("hip_pitch_difference", "双髋 pitch 差：运动周期标记输入",
            [("左髋−右髋", [r["hip"]*degree for r in records],
              [(r["q"][:, 0]-r["q"][:, 6])*degree for r in records])], "角度差 / °",
            "浓线因果 10 Hz 低通；过零事件用于估计重复动作周期，不等于官方策略相位、触地或冲击时刻。")
        figures.extend(diagnostic_figures(figdir, pdf, association, cycles, phase))
    gallery(args.output_dir, figures)
    dump(args.output_dir / "gallery_manifest.json", dict(
        source_trials=[dict(trial=i, prepared_path=str(p), sha256=digest(p)) for i, p in enumerate(paths, 1)],
        script_sha256=digest(Path(__file__)), offline_only=True, frame="fixed H0",
        full_window_s=FULL, detail_window_s=ZOOM, data_grid_dt_s=.002,
        causality="Signal plots use causal prepared data; only clearly labelled retrospective phase plots use future boundaries.",
        figures=figures))
    print(json.dumps(dict(figures=len(figures), html=str(args.output_dir / "gallery.html"),
                          pdf=str(args.output_dir / "gallery.pdf"),
                          heldout_next_event_mae_ms=event_summary["heldout_causal_next_event_mae_ms"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
