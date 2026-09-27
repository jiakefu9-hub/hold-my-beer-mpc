#!/usr/bin/env python3
"""Plot saved held-out forecasts and calculation-ledger metrics; no refitting.

Figures are deliberately outside benchmark/ so its archived artifact inventory
remains unchanged. This script reads actual saved predictions, never manually
types model scores into plots. No SDK, robot output, network, or live control.
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

LABELS = {
    "zoh": "沿用当前滤波测量",
    "clock_average": "按任务时间平均",
    "phase_average": "腿部相位平均模板",
    "hips_clock_ridge": "双髋角度＋任务时间",
    "hips_qdq_knn": "双髋角度＋速度查表",
    "knees_qdq_knn": "双膝角度＋速度查表",
    "legs_qdq_knn": "全部下肢角度＋速度查表",
    "imu_history_ridge": "近期 IMU 线性预测",
    "imu_legs_history_ridge": "近期 IMU＋下肢历史",
    "hybrid_switch": "近端历史＋远端查表（组合）",
}
COLORS = {"zoh": "#555555", "clock_average": "#A87900", "phase_average": "#009E73",
          "legs_qdq_knn": "#0072B2", "imu_history_ridge": "#56B4E9",
          "imu_legs_history_ridge": "#D55E00", "hybrid_switch": "#CC79A7"}
SELECTED = tuple(COLORS)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style(method):
    return dict(color=COLORS.get(method, "black"),
                linewidth=2.4 if method == "hybrid_switch" else 1.3,
                linestyle="--" if method == "hybrid_switch" else "-",
                marker="o" if method == "hybrid_switch" else None,
                markersize=3, label=LABELS[method])


def metric_curve(rows, method, region, group):
    chosen = sorted((r for r in rows if (r["method"], r["region"], r["group"]) ==
                     (method, region, group)), key=lambda r: r["horizon_ms"])
    if len(chosen) != 9:
        raise ValueError(f"Expected nine ledger rows: {method}/{region}/{group}; got {len(chosen)}")
    return np.array([r["horizon_ms"] for r in chosen]), np.array([r["rmse"] for r in chosen])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.study_dir
    out = root / "forecast_figures"
    out.mkdir(parents=True, exist_ok=True)
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10,
                         "axes.grid": True, "grid.alpha": .2})
    metrics_path = root / "benchmark/metrics_aggregate.json"
    prediction_path = root / "benchmark/predictions/trial09.npz"
    rows = json.loads(metrics_path.read_text())
    prediction = dict(np.load(prediction_path))
    figures = []
    with PdfPages(root / "prediction_gallery.pdf") as pdf:
        def save(fig, slug, title, caption):
            fig.savefig(out / (slug+".png"), dpi=140)
            pdf.savefig(fig)
            plt.close(fig)
            figures.append(dict(slug=slug, title=title, caption=caption,
                                image="forecast_figures/"+slug+".png"))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        for ax, group, label, unit in zip(axes, ("acc", "omega", "alpha"),
                                         ("线加速度", "角速度", "角加速度估计"),
                                         ("m/s²", "rad/s", "rad/s²")):
            for method in SELECTED:
                x, y = metric_curve(rows, method, "full", group)
                ax.plot(x, y, **style(method))
            ax.set_title(label)
            ax.set_xlabel("未来区间终点 / 节点时距（ms）")
            ax.set_ylabel("三轴 RMSE / ("+unit+")")
            ax.set_xticks([6, 12, 24, 36, 48, 54])
        axes[0].legend(fontsize=7)
        fig.suptitle("保留轨迹 09–12：从开始前进至停车等待结束的整体预测误差\n固定 H0；滤波目标；数值越低越好", fontsize=13)
        save(fig, "full_error_vs_horizon", "全任务阶段：预测距离越远，误差如何变化？",
             "直接读取 metrics_aggregate.json，预测锚点 5.006–17.942 秒（所有未来目标严格在 18 秒前）。RMSE 按全部保留轨迹的平方误差和/标量数计算。组合在 ≤12ms 用近期 IMU＋下肢历史，≥18ms 用下肢查表；重合曲线是设计结果，不是独立第三种模型。")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        for ax, region, title in zip(axes, ("startup", "steady", "stopping"),
                                    ("起步段", "稳定行走段", "停车请求后／保持平衡段")):
            for method in ("zoh", "clock_average", "phase_average", "legs_qdq_knn", "imu_legs_history_ridge", "hybrid_switch"):
                x, y = metric_curve(rows, method, region, "acc")
                ax.plot(x, y, **style(method))
            ax.set_title(title)
            ax.set_xlabel("未来区间终点时距 / ms")
            ax.set_ylabel("H0 三轴线加速度 RMSE / (m/s²)")
            ax.set_xticks([6, 12, 24, 36, 48, 54])
        axes[0].legend(fontsize=7)
        fig.suptitle("不能只看稳定行走：分别检查起步、稳定、停车阶段", fontsize=13)
        save(fig, "stage_acceleration_errors", "按运动阶段拆开的加速度误差",
             "分段完全沿用 benchmark 的 region 字段；没有为了图更好看重新划窗。停车段表示请求停止后保持平衡，不宣称机器人已经完全静止。相位平均模板没有显式学习启动/停止过渡，因此不能只凭稳定段成绩使用。")

        fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
        t = prediction["time"]
        mask = (t >= 10) & (t <= 11.2)
        for col, horizon in enumerate((6, 24)):
            h = int(np.flatnonzero(prediction["horizons_ms"] == horizon)[0])
            for row, axis in enumerate((0, 2)):
                ax = axes[row, col]
                ax.plot(t[mask], prediction["truth"][mask, h, axis], "k", lw=2, label="未来实际值（计分目标）")
                for method in ("zoh", "legs_qdq_knn", "imu_legs_history_ridge", "hybrid_switch"):
                    kw = style(method)
                    kw.pop("marker")
                    ax.plot(t[mask], prediction[method][mask, h, axis], **kw)
                ax.set_title(f"H0 {'XZ'[row]}：未来 [{horizon-6}, {horizon}) ms 均值")
                ax.set_ylabel("加速度 / (m/s²)")
                ax.set_xlabel("作出预测的当前任务时间 / s（不是未来信号时间）")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("保留轨迹 09 的实际预测片段：图中每一点都能在 predictions/trial09.npz 找到", fontsize=13)
        save(fig, "heldout09_forecast_trace", "实际预测曲线：第09条、未来6ms与24ms",
             "横轴是预测作出的当前时刻。黑线是这一时刻之后指定区间的真实目标，其他线是仅用当前及过去信息作出的预测。加速度采用 LEFT 定义：例如24ms目标取 +18/+20/+22ms 三个采样值的均值；不是把真实信号向前偷移作为输入。图段仅用于展示，结论来自全部保留数据的误差。")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for method in ("zoh", "hybrid_switch"):
            x, y = metric_curve(rows, method, "full", "acc")
            axes[0].plot(x, y, **style(method))
        x, y = metric_curve(rows, "zoh", "full", "raw_acc_hold_baseline")
        axes[1].plot(x, y, color=COLORS["zoh"], lw=1.4, label="保持当前未滤波加速度")
        x, y = metric_curve(rows, "hybrid_switch", "full", "raw_acc_sensitivity")
        axes[1].plot(x, y, **style("hybrid_switch"))
        for ax in axes:
            ax.set_xlabel("未来区间终点时距 / ms")
            ax.set_ylabel("三轴线加速度 RMSE / (m/s²)")
            ax.set_xticks([6, 12, 24, 36, 48, 54])
            ax.legend(fontsize=9)
        axes[0].set_title("计分目标：因果 15 Hz 滤波后的加速度")
        axes[1].set_title("敏感性检查：未滤波 H0 加速度均值")
        fig.suptitle("重要限制：预测滤波后的波形较准，不等于预测到了全部原始冲击", fontsize=13)
        save(fig, "filtered_vs_unfiltered_caveat", "不能把滤波目标成绩说成原始冲击预测成绩",
             "右图组合预测器没有重新训练成原始加速度预测器，只将同一预测与未滤波区间均值比较。最近6ms它不优于保持当前未滤波测量；较远时距仍有改善。15Hz因果滤波有延迟，离线计算未补偿真实消息时延/执行器时延；不能由本图推断MPC真机效果。")

    cards = "\n".join(f'<section><h2>{i+1}. {html.escape(f["title"])}</h2><p>{html.escape(f["caption"])}</p>'
                      f'<a href="{f["image"]}"><img src="{f["image"]}" alt="{html.escape(f["title"])}"></a></section>'
                      for i, f in enumerate(figures))
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 H0 扰动预测：保留轨迹实际结果</title><style>body{{max-width:1350px;margin:30px auto;padding:0 18px;font:16px/1.65 sans-serif}}img{{width:100%;height:auto}}section{{border-top:1px solid #bbb;margin-top:28px}}a{{color:#07578b}}</style>
<h1>G1 H0 扰动预测：保留轨迹实际结果</h1>
<p>仅新十二条中的 09–12 用于这些最终成绩，其他指定开发轨迹选择参数。旧五条牵拉数据不参与。
各方法均有保存的代码、训练文件、逐样本预测及 SSE/计数账本；本图程序直接读取这些文件，不手填成绩。</p>
<p><a href="prediction_gallery.pdf">预测图集 PDF（4页）</a> · <a href="gallery.html">原始轨迹24张图集</a> ·
<a href="benchmark/metrics_aggregate.json">完整误差账本</a> · <a href="forecast_manifest.json">图表来源哈希</a></p>
<p>世界参考为每次运行固定 H0。acc/alpha 按未来 6ms 左采样区间计分；omega/姿态按未来节点计分。
姿态参数只是诊断；本图不是控制效果、足部真实触地预测、网络往返时延或6ms在线控制验收。</p>{cards}</html>'''
    (root / "prediction_gallery.html").write_text(content, encoding="utf-8")
    manifest = dict(offline_only=True, no_refitting=True, methods=LABELS,
                    script_sha256=digest(Path(__file__)),
                    inputs=[dict(path=str(p), sha256=digest(p)) for p in (metrics_path, prediction_path)], figures=figures)
    (root / "forecast_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(figures=len(figures), html=str(root / "prediction_gallery.html")), ensure_ascii=False))


if __name__ == "__main__":
    main()
