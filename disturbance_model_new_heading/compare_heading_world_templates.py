"""比较新 H 模板与 disturbance_model_new 中原有 W 模板。"""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from heading_template_utils import (
    rotation_geodesic_angle,
    rotation_to_rpy,
    rotation_z,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_HEADING_DIR = os.path.join(SCRIPT_DIR, "templates_heading")
DEFAULT_WORLD_DIR = os.path.join(
    REPO_DIR, "disturbance_model_new", "templates_world"
)
DEFAULT_HEADING_DATA = os.path.join(
    SCRIPT_DIR, "torso_disturbance_heading.npz"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "comparison_world_heading"
)

VARIANTS = {
    "raw": (
        "heading_disturbance_template.npz",
        "world_disturbance_template.npz",
    ),
    "half_smoothed": (
        "heading_disturbance_template_half_smoothed.npz",
        "world_disturbance_template_half_smoothed.npz",
    ),
    "fully_smoothed": (
        "heading_disturbance_template_fully_smoothed.npz",
        "world_disturbance_template_fully_smoothed.npz",
    ),
}

VECTOR_FIELDS = {
    "acceleration": "torso_linear_acceleration_template",
    "angular_velocity": "torso_angular_velocity_template",
    "angular_acceleration": "torso_angular_acceleration_template",
}

THRESHOLDS = {
    "acceleration_relative_rmse": 0.10,
    "angular_velocity_relative_rmse": 0.10,
    "angular_acceleration_relative_rmse": 0.15,
    "minimum_trend_correlation": 0.95,
    "maximum_circular_lag_bins": 2,
    "orientation_mean_geodesic_deg": 0.50,
}


def load_npz(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到模板: {path}")
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def circular_mean_angle(angles):
    return float(
        np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    )


def vector_metrics(candidate, reference):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    difference = candidate - reference
    rmse_axis = np.sqrt(np.mean(difference**2, axis=0))
    reference_rms_axis = np.sqrt(np.mean(reference**2, axis=0))
    relative_rmse = float(
        np.sqrt(np.mean(np.sum(difference**2, axis=1)))
        / max(
            np.sqrt(np.mean(np.sum(reference**2, axis=1))),
            1e-12,
        )
    )

    centered_candidate = candidate - np.mean(candidate, axis=0)
    centered_reference = reference - np.mean(reference, axis=0)
    denominator = float(
        np.linalg.norm(centered_candidate)
        * np.linalg.norm(centered_reference)
    )
    trend_correlation = (
        float(
            np.sum(centered_candidate * centered_reference)
            / denominator
        )
        if denominator > 1e-12
        else 1.0
    )

    correlations = []
    lags = []
    best_correlations = []
    num_bins = len(candidate)
    for component in range(candidate.shape[1]):
        x = centered_candidate[:, component]
        y = centered_reference[:, component]
        component_denominator = np.linalg.norm(x) * np.linalg.norm(y)
        correlations.append(
            float(np.dot(x, y) / component_denominator)
            if component_denominator > 1e-12
            else 1.0
        )
        scores = []
        for lag in range(num_bins):
            shifted = np.roll(y, lag)
            score_denominator = np.linalg.norm(x) * np.linalg.norm(
                shifted
            )
            scores.append(
                float(np.dot(x, shifted) / score_denominator)
                if score_denominator > 1e-12
                else 1.0
            )
        best_index = int(np.argmax(scores))
        signed_lag = (
            best_index
            if best_index <= num_bins // 2
            else best_index - num_bins
        )
        lags.append(signed_lag)
        best_correlations.append(float(scores[best_index]))

    return {
        "rmse_axis": rmse_axis.tolist(),
        "reference_rms_axis": reference_rms_axis.tolist(),
        "relative_rmse": relative_rmse,
        "trend_correlation": trend_correlation,
        "correlation_axis": correlations,
        "best_circular_lag_bins_axis": lags,
        "best_circular_correlation_axis": best_correlations,
        "maximum_absolute_lag_bins": int(
            np.max(np.abs(np.asarray(lags)))
        ),
    }


def compare_variant(heading, world, rotation_H_W_reference):
    if not np.allclose(
        heading["phase_centers"], world["phase_centers"], atol=1e-12
    ):
        raise ValueError("H/W 模板 phase_centers 不一致。")
    result = {
        "vectors": {},
        "direct_stored_frame_vectors": {},
    }
    for name, key in VECTOR_FIELDS.items():
        heading_values = np.asarray(heading[key], dtype=np.float64)
        world_values = np.asarray(world[key], dtype=np.float64)
        world_as_heading = np.einsum(
            "ij,nj->ni", rotation_H_W_reference, world_values
        )
        result["vectors"][name] = vector_metrics(
            heading_values, world_as_heading
        )
        result["direct_stored_frame_vectors"][name] = vector_metrics(
            heading_values, world_values
        )

    heading_rotation = np.asarray(
        heading["torso_orientation_rotation_matrix_template"],
        dtype=np.float64,
    )
    world_rotation = np.asarray(
        world["torso_orientation_rotation_matrix_template"],
        dtype=np.float64,
    )
    world_rotation_as_heading = np.einsum(
        "ij,njk->nik", rotation_H_W_reference, world_rotation
    )
    orientation_error = rotation_geodesic_angle(
        heading_rotation, world_rotation_as_heading
    )
    direct_orientation_error = rotation_geodesic_angle(
        heading_rotation, world_rotation
    )
    result["orientation"] = {
        "mean_geodesic_deg": float(
            np.rad2deg(np.mean(orientation_error))
        ),
        "rms_geodesic_deg": float(
            np.rad2deg(np.sqrt(np.mean(orientation_error**2)))
        ),
        "max_geodesic_deg": float(
            np.rad2deg(np.max(orientation_error))
        ),
        "direct_stored_frame_mean_geodesic_deg": float(
            np.rad2deg(np.mean(direct_orientation_error))
        ),
    }
    return result, world_rotation_as_heading


def evaluate_variant(metrics):
    vectors = metrics["vectors"]
    checks = {
        "acceleration_relative_rmse": (
            vectors["acceleration"]["relative_rmse"]
            <= THRESHOLDS["acceleration_relative_rmse"]
        ),
        "angular_velocity_relative_rmse": (
            vectors["angular_velocity"]["relative_rmse"]
            <= THRESHOLDS["angular_velocity_relative_rmse"]
        ),
        "angular_acceleration_relative_rmse": (
            vectors["angular_acceleration"]["relative_rmse"]
            <= THRESHOLDS["angular_acceleration_relative_rmse"]
        ),
        "minimum_trend_correlation": all(
            values["trend_correlation"]
            >= THRESHOLDS["minimum_trend_correlation"]
            for values in vectors.values()
        ),
        "maximum_circular_lag_bins": all(
            values["maximum_absolute_lag_bins"]
            <= THRESHOLDS["maximum_circular_lag_bins"]
            for values in vectors.values()
        ),
        "orientation_mean_geodesic_deg": (
            metrics["orientation"]["mean_geodesic_deg"]
            <= THRESHOLDS["orientation_mean_geodesic_deg"]
        ),
    }
    return checks, bool(all(checks.values()))


def plot_comparison(records, output_path, align_world):
    variants = tuple(VARIANTS)
    colors = ("tab:blue", "tab:orange", "tab:green")
    components = ("x", "y", "z")
    ylabels = (
        "a [m/s²]",
        "omega [rad/s]",
        "alpha [rad/s²]",
    )
    fig, axes = plt.subplots(4, 3, figsize=(25, 15), sharex=True)
    world_label = "W→H" if align_world else "W direct"
    for column, variant in enumerate(variants):
        record = records[variant]
        heading = record["heading_template"]
        world = record["world_template"]
        rotation_H_W = (
            record["rotation_H_W_reference"]
            if align_world
            else np.eye(3)
        )
        phase = heading["phase_centers"]
        for row, ((_, key), ylabel) in enumerate(
            zip(VECTOR_FIELDS.items(), ylabels)
        ):
            heading_values = heading[key]
            world_values = np.einsum(
                "ij,nj->ni", rotation_H_W, world[key]
            )
            for component, (label, color) in enumerate(
                zip(components, colors)
            ):
                axes[row, column].plot(
                    phase,
                    heading_values[:, component],
                    color=color,
                    label=f"H {label}",
                )
                axes[row, column].plot(
                    phase,
                    world_values[:, component],
                    color=color,
                    linestyle="--",
                    alpha=0.65,
                    label=f"{world_label} {label}",
                )
            axes[row, column].set_ylabel(ylabel)

        heading_rpy = np.rad2deg(
            rotation_to_rpy(
                heading[
                    "torso_orientation_rotation_matrix_template"
                ]
            )
        )
        world_rotation = (
            record["world_rotation_as_heading"]
            if align_world
            else world[
                "torso_orientation_rotation_matrix_template"
            ]
        )
        world_rpy = np.rad2deg(rotation_to_rpy(world_rotation))
        for component, (label, color) in enumerate(
            zip(("roll", "pitch", "yaw"), colors)
        ):
            axes[3, column].plot(
                phase,
                heading_rpy[:, component],
                color=color,
                label=f"H {label}",
            )
            axes[3, column].plot(
                phase,
                world_rpy[:, component],
                color=color,
                linestyle="--",
                alpha=0.65,
                label=f"{world_label} {label}",
            )
        axes[3, column].set_ylabel("orientation [deg]")
        axes[3, column].set_xlabel("phase")
        axes[0, column].set_title(variant)
        for row in range(4):
            axes[row, column].grid(True, alpha=0.3)
            axes[row, column].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_summary_csv(metrics_by_variant, path):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            [
                "variant",
                "quantity",
                "relative_rmse",
                "trend_correlation",
                "max_abs_lag_bins",
                "direct_relative_rmse",
                "direct_trend_correlation",
                "direct_max_abs_lag_bins",
                "rmse_x",
                "rmse_y",
                "rmse_z",
                "passed",
            ]
        )
        for variant, variant_metrics in metrics_by_variant.items():
            for quantity, values in variant_metrics["vectors"].items():
                direct = variant_metrics[
                    "direct_stored_frame_vectors"
                ][quantity]
                writer.writerow(
                    [
                        variant,
                        quantity,
                        values["relative_rmse"],
                        values["trend_correlation"],
                        values["maximum_absolute_lag_bins"],
                        direct["relative_rmse"],
                        direct["trend_correlation"],
                        direct["maximum_absolute_lag_bins"],
                        *values["rmse_axis"],
                        int(variant_metrics["passed"]),
                    ]
                )


def main():
    parser = argparse.ArgumentParser(
        description="数值和趋势对比新 H 模板与原有 W 模板"
    )
    parser.add_argument(
        "--heading-template-dir", default=DEFAULT_HEADING_DIR
    )
    parser.add_argument(
        "--world-template-dir", default=DEFAULT_WORLD_DIR
    )
    parser.add_argument(
        "--heading-data", default=DEFAULT_HEADING_DATA
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    with np.load(args.heading_data, allow_pickle=False) as heading_data:
        stable = heading_data["count"] >= float(
            heading_data["discard_time"]
        )
        heading_yaw_reference = circular_mean_angle(
            heading_data["heading_yaw"][stable]
        )
    rotation_H_W_reference = rotation_z(-heading_yaw_reference)

    records = {}
    metrics_by_variant = {}
    for variant, (heading_name, world_name) in VARIANTS.items():
        heading = load_npz(
            os.path.join(args.heading_template_dir, heading_name)
        )
        world = load_npz(
            os.path.join(args.world_template_dir, world_name)
        )
        metrics, world_rotation_as_heading = compare_variant(
            heading, world, rotation_H_W_reference
        )
        checks, passed = evaluate_variant(metrics)
        metrics["checks"] = checks
        metrics["passed"] = passed
        metrics_by_variant[variant] = metrics
        records[variant] = {
            "heading_template": heading,
            "world_template": world,
            "rotation_H_W_reference": rotation_H_W_reference,
            "world_rotation_as_heading": world_rotation_as_heading,
        }

    overall_passed = all(
        values["passed"] for values in metrics_by_variant.values()
    )
    report = {
        "comparison": "heading_template_vs_world_template_rotated_to_heading",
        "heading_yaw_reference_rad": heading_yaw_reference,
        "heading_yaw_reference_deg": float(
            np.rad2deg(heading_yaw_reference)
        ),
        "thresholds": THRESHOLDS,
        "variants": metrics_by_variant,
        "overall_passed": overall_passed,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(
        args.output_dir, "comparison_metrics.json"
    )
    csv_path = os.path.join(
        args.output_dir, "comparison_summary.csv"
    )
    aligned_png_path = os.path.join(
        args.output_dir, "Heading_vs_World_Template_Comparison.png"
    )
    direct_png_path = os.path.join(
        args.output_dir,
        "Heading_vs_World_Template_Direct_Comparison.png",
    )
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    save_summary_csv(metrics_by_variant, csv_path)
    plot_comparison(records, aligned_png_path, align_world=True)
    plot_comparison(records, direct_png_path, align_world=False)

    print(
        "用于 W→H 对齐的稳定段平均 heading yaw: "
        f"{np.rad2deg(heading_yaw_reference):.6f} deg"
    )
    for variant, metrics in metrics_by_variant.items():
        print(f"[{variant}] passed={metrics['passed']}")
        for quantity, values in metrics["vectors"].items():
            print(
                f"  {quantity}: rel_rmse={values['relative_rmse']:.6f}, "
                f"trend_corr={values['trend_correlation']:.6f}, "
                f"max_lag={values['maximum_absolute_lag_bins']}"
            )
        print(
            "  orientation: mean/max geodesic="
            f"{metrics['orientation']['mean_geodesic_deg']:.6f}/"
            f"{metrics['orientation']['max_geodesic_deg']:.6f} deg"
        )
    print(f"整体判断: {'通过' if overall_passed else '未通过'}")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")
    print(f"对齐后 PNG: {aligned_png_path}")
    print(f"直接值 PNG: {direct_png_path}")
    if not overall_passed:
        raise SystemExit("H/W 模板差异超过预设阈值，请检查转换流程。")


if __name__ == "__main__":
    main()
