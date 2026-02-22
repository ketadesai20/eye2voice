"""
evaluate_ceal_predictions.py

Evaluate iTracker predictions on CEAL dataset.

Three-stage evaluation:
  1. Distribution analysis  — How do (pred_x, pred_y) distribute per gt_label?
  2. Threshold classifier   — Map (pred_x, pred_y) → directional labels via thresholds
  3. Classification metrics  — Accuracy, precision, recall, F1, confusion matrix

IMPORTANT: iTracker predicts gaze landing point from the CAMERA's perspective.
  - pred_x is INVERTED relative to the user's left/right (negated before classification)
  - pred_y is correct (positive = up)

Usage:
  python evaluate_ceal_predictions.py \
      --predictions_csv ./ceal_predictions.csv \
      --output_dir      ./ceal_eval_results

Outputs (in output_dir):
  - distribution_summary.csv     Per-label mean/std/median of corrected pred_x, pred_y
  - scatter_by_label.png         Scatter plot of predictions colored by gt_label
  - boxplots.png                 Box plots of pred_x and pred_y by gt_label
  - confusion_matrix.png         Confusion matrix heatmap
  - classification_report.txt    Full sklearn classification report
  - eval_summary.txt             Overall accuracy and key findings
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate correction
# ──────────────────────────────────────────────────────────────────────────────
def apply_coordinate_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply axis corrections to iTracker predictions.

    iTracker outputs gaze landing point from the CAMERA's perspective:
      - pred_x is inverted relative to user's left/right → NEGATE
      - pred_y is correct (positive = up)

    Adds corrected columns: corr_x, corr_y
    """
    df = df.copy()
    df["corr_x"] = -df["pred_x"]  # flip x-axis to user's perspective
    df["corr_y"] = df["pred_y"]   # y-axis is correct
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Distribution Analysis
# ──────────────────────────────────────────────────────────────────────────────
def analyze_distributions(df: pd.DataFrame, output_dir: Path):
    """Compute and save per-label statistics for corrected predictions."""

    labels = ["left", "right", "up", "down", "straight"]

    rows = []
    for label in labels:
        subset = df[df["gt_label"] == label]
        rows.append({
            "label": label,
            "count": len(subset),
            "corr_x_mean": subset["corr_x"].mean(),
            "corr_x_std": subset["corr_x"].std(),
            "corr_x_median": subset["corr_x"].median(),
            "corr_y_mean": subset["corr_y"].mean(),
            "corr_y_std": subset["corr_y"].std(),
            "corr_y_median": subset["corr_y"].median(),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "distribution_summary.csv", index=False)
    print("\n── Distribution Summary (corrected coordinates) ──")
    print(summary.to_string(index=False))

    return summary


def plot_scatter(df: pd.DataFrame, output_dir: Path):
    """Scatter plot of corrected (corr_x, corr_y) colored by gt_label."""

    label_colors = {
        "left": "#e74c3c",
        "right": "#3498db",
        "up": "#2ecc71",
        "down": "#f39c12",
        "straight": "#9b59b6",
    }

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left panel: corrected coordinates
    ax = axes[0]
    for label, color in label_colors.items():
        subset = df[df["gt_label"] == label]
        ax.scatter(
            subset["corr_x"], subset["corr_y"],
            c=color, label=label, alpha=0.3, s=10, edgecolors="none"
        )
    ax.set_xlabel("corr_x (user's left ← → right)", fontsize=12)
    ax.set_ylabel("corr_y (down ← → up)", fontsize=12)
    ax.set_title("Corrected Predictions (x-axis flipped to user perspective)", fontsize=13)
    ax.legend(fontsize=11, markerscale=3)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Right panel: raw predictions for reference
    ax = axes[1]
    for label, color in label_colors.items():
        subset = df[df["gt_label"] == label]
        ax.scatter(
            subset["pred_x"], subset["pred_y"],
            c=color, label=label, alpha=0.3, s=10, edgecolors="none"
        )
    ax.set_xlabel("pred_x (raw, camera perspective)", fontsize=12)
    ax.set_ylabel("pred_y (raw)", fontsize=12)
    ax.set_title("Raw Predictions (before correction)", fontsize=13)
    ax.legend(fontsize=11, markerscale=3)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "scatter_by_label.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: scatter_by_label.png")


def plot_boxplots(df: pd.DataFrame, output_dir: Path):
    """Box plots of corrected pred_x and pred_y by gt_label."""

    label_order = ["left", "straight", "right", "down", "up"]
    colors = ["#e74c3c", "#9b59b6", "#3498db", "#f39c12", "#2ecc71"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # corr_x by label
    data_x = [df[df["gt_label"] == l]["corr_x"].values for l in label_order]
    bp0 = axes[0].boxplot(data_x, tick_labels=label_order, patch_artist=True)
    axes[0].set_title("corr_x by Ground Truth Label (x-flipped)", fontsize=13)
    axes[0].set_ylabel("corr_x (user perspective)")
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=0.5)
    axes[0].grid(True, alpha=0.3)

    # corr_y by label
    data_y = [df[df["gt_label"] == l]["corr_y"].values for l in label_order]
    bp1 = axes[1].boxplot(data_y, tick_labels=label_order, patch_artist=True)
    axes[1].set_title("corr_y by Ground Truth Label", fontsize=13)
    axes[1].set_ylabel("corr_y")
    axes[1].axhline(y=0, color="red", linestyle="--", linewidth=0.5)
    axes[1].grid(True, alpha=0.3)

    for bp in [bp0, bp1]:
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    plt.tight_layout()
    fig.savefig(output_dir / "boxplots.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: boxplots.png")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Threshold-based classification
# ──────────────────────────────────────────────────────────────────────────────
def classify_by_centroid(df: pd.DataFrame, summary: pd.DataFrame) -> pd.Series:
    """
    Classify corrected predictions using nearest centroid.
    Each prediction is assigned to the label whose (mean_x, mean_y)
    centroid it is closest to in Euclidean distance.
    """
    centroids = {}
    for _, row in summary.iterrows():
        centroids[row["label"]] = np.array([row["corr_x_mean"], row["corr_y_mean"]])

    labels = list(centroids.keys())
    centroid_matrix = np.array([centroids[l] for l in labels])  # (5, 2)

    preds = df[["corr_x", "corr_y"]].values  # (N, 2)

    # Euclidean distance from each prediction to each centroid
    dists = np.linalg.norm(preds[:, None, :] - centroid_matrix[None, :, :], axis=2)  # (N, 5)
    nearest = np.argmin(dists, axis=1)

    return pd.Series([labels[i] for i in nearest], index=df.index, name="pred_label_centroid")


def classify_by_axis_thresholds(df: pd.DataFrame) -> pd.Series:
    """
    Classify corrected predictions using axis-dominant logic.

    Uses the median of 'straight' predictions as the center point,
    then assigns direction based on which axis has larger displacement.
    """
    straight = df[df["gt_label"] == "straight"]
    cx = straight["corr_x"].median()
    cy = straight["corr_y"].median()

    dx = df["corr_x"] - cx
    dy = df["corr_y"] - cy

    # Deadzone: within this threshold → straight
    deadzone = min(dx.std(), dy.std()) * 0.25

    def _classify(dx_val, dy_val):
        if abs(dx_val) < deadzone and abs(dy_val) < deadzone:
            return "straight"
        if abs(dx_val) > abs(dy_val):
            return "left" if dx_val < 0 else "right"
        if abs(dy_val) > abs(dx_val):
            return "up" if dy_val > 0 else "down"
        return "left" if dx_val < 0 else "right"

    return pd.Series(
        [_classify(dx_val, dy_val) for dx_val, dy_val in zip(dx, dy)],
        index=df.index,
        name="pred_label_axis",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: Classification metrics
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_classifier(
    df: pd.DataFrame,
    gt_col: str,
    pred_col: str,
    method_name: str,
    output_dir: Path,
):
    """Compute and save classification metrics + confusion matrix."""

    labels = ["left", "right", "up", "down", "straight"]
    y_true = df[gt_col]
    y_pred = df[pred_col]

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    print(f"\n── {method_name} ──")
    print(f"  Overall accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(report)

    # Save text report
    report_path = output_dir / f"classification_report_{method_name}.txt"
    with open(report_path, "w") as f:
        f.write(f"Method: {method_name}\n")
        f.write(f"Overall accuracy: {acc:.4f} ({acc*100:.1f}%)\n\n")
        f.write(report)
    print(f"  Saved: {report_path.name}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {method_name}\nAccuracy: {acc:.1%}", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_{method_name}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: confusion_matrix_{method_name}.png")

    return acc


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation analysis
# ──────────────────────────────────────────────────────────────────────────────
def analyze_augmentation_effect(df: pd.DataFrame, pred_col: str, output_dir: Path):
    """Compare accuracy between original (aug_id=0) and augmented samples."""

    orig = df[df["aug_id"] == 0]
    augs = df[df["aug_id"] > 0]

    acc_orig = accuracy_score(orig["gt_label"], orig[pred_col]) if len(orig) > 0 else float("nan")
    acc_augs = accuracy_score(augs["gt_label"], augs[pred_col]) if len(augs) > 0 else float("nan")

    print(f"\n── Augmentation Breakdown ──")
    print(f"  Original (aug_id=0): {len(orig)} samples, accuracy {acc_orig:.4f}")
    print(f"  Augmented (aug_id>0): {len(augs)} samples, accuracy {acc_augs:.4f}")

    return acc_orig, acc_augs


# ──────────────────────────────────────────────────────────────────────────────
# Per-class accuracy breakdown by head pose
# ──────────────────────────────────────────────────────────────────────────────
def analyze_by_head_pose(df: pd.DataFrame, pred_col: str, output_dir: Path):
    """Show accuracy broken down by head_pose_deg to see pose sensitivity."""

    if "head_pose_deg" not in df.columns:
        return

    print(f"\n── Accuracy by Head Pose ──")
    poses = sorted(df["head_pose_deg"].unique())
    rows = []
    for pose in poses:
        subset = df[df["head_pose_deg"] == pose]
        acc = accuracy_score(subset["gt_label"], subset[pred_col])
        rows.append({"head_pose_deg": pose, "count": len(subset), "accuracy": acc})
        print(f"  {pose:>4d}°: {acc:.4f} ({len(subset)} samples)")

    pose_df = pd.DataFrame(rows)
    pose_df.to_csv(output_dir / "accuracy_by_head_pose.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(pose_df["head_pose_deg"], pose_df["accuracy"], width=3, color="#3498db", alpha=0.8)
    ax.set_xlabel("Head Pose (degrees)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Classification Accuracy by Head Pose ({pred_col})", fontsize=13)
    ax.axhline(y=pose_df["accuracy"].mean(), color="red", linestyle="--",
               label=f"Mean: {pose_df['accuracy'].mean():.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "accuracy_by_head_pose.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: accuracy_by_head_pose.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate CEAL iTracker predictions.")
    parser.add_argument(
        "--predictions_csv", type=str, required=True,
        help="Path to ceal_predictions.csv from run_ceal_inference.py."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./ceal_eval_results",
        help="Directory to save evaluation outputs (will overwrite existing)."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    df = pd.read_csv(args.predictions_csv)
    print(f"Loaded {len(df)} predictions from {args.predictions_csv}")
    print(f"Label distribution:\n{df['gt_label'].value_counts().to_string()}\n")

    # ── Apply coordinate corrections ──────────────────────────────────────
    print("Applying coordinate corrections: negating pred_x (camera→user perspective)")
    df = apply_coordinate_corrections(df)

    # ── Stage 1: Distribution Analysis ────────────────────────────────────
    summary = analyze_distributions(df, output_dir)
    plot_scatter(df, output_dir)
    plot_boxplots(df, output_dir)

    # ── Stage 2 & 3: Classification + Metrics ─────────────────────────────

    # Method A: Centroid nearest-neighbor
    df["pred_label_centroid"] = classify_by_centroid(df, summary)
    acc_centroid = evaluate_classifier(
        df, "gt_label", "pred_label_centroid", "centroid", output_dir
    )

    # Method B: Axis-dominant thresholds
    df["pred_label_axis"] = classify_by_axis_thresholds(df)
    acc_axis = evaluate_classifier(
        df, "gt_label", "pred_label_axis", "axis_threshold", output_dir
    )

    # ── Augmentation analysis (using better method) ───────────────────────
    better_method = "pred_label_centroid" if acc_centroid >= acc_axis else "pred_label_axis"
    better_name = "centroid" if acc_centroid >= acc_axis else "axis_threshold"
    analyze_augmentation_effect(df, better_method, output_dir)

    # ── Head pose analysis ────────────────────────────────────────────────
    analyze_by_head_pose(df, better_method, output_dir)

    # ── Save enriched predictions ─────────────────────────────────────────
    df.to_csv(output_dir / "predictions_with_labels.csv", index=False)
    print(f"\n  Enriched predictions saved to: {output_dir / 'predictions_with_labels.csv'}")

    # ── Summary ───────────────────────────────────────────────────────────
    summary_text = (
        f"CEAL iTracker Evaluation Summary\n"
        f"{'='*50}\n"
        f"Total samples:        {len(df)}\n"
        f"Coordinate correction: pred_x negated (camera→user)\n"
        f"\n"
        f"Centroid accuracy:    {acc_centroid:.4f} ({acc_centroid*100:.1f}%)\n"
        f"Axis threshold acc:   {acc_axis:.4f} ({acc_axis*100:.1f}%)\n"
        f"Best method:          {better_name}\n"
        f"\n"
        f"Key findings:\n"
        f"  - x-axis was inverted (iTracker uses camera perspective)\n"
        f"  - After correction, left/right should separate properly\n"
        f"  - Check scatter plot to verify directional clusters\n"
        f"  - Check head pose analysis for pose sensitivity\n"
    )
    with open(output_dir / "eval_summary.txt", "w") as f:
        f.write(summary_text)

    print(f"\n{'='*50}")
    print(summary_text)
    print(f"All outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()