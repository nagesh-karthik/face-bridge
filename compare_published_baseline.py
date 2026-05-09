"""
Compare a published face-verification baseline against the project approach.

Baseline:
  ArcFace-style embedding cosine similarity, following the highly cited
  ArcFace face-recognition method:
  Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition",
  CVPR 2019.

Project approach:
  The same embedding similarity fused with simple domain-aware image quality
  scores for ID/selfie verification.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from id_selfie_verification_experiment import evaluate_results


def normalized_cosine(series: pd.Series) -> pd.Series:
    return ((series + 1.0) / 2.0).clip(0, 1)


def comparison_rows(clean: pd.DataFrame):
    baseline = clean.copy()
    baseline["arcface_cosine_score"] = normalized_cosine(baseline["raw_cosine_similarity"])

    baseline_metrics = evaluate_results(baseline, "arcface_cosine_score")
    fused_metrics = evaluate_results(clean, "fused_score")

    return pd.DataFrame([
        {
            "method": "Published baseline: ArcFace cosine similarity",
            "score_column": "arcface_cosine_score",
            **baseline_metrics,
        },
        {
            "method": "Proposed: quality-aware fused score",
            "score_column": "fused_score",
            **fused_metrics,
        },
    ])


def save_comparison_roc(clean: pd.DataFrame, output_dir: str) -> None:
    y_true = clean["label"].astype(int).values
    baseline_scores = normalized_cosine(clean["raw_cosine_similarity"]).values
    fused_scores = clean["fused_score"].values

    baseline_fpr, baseline_tpr, _ = roc_curve(y_true, baseline_scores)
    fused_fpr, fused_tpr, _ = roc_curve(y_true, fused_scores)
    baseline_auc = roc_auc_score(y_true, baseline_scores)
    fused_auc = roc_auc_score(y_true, fused_scores)

    plt.figure(figsize=(6.4, 5.2))
    plt.plot(
        baseline_fpr,
        baseline_tpr,
        linewidth=2,
        label=f"ArcFace cosine baseline (AUC={baseline_auc:.3f})",
    )
    plt.plot(
        fused_fpr,
        fused_tpr,
        linewidth=2,
        label=f"Quality-aware fusion (AUC={fused_auc:.3f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Match Rate")
    plt.ylabel("True Match Rate")
    plt.title("Published Baseline vs Quality-Aware Fusion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "baseline_vs_quality_fusion_roc.png"), dpi=300)
    plt.close()


def save_metric_bar(metrics: pd.DataFrame, output_dir: str) -> None:
    plot_data = metrics.set_index("method")[["roc_auc", "accuracy_at_eer_threshold"]]
    plot_data = plot_data.rename(
        columns={
            "roc_auc": "ROC-AUC",
            "accuracy_at_eer_threshold": "Accuracy @ EER threshold",
        }
    )

    ax = plot_data.plot(kind="bar", figsize=(7.2, 4.8), ylim=(0, 1), rot=0)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Baseline Comparison")
    ax.legend(loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "baseline_comparison_metrics.png"), dpi=300)
    plt.close()


def run_comparison(pair_results_csv: str, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(pair_results_csv)
    clean = results[results["status"] == "ok"].copy()
    clean = clean.dropna(subset=["raw_cosine_similarity", "fused_score"])

    if len(clean) < 2 or len(clean["label"].unique()) < 2:
        raise ValueError("Need at least one valid genuine and impostor pair for comparison.")

    metrics = comparison_rows(clean)
    metrics_path = os.path.join(output_dir, "baseline_comparison_metrics.csv")
    metrics.to_csv(metrics_path, index=False)

    save_comparison_roc(clean, output_dir)
    save_metric_bar(metrics, output_dir)

    print("Comparison complete.")
    print(f"Input results: {pair_results_csv}")
    print(f"Valid pairs compared: {len(clean)}")
    print(f"Metrics: {metrics_path}")
    print(f"Plots saved in: {output_dir}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ArcFace-style cosine baseline to quality-aware fusion")
    parser.add_argument(
        "--pair-results",
        default="results_streamed_arcface/pair_results.csv",
        help="Pair result CSV produced by the verification experiment",
    )
    parser.add_argument("--output", default="results_comparison", help="Output directory")
    args = parser.parse_args()

    run_comparison(args.pair_results, args.output)
