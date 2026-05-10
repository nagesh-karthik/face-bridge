"""
Bootstrap confidence intervals for Face Bridge verification metrics.

This script does not retrain or recompute face embeddings. It resamples the
existing evaluated pair rows and estimates how stable ROC-AUC and EER are on a
small test subset.

Default behavior uses stratified bootstrap resampling, which preserves the
original number of genuine and impostor pairs in each bootstrap draw.

Example:
python bootstrap_confidence_intervals.py \
  --pair-results results_streamed_arcface/pair_results.csv \
  --output results_bootstrap_ci \
  --iterations 5000
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


BASELINE_SCORE_COL = "arcface_cosine_score"
FUSED_SCORE_COL = "fused_score"


def normalized_cosine(series: pd.Series) -> pd.Series:
    return ((series + 1.0) / 2.0).clip(0, 1)


def calculate_eer(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return {"eer": eer, "eer_threshold": float(thresholds[idx])}


def metric_row(df: pd.DataFrame, score_col: str) -> Dict[str, float]:
    y_true = df["label"].astype(int).to_numpy()
    scores = df[score_col].astype(float).to_numpy()

    auc = float(roc_auc_score(y_true, scores))
    eer_result = calculate_eer(y_true, scores)
    threshold = eer_result["eer_threshold"]
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": auc,
        "eer": eer_result["eer"],
        "eer_threshold": threshold,
        "accuracy_at_eer_threshold": float(accuracy_score(y_true, y_pred)),
        "false_match_rate": float(fp / (fp + tn)) if (fp + tn) else np.nan,
        "false_non_match_rate": float(fn / (fn + tp)) if (fn + tp) else np.nan,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def prepare_results(pair_results_csv: str) -> pd.DataFrame:
    results = pd.read_csv(pair_results_csv)
    if "status" in results.columns:
        results = results[results["status"] == "ok"].copy()

    required = {"label", "raw_cosine_similarity", FUSED_SCORE_COL}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    clean = results.dropna(subset=["label", "raw_cosine_similarity", FUSED_SCORE_COL]).copy()
    clean["label"] = clean["label"].astype(int)
    clean[BASELINE_SCORE_COL] = normalized_cosine(clean["raw_cosine_similarity"])

    label_counts = clean["label"].value_counts().to_dict()
    if len(label_counts) < 2:
        raise ValueError("Need at least one genuine and one impostor pair.")
    if min(label_counts.values()) < 2:
        raise ValueError("Need at least two pairs in each class for bootstrap resampling.")

    return clean.reset_index(drop=True)


def stratified_sample_indices(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    sampled: List[np.ndarray] = []
    for label in sorted(df["label"].unique()):
        label_indices = df.index[df["label"] == label].to_numpy()
        sampled.append(rng.choice(label_indices, size=len(label_indices), replace=True))
    return np.concatenate(sampled)


def unstratified_sample_indices(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    indices = df.index.to_numpy()
    return rng.choice(indices, size=len(indices), replace=True)


def bootstrap_metrics(
    clean: pd.DataFrame,
    iterations: int,
    seed: int,
    stratified: bool,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    sampler = stratified_sample_indices if stratified else unstratified_sample_indices

    for bootstrap_id in range(iterations):
        sample_indices = sampler(clean, rng)
        sample = clean.loc[sample_indices].copy()

        if len(sample["label"].unique()) < 2:
            continue

        baseline = metric_row(sample, BASELINE_SCORE_COL)
        fused = metric_row(sample, FUSED_SCORE_COL)
        label_counts = sample["label"].value_counts().to_dict()

        rows.append(
            {
                "bootstrap_id": bootstrap_id,
                "valid_pairs": len(sample),
                "genuine_pairs": int(label_counts.get(1, 0)),
                "impostor_pairs": int(label_counts.get(0, 0)),
                "baseline_roc_auc": baseline["roc_auc"],
                "fused_roc_auc": fused["roc_auc"],
                "delta_roc_auc": fused["roc_auc"] - baseline["roc_auc"],
                "baseline_eer": baseline["eer"],
                "fused_eer": fused["eer"],
                "delta_eer": fused["eer"] - baseline["eer"],
                "baseline_accuracy_at_eer_threshold": baseline["accuracy_at_eer_threshold"],
                "fused_accuracy_at_eer_threshold": fused["accuracy_at_eer_threshold"],
            }
        )

    if not rows:
        raise ValueError("No valid bootstrap samples were produced.")
    return pd.DataFrame(rows)


def confidence_interval(values: Iterable[float], confidence: float) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    alpha = 1.0 - confidence
    low = 100.0 * alpha / 2.0
    high = 100.0 * (1.0 - alpha / 2.0)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "ci_lower": float(np.percentile(arr, low)),
        "ci_upper": float(np.percentile(arr, high)),
    }


def summary_rows(
    clean: pd.DataFrame,
    bootstraps: pd.DataFrame,
    confidence: float,
    stratified: bool,
    seed: int,
) -> pd.DataFrame:
    original_baseline = metric_row(clean, BASELINE_SCORE_COL)
    original_fused = metric_row(clean, FUSED_SCORE_COL)

    metric_specs = [
        ("ArcFace cosine baseline", "roc_auc", original_baseline["roc_auc"], "baseline_roc_auc"),
        ("Quality-aware fusion", "roc_auc", original_fused["roc_auc"], "fused_roc_auc"),
        (
            "Quality-aware minus baseline",
            "delta_roc_auc",
            original_fused["roc_auc"] - original_baseline["roc_auc"],
            "delta_roc_auc",
        ),
        ("ArcFace cosine baseline", "eer", original_baseline["eer"], "baseline_eer"),
        ("Quality-aware fusion", "eer", original_fused["eer"], "fused_eer"),
        (
            "Quality-aware minus baseline",
            "delta_eer",
            original_fused["eer"] - original_baseline["eer"],
            "delta_eer",
        ),
    ]

    label_counts = clean["label"].value_counts().to_dict()
    rows = []
    for method, metric, original_value, bootstrap_col in metric_specs:
        interval = confidence_interval(bootstraps[bootstrap_col], confidence)
        rows.append(
            {
                "method": method,
                "metric": metric,
                "original_value": original_value,
                **interval,
                "confidence_level": confidence,
                "bootstrap_iterations": len(bootstraps),
                "stratified_bootstrap": stratified,
                "seed": seed,
                "valid_pairs": len(clean),
                "genuine_pairs": int(label_counts.get(1, 0)),
                "impostor_pairs": int(label_counts.get(0, 0)),
            }
        )

    return pd.DataFrame(rows)


def save_auc_plot(
    bootstraps: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    baseline_original = summary[
        (summary["method"] == "ArcFace cosine baseline") & (summary["metric"] == "roc_auc")
    ]["original_value"].iloc[0]
    fused_original = summary[
        (summary["method"] == "Quality-aware fusion") & (summary["metric"] == "roc_auc")
    ]["original_value"].iloc[0]

    plt.figure(figsize=(7.2, 4.8))
    plt.hist(bootstraps["baseline_roc_auc"], bins=30, alpha=0.62, label="ArcFace baseline")
    plt.hist(bootstraps["fused_roc_auc"], bins=30, alpha=0.62, label="Quality-aware fusion")
    plt.axvline(baseline_original, color="#315eaa", linestyle="--", linewidth=2)
    plt.axvline(fused_original, color="#b95444", linestyle="--", linewidth=2)
    plt.xlabel("ROC-AUC")
    plt.ylabel("Bootstrap count")
    plt.title("Bootstrap ROC-AUC Stability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bootstrap_auc_distribution.png", dpi=300)
    plt.close()


def save_delta_plot(bootstraps: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(7.2, 4.8))
    plt.hist(bootstraps["delta_roc_auc"], bins=30, color="#1f7a68", alpha=0.78)
    plt.axvline(0, color="#18212f", linestyle="--", linewidth=2, label="No improvement")
    plt.xlabel("ROC-AUC delta: quality-aware fusion minus baseline")
    plt.ylabel("Bootstrap count")
    plt.title("Bootstrap Improvement Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bootstrap_delta_auc_distribution.png", dpi=300)
    plt.close()


def run_bootstrap(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean = prepare_results(args.pair_results)
    bootstraps = bootstrap_metrics(
        clean=clean,
        iterations=args.iterations,
        seed=args.seed,
        stratified=not args.unstratified,
    )
    summary = summary_rows(
        clean=clean,
        bootstraps=bootstraps,
        confidence=args.confidence,
        stratified=not args.unstratified,
        seed=args.seed,
    )

    bootstrap_path = output_dir / "bootstrap_metric_samples.csv"
    summary_path = output_dir / "bootstrap_confidence_intervals.csv"
    bootstraps.to_csv(bootstrap_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_auc_plot(bootstraps, summary, output_dir)
    save_delta_plot(bootstraps, output_dir)

    label_counts = clean["label"].value_counts().to_dict()
    print("Bootstrap confidence interval analysis complete.")
    print(f"Input results: {args.pair_results}")
    print(f"Valid scored pairs: {len(clean)}")
    print(f"Genuine pairs: {int(label_counts.get(1, 0))}")
    print(f"Impostor pairs: {int(label_counts.get(0, 0))}")
    print(f"Bootstrap samples used: {len(bootstraps)}")
    print(f"Summary: {summary_path}")
    print(f"Samples: {bootstrap_path}")
    print(f"Plots: {output_dir}")
    print()
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for Face Bridge verification metrics."
    )
    parser.add_argument(
        "--pair-results",
        default="results_streamed_arcface/pair_results.csv",
        help="Pair result CSV produced by the verification experiment.",
    )
    parser.add_argument(
        "--output",
        default="results_bootstrap_ci",
        help="Output directory for bootstrap summaries and plots.",
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--unstratified",
        action="store_true",
        help="Use ordinary bootstrap instead of preserving label counts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_bootstrap(parse_args())
