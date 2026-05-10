"""
Synthetic degradation robustness test for Face Bridge.

This script keeps the main verification experiment intact. It reuses the
existing image-quality, embedding, similarity, fusion, and evaluation helpers
from id_selfie_verification_experiment.py, then applies temporary synthetic
degradations to one image domain before recomputing aggregate metrics.

No degraded face images are saved permanently. Temporary degraded files are
created only so DeepFace can process them, then removed at the end of the run.

Example:
python synthetic_degradation_experiment.py \
  --pairs pairs.csv \
  --output results_synthetic_degradation \
  --model ArcFace \
  --target-domain selfie \
  --max-positive-pairs 52
"""

import argparse
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from id_selfie_verification_experiment import (
    cosine_similarity,
    ensure_dir,
    evaluate_results,
    get_embedding,
    image_quality_scores,
    quality_aware_score,
    read_image_bgr,
)


BASELINE_SCORE_COL = "arcface_cosine_score"
FUSED_SCORE_COL = "fused_score"


def normalized_cosine(value: float) -> float:
    return float(np.clip((value + 1.0) / 2.0, 0, 1))


def identity(image: np.ndarray) -> np.ndarray:
    return image.copy()


def gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def adjust_exposure(image: np.ndarray, alpha: float, beta: float = 0.0) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def jpeg_compress(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image.copy()
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else image.copy()


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, image.shape)
    degraded = image.astype(np.float32) + noise
    return np.clip(degraded, 0, 255).astype(np.uint8)


def degradation_functions(seed: int) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "clean": identity,
        "blur_mild": lambda image: gaussian_blur(image, 5),
        "blur_strong": lambda image: gaussian_blur(image, 11),
        "low_light": lambda image: adjust_exposure(image, alpha=0.55),
        "overexposed": lambda image: adjust_exposure(image, alpha=1.35, beta=38),
        "jpeg_q25": lambda image: jpeg_compress(image, quality=25),
        "gaussian_noise": lambda image: add_gaussian_noise(image, sigma=18.0, seed=seed),
    }


def prepare_pairs(
    pairs_csv: str,
    max_pairs: Optional[int],
    max_positive_pairs: Optional[int],
    seed: int,
) -> pd.DataFrame:
    pairs = pd.read_csv(pairs_csv)
    required_cols = {"id_path", "selfie_path", "label"}
    missing = required_cols - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs CSV is missing columns: {sorted(missing)}")

    pairs = pairs.copy()
    pairs["label"] = pairs["label"].astype(int)
    pairs = pairs.sample(frac=1, random_state=seed).reset_index(drop=True)

    if max_positive_pairs is not None:
        positives = pairs[pairs["label"] == 1].head(max_positive_pairs)
        negatives = pairs[pairs["label"] == 0]
        pairs = pd.concat([positives, negatives], ignore_index=True)
        pairs = pairs.sample(frac=1, random_state=seed).reset_index(drop=True)

    if max_pairs is not None:
        pairs = pairs.head(max_pairs).reset_index(drop=True)

    label_counts = pairs["label"].value_counts().to_dict()
    if len(label_counts) < 2:
        raise ValueError("Need both genuine and impostor pairs for degradation evaluation.")

    return pairs


def save_temp_image(image: np.ndarray, temp_dir: Path, key: str) -> str:
    safe_key = "".join(char if char.isalnum() else "_" for char in key)
    path = temp_dir / f"{safe_key}.jpg"
    cv2.imwrite(str(path), image)
    return str(path)


class ImageAnalysisCache:
    def __init__(
        self,
        model_name: str,
        detector_backend: str,
        temp_dir: Path,
        seed: int,
    ) -> None:
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.temp_dir = temp_dir
        self.seed = seed
        self.embedding_cache: Dict[str, Optional[np.ndarray]] = {}
        self.quality_cache: Dict[str, Dict[str, float]] = {}
        self.degraded_path_cache: Dict[str, Optional[str]] = {}
        self.functions = degradation_functions(seed)

    def materialized_path(self, source_path: str, condition: str, cache_key: str) -> Optional[str]:
        if condition == "clean":
            return source_path
        if cache_key in self.degraded_path_cache:
            return self.degraded_path_cache[cache_key]

        image = read_image_bgr(source_path)
        if image is None:
            self.degraded_path_cache[cache_key] = None
            return None

        degraded = self.functions[condition](image)
        path = save_temp_image(degraded, self.temp_dir, cache_key)
        self.degraded_path_cache[cache_key] = path
        return path

    def analyze(self, source_path: str, condition: str, domain: str) -> Dict[str, object]:
        cache_key = f"{domain}_{condition}_{source_path}"
        path = self.materialized_path(source_path, condition, cache_key)
        if path is None:
            return {
                "embedding": None,
                "quality": {
                    "brightness_score": 0.0,
                    "blur_score": 0.0,
                    "face_presence_score": 0.0,
                    "overall_quality_score": 0.0,
                },
            }

        if cache_key not in self.embedding_cache:
            self.embedding_cache[cache_key] = get_embedding(
                path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
            )
        if cache_key not in self.quality_cache:
            self.quality_cache[cache_key] = image_quality_scores(path, self.detector_backend)

        return {
            "embedding": self.embedding_cache[cache_key],
            "quality": self.quality_cache[cache_key],
        }


def condition_for_domain(condition: str, domain: str, target_domain: str) -> str:
    if target_domain == "both":
        return condition
    return condition if domain == target_domain else "clean"


def safe_metrics(df: pd.DataFrame, score_col: str) -> Dict[str, float]:
    clean = df[df["status"] == "ok"].copy()
    if len(clean) < 2 or len(clean["label"].unique()) < 2:
        return {
            "roc_auc": np.nan,
            "eer": np.nan,
            "eer_threshold": np.nan,
            "accuracy_at_eer_threshold": np.nan,
            "false_match_rate": np.nan,
            "false_non_match_rate": np.nan,
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0,
        }
    return evaluate_results(clean, score_col)


def run_condition(
    pairs: pd.DataFrame,
    condition: str,
    target_domain: str,
    cache: ImageAnalysisCache,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for pair_index, row in tqdm(
        pairs.iterrows(),
        total=len(pairs),
        desc=f"Testing {condition}",
        leave=False,
    ):
        id_condition = condition_for_domain(condition, "id", target_domain)
        selfie_condition = condition_for_domain(condition, "selfie", target_domain)

        id_result = cache.analyze(row["id_path"], id_condition, "id")
        selfie_result = cache.analyze(row["selfie_path"], selfie_condition, "selfie")

        id_quality = id_result["quality"]
        selfie_quality = selfie_result["quality"]
        id_embedding = id_result["embedding"]
        selfie_embedding = selfie_result["embedding"]

        if id_embedding is None or selfie_embedding is None:
            raw_similarity = np.nan
            baseline_score = np.nan
            fused_score = np.nan
            status = "embedding_failed"
        else:
            raw_similarity = cosine_similarity(id_embedding, selfie_embedding)
            baseline_score = normalized_cosine(raw_similarity)
            fused_score = quality_aware_score(
                raw_similarity,
                id_quality["overall_quality_score"],
                selfie_quality["overall_quality_score"],
            )
            status = "ok"

        rows.append(
            {
                "condition": condition,
                "pair_index": pair_index,
                "label": int(row["label"]),
                "status": status,
                "raw_cosine_similarity": raw_similarity,
                BASELINE_SCORE_COL: baseline_score,
                FUSED_SCORE_COL: fused_score,
                "id_overall_quality_score": id_quality["overall_quality_score"],
                "selfie_overall_quality_score": selfie_quality["overall_quality_score"],
                "id_brightness_score": id_quality["brightness_score"],
                "id_blur_score": id_quality["blur_score"],
                "id_face_presence_score": id_quality["face_presence_score"],
                "selfie_brightness_score": selfie_quality["brightness_score"],
                "selfie_blur_score": selfie_quality["blur_score"],
                "selfie_face_presence_score": selfie_quality["face_presence_score"],
            }
        )

    return pd.DataFrame(rows)


def summarize_condition(condition_df: pd.DataFrame) -> Dict[str, object]:
    baseline_metrics = safe_metrics(condition_df, BASELINE_SCORE_COL)
    fused_metrics = safe_metrics(condition_df, FUSED_SCORE_COL)
    valid = condition_df[condition_df["status"] == "ok"]
    label_counts = condition_df["label"].value_counts().to_dict()

    return {
        "condition": condition_df["condition"].iloc[0],
        "total_pairs": len(condition_df),
        "valid_pairs": int(len(valid)),
        "embedding_failed_pairs": int((condition_df["status"] != "ok").sum()),
        "genuine_pairs": int(label_counts.get(1, 0)),
        "impostor_pairs": int(label_counts.get(0, 0)),
        "mean_id_quality": float(condition_df["id_overall_quality_score"].mean()),
        "mean_selfie_quality": float(condition_df["selfie_overall_quality_score"].mean()),
        "baseline_roc_auc": baseline_metrics["roc_auc"],
        "fused_roc_auc": fused_metrics["roc_auc"],
        "delta_roc_auc": fused_metrics["roc_auc"] - baseline_metrics["roc_auc"],
        "baseline_eer": baseline_metrics["eer"],
        "fused_eer": fused_metrics["eer"],
        "delta_eer": fused_metrics["eer"] - baseline_metrics["eer"],
        "baseline_accuracy_at_eer_threshold": baseline_metrics["accuracy_at_eer_threshold"],
        "fused_accuracy_at_eer_threshold": fused_metrics["accuracy_at_eer_threshold"],
    }


def save_auc_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    x = np.arange(len(summary))
    width = 0.36

    plt.figure(figsize=(8.4, 4.8))
    plt.bar(x - width / 2, summary["baseline_roc_auc"], width, label="ArcFace baseline")
    plt.bar(x + width / 2, summary["fused_roc_auc"], width, label="Quality-aware fusion")
    plt.xticks(x, summary["condition"], rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("ROC-AUC")
    plt.title("Synthetic Degradation Robustness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "synthetic_degradation_auc.png", dpi=300)
    plt.close()


def save_quality_plot(summary: pd.DataFrame, output_dir: Path, target_domain: str) -> None:
    quality_col = "mean_selfie_quality" if target_domain == "selfie" else "mean_id_quality"
    if target_domain == "both":
        values = (summary["mean_id_quality"] + summary["mean_selfie_quality"]) / 2.0
        label = "Mean pair quality"
    else:
        values = summary[quality_col]
        label = f"Mean {target_domain} quality"

    plt.figure(figsize=(8.4, 4.8))
    plt.plot(summary["condition"], values, marker="o", linewidth=2)
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel(label)
    plt.title("Quality Score Under Synthetic Degradation")
    plt.tight_layout()
    plt.savefig(output_dir / "synthetic_degradation_quality.png", dpi=300)
    plt.close()


def save_delta_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8.4, 4.8))
    colors = ["#1f7a68" if value >= 0 else "#b95444" for value in summary["delta_roc_auc"]]
    plt.bar(summary["condition"], summary["delta_roc_auc"], color=colors)
    plt.axhline(0, color="#18212f", linestyle="--", linewidth=1.5)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("ROC-AUC delta")
    plt.title("Quality-Aware Fusion Minus Baseline")
    plt.tight_layout()
    plt.savefig(output_dir / "synthetic_degradation_delta_auc.png", dpi=300)
    plt.close()


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    ensure_dir(str(output_dir))

    pairs = prepare_pairs(
        pairs_csv=args.pairs,
        max_pairs=args.max_pairs,
        max_positive_pairs=args.max_positive_pairs,
        seed=args.seed,
    )
    functions = degradation_functions(args.seed)
    selected_conditions = args.conditions or list(functions.keys())

    unknown = [condition for condition in selected_conditions if condition not in functions]
    if unknown:
        raise ValueError(f"Unknown degradation conditions: {unknown}")

    all_rows: List[pd.DataFrame] = []
    summaries: List[Dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="face_bridge_degraded_") as temp_name:
        cache = ImageAnalysisCache(
            model_name=args.model,
            detector_backend=args.detector,
            temp_dir=Path(temp_name),
            seed=args.seed,
        )

        for condition in selected_conditions:
            condition_df = run_condition(pairs, condition, args.target_domain, cache)
            all_rows.append(condition_df)
            summaries.append(summarize_condition(condition_df))

    pair_scores = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame(summaries)

    pair_scores_path = output_dir / "synthetic_degradation_pair_scores.csv"
    summary_path = output_dir / "synthetic_degradation_summary.csv"
    pair_scores.to_csv(pair_scores_path, index=False)
    summary.to_csv(summary_path, index=False)

    save_auc_plot(summary, output_dir)
    save_quality_plot(summary, output_dir, args.target_domain)
    save_delta_plot(summary, output_dir)

    print("Synthetic degradation experiment complete.")
    print(f"Input pairs: {args.pairs}")
    print(f"Target domain: {args.target_domain}")
    print(f"Model: {args.model}")
    print(f"Detector: {args.detector}")
    print(f"Pairs evaluated per condition: {len(pairs)}")
    print(f"Summary: {summary_path}")
    print(f"Pair scores without image paths: {pair_scores_path}")
    print(f"Plots: {output_dir}")
    print()
    display_cols = [
        "condition",
        "valid_pairs",
        "embedding_failed_pairs",
        "mean_selfie_quality",
        "baseline_roc_auc",
        "fused_roc_auc",
        "delta_roc_auc",
        "baseline_eer",
        "fused_eer",
    ]
    print(summary[display_cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic degradation robustness tests.")
    parser.add_argument("--pairs", default="pairs.csv", help="Private pairs CSV with id_path,selfie_path,label.")
    parser.add_argument("--output", default="results_synthetic_degradation")
    parser.add_argument("--model", default="ArcFace", help="DeepFace model name.")
    parser.add_argument("--detector", default="opencv", help="DeepFace detector backend.")
    parser.add_argument(
        "--target-domain",
        choices=["id", "selfie", "both"],
        default="selfie",
        help="Which side of the pair to degrade.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="Optional subset of conditions: clean blur_mild blur_strong low_light overexposed jpeg_q25 gaussian_noise.",
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--max-positive-pairs",
        type=int,
        default=None,
        help="Use all impostor pairs and at most this many genuine pairs to control runtime.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
