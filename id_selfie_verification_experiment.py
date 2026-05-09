"""
ID-to-Selfie Face Verification Experiment
----------------------------------------
Purpose:
Generate testable poster results for a domain-aware ID photo vs selfie verification idea.

What this script does:
1. Reads a CSV of image pairs.
2. Computes basic image quality scores: brightness, blur, and face detection confidence proxy.
3. Extracts facial embeddings using DeepFace.
4. Computes cosine similarity between ID and selfie embeddings.
5. Applies a simple quality-aware score fusion.
6. Evaluates accuracy, ROC-AUC, EER, false match rate, and false non-match rate.
7. Saves results and poster-ready plots.

Expected folder structure:
project/
  id_selfie_verification_experiment.py
  pairs.csv
  data/
    id/
      person001_id.jpg
      person002_id.jpg
    selfie/
      person001_selfie.jpg
      person002_selfie.jpg
      impostor_selfie.jpg

Expected pairs.csv format:
id_path,selfie_path,label
./data/id/person001_id.jpg,./data/selfie/person001_selfie.jpg,1
./data/id/person001_id.jpg,./data/selfie/impostor_selfie.jpg,0

label = 1 means genuine match, label = 0 means non-match/impostor pair.

Install:
pip install deepface opencv-python pandas numpy scikit-learn matplotlib tqdm

Run:
python id_selfie_verification_experiment.py --pairs pairs.csv --output results
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from deepface import DeepFace


# -----------------------------
# Utility functions
# -----------------------------

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_image_bgr(path: str) -> Optional[np.ndarray]:
    image = cv2.imread(path)
    if image is None:
        print(f"Warning: Could not read image: {path}")
    return image


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# -----------------------------
# Image quality assessment
# -----------------------------

def brightness_score(image_bgr: np.ndarray) -> float:
    """
    Returns a normalized brightness quality score between 0 and 1.
    Ideal brightness is assumed near the middle of the 0-255 range.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    score = 1.0 - abs(mean_brightness - 127.5) / 127.5
    return float(np.clip(score, 0, 1))


def blur_score(image_bgr: np.ndarray) -> float:
    """
    Returns a normalized sharpness score between 0 and 1 using variance of Laplacian.
    Higher values indicate sharper images.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 300 is a practical normalization constant; tune if needed.
    score = variance / 300.0
    return float(np.clip(score, 0, 1))


def face_presence_score(image_path: str, detector_backend: str = "opencv") -> float:
    """
    Simple proxy for face visibility.
    Returns 1 if DeepFace detects a face, otherwise 0.
    """
    try:
        _ = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=detector_backend,
            enforce_detection=True,
            align=True,
        )
        return 1.0
    except Exception:
        return 0.0


def image_quality_scores(image_path: str, detector_backend: str = "opencv") -> Dict[str, float]:
    image = read_image_bgr(image_path)
    if image is None:
        return {
            "brightness_score": 0.0,
            "blur_score": 0.0,
            "face_presence_score": 0.0,
            "overall_quality_score": 0.0,
        }

    b_score = brightness_score(image)
    sh_score = blur_score(image)
    fp_score = face_presence_score(image_path, detector_backend)

    # Simple weighted quality fusion.
    # You can adjust these weights and report them in your methodology.
    overall = (0.35 * b_score) + (0.35 * sh_score) + (0.30 * fp_score)

    return {
        "brightness_score": b_score,
        "blur_score": sh_score,
        "face_presence_score": fp_score,
        "overall_quality_score": float(np.clip(overall, 0, 1)),
    }


# -----------------------------
# Embedding extraction
# -----------------------------

def get_embedding(
    image_path: str,
    model_name: str = "Facenet512",
    detector_backend: str = "opencv",
) -> Optional[np.ndarray]:
    """
    Extracts a face embedding using DeepFace.
    Good model options: Facenet512, ArcFace, VGG-Face.
    """
    try:
        reps = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True,
            align=True,
        )
        if not reps:
            return None
        return np.array(reps[0]["embedding"], dtype=np.float32)
    except Exception as exc:
        print(f"Embedding failed for {image_path}: {exc}")
        return None


# -----------------------------
# Scoring and evaluation
# -----------------------------

def quality_aware_score(similarity: float, id_quality: float, selfie_quality: float) -> float:
    """
    Combines embedding similarity with image quality.
    Cosine similarity usually ranges from -1 to 1, so it is normalized to 0 to 1.
    """
    normalized_similarity = (similarity + 1.0) / 2.0
    pair_quality = (id_quality + selfie_quality) / 2.0

    # Score fusion: mostly identity similarity, partially quality confidence.
    fused = (0.80 * normalized_similarity) + (0.20 * pair_quality)
    return float(np.clip(fused, 0, 1))


def calculate_eer(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Calculates Equal Error Rate and the threshold at which it occurs.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    return float(eer), float(threshold)


def evaluate_results(df: pd.DataFrame, score_col: str = "fused_score") -> Dict[str, float]:
    y_true = df["label"].astype(int).values
    scores = df[score_col].values

    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan
    eer, eer_threshold = calculate_eer(y_true, scores)

    y_pred = (scores >= eer_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    false_match_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    false_non_match_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    return {
        "roc_auc": float(auc),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "accuracy_at_eer_threshold": float(acc),
        "false_match_rate": float(false_match_rate),
        "false_non_match_rate": float(false_non_match_rate),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


# -----------------------------
# Plotting
# -----------------------------

def save_roc_plot(df: pd.DataFrame, output_dir: str, score_col: str = "fused_score") -> None:
    y_true = df["label"].astype(int).values
    scores = df[score_col].values
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Match Rate")
    plt.ylabel("True Match Rate")
    plt.title("ROC Curve: ID-to-Selfie Verification")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
    plt.close()


def save_score_distribution_plot(df: pd.DataFrame, output_dir: str, score_col: str = "fused_score") -> None:
    genuine = df[df["label"] == 1][score_col]
    impostor = df[df["label"] == 0][score_col]

    plt.figure(figsize=(6, 5))
    plt.hist(genuine, bins=20, alpha=0.7, label="Genuine pairs")
    plt.hist(impostor, bins=20, alpha=0.7, label="Impostor pairs")
    plt.xlabel("Fused Match Score")
    plt.ylabel("Number of Pairs")
    plt.title("Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distribution.png"), dpi=300)
    plt.close()


def save_quality_plot(df: pd.DataFrame, output_dir: str) -> None:
    labels = ["ID Quality", "Selfie Quality"]
    means = [df["id_overall_quality_score"].mean(), df["selfie_overall_quality_score"].mean()]

    plt.figure(figsize=(5, 4))
    plt.bar(labels, means)
    plt.ylim(0, 1)
    plt.ylabel("Average Quality Score")
    plt.title("Average Image Quality by Domain")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_quality_by_domain.png"), dpi=300)
    plt.close()


# -----------------------------
# Main experiment
# -----------------------------

def run_experiment(
    pairs_csv: str,
    output_dir: str,
    model_name: str = "Facenet512",
    detector_backend: str = "opencv",
) -> None:
    ensure_dir(output_dir)

    pairs = pd.read_csv(pairs_csv)
    required_cols = {"id_path", "selfie_path", "label"}
    missing = required_cols - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs.csv is missing columns: {missing}")

    embedding_cache: Dict[str, Optional[np.ndarray]] = {}
    quality_cache: Dict[str, Dict[str, float]] = {}

    rows = []

    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Running verification"):
        id_path = row["id_path"]
        selfie_path = row["selfie_path"]
        label = int(row["label"])

        for path in [id_path, selfie_path]:
            if path not in embedding_cache:
                embedding_cache[path] = get_embedding(path, model_name, detector_backend)
            if path not in quality_cache:
                quality_cache[path] = image_quality_scores(path, detector_backend)

        id_embedding = embedding_cache[id_path]
        selfie_embedding = embedding_cache[selfie_path]

        id_quality = quality_cache[id_path]
        selfie_quality = quality_cache[selfie_path]

        if id_embedding is None or selfie_embedding is None:
            raw_similarity = np.nan
            fused_score = 0.0
            status = "embedding_failed"
        else:
            raw_similarity = cosine_similarity(id_embedding, selfie_embedding)
            fused_score = quality_aware_score(
                raw_similarity,
                id_quality["overall_quality_score"],
                selfie_quality["overall_quality_score"],
            )
            status = "ok"

        rows.append({
            "id_path": id_path,
            "selfie_path": selfie_path,
            "label": label,
            "raw_cosine_similarity": raw_similarity,
            "fused_score": fused_score,
            "status": status,
            "id_brightness_score": id_quality["brightness_score"],
            "id_blur_score": id_quality["blur_score"],
            "id_face_presence_score": id_quality["face_presence_score"],
            "id_overall_quality_score": id_quality["overall_quality_score"],
            "selfie_brightness_score": selfie_quality["brightness_score"],
            "selfie_blur_score": selfie_quality["blur_score"],
            "selfie_face_presence_score": selfie_quality["face_presence_score"],
            "selfie_overall_quality_score": selfie_quality["overall_quality_score"],
        })

    results = pd.DataFrame(rows)
    results_path = os.path.join(output_dir, "pair_results.csv")
    results.to_csv(results_path, index=False)

    clean = results[results["status"] == "ok"].copy()
    if len(clean) < 2 or len(clean["label"].unique()) < 2:
        print("Not enough valid genuine and impostor pairs for full evaluation.")
        print(f"Saved pair-level results to: {results_path}")
        return

    metrics = evaluate_results(clean, "fused_score")
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(output_dir, "summary_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    save_roc_plot(clean, output_dir, "fused_score")
    save_score_distribution_plot(clean, output_dir, "fused_score")
    save_quality_plot(clean, output_dir)

    print("\nExperiment complete.")
    print(f"Pair-level results: {results_path}")
    print(f"Summary metrics: {metrics_path}")
    print(f"Plots saved in: {output_dir}")
    print("\nSummary Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ID-to-selfie verification experiment")
    parser.add_argument("--pairs", required=True, help="Path to pairs.csv")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--model", default="Facenet512", help="DeepFace model name: Facenet512, ArcFace, VGG-Face, etc.")
    parser.add_argument("--detector", default="opencv", help="Detector backend: opencv, mtcnn, retinaface, ssd, etc.")
    args = parser.parse_args()

    run_experiment(
        pairs_csv=args.pairs,
        output_dir=args.output,
        model_name=args.model,
        detector_backend=args.detector,
    )
