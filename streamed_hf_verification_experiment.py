"""
Streamed Hugging Face ID-to-selfie verification experiment.

This runner avoids cloning or pre-downloading the full image dataset. It streams
the Hugging Face dataset metadata, forms genuine/impostor pairs from remote
paths, then downloads each required image only to a temporary file while DeepFace
extracts quality scores and embeddings.
"""

import argparse
import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


HF_DATASET = "ud-biometrics/Selfie-and-ID-Dataset"
HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def hf_path_to_repo_path(hf_path: str) -> str:
    marker = "/files/"
    if marker not in hf_path:
        raise ValueError(f"Unexpected Hugging Face image path: {hf_path}")
    return f"files/{hf_path.split(marker, 1)[1]}"


def hf_path_to_revision(hf_path: str) -> str:
    if "@" not in hf_path:
        return "main"
    after_at = hf_path.split("@", 1)[1]
    return after_at.split("/", 1)[0]


def hf_path_to_url(hf_path: str) -> str:
    revision = hf_path_to_revision(hf_path)
    repo_path = hf_path_to_repo_path(hf_path)
    quoted_path = urllib.parse.quote(repo_path)
    return f"{HF_RESOLVE_BASE}/{revision}/{quoted_path}?download=true"


def iter_hf_tree_files(dataset_name: str, revision: str):
    api_url = f"https://huggingface.co/api/datasets/{dataset_name}/tree/{revision}/files?recursive=true"
    with urllib.request.urlopen(api_url) as response:
        entries = json.loads(response.read().decode("utf-8"))

    for entry in entries:
        if entry.get("type") == "file":
            yield entry["path"]


def collect_streamed_image_paths(dataset_name: str, revision: str, max_people: Optional[int]) -> Dict[int, Dict[str, List[str]]]:
    people: Dict[int, Dict[str, List[str]]] = {}

    for repo_path in tqdm(iter_hf_tree_files(dataset_name, revision), desc="Streaming metadata"):
        extension = Path(repo_path).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue

        parts = repo_path.split("/")
        if len(parts) < 3 or parts[0] != "files":
            continue

        person_label = int(parts[1])
        hf_path = f"hf://datasets/{dataset_name}@{revision}/{repo_path}"
        people.setdefault(person_label, {"id": [], "selfie": []})
        filename = Path(repo_path).name.lower()
        if filename.startswith("id_"):
            people[person_label]["id"].append(hf_path)
        elif filename.startswith("selfie_"):
            people[person_label]["selfie"].append(hf_path)

        if max_people is not None and len(people) >= max_people:
            complete_people = [
                person
                for person, paths in people.items()
                if paths["id"] and paths["selfie"]
            ]
            if len(complete_people) >= max_people:
                break

    return people


def build_pairs(people: Dict[int, Dict[str, List[str]]]) -> List[Dict[str, object]]:
    complete_people = [
        person
        for person, paths in people.items()
        if paths["id"] and paths["selfie"]
    ]
    sorted_people = sorted(complete_people)
    pairs: List[Dict[str, object]] = []

    for index, person in enumerate(sorted_people):
        id_path = sorted(people[person]["id"])[0]
        selfies = sorted(people[person]["selfie"])
        for selfie_path in selfies:
            pairs.append({
                "id_path": id_path,
                "selfie_path": selfie_path,
                "label": 1,
            })

        next_person = sorted_people[(index + 1) % len(sorted_people)]
        impostor_selfies = sorted(people[next_person]["selfie"])
        if impostor_selfies:
            pairs.append({
                "id_path": id_path,
                "selfie_path": impostor_selfies[0],
                "label": 0,
            })

    return pairs


def download_to_tempfile(hf_path: str, temp_dir: str) -> str:
    suffix = Path(hf_path_to_repo_path(hf_path)).suffix.lower()
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir)
    handle.close()

    with urllib.request.urlopen(hf_path_to_url(hf_path)) as response:
        Path(handle.name).write_bytes(response.read())
    return handle.name


def analyze_remote_image(
    hf_path: str,
    temp_dir: str,
    model_name: str,
    detector_backend: str,
) -> Dict[str, object]:
    temp_path = download_to_tempfile(hf_path, temp_dir)
    try:
        return {
            "embedding": get_embedding(temp_path, model_name, detector_backend),
            "quality": image_quality_scores(temp_path, detector_backend),
        }
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass


def run_streamed_experiment(
    dataset_name: str,
    revision: str,
    output_dir: str,
    model_name: str,
    detector_backend: str,
    max_people: Optional[int],
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    people = collect_streamed_image_paths(dataset_name, revision, max_people)
    pairs = build_pairs(people)
    if not pairs:
        raise RuntimeError("No ID/selfie pairs could be formed from the streamed dataset.")

    pd.DataFrame(pairs).to_csv(os.path.join(output_dir, "streamed_pairs.csv"), index=False)

    # Import DeepFace-backed helpers after Hugging Face metadata streaming to
    # avoid TensorFlow/HF runtime lock interactions seen on some local setups.
    global cosine_similarity
    global evaluate_results
    global get_embedding
    global image_quality_scores
    global quality_aware_score
    global save_quality_plot
    global save_roc_plot
    global save_score_distribution_plot
    from id_selfie_verification_experiment import (
        cosine_similarity,
        evaluate_results,
        get_embedding,
        image_quality_scores,
        quality_aware_score,
        save_quality_plot,
        save_roc_plot,
        save_score_distribution_plot,
    )

    image_cache: Dict[str, Dict[str, object]] = {}
    rows = []

    with tempfile.TemporaryDirectory(prefix="hf_streamed_faces_") as temp_dir:
        for pair in tqdm(pairs, desc="Running streamed verification"):
            id_path = str(pair["id_path"])
            selfie_path = str(pair["selfie_path"])
            label = int(pair["label"])

            for hf_path in [id_path, selfie_path]:
                if hf_path not in image_cache:
                    image_cache[hf_path] = analyze_remote_image(
                        hf_path,
                        temp_dir,
                        model_name,
                        detector_backend,
                    )

            id_result = image_cache[id_path]
            selfie_result = image_cache[selfie_path]
            id_embedding = id_result["embedding"]
            selfie_embedding = selfie_result["embedding"]
            id_quality = id_result["quality"]
            selfie_quality = selfie_result["quality"]

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
    metrics_path = os.path.join(output_dir, "summary_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    save_roc_plot(clean, output_dir, "fused_score")
    save_score_distribution_plot(clean, output_dir, "fused_score")
    save_quality_plot(clean, output_dir)

    print("\nStreamed experiment complete.")
    print(f"Streamed pairs: {os.path.join(output_dir, 'streamed_pairs.csv')}")
    print(f"Pair-level results: {results_path}")
    print(f"Summary metrics: {metrics_path}")
    print(f"Plots saved in: {output_dir}")
    print("\nSummary Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamed Hugging Face ID-to-selfie verification experiment")
    parser.add_argument("--dataset", default=HF_DATASET, help="Hugging Face dataset repo ID")
    parser.add_argument("--revision", default="main", help="Dataset revision to stream")
    parser.add_argument("--output", default="results_streamed", help="Output directory")
    parser.add_argument("--model", default="Facenet512", help="DeepFace model name")
    parser.add_argument("--detector", default="opencv", help="DeepFace detector backend")
    parser.add_argument("--max-people", type=int, default=None, help="Optional cap on streamed people")
    args = parser.parse_args()

    run_streamed_experiment(
        dataset_name=args.dataset,
        revision=args.revision,
        output_dir=args.output,
        model_name=args.model,
        detector_backend=args.detector,
        max_people=args.max_people,
    )
