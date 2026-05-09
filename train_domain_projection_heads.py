"""
Prototype two-domain projection heads for ID-to-selfie verification.

This is Plan B: keep ArcFace frozen as the base face encoder, then train two
small residual domain adapters:
  ID image embedding      -> ID adapter
  selfie image embedding  -> selfie adapter

The heads are trained with binary contrastive verification loss on ID/selfie
pairs. This is a prototype because the public Kaggle sample is small.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ID_PREFIXES = ("passport", "national_id")


@dataclass
class IdentityImages:
    identity: str
    id_paths: List[str]
    selfie_paths: List[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def collect_public_sample(root: Path) -> List[IdentityImages]:
    sample_root = root / "Selfie & id data - public sample"
    identities = []
    if not sample_root.exists():
        return identities

    for person_dir in sorted(path for path in sample_root.iterdir() if path.is_dir()):
        id_paths = sorted(str(path) for path in (person_dir / "docs").glob("*") if is_image(path))
        selfie_paths = []
        for folder in ["selfies", "archive_selfies"]:
            folder_path = person_dir / folder
            if folder_path.exists():
                selfie_paths.extend(str(path) for path in folder_path.glob("*") if is_image(path))

        if id_paths and selfie_paths:
            identities.append(IdentityImages(f"public_{person_dir.name}", id_paths, sorted(selfie_paths)))
    return identities


def collect_axon_samples(root: Path) -> List[IdentityImages]:
    sample_root = root / "AxonLabs_Diverse Selfie & ID Photo Dataset - samples"
    identities = []
    if not sample_root.exists():
        return identities

    for person_dir in sorted(path for path in sample_root.glob("*/*") if path.is_dir()):
        image_paths = [path for path in person_dir.iterdir() if path.is_file() and is_image(path)]
        id_paths = [
            str(path)
            for path in image_paths
            if path.name.lower().startswith(ID_PREFIXES)
        ]
        selfie_paths = [
            str(path)
            for path in image_paths
            if not path.name.lower().startswith(ID_PREFIXES)
        ]
        if id_paths and selfie_paths:
            identity = "axon_" + "_".join(person_dir.parts[-2:])
            identities.append(IdentityImages(identity, sorted(id_paths), sorted(selfie_paths)))
    return identities


def collect_identities(dataset_root: Path) -> List[IdentityImages]:
    identities = collect_public_sample(dataset_root) + collect_axon_samples(dataset_root)
    return sorted(identities, key=lambda item: item.identity)


def embedding_cache_key(path: str) -> str:
    return str(Path(path).resolve())


def load_embedding_cache(cache_path: Path) -> Dict[str, np.ndarray]:
    if not cache_path.exists():
        return {}
    loaded = np.load(cache_path, allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}


def save_embedding_cache(cache_path: Path, cache: Dict[str, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **cache)


def extract_embeddings(
    identities: List[IdentityImages],
    cache_path: Path,
    model_name: str,
    detector: str,
) -> Dict[str, np.ndarray]:
    from id_selfie_verification_experiment import get_embedding

    cache = load_embedding_cache(cache_path)
    all_paths = sorted({
        path
        for identity in identities
        for path in identity.id_paths + identity.selfie_paths
    })

    changed = False
    for path in tqdm(all_paths, desc="Caching ArcFace embeddings"):
        key = embedding_cache_key(path)
        if key in cache:
            continue
        embedding = get_embedding(path, model_name=model_name, detector_backend=detector)
        if embedding is not None:
            cache[key] = embedding.astype(np.float32)
        changed = True

    if changed:
        save_embedding_cache(cache_path, cache)
    return cache


def filter_identities_with_embeddings(
    identities: List[IdentityImages],
    embeddings: Dict[str, np.ndarray],
) -> List[IdentityImages]:
    filtered = []
    for identity in identities:
        id_paths = [path for path in identity.id_paths if embedding_cache_key(path) in embeddings]
        selfie_paths = [path for path in identity.selfie_paths if embedding_cache_key(path) in embeddings]
        if id_paths and selfie_paths:
            filtered.append(IdentityImages(identity.identity, id_paths, selfie_paths))
    return filtered


def split_identities(identities: List[IdentityImages], seed: int):
    shuffled = identities[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = max(1, int(0.7 * n))
    val_end = max(train_end + 1, int(0.85 * n))
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def make_pairs(
    identities: List[IdentityImages],
    embeddings: Dict[str, np.ndarray],
    negatives_per_positive: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    all_selfies = [
        (identity.identity, path)
        for identity in identities
        for path in identity.selfie_paths
    ]

    for identity in identities:
        for id_path in identity.id_paths:
            for selfie_path in identity.selfie_paths:
                rows.append({
                    "id_path": id_path,
                    "selfie_path": selfie_path,
                    "id_identity": identity.identity,
                    "selfie_identity": identity.identity,
                    "label": 1,
                })

                negative_pool = [
                    candidate
                    for candidate in all_selfies
                    if candidate[0] != identity.identity
                ]
                for negative_identity, negative_selfie in rng.sample(
                    negative_pool,
                    k=min(negatives_per_positive, len(negative_pool)),
                ):
                    rows.append({
                        "id_path": id_path,
                        "selfie_path": negative_selfie,
                        "id_identity": identity.identity,
                        "selfie_identity": negative_identity,
                        "label": 0,
                    })

    pairs = pd.DataFrame(rows)
    pairs["id_key"] = pairs["id_path"].map(embedding_cache_key)
    pairs["selfie_key"] = pairs["selfie_path"].map(embedding_cache_key)
    pairs = pairs[
        pairs["id_key"].isin(embeddings)
        & pairs["selfie_key"].isin(embeddings)
    ].copy()
    return pairs.drop(columns=["id_key", "selfie_key"])


def pair_tensors(pairs: pd.DataFrame, embeddings: Dict[str, np.ndarray]):
    id_vectors = np.stack([
        embeddings[embedding_cache_key(path)]
        for path in pairs["id_path"]
    ]).astype(np.float32)
    selfie_vectors = np.stack([
        embeddings[embedding_cache_key(path)]
        for path in pairs["selfie_path"]
    ]).astype(np.float32)
    labels = pairs["label"].astype(np.float32).values
    return (
        torch.from_numpy(id_vectors),
        torch.from_numpy(selfie_vectors),
        torch.from_numpy(labels),
    )


class ResidualDomainAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, values):
        adapted = values + self.residual_scale.clamp(0.0, 1.0) * self.adapter(values)
        return nn.functional.normalize(adapted, dim=1)


class DomainProjectionModel(nn.Module):
    def __init__(self, input_dim: int, adapter_hidden_dim: int):
        super().__init__()
        self.id_head = ResidualDomainAdapter(input_dim, adapter_hidden_dim)
        self.selfie_head = ResidualDomainAdapter(input_dim, adapter_hidden_dim)
        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, id_vectors, selfie_vectors):
        id_projected = self.id_head(id_vectors)
        selfie_projected = self.selfie_head(selfie_vectors)
        cosine = torch.sum(id_projected * selfie_projected, dim=1)
        logits = self.logit_scale.clamp(1.0, 30.0) * cosine + self.bias
        return logits, cosine


def baseline_scores(pairs: pd.DataFrame, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    scores = []
    for _, row in pairs.iterrows():
        id_embedding = embeddings[embedding_cache_key(row["id_path"])]
        selfie_embedding = embeddings[embedding_cache_key(row["selfie_path"])]
        denom = np.linalg.norm(id_embedding) * np.linalg.norm(selfie_embedding)
        score = 0.0 if denom == 0 else float(np.dot(id_embedding, selfie_embedding) / denom)
        scores.append((score + 1.0) / 2.0)
    return np.array(scores, dtype=np.float32)


def calculate_eer(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def metrics_for_scores(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    auc = roc_auc_score(y_true, scores)
    eer, threshold = calculate_eer(y_true, scores)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(auc),
        "eer": float(eer),
        "eer_threshold": float(threshold),
        "accuracy_at_eer_threshold": float(accuracy_score(y_true, y_pred)),
        "false_match_rate": float(fp / (fp + tn)) if (fp + tn) else np.nan,
        "false_non_match_rate": float(fn / (fn + tp)) if (fn + tp) else np.nan,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def train_model(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
    adapter_hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Tuple[DomainProjectionModel, List[Dict[str, float]]]:
    train_id, train_selfie, train_y = pair_tensors(train_pairs, embeddings)
    val_id, val_selfie, val_y = pair_tensors(val_pairs, embeddings)
    input_dim = train_id.shape[1]

    model = DomainProjectionModel(input_dim, adapter_hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(train_id, train_selfie, train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    model.eval()
    with torch.no_grad():
        val_logits, _ = model(val_id, val_selfie)
        val_scores = torch.sigmoid(val_logits).numpy()
        best_val_auc = float(roc_auc_score(val_y.numpy(), val_scores))
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_id, batch_selfie, batch_y in loader:
            optimizer.zero_grad()
            logits, _ = model(batch_id, batch_selfie)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)

        train_loss = total_loss / len(train_y)
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(val_id, val_selfie)
            val_scores = torch.sigmoid(val_logits).numpy()
            val_auc = roc_auc_score(val_y.numpy(), val_scores)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_roc_auc": float(val_auc)})

        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history


def projection_scores(model: DomainProjectionModel, pairs: pd.DataFrame, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    id_vectors, selfie_vectors, _ = pair_tensors(pairs, embeddings)
    model.eval()
    with torch.no_grad():
        logits, _ = model(id_vectors, selfie_vectors)
    return torch.sigmoid(logits).numpy()


def save_comparison_roc(y_true, baseline, projected, output_dir: Path) -> None:
    baseline_fpr, baseline_tpr, _ = roc_curve(y_true, baseline)
    projected_fpr, projected_tpr, _ = roc_curve(y_true, projected)
    baseline_auc = roc_auc_score(y_true, baseline)
    projected_auc = roc_auc_score(y_true, projected)

    plt.figure(figsize=(6.4, 5.2))
    plt.plot(baseline_fpr, baseline_tpr, label=f"Frozen ArcFace baseline (AUC={baseline_auc:.3f})")
    plt.plot(projected_fpr, projected_tpr, label=f"Two residual adapters (AUC={projected_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Match Rate")
    plt.ylabel("True Match Rate")
    plt.title("Two-Domain Residual Adapter Prototype")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "projection_heads_vs_arcface_roc.png", dpi=300)
    plt.close()


def save_history_plot(history: List[Dict[str, float]], output_dir: Path) -> None:
    history_df = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(6.4, 4.6))
    ax1.plot(history_df["epoch"], history_df["train_loss"], label="Train loss", color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(history_df["epoch"], history_df["val_roc_auc"], label="Val ROC-AUC", color="tab:orange")
    ax2.set_ylabel("Val ROC-AUC", color="tab:orange")
    plt.title("Projection Head Training")
    fig.tight_layout()
    fig.savefig(output_dir / "projection_head_training_history.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train prototype ID/selfie domain projection heads")
    parser.add_argument("--dataset-root", default="kaggle_downloads/axon_id_selfie")
    parser.add_argument("--output", default="results_projection_heads")
    parser.add_argument("--cache", default=".cache/axon_arcface_embeddings.npz")
    parser.add_argument("--model", default="ArcFace")
    parser.add_argument("--detector", default="opencv")
    parser.add_argument("--adapter-hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    identities = collect_identities(Path(args.dataset_root))
    embeddings = extract_embeddings(
        identities,
        Path(args.cache),
        model_name=args.model,
        detector=args.detector,
    )
    identities = filter_identities_with_embeddings(identities, embeddings)
    train_ids, val_ids, test_ids = split_identities(identities, args.seed)

    train_pairs = make_pairs(train_ids, embeddings, args.negatives_per_positive, args.seed)
    val_pairs = make_pairs(val_ids, embeddings, args.negatives_per_positive, args.seed + 1)
    test_pairs = make_pairs(test_ids, embeddings, args.negatives_per_positive, args.seed + 2)

    for name, ids, pairs in [
        ("train", train_ids, train_pairs),
        ("val", val_ids, val_pairs),
        ("test", test_ids, test_pairs),
    ]:
        pairs.to_csv(output_dir / f"{name}_pairs.csv", index=False)
        print(f"{name}: {len(ids)} identities, {len(pairs)} pairs")

    model, history = train_model(
        train_pairs,
        val_pairs,
        embeddings,
        adapter_hidden_dim=args.adapter_hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    save_history_plot(history, output_dir)

    y_true = test_pairs["label"].astype(int).values
    baseline = baseline_scores(test_pairs, embeddings)
    projected = projection_scores(model, test_pairs, embeddings)

    metrics = pd.DataFrame([
        {
            "method": "Frozen ArcFace cosine baseline",
            **metrics_for_scores(y_true, baseline),
        },
        {
            "method": "Prototype ID/selfie residual adapters",
            **metrics_for_scores(y_true, projected),
        },
    ])
    metrics.to_csv(output_dir / "projection_head_comparison_metrics.csv", index=False)
    save_comparison_roc(y_true, baseline, projected, output_dir)

    torch.save(model.state_dict(), output_dir / "projection_heads.pt")

    summary = {
        "dataset_root": args.dataset_root,
        "usable_identities": len(identities),
        "train_identities": [identity.identity for identity in train_ids],
        "val_identities": [identity.identity for identity in val_ids],
        "test_identities": [identity.identity for identity in test_ids],
        "embedding_count": len(embeddings),
        "adapter_hidden_dim": args.adapter_hidden_dim,
        "epochs": args.epochs,
        "note": "Prototype only: the public sample is small for training domain-specific heads.",
    }
    with (output_dir / "run_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("\nProjection-head prototype complete.")
    print(metrics.to_string(index=False))
    print(f"Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
