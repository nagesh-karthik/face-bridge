# Face Bridge

Quality-aware ID-to-selfie face verification experiments for poster-ready
computer vision results.

Bridging the gap between ID photos and selfies: a modern face verification
approach.

## What This Project Does

This project compares standard face-embedding verification against a
quality-aware score that uses:

- face embedding cosine similarity
- blur/sharpness measured with variance of Laplacian
- brightness/exposure quality
- face-detection visibility

The main idea is to keep a strong published face-recognition baseline, such as
ArcFace or FaceNet, and add interpretable computer-vision quality signals for
ID-photo to selfie verification.

## Repository Contents

- `id_selfie_verification_experiment.py` - main local pair-based experiment
- `streamed_hf_verification_experiment.py` - streamed Hugging Face experiment
- `compare_published_baseline.py` - ArcFace cosine baseline comparison
- `train_domain_projection_heads.py` - prototype ID/selfie adapter-head training
- `pairs.example.csv` - non-sensitive example pair file format
- `results*/` - aggregate metrics and poster-ready plots only

## Data Notice

Raw biometric datasets, face images, identity documents, Kaggle downloads,
local pair lists, model weights, and embedding caches are intentionally excluded
from this repository.

To run locally, place permitted data on your machine and create a private
`pairs.csv` using the format in `pairs.example.csv`:

```csv
id_path,selfie_path,label
./data/id/example_person_id.jpg,./data/selfie/example_person_selfie.jpg,1
./data/id/example_person_id.jpg,./data/selfie/example_impostor_selfie.jpg,0
```

## Example Commands

Run the local pair-based experiment:

```bash
python id_selfie_verification_experiment.py --pairs pairs.csv --output results
```

Compare the published ArcFace-style cosine baseline to quality-aware fusion:

```bash
python compare_published_baseline.py \
  --pair-results results_streamed_arcface/pair_results.csv \
  --output results_comparison_arcface
```

Train the prototype residual ID/selfie adapter heads:

```bash
python train_domain_projection_heads.py \
  --output results_projection_heads_residual \
  --epochs 150
```

## Licensing

Project code is licensed under the MIT License. Original documentation,
analysis, and aggregate figures are licensed under CC BY-NC 4.0 unless
otherwise noted.

Datasets, face images, identity documents, pretrained model weights, and
third-party libraries remain under their own licenses and terms. See
`THIRD_PARTY_NOTICES.md`.
