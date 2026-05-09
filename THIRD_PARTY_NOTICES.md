# Third-Party Notices

This repository uses third-party datasets, models, and libraries. They are not
relicensed by this project.

## Project Code

Original source code in this repository is licensed under the MIT License.
See [LICENSE](LICENSE).

## Documentation and Figures

Original project documentation, poster text, written analysis, and aggregate
figures are licensed under CC BY-NC 4.0 unless otherwise noted.
See [LICENSE-DOCS](LICENSE-DOCS).

## Datasets

Raw biometric images, identity document photos, Kaggle downloads, Hugging Face
downloads, and derived local data files should not be committed to this
repository.

Datasets used or evaluated during this project include:

- Axon/Kaggle Selfie and Official ID Photo Dataset:
  https://www.kaggle.com/datasets/axondata/selfie-and-official-id-photo-dataset-18k-images
  Listed license: CC BY-NC 4.0.
- Hugging Face `ud-biometrics/Selfie-and-ID-Dataset`:
  https://huggingface.co/datasets/ud-biometrics/Selfie-and-ID-Dataset
  Listed license: CC BY-NC-ND 4.0.

Follow the original dataset terms, attribution requirements, privacy
requirements, and any platform terms of use. Do not redistribute face images or
identity documents from these datasets through this repository.

## Models and Libraries

This project uses DeepFace and pretrained face-recognition models such as
ArcFace/FaceNet through DeepFace. DeepFace is MIT licensed, but its wrapped
models and pretrained weights may carry their own original terms. Verify those
terms before production or commercial use.

Other Python dependencies, including PyTorch, TensorFlow, OpenCV, scikit-learn,
pandas, NumPy, Matplotlib, and tqdm, remain under their respective licenses.

## Sensitive Files

The `.gitignore` excludes common sensitive artifacts, including:

- Kaggle credentials and environment files
- Raw datasets and local image folders
- Local caches and pretrained weights
- Embedding arrays and trained checkpoints
- Pair-level result CSVs containing local image paths or identity split details

