# ResNet50V2 for Binary COVID-19 CT Image Classification

A modular TensorFlow/Keras implementation of a ResNet50V2-based binary image-classification workflow for **healthy** and **infected** CT images.

> **Research-code note:** This repository provides the implementation only. It does not include trained weights, dataset files, or performance claims. No experimental results are reported in this repository.

## Overview

The pipeline is organized into separate stages:

1. Discover images from class-specific folders.
2. Create reproducible, stratified training, validation, and test partitions.
3. Train a ResNet50V2 feature extractor with a global-average-pooling classification head.
4. Select the checkpoint using validation performance.
5. Evaluate the selected checkpoint on the held-out test set and save diagnostic artifacts.

The default model initialization is `none`, matching the supplied notebook's `weights=None` setting. ImageNet initialization can be enabled explicitly with `--weights imagenet`.

## Dataset layout

Arrange the dataset as follows:

```text
dataset/
├── Merged_healthy_images/
│   ├── image_001.png
│   └── ...
└── Merged_infected_images/
    ├── image_001.png
    └── ...
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.tif`/`.tiff`.

The class mapping is deterministic:

| Class name | Class index |
|---|---:|
| `Merged_healthy_images` | 0 |
| `Merged_infected_images` | 1 |

If your folder names differ, update `CLASS_NAMES` in `src/resnet50v2_covid/data.py`.

### Important split-design consideration

The supplied notebook performs a random image-level train/test split. This repository uses stratified train/validation/test partitions, but it cannot infer patient identity from image filenames. **If multiple slices or images belong to the same patient, provide patient-level grouping before splitting.** Images from one patient must not occur in more than one partition; otherwise, data leakage may inflate the estimated generalization performance. The current implementation is appropriate only when each image can be treated as an independent sample or when patient-level separation has already been ensured in the folder organization.

## Environment setup

Python 3.10 or 3.11 is recommended.

```bash
git clone <YOUR_REPOSITORY_URL>
cd resnet50v2-covid-hrct

python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For GPU training, install a TensorFlow build compatible with your CUDA/cuDNN environment. Refer to the official TensorFlow installation documentation for the supported configuration.

## Training

```bash
python -m resnet50v2_covid.train \
  --data-dir /path/to/dataset \
  --output-dir outputs/resnet50v2 \
  --epochs 20 \
  --batch-size 20 \
  --image-size 224 \
  --learning-rate 2e-5 \
  --weights none
```

On Windows, use a path such as `D:\dataset` (quote it if it contains spaces).

The default partition proportions are 70% training, 15% validation, and 15% testing. Set the random seed with `--seed`. The test partition is not used for checkpoint selection.

To initialize the backbone with ImageNet weights:

```bash
python -m resnet50v2_covid.train \
  --data-dir /path/to/dataset \
  --output-dir outputs/resnet50v2_imagenet \
  --weights imagenet
```

### Training options

```bash
python -m resnet50v2_covid.train --help
```

Key options include:

- `--data-dir`: dataset root directory.
- `--output-dir`: directory for checkpoints, logs, and split manifests.
- `--image-size`: square input size (default: 224).
- `--batch-size`: mini-batch size (default: 20).
- `--epochs`: maximum number of epochs (default: 20).
- `--learning-rate`: Adam learning rate (default: `2e-5`).
- `--weights`: `none` or `imagenet`.
- `--seed`: random seed (default: 42).

## Evaluation

Evaluation is performed separately after training:

```bash
python -m resnet50v2_covid.evaluate \
  --run-dir outputs/resnet50v2
```

This loads the saved test manifest and best checkpoint, then writes evaluation artifacts to `outputs/resnet50v2/evaluation/`. The test set is evaluated only in this stage.

## Output structure

```text
outputs/resnet50v2/
├── best_model.keras
├── training_history.csv
├── training_curves.png
├── class_mapping.json
├── split_manifest.csv
└── evaluation/
    ├── classification_report.json
    ├── confusion_matrix.csv
    ├── roc_curve.png
    └── evaluation_summary.json
```

These files are generated locally when the pipeline is run. They are not included in the repository.

## Methodological and implementation notes

- **Input preprocessing:** ResNet50V2 expects inputs preprocessed to the range used by `tf.keras.applications.resnet_v2.preprocess_input` (approximately `[-1, 1]`). The pipeline applies this preprocessing consistently.
- **Augmentation:** Random rotation, translation, zoom, and horizontal/vertical flips are applied to training images only.
- **Model selection:** `val_loss` is monitored for checkpointing and early stopping. The held-out test set is not used during training.
- **Reproducibility:** Python, NumPy, and TensorFlow seeds are set. Exact determinism can still depend on hardware and TensorFlow operations.
- **Patient-level independence:** Confirm that the split unit is the patient, not merely the image, whenever multiple images originate from one patient.
- **Clinical interpretation:** This is a research implementation, not a clinically validated diagnostic system.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── notebooks/
│   └── legacy_original_workflow.ipynb
└── src/
    └── resnet50v2_covid/
        ├── __init__.py
        ├── data.py
        ├── model.py
        ├── train.py
        └── evaluate.py
```

## Citation

If you use or adapt this code, cite the associated research work once it has been published. A placeholder `CITATION.cff` is included; update it with the final bibliographic details before making the repository public.

## License

The code is distributed under the MIT License. Dataset access, redistribution, and use must follow the dataset owners' terms and applicable institutional approvals. The license in this repository does not grant rights to redistribute medical images.
