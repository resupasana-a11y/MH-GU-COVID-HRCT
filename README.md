# HRCT Image Analysis: Enhancement, Classification, and Dataset Validation

This repository brings together three Python workflows for HRCT/CT image research:

1. **Mamdani fuzzy CT image enhancement** using a rule-based Mamdani Fuzzy Inference System (FIS).
2. **Binary HRCT image classification** using eight TensorFlow/Keras deep-learning architectures.
3. **MH-GU COVID-HRCT dataset validation** for checking the released TIFF images and their directory organization.

The workflows are provided in one repository, but they serve different purposes and have different input-data requirements. The dataset validation script checks the MH-GU dataset structure and TIFF files; the classification pipeline expects a separate two-class image-folder layout; and the enhancement script processes an individual grayscale CT image.

> \\\*\\\*Important scope note:\\\*\\\* The validation script checks TIFF images and folder organization only. It does not read, parse, or validate DICOM files or DICOM metadata. The Mamdani enhancement implementation is a conventional rule-based fuzzy system, not a trainable fuzzy neural network or ANFIS model.

\---

## 1\. Repository structure

The main files are organized in a single folder as follows:

```text
HRCT-Image-Analysis/
├── models/
│   ├── densenet121.py
│   ├── efficientnetb0.py
│   ├── inception\\\_v3.py
│   ├── mobilenet.py
│   ├── mobilenet\\\_v2.py
│   ├── resnet50v2.py
│   ├── resnet101.py
│   └── xception.py
├── mamdani\\\_ct\\\_enhancement.py
├── preprocessing.py
├── validation.py
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `models/` directory contains the eight classification model implementations. The remaining Python files provide image enhancement, preprocessing, model training, and dataset validation functionality.

Generated classification outputs and validation reports are created when the corresponding scripts are run; they are not required to be present in the repository beforehand.

\---

## 2\. Environment setup

Python 3.10 or 3.11 is recommended for the classification pipeline. The enhancement and validation scripts specify Python 3.9 or later.

Install the dependencies listed in the repository's combined `requirements.txt` file:

```bash
python -m pip install -r requirements.txt
```

Use a virtual environment if you want to keep the project dependencies isolated:

```bash
python -m venv .venv
```

Activate it, then install the requirements.

**Windows:**

```bash
.venv\\\\Scripts\\\\activate
python -m pip install -r requirements.txt
```

**Linux/macOS:**

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The workflows use libraries including TensorFlow/Keras, NumPy, OpenCV, Matplotlib, scikit-fuzzy, pandas, and Pillow. The `requirements.txt` file should contain the complete dependency list for the scripts included in this repository.

\---

## 3\. Mamdani fuzzy CT image enhancement

### 3.1 Overview

The enhancement module implements a **rule-based Mamdani Fuzzy Inference System (FIS)** for grayscale CT image enhancement. It follows the workflow, fuzzy membership functions, and rules specified in the supplied fuzzy-logic notebook.

For each normalized pixel intensity, the system estimates a fuzzy enhancement adjustment and adds the resulting crisp adjustment to the pixel intensity.

### 3.2 Processing pipeline

1. Read the input CT image in grayscale.
2. Convert pixel values from the integer range `\\\[0, 255]` to floating-point values in `\\\[0, 1]`.
3. Apply a `3 × 3` Gaussian blur for light denoising (configurable).
4. Fuzzify each pixel intensity.
5. Evaluate the three Mamdani IF–THEN rules.
6. Aggregate the fuzzy output sets and defuzzify the result. `scikit-fuzzy` uses centroid defuzzification by default for a `Consequent`.
7. Add the crisp adjustment to the original normalized pixel intensity.
8. Clip the enhanced intensity to `\\\[0, 1]`, save the output, and optionally display a comparison.

### 3.3 Fuzzy system design

**Input variable: `intensity`**

* Universe: `\\\[0, 1]`, sampled at 256 points.

|Fuzzy set|Membership function|Parameters|
|-|-|-|
|Dark|Trapezoidal|`\\\[0.0, 0.0, 0.2, 0.4]`|
|Medium|Triangular|`\\\[0.3, 0.5, 0.7]`|
|Bright|Trapezoidal|`\\\[0.6, 0.75, 1.0, 1.0]`|

**Output variable: `enhancement`**

* Universe: `\\\[-0.3, 0.3]`, sampled at 256 points.

|Fuzzy set|Membership function|Parameters|
|-|-|-|
|Decrease|Triangular|`\\\[-0.3, -0.15, 0.0]`|
|None|Triangular|`\\\[-0.05, 0.0, 0.05]`|
|Increase|Triangular|`\\\[0.0, 0.15, 0.3]`|

**Rule base**

|Rule|Antecedent (IF)|Consequent (THEN)|
|-|-|-|
|R1|Intensity is Dark|Enhancement is Decrease|
|R2|Intensity is Medium|Enhancement is Increase|
|R3|Intensity is Bright|Enhancement is None|

These rules are preserved from the supplied notebook. In particular, the rule for dark pixels decreases intensity. This may not match the intended clinical or image-enhancement objective; review and validate the rule base before changing or relying on it.

### 3.4 Usage

Run the script from the repository's root directory:

```bash
python mamdani\\\_ct\\\_enhancement.py --input "path/to/ct\\\_image.jpg"
```

Specify an output filename:

```bash
python mamdani\\\_ct\\\_enhancement.py \\\\
  --input "path/to/ct\\\_image.jpg" \\\\
  --output "results/enhanced\\\_ct.png"
```

Disable the display plot:

```bash
python mamdani\\\_ct\\\_enhancement.py \\\\
  --input "path/to/ct\\\_image.jpg" \\\\
  --output "results/enhanced\\\_ct.png" \\\\
  --no-plot
```

Disable Gaussian smoothing by setting the blur kernel size to 1:

```bash
python mamdani\\\_ct\\\_enhancement.py \\\\
  --input "path/to/ct\\\_image.jpg" \\\\
  --blur-kernel 1
```

The input can be a grayscale-readable image such as JPG or PNG. The output is saved as an 8-bit PNG when the output filename ends in `.png`.

### 3.5 Output

The script saves the enhanced image and, unless `--no-plot` is specified, displays:

* Original grayscale CT image after denoising.
* Mamdani fuzzy-enhanced image.
* Difference image (`enhanced - original`).

### 3.6 Implementation notes and limitations

* **Pixel-wise processing:** The fuzzy system is evaluated separately for every pixel. This follows the supplied notebook but can be slow for large CT images.
* **No adaptive learning:** Membership functions and rules are manually specified; the system does not learn them from data.
* **Rule validation:** The original rule base maps Dark to Decrease and Medium to Increase. These choices are preserved for reproducibility, not asserted to be optimal.
* **Clinical validation:** This is an image-processing demonstration, not a clinically validated diagnostic or enhancement tool. Evaluate image quality and downstream task performance on appropriate data before research or clinical use.
* **Intensity information:** The script reads an image as an 8-bit grayscale image. For quantitative CT/Hounsfield Unit processing, use the original DICOM data and a carefully defined HU/windowing pipeline rather than treating a rendered JPG/PNG as raw CT data.

The pixel update is:

\[
I\_{\\mathrm{enhanced}}(i,j)
===

\\operatorname{clip}
\\left(
I\_{\\mathrm{input}}(i,j)+\\Delta I(i,j),\\ 0,\\ 1
\\right),
]

where (\\Delta I(i,j)) is the crisp output of the Mamdani FIS for the input pixel intensity.

\---

## 4\. Binary HRCT image classification using deep learning

### 4.1 Overview

The classification pipeline contains eight TensorFlow/Keras implementations for binary HRCT image classification:

* DenseNet121
* EfficientNetB0
* InceptionV3
* MobileNet
* MobileNetV2
* ResNet50V2
* ResNet101
* Xception

The implementation is based on the supplied `Aug\\\_res50v2.ipynb` notebook. All backbones are initialized **without pretrained weights** (`weights=None`).

### 4.2 Dataset organization

The classification pipeline expects two class folders. Keep the folder names and class order shown below:

```text
your\\\_dataset/
├── Merged\\\_healthy\\\_images/
│   ├── image\\\_001.jpg
│   └── ...
└── Merged\\\_infected\\\_images/
    ├── image\\\_001.jpg
    └── ...
```

Supported image extensions are `.jpg`, `.jpeg`, `.png`, and `.bmp`.

The dataset itself is not included in this repository. Set `--data-dir` to the local directory containing these two class folders.

> \\\*\\\*Note:\\\*\\\* This is the input structure documented for the classification pipeline. It is distinct from the MH-GU COVID-HRCT directory structure used by `validation.py` in Section 5. Do not assume that the two layouts are interchangeable.

### 4.3 Experimental configuration

The following configuration is taken from the supplied notebook:

|Setting|Value|
|-|-|
|Image size|224 × 224|
|Input channels|3|
|Pixel scaling|Divide by 255|
|Classes|Healthy, infected|
|Train/validation/test split|80% / 10% / 10%|
|Split random state|42|
|NumPy seed|3|
|TensorFlow seed|7|
|Batch size|20|
|Epochs|20|
|Optimizer|Adam|
|Learning rate|0.00002|
|Loss|Categorical cross-entropy|
|Classifier head|GlobalAveragePooling2D → Dropout(0.5) → Dense(512, ReLU) → Dropout(0.5) → Dense(2, softmax)|
|Backbone weights|None (random initialization)|
|Augmentation|Rotation 180°, width/height shift 0.2, zoom 0.2, horizontal and vertical flips|
|Checkpoint|Best validation accuracy|

### 4.4 Training

Train one model at a time from the repository's root directory. For example:

```bash
python train.py --model ResNet50V2 --data-dir "/path/to/your\\\_dataset"
```

Other model names accepted by `--model` are:

```text
DenseNet121
EfficientNetB0
InceptionV3
MobileNet
MobileNetV2
ResNet50V2
ResNet101
Xception
```

Optional parameters:

```bash
python train.py --model DenseNet121 \\\\
  --data-dir "/path/to/your\\\_dataset" \\\\
  --image-size 224 \\\\
  --batch-size 20 \\\\
  --epochs 20 \\\\
  --learning-rate 0.00002 \\\\
  --seed 42
```

Each run saves its best validation-accuracy checkpoint, training history, and experiment configuration under:

```text
outputs/<model-name>/
```

### 4.5 Validation split and evaluation methodology

The original notebook creates an 80/10/10 split, but passes `X\\\_test, Y\\\_test` as `validation\\\_data` during fitting and does not use `X\\\_val`. This allows the held-out test set to influence checkpoint selection.

In this repository, the intended validation split (`X\\\_val, Y\\\_val`) is used during training, and the test split is used only for final evaluation. This is a methodological correction; consequently, results from this cleaned pipeline may not exactly match the numbers obtained with the original notebook.

The notebook also creates a `ReduceLROnPlateau` callback named `annealer` but does not include it in the `model.fit` callback list. To match the executed training call, this repository uses the checkpoint callback only.

The notebook's printed F1 value is the harmonic mean of weighted precision and weighted recall. The evaluation script reports that value and standard weighted F1 separately to make the calculation transparent.

For a strictly identical rerun of the original notebook, the `validation\\\_data` argument in `train.py` would need to be changed to `(x\\\_test, y\\\_test)`. This is **not recommended** for a scientifically independent test evaluation.

\---

## 5\. MH-GU COVID-HRCT dataset validation

### 5.1 Overview and scope

`validation.py` is a lightweight, reproducible Python quality-control script for the **released TIFF images** of the MH-GU COVID-HRCT dataset.

It checks the dataset directory organization, patient/image counts, TIFF file integrity, image dimensions, pixel-intensity statistics, blank images, and exact duplicate files. It exports machine-readable CSV reports to support transparent dataset documentation and independent verification.

**Scope:** This repository validates the TIFF dataset and its folder structure. It does not read, parse, or validate DICOM files or DICOM metadata. The `Raw DICOM Files` directory is checked only for its presence as part of the expected top-level organization.

### 5.2 Expected dataset organization

The script expects the following structure. The names of the pattern folders must match the entries in `PATTERNS` in `validation.py` (folder matching ignores spaces and punctuation):

```text
MH-GU-COVID-HRCT/
├── Healthy Cases/
│   ├── patient\\\_1/
│   │   ├── image\\\_001.tiff
│   │   └── ...
│   ├── patient\\\_2/
│   └── ...
└── Infected Cases/
    ├── Minimal GGO/
    │   ├── patient\\\_1/
    │   │   ├── image\\\_001.tiff
    │   │   └── Annotations/
    │   │       └── ...
    │   └── ...
    ├── GGO/
    ├── Consolidation/
    ├── GGO with Consolidation/
    ├── Crazy paving/
    ├── GGO with CP/
    └── Halo sign/
```

The example illustrates the expected organization. Actual filenames and annotation-folder names may differ. The script identifies annotation folders by the presence of `annot` in the folder name.

### 5.3 Validation checks

**Directory structure**

* Checks for the three expected top-level directories: `Raw DICOM Files`, `Healthy Cases`, and `Infected Cases`.
* Checks that healthy patient folders contain TIFF images.
* Checks that infected cases are organized by pattern and patient.
* Checks that each infected patient folder has exactly one annotation folder.
* Checks that annotation folders are not empty.
* Reports unexpected nested folders and unexpected pattern folders.

**Patient and image counts**

The default expected counts are:

|Category|Expected patients|Expected TIFF images|
|-|-:|-:|
|Healthy|82|330|
|Infected|106|720|
|**Total**|**188**|**1,050**|

These values are configurable in `EXPECTED\\\_COUNTS`. A count mismatch is reported as a warning so that the script does not silently treat a discrepancy as proof of file corruption.

**TIFF image integrity**

For each TIFF image, the script checks or records:

* Whether the image can be opened and verified.
* Image width and height.
* PIL image mode and NumPy data type.
* Whether the image is grayscale.
* Minimum and maximum pixel intensity.
* Mean and standard deviation of pixel intensity.
* Constant-intensity images (zero standard deviation).
* Exact byte-identical duplicates (MD5 hash).

Observed intensity values are reported rather than compared with an arbitrary universal threshold. Duplicate and blank-image findings are flagged for review; files are not automatically deleted or modified.

### 5.4 Configuration

Open `validation.py` and edit the dataset path:

```python
ROOT = Path(r"/path/to/MH-GU-COVID-HRCT")
```

`ROOT` must point to the directory containing the three top-level folders.

Check the pattern-folder names against the actual dataset. The default list includes `GGO with CP`; if the actual directory is named `GGO with Crazy paving`, update the corresponding entry in `PATTERNS`.

The expected patient and image counts can be changed in `EXPECTED\\\_COUNTS`. Set a value to `None` to skip that count comparison.

If the TIFF images are not all expected to be grayscale or `uint8`, change:

```python
EXPECT\\\_GRAYSCALE = True
EXPECT\\\_UINT8 = True
```

to `False` for the relevant property. These options control whether the script reports deviations from the expected representation; the actual mode and data type are recorded in the output metadata.

### 5.5 Run the validation

From a terminal in the repository's root directory:

```bash
python validation.py
```

The script prints validation outcomes to the terminal and creates a `validation\\\_output/` directory containing the reports.

### 5.6 Output files

|File|Description|
|-|-|
|`validation\\\_results.csv`|Individual validation checks with `PASS`, `WARN`, `FAIL`, or `INFO` status|
|`folder\\\_summary.csv`|Patient, image, and annotation-file counts by category/pattern|
|`tiff\\\_image\\\_metadata.csv`|Per-image readability, dimensions, mode, data type, intensity statistics, and duplicate hash|

The CSV files are generated from the current dataset when the script is run. They are not pre-populated or assumed results.

### 5.7 Interpreting results

* **PASS:** The tested condition was satisfied.
* **WARN:** A deviation was found that should be reviewed; it may be legitimate depending on the dataset.
* **FAIL:** A required folder or validation condition was not satisfied.
* **INFO:** An observed value was recorded without a pass/fail criterion.

Review all warnings and failures before reporting validation outcomes in a manuscript or releasing the dataset. The script does not modify, rename, move, or delete source data.

### 5.8 Scope and limitations

* The script validates TIFF files and directory organization only.
* It does not inspect DICOM headers, DICOM pixel data, or DICOM anonymization.
* An MD5 match identifies byte-identical files; it does not detect visually similar images saved with different encodings.
* A zero-standard-deviation test detects constant-intensity images, not every possible clinically uninformative image.
* A grayscale check based on image array dimensions does not establish that an image is diagnostically adequate.
* A folder-structure check cannot verify the clinical correctness of labels or annotations.
* The script does not replace clinical review, annotation quality assessment, or a complete privacy assessment.

### 5.9 Reproducibility and responsible release

For transparent reporting, publish:

1. This validation script.
2. The `requirements.txt` file (and, if desired, a pinned environment file).
3. The generated CSV validation reports, after reviewing them for unintended disclosure of sensitive filenames or identifiers.
4. The dataset version or release identifier and the date on which validation was run.

Do not publish patient-identifying information in the reports. Use anonymized or pseudonymized folder and filename conventions in any public release.

\---

## 6\. General notes for running the repository

* Run commands from the repository's root directory unless a command specifies otherwise.
* Provide the correct input path for the workflow being run.
* The classification dataset and the MH-GU validation dataset have different documented directory structures.
* The enhancement script operates on an individual grayscale-readable image; it is not a DICOM validation tool.
* Review generated outputs and warnings before using results in research reports or publications.
* Do not interpret image enhancement, dataset quality-control results, or model predictions as a substitute for clinical assessment.

## 7\. Citation and licensing

If you use the classification code or reported results, cite the associated research publication. Add the publication citation or DOI here when available.

Add the license applicable to the dataset and code before public distribution. The code license and dataset license may be different.

