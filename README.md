# MH-GU COVID-HRCT Dataset Validation

A lightweight, reproducible Python quality-control script for the **released TIFF images** of the MH-GU COVID-HRCT dataset.

The script checks the dataset directory organization, patient/image counts, TIFF file integrity, image dimensions, pixel-intensity statistics, blank images, and exact duplicate files. It exports machine-readable CSV reports to support transparent dataset documentation and independent verification.

> **Scope:** This repository validates the TIFF dataset and its folder structure. It does **not** read, parse, or validate DICOM files or DICOM metadata. The `Raw DICOM Files` directory is checked only for its presence as part of the expected top-level organization.

## Contents

```text
MH-GU-COVID-HRCT-validation/
├── validation.py
├── requirements.txt
├── README.md
├── .gitignore
└── validation_output/          # generated after execution
    ├── validation_results.csv
    ├── folder_summary.csv
    └── tiff_image_metadata.csv
```

## 1. Expected dataset organization

The script expects the following structure. The names of the pattern folders must match the entries in `PATTERNS` in `validation.py` (folder matching ignores spaces and punctuation).

```text
MH-GU-COVID-HRCT/
├── Raw DICOM Files/
├── Healthy Cases/
│   ├── patient_1/
│   │   ├── image_001.tiff
│   │   └── ...
│   ├── patient_2/
│   └── ...
└── Infected Cases/
    ├── Minimal GGO/
    │   ├── patient_1/
    │   │   ├── image_001.tiff
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

The example above illustrates the expected organization. Actual filenames and annotation-folder names may differ. The script identifies annotation folders by the presence of `annot` in the folder name.

## 2. Validation checks

### Directory structure

- Checks for the three expected top-level directories: `Raw DICOM Files`, `Healthy Cases`, and `Infected Cases`.
- Checks that healthy patient folders contain TIFF images.
- Checks that infected cases are organized by pattern and patient.
- Checks that each infected patient folder has exactly one annotation folder.
- Checks that annotation folders are not empty.
- Reports unexpected nested folders and unexpected pattern folders.

### Patient and image counts

The default expected counts are configured as:

| Category | Expected patients | Expected TIFF images |
|---|---:|---:|
| Healthy | 82 | 330 |
| Infected | 106 | 720 |
| **Total** | **188** | **1,050** |

These are configurable in `EXPECTED_COUNTS`. A count mismatch is reported as a warning so that the script does not silently treat a discrepancy as proof of file corruption.

### TIFF image integrity

For each TIFF image, the script checks or records:

- Whether the image can be opened and verified.
- Image width and height.
- PIL image mode and NumPy data type.
- Whether the image is grayscale.
- Minimum and maximum pixel intensity.
- Mean and standard deviation of pixel intensity.
- Constant-intensity images (zero standard deviation).
- Exact byte-identical duplicates (MD5 hash).

Observed intensity values are reported rather than compared with an arbitrary universal threshold. Duplicate and blank-image findings are flagged for review; files are not automatically deleted or modified.

## 3. Requirements

- Python 3.9 or later
- NumPy
- pandas
- Pillow

Install dependencies:

```bash
pip install -r requirements.txt
```

## 4. Configuration

Open `validation.py` and edit the dataset path:

```python
ROOT = Path(r"/path/to/MH-GU-COVID-HRCT")
```

`ROOT` must point to the directory containing the three top-level folders.

Check the pattern-folder names against the actual dataset. The default list includes `GGO with CP`; if the actual directory is named `GGO with Crazy paving`, update the corresponding entry in `PATTERNS`.

The expected patient and image counts can be changed in `EXPECTED_COUNTS`. Set a value to `None` to skip that count comparison.

If the TIFF images are not all expected to be grayscale or `uint8`, change:

```python
EXPECT_GRAYSCALE = True
EXPECT_UINT8 = True
```

to `False` for the relevant property. These options control whether the script reports deviations from the expected representation; the actual mode and data type are recorded in the output metadata.

## 5. Run the validation

From a terminal in the repository directory:

```bash
python validation.py
```

The script prints validation outcomes to the terminal and creates a `validation_output/` directory containing the reports.

## 6. Output files

| File | Description |
|---|---|
| `validation_results.csv` | Individual validation checks with `PASS`, `WARN`, `FAIL`, or `INFO` status |
| `folder_summary.csv` | Patient, image, and annotation-file counts by category/pattern |
| `tiff_image_metadata.csv` | Per-image readability, dimensions, mode, data type, intensity statistics, and duplicate hash |

The CSV files are generated from the current dataset when the script is run. They are not pre-populated or assumed results.

## 7. Interpreting results

- **PASS:** The tested condition was satisfied.
- **WARN:** A deviation was found that should be reviewed; it may be legitimate depending on the dataset.
- **FAIL:** A required folder or validation condition was not satisfied.
- **INFO:** An observed value was recorded without a pass/fail criterion.

Review all warnings and failures before reporting validation outcomes in a manuscript or releasing the dataset. The script does not modify, rename, move, or delete source data.

## 8. Scope and limitations

- The script validates TIFF files and directory organization only.
- It does not inspect DICOM headers, DICOM pixel data, or DICOM anonymization.
- An MD5 match identifies byte-identical files; it does not detect visually similar images saved with different encodings.
- A zero-standard-deviation test detects constant-intensity images, not every possible clinically uninformative image.
- A grayscale check based on image array dimensions does not establish that an image is diagnostically adequate.
- A folder-structure check cannot verify the clinical correctness of labels or annotations.
- The script does not replace clinical review, annotation quality assessment, or a complete privacy assessment.

## 9. Reproducibility

For transparent reporting, publish:

1. This validation script.
2. The `requirements.txt` file (and, if desired, a pinned environment file).
3. The generated CSV validation reports, after reviewing them for unintended disclosure of sensitive filenames or identifiers.
4. The dataset version or release identifier and the date on which validation was run.

Do not publish patient-identifying information in the reports. Use anonymized or pseudonymized folder and filename conventions in any public release.

## License

Add the license applicable to the dataset and code before public distribution. The code license and dataset license may be different.
