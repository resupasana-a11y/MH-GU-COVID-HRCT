"""
MH-GU COVID-HRCT Dataset Validation
===================================

Validates the released TIFF dataset and its directory organization.

Checks:
1. Required top-level folders
2. Healthy and infected patient-folder organization
3. Infected pattern and annotation-folder organization
4. Patient and TIFF image counts
5. TIFF readability and basic image properties
6. Image dimensions, pixel-intensity ranges, blank images, and exact duplicates

This script intentionally does NOT inspect DICOM files or DICOM metadata.
"""

from pathlib import Path
from collections import Counter
import hashlib
import re

import numpy as np
import pandas as pd
from PIL import Image


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Set this to the directory containing:
#   Raw DICOM Files/
#   Healthy Cases/
#   Infected Cases/
#
# Raw DICOM Files is only checked for its presence; its contents are not read.
ROOT = Path(r"/path/to/MH-GU-COVID-HRCT")

PATTERNS = [
    "Minimal GGO",
    "GGO",
    "Consolidation",
    "GGO with Consolidation",
    "Crazy paving",
    "GGO with CP",
    "Halo sign",
]

# Set any expected value to None to disable that count check.
EXPECTED_COUNTS = {
    "healthy_patients": 82,
    "healthy_images": 330,
    "infected_patients": 106,
    "infected_images": 720,
    "total_patients": 188,
}

# A patient folder is expected to be named patient_<number>.
# Set to None if folder naming is not standardized.
PATIENT_FOLDER_PATTERN = re.compile(r"^patient_\d+$", re.IGNORECASE)

# Set to True only if all released images are expected to be grayscale.
EXPECT_GRAYSCALE = True

# Set to True only if all released images are expected to be 8-bit unsigned.
EXPECT_UINT8 = True

# Output directory for CSV reports. Relative paths are resolved from this script's
# current working directory.
OUTPUT_DIR = Path("validation_output")


# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

results = []


def record(status, check_name, message):
    """Store and print one validation result."""
    results.append({
        "status": status,
        "check": check_name,
        "message": str(message),
    })
    print(f"[{status}] {check_name}: {message}")


def check(condition, check_name, success_message, failure_message, failure_status="FAIL"):
    """Record a PASS or a configurable failure status."""
    record(
        "PASS" if condition else failure_status,
        check_name,
        success_message if condition else failure_message,
    )


def normalize(name):
    """Normalize folder names for comparison, ignoring spaces and punctuation."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def subfolders(path):
    return sorted(p for p in path.iterdir() if p.is_dir()) if path.is_dir() else []


def direct_files(path):
    return sorted(p for p in path.iterdir() if p.is_file()) if path.is_dir() else []


def is_tiff(path):
    return path.suffix.lower() in {".tif", ".tiff"}


def relative_paths(paths, limit=5):
    return [str(Path(p).relative_to(ROOT)) for p in list(paths)[:limit]]


def locate_unique_folder(root, folder_name):
    """
    Find exactly one folder with the requested name under root.
    Returns None if no match or multiple matches are found.
    """
    matches = [
        p for p in root.rglob("*")
        if p.is_dir() and normalize(p.name) == normalize(folder_name)
    ]

    if len(matches) == 1:
        record("PASS", f"locate/{folder_name}", f"Found: {matches[0]}")
        return matches[0]

    if len(matches) == 0:
        record("FAIL", f"locate/{folder_name}", f"Folder not found under {root}")
        return None

    record(
        "FAIL",
        f"locate/{folder_name}",
        f"Multiple matching folders found; set ROOT to the correct dataset directory: {matches}",
    )
    return None


# =============================================================================
# 3. ROOT AND TOP-LEVEL DIRECTORY VALIDATION
# =============================================================================

print("\n" + "=" * 72)
print("1. DIRECTORY STRUCTURE")
print("=" * 72)

if not ROOT.is_dir():
    raise FileNotFoundError(
        f"ROOT does not exist or is not a directory: {ROOT}\n"
        "Edit ROOT in the configuration section before running the script."
    )

raw_dir = locate_unique_folder(ROOT, "Raw DICOM Files")
healthy_dir = locate_unique_folder(ROOT, "Healthy Cases")
infected_dir = locate_unique_folder(ROOT, "Infected Cases")

if any(folder is None for folder in (raw_dir, healthy_dir, infected_dir)):
    raise RuntimeError(
        "One or more required folders could not be located uniquely. "
        "Correct ROOT or resolve duplicate folder names, then rerun."
    )

print("\nResolved dataset folders:")
print("Raw DICOM Files:", raw_dir)
print("Healthy Cases:  ", healthy_dir)
print("Infected Cases: ", infected_dir)

# Note: Raw DICOM Files is not opened or inspected by this script.


# =============================================================================
# 4. HEALTHY FOLDER STRUCTURE
# =============================================================================

print("\n" + "=" * 72)
print("2. HEALTHY CASES STRUCTURE")
print("=" * 72)

healthy_patient_dirs = subfolders(healthy_dir)
healthy_items = []

loose_healthy_files = direct_files(healthy_dir)
check(
    len(loose_healthy_files) == 0,
    "healthy/no_loose_files",
    "No files are stored directly inside Healthy Cases.",
    f"Files found directly inside Healthy Cases: {relative_paths(loose_healthy_files)}",
    "WARN",
)

for patient_dir in healthy_patient_dirs:
    patient_tiffs = [f for f in direct_files(patient_dir) if is_tiff(f)]
    healthy_items.extend((f, "Healthy", patient_dir.name) for f in patient_tiffs)

    check(
        len(patient_tiffs) > 0,
        f"healthy/images/{patient_dir.name}",
        f"{len(patient_tiffs)} TIFF image(s) found.",
        "No TIFF images found in this patient folder.",
    )

    nested_dirs = subfolders(patient_dir)
    check(
        len(nested_dirs) == 0,
        f"healthy/no_nested_folders/{patient_dir.name}",
        "No nested folders found.",
        f"Unexpected nested folders: {[p.name for p in nested_dirs]}",
        "WARN",
    )

    if PATIENT_FOLDER_PATTERN is not None:
        check(
            bool(PATIENT_FOLDER_PATTERN.match(patient_dir.name)),
            f"healthy/patient_folder_name/{patient_dir.name}",
            "Patient folder name follows patient_<number> convention.",
            "Patient folder name differs from patient_<number> convention.",
            "WARN",
        )


# =============================================================================
# 5. INFECTED FOLDER STRUCTURE
# =============================================================================

print("\n" + "=" * 72)
print("3. INFECTED CASES STRUCTURE")
print("=" * 72)

infected_pattern_dirs = {
    normalize(folder.name): folder
    for folder in subfolders(infected_dir)
}

infected_items = []
infected_patient_records = []

for pattern_name in PATTERNS:
    pattern_dir = infected_pattern_dirs.get(normalize(pattern_name))

    if pattern_dir is None:
        record("FAIL", "infected/missing_pattern", f"Pattern folder not found: {pattern_name}")
        continue

    loose_pattern_files = direct_files(pattern_dir)
    check(
        len(loose_pattern_files) == 0,
        f"infected/no_loose_files/{pattern_name}",
        "No files stored directly inside the pattern folder.",
        f"Unexpected loose files: {relative_paths(loose_pattern_files)}",
        "WARN",
    )

    for patient_dir in subfolders(pattern_dir):
        child_dirs = subfolders(patient_dir)
        annotation_dirs = [
            folder for folder in child_dirs
            if "annot" in folder.name.lower()
        ]
        other_dirs = [
            folder for folder in child_dirs
            if folder not in annotation_dirs
        ]

        patient_tiffs = [f for f in direct_files(patient_dir) if is_tiff(f)]

        annotation_file_count = sum(
            1
            for annotation_dir in annotation_dirs
            for item in annotation_dir.rglob("*")
            if item.is_file()
        )

        infected_patient_records.append({
            "pattern": pattern_name,
            "patient": patient_dir.name,
            "directory": patient_dir,
            "images": patient_tiffs,
            "annotation_folders": annotation_dirs,
            "annotation_files": annotation_file_count,
        })

        infected_items.extend(
            (f, pattern_name, patient_dir.name)
            for f in patient_tiffs
        )

        check(
            len(patient_tiffs) > 0,
            f"infected/images/{pattern_name}/{patient_dir.name}",
            f"{len(patient_tiffs)} TIFF image(s) found.",
            "No TIFF images found in this patient folder.",
        )

        check(
            len(annotation_dirs) == 1,
            f"infected/annotation_folder/{pattern_name}/{patient_dir.name}",
            "Exactly one annotation folder found.",
            f"Expected one annotation folder; found {len(annotation_dirs)}.",
        )

        if annotation_dirs:
            check(
                annotation_file_count > 0,
                f"infected/annotation_files/{pattern_name}/{patient_dir.name}",
                f"{annotation_file_count} annotation file(s) found.",
                "Annotation folder exists but contains no files.",
                "WARN",
            )

        check(
            len(other_dirs) == 0,
            f"infected/unexpected_subfolders/{pattern_name}/{patient_dir.name}",
            "No unexpected patient subfolders found.",
            f"Unexpected subfolders: {[p.name for p in other_dirs]}",
            "WARN",
        )

        if PATIENT_FOLDER_PATTERN is not None:
            check(
                bool(PATIENT_FOLDER_PATTERN.match(patient_dir.name)),
                f"infected/patient_folder_name/{pattern_name}/{patient_dir.name}",
                "Patient folder name follows patient_<number> convention.",
                "Patient folder name differs from patient_<number> convention.",
                "WARN",
            )

expected_pattern_names = {normalize(p) for p in PATTERNS}
unexpected_pattern_dirs = [
    folder.name
    for key, folder in infected_pattern_dirs.items()
    if key not in expected_pattern_names
]
check(
    len(unexpected_pattern_dirs) == 0,
    "infected/unexpected_pattern_folders",
    "No unexpected pattern folders found.",
    f"Unexpected folders: {unexpected_pattern_dirs}",
    "WARN",
)


# =============================================================================
# 6. PATIENT AND IMAGE COUNTS
# =============================================================================

print("\n" + "=" * 72)
print("4. PATIENT AND IMAGE COUNTS")
print("=" * 72)

observed_counts = {
    "healthy_patients": len(healthy_patient_dirs),
    "healthy_images": len(healthy_items),
    "infected_patients": len(infected_patient_records),
    "infected_images": len(infected_items),
    "total_patients": len(healthy_patient_dirs) + len(infected_patient_records),
}

for count_name, observed in observed_counts.items():
    expected = EXPECTED_COUNTS.get(count_name)

    if expected is None:
        record("INFO", f"count/{count_name}", f"Observed count: {observed}")
    else:
        check(
            observed == expected,
            f"count/{count_name}",
            f"Observed count {observed} matches expected count {expected}.",
            f"Observed count {observed}; expected count {expected}.",
            "WARN",
        )

pattern_summary = []
for pattern_name in PATTERNS:
    records_for_pattern = [
        row for row in infected_patient_records
        if row["pattern"] == pattern_name
    ]

    pattern_summary.append({
        "category": pattern_name,
        "patients": len(records_for_pattern),
        "images": sum(len(row["images"]) for row in records_for_pattern),
        "annotation_files": sum(row["annotation_files"] for row in records_for_pattern),
    })

folder_summary = pd.DataFrame([
    {
        "category": "Healthy",
        "patients": len(healthy_patient_dirs),
        "images": len(healthy_items),
        "annotation_files": 0,
    },
    *pattern_summary,
])

print(folder_summary.to_string(index=False))


# =============================================================================
# 7. TIFF IMAGE INTEGRITY AND PIXEL-INTENSITY VALIDATION
# =============================================================================

print("\n" + "=" * 72)
print("5. TIFF IMAGE VALIDATION")
print("=" * 72)

all_image_items = healthy_items + infected_items
image_rows = []

for file_path, category, patient_name in all_image_items:
    row = {
        "category": category,
        "patient": patient_name,
        "file": str(file_path.relative_to(ROOT)),
        "readable": False,
    }

    try:
        # Verify file structure, then reopen to read actual pixel values.
        with Image.open(file_path) as image:
            image.verify()

        with Image.open(file_path) as image:
            pixel_array = np.asarray(image)
            image_mode = image.mode

        if pixel_array.size == 0:
            raise ValueError("Image contains no pixel data.")

        if pixel_array.ndim == 2:
            grayscale = True
            height, width = pixel_array.shape
        elif pixel_array.ndim == 3:
            grayscale = False
            height, width = pixel_array.shape[:2]
        else:
            raise ValueError(f"Unsupported image array shape: {pixel_array.shape}")

        row.update({
            "readable": True,
            "mode": image_mode,
            "dtype": str(pixel_array.dtype),
            "width": int(width),
            "height": int(height),
            "grayscale": bool(grayscale),
            "minimum": float(np.min(pixel_array)),
            "maximum": float(np.max(pixel_array)),
            "mean": float(np.mean(pixel_array)),
            "standard_deviation": float(np.std(pixel_array)),
            "md5": hashlib.md5(file_path.read_bytes()).hexdigest(),
        })

    except Exception as error:
        row["error"] = repr(error)

    image_rows.append(row)

image_metadata = pd.DataFrame(image_rows)
print(f"TIFF files examined: {len(image_metadata)}")

if not image_metadata.empty:
    unreadable = image_metadata[~image_metadata["readable"]]

    check(
        unreadable.empty,
        "tiff/readability",
        f"All {len(image_metadata)} TIFF files are readable.",
        f"{len(unreadable)} unreadable TIFF file(s): "
        f"{unreadable['file'].head(10).tolist()}",
    )

    readable = image_metadata[image_metadata["readable"]].copy()

    if not readable.empty:
        # Image dimensions
        dimension_counts = (
            readable.groupby(["width", "height"])
            .size()
            .to_dict()
        )
        check(
            len(dimension_counts) == 1,
            "tiff/dimensions",
            f"All readable images have dimensions {next(iter(dimension_counts))}.",
            f"Multiple image dimensions found: {dimension_counts}",
            "WARN",
        )

        # Grayscale check
        if EXPECT_GRAYSCALE:
            non_grayscale = readable[~readable["grayscale"]]
            check(
                non_grayscale.empty,
                "tiff/grayscale",
                "All readable images are grayscale.",
                f"{len(non_grayscale)} non-grayscale image(s): "
                f"{non_grayscale['file'].head(10).tolist()}",
                "WARN",
            )

        # Data type check
        if EXPECT_UINT8:
            non_uint8 = readable[readable["dtype"] != "uint8"]
            check(
                non_uint8.empty,
                "tiff/data_type",
                "All readable images use uint8 pixel data.",
                f"Other data types found: {readable['dtype'].value_counts().to_dict()}",
                "WARN",
            )

        # Record the observed intensity range; do not impose an arbitrary
        # universal threshold on the images.
        global_min = readable["minimum"].min()
        global_max = readable["maximum"].max()
        print(f"Observed pixel-intensity range across images: {global_min} to {global_max}")

        # Constant images have zero standard deviation.
        blank_images = readable[readable["standard_deviation"] == 0]
        check(
            blank_images.empty,
            "tiff/blank_images",
            "No constant-intensity images detected.",
            f"{len(blank_images)} constant-intensity image(s): "
            f"{blank_images['file'].head(10).tolist()}",
            "WARN",
        )

        # Exact file duplicates are flagged for manual review.
        duplicate_images = readable[
            readable.duplicated("md5", keep=False)
        ]
        check(
            duplicate_images.empty,
            "tiff/exact_duplicates",
            "No byte-identical TIFF files detected.",
            f"{len(duplicate_images)} file(s) belong to duplicate groups: "
            f"{duplicate_images['file'].head(10).tolist()}",
            "WARN",
        )

        # Save image-property distributions for reporting.
        print("\nImage mode distribution:")
        print(readable["mode"].value_counts(dropna=False).to_string())

        print("\nData type distribution:")
        print(readable["dtype"].value_counts(dropna=False).to_string())

        print("\nImage dimensions distribution:")
        print(readable.groupby(["width", "height"]).size().to_string())

else:
    record("FAIL", "tiff/no_images", "No TIFF images were found in the expected folders.")


# =============================================================================
# 8. EXPORT REPORTS
# =============================================================================

print("\n" + "=" * 72)
print("6. EXPORT VALIDATION REPORTS")
print("=" * 72)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(
    results,
    columns=["status", "check", "message"],
)

results_df.to_csv(OUTPUT_DIR / "validation_results.csv", index=False)
folder_summary.to_csv(OUTPUT_DIR / "folder_summary.csv", index=False)
image_metadata.to_csv(OUTPUT_DIR / "tiff_image_metadata.csv", index=False)

print("\nValidation status:")
print(results_df["status"].value_counts(dropna=False).to_string())

print("\nReports saved in:", OUTPUT_DIR.resolve())
print("- validation_results.csv")
print("- folder_summary.csv")
print("- tiff_image_metadata.csv")

print("\nValidation run completed.")
