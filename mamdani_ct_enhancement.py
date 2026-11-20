"""
Mamdani Fuzzy Inference System for grayscale CT image enhancement.

This script reproduces the fuzzy logic workflow in the supplied notebook:
  1. Read a CT image in grayscale.
  2. Normalize pixel intensities to [0, 1].
  3. Apply light Gaussian denoising.
  4. Apply a Mamdani fuzzy inference system to each pixel.
  5. Add the defuzzified adjustment to the input intensity.
  6. Clip the result to [0, 1] and save/display the output.

Example:
    python mamdani_ct_enhancement.py --input path/to/ct.jpg

Dependencies:
    numpy, opencv-python, matplotlib, scikit-fuzzy
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def build_mamdani_system():
    """Create and return the Mamdani fuzzy control system."""
    # Input: normalized pixel intensity in [0, 1].
    intensity = ctrl.Antecedent(
        np.linspace(0.0, 1.0, 256), "intensity"
    )
    intensity["dark"] = fuzz.trapmf(
        intensity.universe, [0.0, 0.0, 0.2, 0.4]
    )
    intensity["medium"] = fuzz.trimf(
        intensity.universe, [0.3, 0.5, 0.7]
    )
    intensity["bright"] = fuzz.trapmf(
        intensity.universe, [0.6, 0.75, 1.0, 1.0]
    )

    # Output: fuzzy intensity adjustment in [-0.3, 0.3].
    enhancement = ctrl.Consequent(
        np.linspace(-0.3, 0.3, 256), "enhancement"
    )
    enhancement["decrease"] = fuzz.trimf(
        enhancement.universe, [-0.3, -0.15, 0.0]
    )
    enhancement["none"] = fuzz.trimf(
        enhancement.universe, [-0.05, 0.0, 0.05]
    )
    enhancement["increase"] = fuzz.trimf(
        enhancement.universe, [0.0, 0.15, 0.3]
    )

    # Original notebook's three rules, retained unchanged.
    rule1 = ctrl.Rule(
        intensity["dark"], enhancement["decrease"]
    )
    rule2 = ctrl.Rule(
        intensity["medium"], enhancement["increase"]
    )
    rule3 = ctrl.Rule(
        intensity["bright"], enhancement["none"]
    )

    control_system = ctrl.ControlSystem([rule1, rule2, rule3])
    return control_system


def enhance_image(image_path, blur_kernel=3):
    """
    Enhance one grayscale image.

    Returns:
        original: normalized grayscale image after denoising
        enhanced: fuzzy-enhanced image, both float32 arrays in [0, 1]
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}. "
            "Check that the file is a supported image format."
        )

    original = image.astype(np.float32) / 255.0

    if blur_kernel < 1 or blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be a positive odd integer.")
    if blur_kernel > 1:
        original = cv2.GaussianBlur(
            original, (blur_kernel, blur_kernel), 0
        )

    control_system = build_mamdani_system()
    fuzzy_simulation = ctrl.ControlSystemSimulation(control_system)

    enhanced = np.zeros_like(original, dtype=np.float32)

    # Pixel-wise Mamdani inference, following the supplied notebook.
    for row in range(original.shape[0]):
        for col in range(original.shape[1]):
            pixel_intensity = float(original[row, col])
            fuzzy_simulation.input["intensity"] = pixel_intensity
            fuzzy_simulation.compute()
            adjustment = fuzzy_simulation.output["enhancement"]
            enhanced[row, col] = pixel_intensity + adjustment

    enhanced = np.clip(enhanced, 0.0, 1.0)
    return original, enhanced


def save_and_plot(original, enhanced, output_path, show_plot=True):
    """Save the enhanced image and optionally display comparison plots."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_uint8 = np.round(enhanced * 255.0).astype(np.uint8)
    if not cv2.imwrite(str(output_path), output_uint8):
        raise IOError(f"Could not save enhanced image to: {output_path}")

    print(f"Enhanced image saved to: {output_path}")

    if show_plot:
        difference = enhanced - original

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("Original CT (after denoising)")
        plt.imshow(original, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Enhanced (Mamdani Fuzzy)")
        plt.imshow(enhanced, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Difference")
        plt.imshow(difference, cmap="seismic")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")

        plt.tight_layout()
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mamdani fuzzy inference-based CT image enhancement."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CT image."
    )
    parser.add_argument(
        "--output",
        default="enhanced_ct.png",
        help="Output image path (default: enhanced_ct.png)."
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=3,
        help="Odd Gaussian blur kernel size; use 1 to disable blur (default: 3)."
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Save the output without displaying comparison plots."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    original, enhanced = enhance_image(
        args.input, blur_kernel=args.blur_kernel
    )
    save_and_plot(
        original,
        enhanced,
        args.output,
        show_plot=not args.no_plot
    )


if __name__ == "__main__":
    main()
