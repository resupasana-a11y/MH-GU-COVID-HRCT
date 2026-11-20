"""Dataset loading and train/validation/test splitting."""

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import load_img, img_to_array, to_categorical

CLASS_NAMES = ["Merged_healthy_images", "Merged_infected_images"]


def load_dataset(data_dir, image_size=224):
    """Load images from one subfolder per class, as in the supplied notebook."""
    data_dir = Path(data_dir)
    images, labels = [], []

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Expected class folder not found: {class_dir}. "
                f"Create folders named: {', '.join(CLASS_NAMES)}"
            )
        image_paths = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        for image_path in image_paths:
            image = load_img(image_path, target_size=(image_size, image_size))
            images.append(img_to_array(image) / 255.0)
            labels.append(class_index)

    if not images:
        raise ValueError(f"No supported images found in {data_dir}")

    x = np.asarray(images, dtype=np.float32)
    y = to_categorical(np.asarray(labels, dtype=np.int64), num_classes=2)
    return x, y


def split_dataset(x, y, seed=42):
    """Reproduce the notebook's 80/10/10 split and random_state=42."""
    x_train, x_remaining, y_train, y_remaining = train_test_split(
        x, y, test_size=0.20, random_state=seed
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_remaining, y_remaining, test_size=0.50, random_state=seed
    )
    return x_train, x_val, x_test, y_train, y_val, y_test
