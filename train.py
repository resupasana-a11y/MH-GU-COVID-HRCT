"""Train one HRCT classification pipeline."""

import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

from models import MODEL_BUILDERS
from preprocessing import load_dataset, split_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train an HRCT image classifier.")
    parser.add_argument("--model", choices=list(MODEL_BUILDERS), required=True)
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing Merged_healthy_images/ and Merged_infected_images/.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    tf.random.set_seed(7)
    np.random.seed(3)

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    x, y = load_dataset(args.data_dir, image_size=args.image_size)
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        x, y, seed=args.seed
    )
    print(f"Train: {x_train.shape}, validation: {x_val.shape}, test: {x_test.shape}")

    # Match the notebook's ImageDataGenerator augmentation settings.
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )

    model = MODEL_BUILDERS[args.model](
        image_size=args.image_size, learning_rate=args.learning_rate
    )
    model.summary()

    checkpoint_path = output_dir / "best_model.keras"
    checkpoint = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size, shuffle=True),
        steps_per_epoch=max(1, x_train.shape[0] // args.batch_size),
        epochs=args.epochs,
        verbose=1,
        callbacks=[checkpoint],
        # The notebook accidentally used X_test here. Use X_val to keep the
        # held-out test set independent until final evaluation.
        validation_data=(x_val, y_val),
    )

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(
            {key: [float(v) for v in values] for key, values in history.history.items()},
            f, indent=2
        )

    # Save the split sizes and class ordering for experiment documentation.
    metadata = {
        "model": args.model,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed_split": args.seed,
        "numpy_seed": 3,
        "tensorflow_seed": 7,
        "class_names": ["Merged_healthy_images", "Merged_infected_images"],
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "backbone_weights": None,
    }
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Best checkpoint saved to: {checkpoint_path}")
    print("Next, run evaluate.py using the same dataset directory.")


if __name__ == "__main__":
    main()
