"""Build the ResNet101 HRCT classifier, following the supplied Aug_res50v2 notebook."""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(image_size: int = 224, learning_rate: float = 2e-5):
    """Create an unweighted two-class classifier.

    Backbone weights are intentionally initialized randomly (weights=None),
    matching the supplied notebook. Data augmentation is applied by train.py
    through ImageDataGenerator, also matching the notebook.
    """
    backbone = keras.applications.ResNet101(
        include_top=False,
        weights=None,
        input_shape=(image_size, image_size, 3),
    )
    inputs = keras.Input(shape=(image_size, image_size, 3), name="ct_image")
    x = backbone(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation="softmax", name="class_probabilities")(x)

    model = keras.Model(inputs, outputs, name="resnet101_hrct_classifier")
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model
