"""Model builders for the HRCT classification experiments."""

from .densenet121 import build_model as build_densenet121
from .efficientnetb0 import build_model as build_efficientnetb0
from .inception_v3 import build_model as build_inception_v3
from .mobilenet import build_model as build_mobilenet
from .mobilenet_v2 import build_model as build_mobilenet_v2
from .resnet50v2 import build_model as build_resnet50v2
from .resnet101 import build_model as build_resnet101
from .xception import build_model as build_xception

MODEL_BUILDERS = {
    "DenseNet121": build_densenet121,
    "EfficientNetB0": build_efficientnetb0,
    "InceptionV3": build_inception_v3,
    "MobileNet": build_mobilenet,
    "MobileNetV2": build_mobilenet_v2,
    "ResNet50V2": build_resnet50v2,
    "ResNet101": build_resnet101,
    "Xception": build_xception,
}
