from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4, resnet14x05, resnet20x05, resnet20x0375
from .mcdo_resnet import mcdo_resnet8, mcdo_resnet14, mcdo_resnet20, mcdo_resnet32, mcdo_resnet44, mcdo_resnet56, mcdo_resnet110, mcdo_resnet8x4, mcdo_resnet32x4
from .resnetv2 import resnet18, resnet34, resnet50, resnet101, resnet152
from .mcdo_resnetv2 import mcdo_resnet18, mcdo_resnet34, mcdo_resnet50, mcdo_resnet101, mcdo_resnet152
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,

    'mcdo_resnet8': mcdo_resnet8,
    'mcdo_resnet14': mcdo_resnet14,
    'mcdo_resnet20': mcdo_resnet20,
    'mcdo_resnet32': mcdo_resnet32,
    'mcdo_resnet44': mcdo_resnet44,
    'mcdo_resnet56': mcdo_resnet56,
    'mcdo_resnet110': mcdo_resnet110,
    'mcdo_resnet8x4': mcdo_resnet8x4,
    'mcdo_resnet32x4': mcdo_resnet32x4,

    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,

    'mcdo_resnet18': mcdo_resnet18,
    'mcdo_resnet34': mcdo_resnet34,
    'mcdo_resnet50': mcdo_resnet50,
    'mcdo_resnet101': mcdo_resnet101,
    'mcdo_resnet152': mcdo_resnet152,

    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'resnet14x05': resnet14x05,
    'resnet20x05': resnet20x05,
    'resnet20x0375': resnet20x0375,
}
