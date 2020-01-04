from imports import *
from model_zoo.resnet import *


SETTINGS = {
    "resnet34": {
        "name": "resnet34",
        "encoder": ResNet,
        "pretrained_settings": pretrained_settings["resnet34"]["imagenet"],
        "out_shapes": (512, 256, 128, 64, 64),
        "params": {"block": BasicBlock, "layers": [3, 4, 6, 3],},
    },
    "resnet50": {
        "name": "resnet50",
        "encoder": ResNet,
        "pretrained_settings": pretrained_settings["resnet50"]["imagenet"],
        "out_shapes": (2048, 1024, 512, 256, 64),
        "params": {"block": Bottleneck, "layers": [3, 4, 6, 3],},
    },
    "resnet101": {
        "name": "resnet101",
        "encoder": ResNet,
        "pretrained_settings": pretrained_settings["resnet101"]["imagenet"],
        "out_shapes": (2048, 1024, 512, 256, 64),
        "params": {"block": Bottleneck,"layers": [3, 4, 23, 3],},
    },
} 


def get_encoder(settings):
    Encoder = settings["encoder"]
    encoder = Encoder(**settings["params"])
    encoder.out_shapes = settings["out_shapes"]

    if settings["pretrained_settings"] is not None:
        encoder.load_state_dict(
            model_zoo.load_url(settings["pretrained_settings"]["url"])
        )
    return encoder


def adaptive_concat_pool2d(x, sz=1):
    out1 = F.adaptive_avg_pool2d(x, sz)
    out2 = F.adaptive_max_pool2d(x, sz)
    return torch.cat([out1, out2], 1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



