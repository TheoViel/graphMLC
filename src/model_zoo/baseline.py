from model_zoo.common import *

class Classifier(Model):
    def __init__(self, backbone, num_classes=20):
        super().__init__()

        self.num_classes = num_classes

        if backbone == "resnext101":
            self.backbone = resnext101_32x8d_wsl()
            self.nb_ft = 2048
        else:
            self.backbone = get_encoder(SETTINGS[backbone])
            self.nb_ft = SETTINGS[backbone]["out_shapes"][0]

        self.logit = nn.Linear(self.nb_ft, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_max_pool2d(x, 1).view(-1, self.nb_ft)
        return self.logit(x)