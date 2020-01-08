from model_zoo.common import *

class Classifier(Model):
    """
    Baseline classifier
    """
    def __init__(self, backbone, num_classes=20):
        """
        Constructor
        
        Arguments:
            backbone {string} -- Name of the network to use as backbone. Expected in ["resnet34", "resnet101", "resnext101"]
            num_classes {int} -- Number of classes of the problem (default: {20})
        """
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
        """
        Usual torch forward function
        
        Arguments:
            x {torch tensor} -- Batch of images, expect of size (batch_size x 3 x img_size x img_size)
        
        Returns:
            torch tensor -- Logits, should be of size (batch_size x num_classes)
        """
        x = self.backbone(x)
        x = F.adaptive_max_pool2d(x, 1).view(-1, self.nb_ft)
        return self.logit(x)