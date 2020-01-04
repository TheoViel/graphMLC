from params import *
from imports import *
from model_zoo.common import *


class GCN(nn.Module):
    def __init__(self, input_features, output_features, A, use_bias=False):
        super().__init__()
        self.use_bias = use_bias

        self.A = torch.tensor(A).float().to(DEVICE)
        # self.A = nn.Parameter(torch.tensor(A).float(), requires_grad=True)

        self.activation = nn.LeakyReLU(0.2)
        
        self.W = nn.Parameter(torch.Tensor(input_features, output_features), requires_grad=True)
        self.W.data.uniform_(- 1 / np.sqrt(output_features), 1 / np.sqrt(output_features))
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1, output_features), requires_grad=True)

    def forward(self, h):
        prod = torch.matmul(torch.matmul(self.A, h), self.W)
        return self.activation(prod + self.bias) if self.use_bias else self.activation(prod)


class GCNClassifier(Model):
    def __init__(self, backbone, num_classes, A, class_embeddings, n_gcns=2, use_bias=False):
        super().__init__()

        self.num_classes = num_classes
        self.use_bias = use_bias
        self.embeddings = torch.tensor(class_embeddings).float().to(DEVICE)
        
        if backbone == "resnext101":
            self.backbone = resnext101_32x8d_wsl()
            self.nb_ft = 2048
        else:
            self.backbone = get_encoder(SETTINGS[backbone])
            self.nb_ft = SETTINGS[backbone]["out_shapes"][0]
        
        gcn_dims = [class_embeddings.shape[1]] + [self.nb_ft // 2] * (n_gcns - 1) + [self.nb_ft]
        gcns = [GCN(gcn_dims[i], gcn_dims[i + 1], A) for i in range(n_gcns)]
        self.gcns = nn.Sequential(*gcns)
        
        if use_bias:
            self.clf_bias = nn.Parameter(torch.zeros(1, num_classes), requires_grad=True)
            
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_max_pool2d(x, 1).view(-1, self.nb_ft)
        
        classifiers = self.gcns(self.embeddings)
        output = torch.matmul(x, classifiers.T)
        
        return output + self.clf_bias if self.use_bias else output

    def get_classifiers(self):
        return self.gcns(self.embeddings)