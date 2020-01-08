from params import *
from imports import *
from model_zoo.common import *


class GCN(nn.Module):
    """
    Implementation of the Graph Convolutional Network layer from :
    Thomas N. Kipf, Max Welling. Semi-Supervised Classification with Graph Convolutional Networks
    (https://openreview.net/pdf?id=SJU4ayYgl)
    """
    def __init__(self, input_features, output_features, A, use_bias=False):
        """
        Constructor
        
        Arguments:
            input_features {int} -- Size of the input
            output_features {int} -- Size of the output
            A {numpy array} -- Matrix of the GCN, expected of size (num_classes x num_classes)
        
        Keyword Arguments:
            use_bias {bool} -- Whether to use bias (default: {False})
        """
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
        """
        Usual torch forward function
        
        Arguments:
            h {torch tensor} -- Input features, expected of size (batch_size x num_classes x input_features)
        
        Returns:
            torch tensor-- Output features, should be of size (batch_size x num_classes x output_features)
        """
        prod = torch.matmul(torch.matmul(self.A, h), self.W)
        return self.activation(prod + self.bias) if self.use_bias else self.activation(prod)


class GCNClassifier(Model):
    """
    Implementation of the proposed Graph Convolutional Network for multi-label classfication from :
    Zhao-Min Chen, Xiu-Shen Wei, Peng Wang, and Yanwen Guo. Multi-label image recognition with graph convolutional networks
    (http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Multi-Label_Image_Recognition_With_Graph_Convolutional_Networks_CVPR_2019_paper.pdf)
    """
    def __init__(self, backbone, num_classes, A, class_embeddings, n_gcns=2, use_bias=False):
        """
        Constructor
        
        Arguments:
            backbone {string} -- Name of the network to use as backbone. Expected in ["resnet34", "resnet101", "resnext101"]
            num_classes {int} -- Number of classes of the problem
            A {numpy array} -- Matrix of the GCN, expected of size (num_classes x num_classes)
            class_embeddings {numpy array} -- Word embeddings of the classes. Expected of size (num_classes x E)
        
        Keyword Arguments:
            n_gcns {int} -- Number of GCN layers to use (default: {2})
            use_bias {bool} -- Whether to use bias for the output dot product (default: {False})
        """
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
        """
        Usual torch forward function
        
        Arguments:
            x {torch tensor} -- Batch of images, expect of size (batch_size x 3 x img_size x img_size)
        
        Returns:
            torch tensor -- Logits, should be of size (batch_size x num_classes)
        """
        x = self.backbone(x)
        x = F.adaptive_max_pool2d(x, 1).view(-1, self.nb_ft)
        
        classifiers = self.gcns(self.embeddings)
        output = torch.matmul(x, classifiers.T)
        
        return output + self.clf_bias if self.use_bias else output

    def get_classifiers(self):
        """
        Gets the learned classifier of the model
        
        Returns:
            torch tensor -- classifiers, of size (num_classes x nb_ft)
        """
        return self.gcns(self.embeddings)