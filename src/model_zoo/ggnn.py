from params import *
from imports import *
from model_zoo.common import *

class GGNN(nn.Module):
    """
    Implementation of the Graph Gated Neural Network layer from :
    Yujia Li, Daniel Tarlow, Marc Brockschmidt, Richard Zemel. Gated Graph Sequence Neural Networks
    (https://arxiv.org/pdf/1511.05493.pdf)
    """
    def __init__(self, input_features, A_in, A_out, time_steps=3):
        """
        Constructor
        
        Arguments:
            input_features {int} -- Size of the input features
            A_in {numpy array} -- Matrix (in) to build the graph. Expected of size (num_edges_label x num_classes x num_classes)
            A_out {numpy array} -- Matrix (out) to build the graph. Expected of size (num_edges_label x num_classes x num_classes)
        
        Keyword Arguments:
            time_steps {int} -- Number of steps in the message passing loop (default: {3})
        """
        super().__init__()
        
        self.input_features = input_features
        self.time_steps = time_steps
        
        self.d = A_in.shape[0]
        self.n_nodes = A_in.shape[1]
        
        self.A_in = torch.tensor(A_in).float().to(DEVICE)
        self.A_out= torch.tensor(A_out).float().to(DEVICE)
        
        self.in_fcs = nn.ModuleList([])
        self.out_fcs = nn.ModuleList([])
        for i in range(self.d):
            self.in_fcs.append(nn.Linear(input_features, input_features))
            self.out_fcs.append(nn.Linear(input_features, input_features))

        self.Wz = nn.Linear(2 * input_features, input_features, bias=False)
        self.Uz = nn.Linear(input_features, input_features, bias=False)
        self.Wr = nn.Linear(2 * input_features, input_features, bias=False)
        self.Ur = nn.Linear(input_features, input_features, bias=False)
        self.W = nn.Linear(2 * input_features, input_features, bias=False)
        self.U = nn.Linear(input_features, input_features, bias=False)

    def forward(self, xv):
        """
        Usual torch forward function
        
        Arguments:
            xv {torch tensor} -- Input features, expected of size (batch_size x num_classes x input_features)
        
        Returns:
            torch tensor -- Output of the GGNN, should be of size (batch_size x num_classes x input_features)
        """
        bs, n_nodes, input_fts = xv.size()
        h = xv
        
        in_states = []
        out_states = []
        for i in range(self.d):
            in_states.append(self.in_fcs[i](h))
            out_states.append(self.out_fcs[i](h))
        
        in_states = torch.cat(in_states, 1) # (bs, n_nodes * d,  input_fts)
        out_states = torch.cat(out_states, 1)
        
        A_in = self.A_in.view(1, -1, self.n_nodes).transpose(1, 2).repeat(bs, 1, 1)  # (bs, n_nodes, n_nodes * d)
        A_out = self.A_out.view(1, -1, self.n_nodes).transpose(1, 2).repeat(bs, 1, 1)  # (bs, n_nodes, n_nodes * d)

        for t in range(self.time_steps):
            a_in = torch.bmm(A_in, in_states)
            a_out = torch.bmm(A_out, out_states)
        
            a = torch.cat([a_in, a_out], 2)
            z = F.sigmoid(self.Wz(a) + self.Uz(h))
            r = F.sigmoid(self.Wr(a) + self.Ur(h))
            
            h_tilde = F.tanh(self.W(a) + self.U(r * h))
            h = (1 - z) * h + (z * h_tilde)

        return h


class GGNNClassifier(Model):
    """
    Implementation of a GGNN for multi-label classfication inspired from :
    Kenneth Marino, Ruslan Salakhutdinov, and Abhinav Gupta. The more you know: Using knowledge graphs for image classification
    (https://arxiv.org/pdf/1612.04844.pdf)
    """
    def __init__(self, backbone, num_classes, A_in, A_out, ggnn_dim=10, time_steps=3, use_ggnn=True):
        """
        Constructor
        
        Arguments:
            backbone {string} -- Name of the network to use as backbone. Expected in ["resnet34", "resnet101", "resnext101"]
            num_classes {int} -- Number of classes of the problem (default: {20})
            A_in {numpy array} -- Matrix (in) to build the graph of the GGNN. Expected of size (num_edges_label x num_classes x num_classes)
            A_out {numpy array} -- Matrix (out) to build the graph of the GGNN. Expected of size (num_edges_label x num_classes x num_classes)
        
        Keyword Arguments:
            ggnn_dim {int} -- Number of features of the GGNN (default: {10})
            time_steps {int} --  Number of steps in the message passing loop of the GGNN (default: {3})
            use_ggnn {bool} -- Whether to use the GGNN layer (default: {True})
        """
        super().__init__()

        self.num_classes = num_classes
        self.ggnn_dim = ggnn_dim
        self.use_ggnn = use_ggnn
        self.nb_edges = A_in.shape[1]
        
        if backbone == "resnext101":
            self.backbone = resnext101_32x8d_wsl()
            self.nb_ft = 2048
        else:
            self.backbone = get_encoder(SETTINGS[backbone])
            self.nb_ft = SETTINGS[backbone]["out_shapes"][0]
            
        self.logits = nn.Linear(self.nb_ft, self.nb_edges)

        if use_ggnn:
            self.ggnn = GGNN(ggnn_dim, A_in, A_out, time_steps=time_steps)
            self.out_fts = self.nb_edges  * ggnn_dim
            self.out = nn.Linear(self.nb_edges + self.out_fts, num_classes)
            
            
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
        
        annotations = self.logits(x)
        
        if self.use_ggnn:
            h = self.ggnn(annotations.unsqueeze(-1).repeat(1, 1, self.ggnn_dim))
            h = h.view(-1, self.out_fts)
            return self.out(torch.cat([annotations, h], 1))
        
        else:
            return annotations[:, :20]