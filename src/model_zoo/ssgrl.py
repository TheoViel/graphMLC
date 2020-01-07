from params import *
from imports import *
from model_zoo.common import *


class SemanticDecoupling(nn.Module):
    """
    Implementation of the Semantic Decoupling module from :
    Tianshui Chen, Muxin Xu, Xiaolu Hui, Hefeng Wu, and Liang Lin. Learning semantic-specific graphrepresentation for multi-label image recognition
    (http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Learning_Semantic-Specific_Graph_Representation_for_Multi-Label_Image_Recognition_ICCV_2019_paper.pdf)
    """
    def __init__(self, input_features, class_embeddings, d1=1024, d2=1024):
        """
        Constructor
        
        Arguments:
            input_features {int} -- Number of features of the input
            class_embeddings {numpy array} -- Word embeddings of the classes
        
        Keyword Arguments:
            d1 {int} -- First hidden size (default: {1024})
            d2 {int} -- Second hidden size (default: {1024})
        """

        super().__init__()
        self.input_features = input_features
        self.c, self.embed_fts = class_embeddings.shape
        
        self.xc = torch.tensor(class_embeddings).float().to(DEVICE)
        self.dense_img = nn.Linear(self.input_features, d1, bias=False)
        self.dense_embed = nn.Linear(self.embed_fts, d1, bias=False)
        self.dense_sgam = nn.Linear(d1, d2)
        
        self.attention_fct = nn.Linear(d2, 1)

    def forward(self, fi, return_att=False):
        """
        Usual torch forward function
        
        Arguments:
            fi {torch tensor} -- Feature maps, expected of size (batch_size x height x width x input_features)
            return_att {bool} -- Whether to return the attention maps in addition (default: {False})
        
        Returns:
            torch tensor -- Semantic decoupled features, should be of size (batch_size x num_classes x input_features)
        """

        bs, n, h, w = fi.size()
        fi = fi.view(bs, n, -1).transpose(1, 2) # (Bs, h*w, n)
        img_fts = self.dense_img(fi)  # (Bs, h*w, d1)
        word_fts = self.dense_embed(self.xc)  # (c, d1)
        
        img_fts = img_fts.view(bs, h*w, 1, -1).repeat((1, 1, self.c, 1))  # (Bs, h*w, c, d1)
        word_fts = word_fts.view(1, 1, self.c, -1).repeat((bs, h*w, 1, 1))  # (Bs, h*w, c, d1)
        
        fi_tilde = self.dense_sgam(F.tanh(img_fts * word_fts))  # (Bs, h*w, c, d2)
        
        a = self.attention_fct(fi_tilde).view(bs, h*w, self.c, 1)  # (Bs, h*w, c, 1)
        a = F.softmax(a, 1).repeat(1, 1, 1, n)  # (Bs, h*w, c, n)

        fi = fi.view(bs, h*w, 1, n).repeat(1, 1, self.c, 1) # (Bs, h*w, c, n)

        if return_att:
            return (a * fi).sum(1), a
        return (a * fi).sum(1)  # (bs, c, n)


class SemanticInteraction(nn.Module):
    """        
    Implementation of the Semantic Interaction module from :
    Tianshui Chen, Muxin Xu, Xiaolu Hui, Hefeng Wu, and Liang Lin. Learning semantic-specific graphrepresentation for multi-label image recognition
    (http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Learning_Semantic-Specific_Graph_Representation_for_Multi-Label_Image_Recognition_ICCV_2019_paper.pdf)
    It is a simplified GGNN.
    """
    def __init__(self, input_features, coocurence_matrix, time_steps=3):
        """
        Constructor
        
        Arguments:
            input_features {int} -- Size of the input features
            coocurence_matrix {numpy array} -- Coocurence matrix to build the graph. Expected of size (num_classes x num_classes)
            time_steps {int} -- Number of steps in the message passing loop (default: {3})
        """
        super().__init__()
        self.input_features = input_features
        self.time_steps = time_steps
        
        self.A = torch.tensor(coocurence_matrix).float().to(DEVICE)

        self.Wz = nn.Linear(2 * input_features, input_features, bias=False)
        self.Uz = nn.Linear(input_features, input_features, bias=False)
        self.Wr = nn.Linear(2 * input_features, input_features, bias=False)
        self.Ur = nn.Linear(input_features, input_features, bias=False)
        self.W = nn.Linear(2 * input_features, input_features, bias=False)
        self.U = nn.Linear(input_features, input_features, bias=False)
        """
        fc : Features vector of all categories (bs, c, input_ft)
        """
    def forward(self, fc):
        """
        Usual torch forward function
        
        Arguments:
            fc {torch tensor} -- Input features, expected of size (batch_size x num_classes x input_features)
        
        Returns:
            torch tensor -- Output of the semantic interaction module, should be of size (batch_size x num_classes x input_features)
        """

        bs, c, input_features = fc.size()
        h = fc
        
        for t in range(self.time_steps):
            A = self.A.unsqueeze(0).repeat(bs, 1, 1)

            a_in = torch.bmm(A, h)
            a_out = torch.bmm(A.transpose(1, 2), h)
            a = torch.cat([a_in, a_out], 2)

            z = F.sigmoid(self.Wz(a) + self.Uz(h))
            r = F.sigmoid(self.Wr(a) + self.Ur(h))
            
            h_tilde = F.tanh(self.W(a) + self.U(r * h))
            h = (1 - z) * h + (z * h_tilde)

        return h


class SSGRLClassifier(Model):
    """
    Implementation of the Semantic-Specific Graph Representation Learning framework from :
    Tianshui Chen, Muxin Xu, Xiaolu Hui, Hefeng Wu, and Liang Lin. Learning semantic-specific graphrepresentation for multi-label image recognition
    (http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Learning_Semantic-Specific_Graph_Representation_for_Multi-Label_Image_Recognition_ICCV_2019_paper.pdf)

    """
    def __init__(self, backbone, coocurence_matrix, class_embeddings, d1=1024, d2=1024, time_steps=3, logit_fts=2048, use_si=True):
        """
        Constructor
        
        Arguments:
            backbone {string} -- Name of the network to use as backbone. Expected in ["resnet34", "resnet101", "resnext101"]
            coocurence_matrix {numpy array} -- Coocurence matrix to build the semantic interaction graph, expected of size (num_classes x num_classes)
            class_embeddings {[type]} -- Word embeddings of the classes for the semantic decoupling module
        
        Keyword Arguments:
            d1 {int} -- First hidden size of the semantic decoupling module (default: {1024})
            d2 {int} -- Second hidden size of the semantic decoupling module (default: {1024})
            time_steps {int} -- Number of steps in the message passing loop of the semantic interaction module (default: {3})
            logit_fts {int} -- Number of features of the second to last dense layer (default: {2048})
            use_si {bool} -- Whether to use the semantic interaction module (default: {True})
        """
        super().__init__()

        self.num_classes = coocurence_matrix.shape[0]

        if backbone == "resnext101":
            self.backbone = resnext101_32x8d_wsl()
            self.nb_ft = 2048
        else:
            self.backbone = get_encoder(SETTINGS[backbone])
            self.nb_ft = SETTINGS[backbone]["out_shapes"][0]

        self.pooler = nn.AvgPool2d(2, stride=2)

        self.semantic_decoupling = SemanticDecoupling(self.nb_ft, class_embeddings, d1=d1, d2=d2)
        
        if use_si:
            self.semantic_interaction = SemanticInteraction(self.nb_ft, coocurence_matrix, time_steps=time_steps)
            self.semantic_fts = self.nb_ft * 2
        else:
            self.semantic_interaction = None 
            self.semantic_fts = self.nb_ft

        self.fo = nn.Sequential(
            nn.Linear(self.semantic_fts, logit_fts), 
            nn.Tanh(),
        )
        
        self.logits = nn.ModuleList([])
        for i in range(self.num_classes):
            self.logits.append(nn.Linear(logit_fts, 1))

    def forward(self, x):
        """
        Usual torch forward function
        
        Arguments:
            x {torch tensor} -- Batch of images, expect of size (batch_size x 3 x img_size x img_size)
        
        Returns:
            [type] -- Logits, should be of size (batch_size x num_classes)
        """
        x = self.backbone(x)
        x = self.pooler(x)
    
        h0 = self.semantic_decoupling(x)


        if self.semantic_interaction is not None:
            hT = self.semantic_interaction(h0)
            o = self.fo(torch.cat([h0, hT], 2))
        else:
            o = self.fo(h0)
        
        outputs = []
        for i in range(self.num_classes):
            outputs.append(self.logits[i](o[:, i, :]))
            
        return torch.cat(outputs, 1) 

    def get_attention(self, x):
        """
        Get the attention weights of the semantic decoupling layer 
        
        Arguments:
            x {torch tensor} -- Batch of images, expect of size (batch_size x 3 x img_size x img_size)
        
        Returns:
            torch tensor -- Logits, should be of size (batch_size x num_classes)
            torch tensor -- Attention maps, should be of size (batch_size x num_classes x H x W) where H and W depend on the backbone and on the input img_size
        """
        x = self.backbone(x)
        x = self.pooler(x)
    
        h0, att = self.semantic_decoupling(x, return_att=True)

        if self.semantic_interaction is not None:
            hT = self.semantic_interaction(h0)
            o = F.tanh(self.fo(torch.cat([h0, hT], 2)))
        else:
            o = F.tanh(self.fo(h0))
        
        outputs = []
        for i in range(self.num_classes):
            outputs.append(self.logits[i](o[:, i, :]))
            
        return torch.cat(outputs, 1), att