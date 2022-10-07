import torch
import torch.nn as nn

from core.cnn import simple_cnn

class MixtureOfLogits(nn.Module):
    def __init__(self,
                 in_dim     = 64,   # input feature dimension 
                 y_dim      = 10,   # number of classes 
                 k          = 5,    # number of mixtures
                 sigma      = True,  # use sigma
                 sig_min    = 1e-4, # minimum sigma
                 sig_max    = None, # maximum sigma
                 mu_min     = -3,   # minimum mu (init)
                 mu_max     = +3,   # maximum mu (init)
                 SHARE_SIG  = True  # share sigma among mixture
                 ):
        super(MixtureOfLogits,self).__init__()
        self.in_dim     = in_dim    # Q
        self.y_dim      = y_dim     # D
        self.k          = k         # K
        self.sigma      = sigma
        self.sig_min    = sig_min
        self.sig_max    = sig_max
        self.mu_min     = mu_min
        self.mu_max     = mu_max
        self.SHARE_SIG  = SHARE_SIG
        self.build_graph()
        self.init_param()

    def build_graph(self):
        self.fc_pi      = nn.Linear(self.in_dim,self.k)
        self.fc_mu      = nn.Linear(self.in_dim,self.k*self.y_dim)
        if self.sigma:
            if self.SHARE_SIG:
                self.fc_sigma   = nn.Linear(self.in_dim,self.k)
            else:
                self.fc_sigma   = nn.Linear(self.in_dim,self.k*self.y_dim)

    def forward(self,x):
        """
            :param x: [N x Q]
        """
        pi_logit        = self.fc_pi(x)                                 # [N x K]
        pi              = torch.softmax(pi_logit,dim=1)                 # [N x K]
        mu              = self.fc_mu(x)                                 # [N x KD]
        mu              = torch.reshape(mu,(-1,self.k,self.y_dim))      # [N x K x D]
        if self.sigma:
            if self.SHARE_SIG:
                sigma       = self.fc_sigma(x)                              # [N x K]
                sigma       = sigma.unsqueeze(dim=-1)                       # [N x K x 1]
                sigma       = sigma.expand_as(mu)                           # [N x K x D]
            else:
                sigma       = self.fc_sigma(x)                              # [N x KD]
            sigma           = torch.reshape(sigma,(-1,self.k,self.y_dim))   # [N x K x D]
            if self.sig_max is None:
                sigma = self.sig_min + torch.exp(sigma)                     # [N x K x D]
            else:
                sig_range = (self.sig_max-self.sig_min)
                sigma = self.sig_min + sig_range*torch.sigmoid(sigma)       # [N x K x D]
            mol_out = {'pi':pi,'mu':mu,'sigma':sigma}
        else:
            mol_out = {'pi':pi, 'mu':mu}
            # print(pi)
        return mol_out

    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d): # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
        self.fc_mu.bias.data.uniform_(self.mu_min,self.mu_max)


class mln_cnn(nn.Module):
    def __init__(self,
                x_dim      = [1,28,28],    # input dimension
                k_size     = 3,            # kernel size
                c_dims     = [32,64],      # conv channel dimensions
                p_sizes    = [2,2],        # pooling sizes
                h_dims     = [128],        # hidden dimensions
                y_dim      = 10,           # output dimension
                USE_BN     = True,         # whether to use batch-norm)
                in_dim     = 128,   # input feature dimension 
                k          = 5,    # number of mixtures
                sigma      = True,  # use sigma
                sig_min    = 1e-4, # minimum sigma
                sig_max    = None, # maximum sigma
                mu_min     = -3,   # minimum mu (init)
                mu_max     = +3,   # maximum mu (init)
                SHARE_SIG  = True  # share sigma among mixture
                ):
        super(mln_cnn,self).__init__()
        self.backbone = simple_cnn(
            x_dim = x_dim,
            k_size= k_size,
            c_dims = c_dims,
            p_sizes= p_sizes,
            h_dims = h_dims,
            y_dim = y_dim,
            USE_BN=USE_BN,
            HEAD = False, DROPOUT=False
        )
        self.mln_head = MixtureOfLogits(
            in_dim= in_dim, 
            y_dim = y_dim,
            k = k,
            sigma = sigma,
            sig_min = sig_min,
            sig_max = sig_max,
            mu_max= mu_max,
            mu_min = mu_min,
            SHARE_SIG = SHARE_SIG
        )
        self.init_param()
    def forward(self,x):
        feature = self.backbone.net(x)
        return self.mln_head(feature)

    def init_param(self):
        self.backbone.init_param()
        self.mln_head.init_param()