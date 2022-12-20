import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from core.cnn import simple_cnn
import torch.nn as nn

class vos():
    def __init__(self,num_classes, feature_dim = 128, T =1, queue_size = 100):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.mu = torch.zeros((num_classes,feature_dim))
        self.var = torch.zeros((feature_dim,feature_dim))
        self.num_outlier = T
        self.queue_size = queue_size

    def append(self,features,labels):
        pass

    def fit(self,features, labels):
        self.epsilons = []
        for c in range(self.num_classes):
            index = torch.where(labels == c)[0]
            data = features[index]
            mu = torch.mean(data,axis=0)
            self.mu[c] = mu
                
            log_probs = []
            m = MultivariateNormal(self.mu[c], self.var[c])
            for i in range(10,000):
                sa = m.sample()
                log_prob = torch.exp(m.log_prob(sa))
                log_probs.append(log_prob)
            log_probs = np.asarray(log_probs)
            small_probs = np.sort(log_probs)
            ep = small_probs[:self.num_outlier].mean()
            self.epsilons.append(ep)


    def sample(self, label):
        stotal = 0
        m = MultivariateNormal(self.mu[label], self.var[label])
        outliers = torch.zeros((self.num_outlier,self.feature_fim))
        while stotal < self.num_outlier:
            sa = m.sample()
            log_prob = torch.exp(m.log_prob(sa))
            if log_prob < self.epsilons[label]:
                outliers[stotal] = sa
                stotal += 1
        return outliers


class vos_model(nn.Module):
    def __init__(self,
                 x_dim      = [1,28,28],    # input dimension
                 k_size     = 3,            # kernel size
                 c_dims     = [32,64],      # conv channel dimensions
                 p_sizes    = [2,2],        # pooling sizes
                 h_dims     = [128],        # hidden dimensions
                 y_dim      = 10,           # output dimension
                 USE_BN     = True,         # whether to use batch-norm
                 HEAD       = True,         # whether to use linear head
                 DROPOUT    = False
                 ):
        super(vos_model,self).__init__()
        self.simple_cnn = simple_cnn(x_dim      = x_dim,
                 k_size     = k_size,            # kernel size
                 c_dims     = c_dims,      # conv channel dimensions
                 p_sizes    = p_sizes,        # pooling sizes
                 h_dims     = h_dims,        # hidden dimensions
                 y_dim      = y_dim,           # output dimension
                 USE_BN     = USE_BN,         # whether to use batch-norm
                 HEAD       = HEAD,         # whether to use linear head
                 DROPOUT    = DROPOUT
                 )
        self.energy_surface = nn.Linear(1,1)
        self.weight_energy = nn.Linear(y_dim, 1).cuda()
        self.y_dim = y_dim
        self.beta = .5
        nn.init.kaiming_normal(self.weight_energy.weight)
        nn.init.kaiming_normal(self.energy_surface.weight)

    def forward(self,x):
        feature = self.simple_cnn.net(x)
        return self.simple_cnn.head(feature), feature

    def energy(self,logits):
        weight = torch.softmax(self.weight_energy.weight, dim=-1)
        weighted_energy = torch.matmul(weight,(torch.exp(logits)).T)
        return -torch.log(weighted_energy+1e-6).squeeze(0).unsqueeze(-1)

    def cross_entropy(self,out,target):
        out = torch.softmax(out, dim=-1)
        loss = torch.sum(-target*torch.log(out+1e-6),dim=-1)
        if loss.shape[0] == 0:
            return 0
        return loss.mean()

    def binary_sigmoid_loss(self,energy, en_target):
        return torch.mean(-torch.log(torch.exp(-en_target*energy)/
                            (1+torch.exp(-energy))+1e-6))

    def oe_criterion(self,out, target):
        oe = (target == -1)
        id_ =~ oe 
        in_en = self.energy(out[id_])
        in_en = self.energy_surface(in_en).squeeze(-1)
        in_en = torch.relu(in_en)
        
        in_target = torch.ones((in_en.shape[0])).to('cuda')
        
        od_en = self.energy(out[oe])
        od_en = self.energy_surface(od_en).squeeze(-1)
        od_en = torch.relu(od_en)
        od_target = torch.zeros((od_en.shape[0])).to('cuda')

        en = torch.cat((in_en,od_en),dim=-1)
        en_target = torch.cat((in_target, od_target),dim=-1)
        target_exp = torch.eye(self.y_dim)[target[id_]].to('cuda')

        loss_ce = self.cross_entropy(out[id_], target_exp)
        loss_vos = self.binary_sigmoid_loss(en, en_target)
        return self.beta*loss_vos + loss_ce