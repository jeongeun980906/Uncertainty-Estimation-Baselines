from core.cnn import simple_cnn
import torch
import torch.nn as nn
import math

class SNGP_module(nn.Module):
    def __init__(self,
                 x_dim      = [1,28,28],    # input dimension
                 k_size     = 3,            # kernel size
                 c_dims     = [32,64],      # conv channel dimensions
                 p_sizes    = [2,2],        # pooling sizes
                 h_dims     = [128],        # hidden dimensions
                 y_dim      = 10,           # output dimension
                 USE_BN     = True,         # whether to use batch-norm
                 HEAD       = True,         # whether to use linear head
                 DROPOUT    = False):
        super(SNGP_module,self).__init__()
        self.simple_cnn = simple_cnn(x_dim      = x_dim,
                 k_size     = k_size,            # kernel size
                 c_dims     = c_dims,      # conv channel dimensions
                 p_sizes    = p_sizes,        # pooling sizes
                 h_dims     = h_dims,        # hidden dimensions
                 y_dim      = y_dim,           # output dimension
                 USE_BN     = USE_BN,         # whether to use batch-norm
                 HEAD       = HEAD,         # whether to use linear head
                 DROPOUT    = DROPOUT,
                 SN         = True
                 )
        self.D_L = h_dims[-1]
        self.fixed = torch.nn.Linear(self.D_L, self.D_L, bias=True)
        self.fixed.weight.requires_grad = False
        self.fixed.bias.requires_grad = False
        self.fixed.weight.data.normal_(0,1)
        self.fixed.bias.data.uniform_(0, math.pi*2)
        self.learnable_d = torch.nn.Linear(self.D_L, y_dim, bias=False)
        self.learnable_d.weight.data.normal_(0,1)
        self.Sigma_inv = [torch.eye(self.D_L)]*y_dim
        self.device = 'cuda'
        self.y_dim = y_dim

    def forward(self, x, update = False):
        feature = self.simple_cnn.net(x) # [N x D]
        rff_approx = math.sqrt(2/self.D_L)* torch.cos(self.fixed(feature))
        g = self.learnable_d(rff_approx)
        if update:
            temp1 = rff_approx.unsqueeze(-1)
            temp2 = rff_approx.unsqueeze(1)
            g = torch.softmax(g, dim=-1)
            temp = torch.matmul(temp1, temp2)
            for c in range(self.y_dim):
                p_ = g[:,c]*(1-g[:,c])
                res = []
                for p, el in zip(p_, temp):
                    temp2 = p*el
                    res.append(el.unsqueeze(0))
                res = torch.cat(res)
                self.Sigma_inv[c] = torch.eye(self.D_L).to(self.device)+ torch.sum(res, dim=0)
                # print(self.Sigma_inv[c].shape)
            # print(self.Sigma_inv)
        return g

    def inference(self,x):
        feature = self.simple_cnn.net(x) # [N x D]
        feature = math.sqrt(2/self.D_L)*torch.cos(self.fixed(feature)) # [N x D]
        logit = self.learnable_d(feature)
        temp = feature.unsqueeze(1)
        temp2 = feature.unsqueeze(-1)
        pred = []
        for c, sig_inv in enumerate(self.Sigma_inv):
            sig = torch.inverse(sig_inv)
            var = torch.matmul(temp, sig)
            var = torch.matmul(var,temp2)
            m = torch.randn(var.shape[0]).to(self.device)* var[:,0,0] + logit[:,c]
            pred.append(m.unsqueeze(-1))
        pred = torch.cat(pred, dim=-1)
        demp = self.y_dim/(self.y_dim + torch.sum(torch.exp(pred), dim = -1))
        return pred, demp

if __name__ == '__main__':
    model = SNGP_module()
    x = torch.randn((64,1,28,28))
    a = model(x, update= True)
    a = model.inference(x)