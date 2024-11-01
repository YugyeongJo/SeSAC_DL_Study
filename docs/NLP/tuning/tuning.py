import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

class BatchNormalization(nn.Module):
    def __init__(self, hidden_dim, batch_dim = 0):
        super(BatchNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = 1e-6
        self.batch_dim = 0
    
    def forward(self, x):
        mean = x.mean(dim = self.batch_dim) 
        std = x.var(dim = self.batch_dim) 
        x_hat = (x - mean) / torch.sqrt(std + self.eps)

        return self.gamma * x_hat + self.beta

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
        
    def forward(self, x):
        if self.training:
            mask = (torch.rand(x.shape) > self.p).float()
            return x * mask / (1 - self.p)
        return x
    