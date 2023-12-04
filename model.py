import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

    
class ConvMalware(nn.Module):
    def __init__(self):
        super(ConvMalware, self).__init__()
        self.embedding = nn.Embedding(257, 8)
        self.conv = nn.Conv1d(8, 128, 500, 500)
        self.convsig = nn.Conv1d(8, 128, 500, 500)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 1)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x): # x : (batch_size, seqlen)
        x = self.embedding(x) # (batch_size, seqlen, emb_dim)
        x = x.transpose(1, 2) # (batch_size, emb_dim, seqlen)
        x1 = self.conv(x) # (batch_size, 128, seqlen)
        x1 = self.batchnorm1(x1)
        x2 = F.sigmoid(self.convsig(x)) # (batch_size, 128, seqlen)
        x2 = self.batchnorm2(x2)
        x3 = torch.mul(x1, x2) # (batch_size, 128, seqlen)
        x3 = self.relu1(x3)
        x3 = self.dropout(x3)
        x3 = self.maxpool(x3) # (batch_size, 128, 1)
        x3 = x3.squeeze() # (batch_size, 128)
        return self.fc(x3) # (batch_size, 1)