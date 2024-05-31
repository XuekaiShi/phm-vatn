import math

import torch
from torch import Tensor
import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class VATN(nn.Module):
    def __init__(self, ):
        super(VATN, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=50, stride=25)
        self.pos_encoder = PositionalEncoding(256)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, batch_first=True),
            num_layers=8)
        self.max_pool = nn.AdaptiveMaxPool1d(4)
        self.avg_pool = nn.AdaptiveAvgPool1d(4)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(312, 64)
        self.fc_2 = nn.Linear(64, 4)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x_1 = self.max_pool(x)
        x_2 = self.avg_pool(x) 
        x = torch.cat((x_1, x_2), dim=2)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = self.softmax(x)
        return x