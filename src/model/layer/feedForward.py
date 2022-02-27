import torch.nn as nn
import torch.nn.functional as F


class FeedforwardLayer(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.3):
        super(FeedforwardLayer,self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(self.dropout(F.relu(self.w_1(x))))
        x = residual + self.dropout2(x)
        return x