import torch
from torch import nn

import utils


# Simple MLP, decoder is used for pre-training
class MLP(nn.Module):
    def __init__(self, input_size, num_features, rep_dim, bias=True):
        super().__init__()
        self.input_size = input_size
        self.num_features = num_features
        self.rep_dim = rep_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = nn.Sequential(
            nn.Linear(input_size, num_features, bias=bias),
            nn.ReLU(),
            nn.Linear(num_features, num_features // 2, bias=bias),
            nn.ReLU(),
            nn.Linear(num_features // 2, rep_dim, bias=bias),
        ).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(rep_dim, num_features // 2, bias=bias),
            nn.ReLU(),
            nn.Linear(num_features // 2, num_features, bias=bias),
            nn.ReLU(),
            nn.Linear(num_features, input_size, bias=bias)
        ).to(self.device)

        # utils.init_weights(self.encoder, init_type='normal')
        # utils.init_weights(self.decoder, init_type='normal')

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))
