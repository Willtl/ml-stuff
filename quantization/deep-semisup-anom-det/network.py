import torch
import torch.jit as jit
from torch import nn

import utils


# Simple MLP, decoder is used for pre-training
class MLP(nn.Module):
    def __init__(self, input_size, num_features, rep_dim, bias=True):
        super().__init__()
        self.input_size = input_size
        self.num_features = num_features
        self.rep_dim = rep_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_size, num_features, bias=bias),
            nn.ReLU(),
            nn.Linear(num_features, num_features // 2, bias=bias),
            nn.ReLU(),
            nn.Linear(num_features // 2, rep_dim, bias=bias),
        ).to(utils.device)

        self.decoder = nn.Sequential(
            nn.Linear(rep_dim, num_features // 2, bias=bias),
            nn.ReLU(),
            nn.Linear(num_features // 2, num_features, bias=bias),
            nn.ReLU(),
            nn.Linear(num_features, input_size, bias=bias)
        ).to(utils.device)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # utils.init_weights(self.encoder, init_type='normal')
        # utils.init_weights(self.decoder, init_type='normal')

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)
        return x


# Simple MLP, decoder is used for pre-training
class JITMLP(jit.ScriptModule):
    def __init__(self, input_size, num_features, rep_dim, bias=True):
        super().__init__()
        self.input_size = input_size
        self.num_features = num_features
        self.rep_dim = rep_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_size, num_features, bias=bias),
            nn.ReLU(),
            nn.Linear(num_features, num_features // 2, bias=bias),
            nn.ReLU(),
            nn.Linear(num_features // 2, rep_dim, bias=bias),
        ).to(utils.device)

        self.decoder = nn.Sequential(
            nn.Linear(rep_dim, num_features // 2, bias=bias),
            nn.ReLU(),
            nn.Linear(num_features // 2, num_features, bias=bias),
            nn.ReLU(),
            nn.Linear(num_features, input_size, bias=bias)
        ).to(utils.device)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # utils.init_weights(self.encoder, init_type='normal')
        # utils.init_weights(self.decoder, init_type='normal')

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    @jit.script_method
    def forward(self, x):
        x = self.encoder(x)
        return x
