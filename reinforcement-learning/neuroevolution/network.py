import torch
import torch.nn as nn
from torch.distributions import normal


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = 2
        self.outputs = 4
        self.model = nn.Sequential(
            nn.Linear(self.inputs, 4),
            nn.LeakyReLU(),
            nn.Linear(4, self.outputs)
        )
        self.init_weights(self.model)
        self.mutation_operator = normal.Normal(0.0, 1.0)

    def forward(self, x):
        output = self.model(x)
        return output

    def init_weights(self, net, init_type='normal', gain=0.02):
        def init_func(m):
            with torch.no_grad():
                classname = m.__class__.__name__
                if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                    if init_type == 'normal':
                        torch.nn.init.normal_(m.weight.data, mean=0.0, std=gain)
                    elif init_type == 'xavier':
                        torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                    elif init_type == 'kaiming':
                        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                    elif init_type == 'orthogonal':
                        torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                    else:
                        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                    if hasattr(m, 'bias') and m.bias is not None:
                        torch.nn.init.constant_(m.bias.data, 0.0)
                elif classname.find('BatchNorm2d') != -1:
                    torch.nn.init.normal_(m.weight.data, 1.0, gain)
                    torch.nn.init.constant_(m.bias.data, 0.0)

        net.apply(init_func)

    def disable_grad(self):
        for param in self.parameters():
            param.requires_grad = False
