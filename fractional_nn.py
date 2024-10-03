# fractional_nn.py

import torch
import torch.nn as nn

class FractionalLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.5):
        super(FractionalLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha  # Order of the fractional derivative
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        out = torch.matmul(input, self.weight.t()) + self.bias
        # Apply fractional activation (e.g., power)
        out = torch.pow(out.abs(), self.alpha) * torch.sign(out)
        return out

class FractionalNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, alpha=0.5):
        super(FractionalNeuralNetwork, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.append(FractionalLayer(dims[i], dims[i+1], alpha=alpha))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
