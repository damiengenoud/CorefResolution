"""
Basic FFNN, the same as the ones used in e2e-coref, Lee et al. (2017).
"""

import torch


class FFNN(torch.nn.Module):

    def __init__(self, in_features: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, 150)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(150, 150)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(150, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return x
