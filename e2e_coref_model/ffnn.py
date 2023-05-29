import torch
import config


class FFNN(torch.nn.Module):

    def __init__(self, in_features: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, config.HIDDEN_LAYERS_SIZE)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(config.HIDDEN_LAYERS_SIZE, config.HIDDEN_LAYERS_SIZE)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(config.HIDDEN_LAYERS_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return x
