import torch


class FFNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1024, 150)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(150, 150)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(150, 1)

    def forward(self, spanbert_out: torch.Tensor) -> torch.Tensor:
        spanbert_out = self.linear1(spanbert_out)
        spanbert_out = self.activation1(spanbert_out)
        spanbert_out = self.linear2(spanbert_out)
        spanbert_out = self.activation2(spanbert_out)
        spanbert_out = self.linear3(spanbert_out)
        return spanbert_out
