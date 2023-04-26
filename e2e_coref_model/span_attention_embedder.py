"""
Describes the model that creates the final span representation, based on attention mechanism.
"""
from typing import Tuple

import torch
from torch import Tensor

import ffnn


class SpanAttentionEmbedder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ffnn = ffnn.FFNN(1024)

    def forward(self, spanbert_out: torch.Tensor) -> tuple[Tensor, Tensor]:
        max_span_length = 15
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        alpha_t = torch.flatten(self.ffnn(spanbert_out))

        spans_representations = []
        spans_representations_boundaries = []

        # TODO make this more efficient... but how ?
        for start_index in range(0, alpha_t.size(dim=0)):
            for end_index in range(start_index, min(start_index + max_span_length, alpha_t.size(dim=0))):
                indices = torch.arange(start_index, end_index + 1).to(device)
                tokens_softmax = torch.softmax(alpha_t[indices], 0)
                subset = spanbert_out[indices]
                weighted_sum = torch.sum(subset * tokens_softmax[:, None], 0, False)
                span_rep = torch.cat([spanbert_out[start_index], spanbert_out[end_index], weighted_sum])
                spans_representations.append(span_rep)
                spans_representations_boundaries.append(torch.tensor([start_index, end_index]))

        return torch.stack(spans_representations), torch.stack(spans_representations_boundaries)
