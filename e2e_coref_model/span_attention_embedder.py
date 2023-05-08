"""
Describes the model that creates the final span representation, based on attention mechanism.
"""

import torch
from torch import Tensor

import ffnn


class SpanAttentionEmbedder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ffnn = ffnn.FFNN(768)

    def forward(self, spanbert_out: torch.Tensor) -> tuple[Tensor, Tensor]:
        max_span_length = 15
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # compute the tokens attention representation
        tokens_ffnn_attn = torch.flatten(self.ffnn(spanbert_out))  # alpha

        spans_representations = []
        spans_representations_boundaries = []

        # TODO make this more efficient... but how ?
        for start_index in range(0, tokens_ffnn_attn.size(dim=0)):
            for end_index in range(start_index, min(start_index + max_span_length, tokens_ffnn_attn.size(dim=0))):
                # get indices of span
                indices = torch.arange(start_index, end_index + 1).to(device)
                # compute the softmax over each token
                tokens_softmax = torch.softmax(tokens_ffnn_attn[indices], 0)
                # get the tokens representation of the span
                subset = spanbert_out[indices]
                # compute the weighted sum
                weighted_sum = torch.sum(subset * tokens_softmax[:, None], 0, False)
                # create the final representation of the span
                span_rep = torch.cat([spanbert_out[start_index], spanbert_out[end_index], weighted_sum])
                # add this representation to the list of representations
                spans_representations.append(span_rep)
                # add the boundary indices to the list of span boundaries
                spans_representations_boundaries.append(torch.tensor([start_index, end_index]))

        return torch.stack(spans_representations), torch.stack(spans_representations_boundaries)
