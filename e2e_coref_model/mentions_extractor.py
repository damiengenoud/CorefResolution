"""
Describes the model that evaluates if the spans are mentions and retains only the most relevant ones.
"""

import torch
from torch import Tensor

import ffnn
import math


class MentionsExtractor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ffnn = ffnn.FFNN(1024 * 3)

    def forward(self, attention_scorer_out: torch.Tensor) -> tuple[Tensor, Tensor]:
        n_high_score = math.ceil(attention_scorer_out.size(dim=0) * 0.25)

        # compute mention scores of each span
        score = self.ffnn(attention_scorer_out)

        # keep only high scoring spans
        _, top_score_indices = torch.topk(score, n_high_score, dim=0, largest=True, sorted=False)
        high_score_spans = attention_scorer_out[top_score_indices.squeeze()]
        return high_score_spans, top_score_indices
