"""
Describes the model that evaluates if the spans are mentions and retains only the most relevant ones.
"""

import torch
from torch import Tensor

import ffnn


class MentionsExtractor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ffnn = ffnn.FFNN(1024 * 3)

    def forward(self, attention_scorer_out: torch.Tensor, n_retained_spans: int) -> tuple[Tensor, Tensor, Tensor]:
        # compute mention scores of each span
        score = self.ffnn(attention_scorer_out)

        # keep only high scoring spans
        top_scores, top_scores_indices = torch.topk(score, n_retained_spans, dim=0, largest=True, sorted=False)
        high_score_spans = attention_scorer_out[top_scores_indices.squeeze()]
        return high_score_spans, top_scores, top_scores_indices
