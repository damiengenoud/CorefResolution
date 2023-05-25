import torch
import ffnn


class AntecedentsScorer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ffnn = ffnn.FFNN(3 * 3 * 768)

    def forward(self, top_scoring_spans: torch.Tensor, top_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pairs_scores = []
        pairs_indices = []
        # iterate over all pairs of spans
        for target_index in range(1, top_scoring_spans.size(0)):
            for antecedent_index in range(0, target_index):
                # multiply the two tensors element-wise
                multiplied = torch.mul(top_scoring_spans[target_index], top_scoring_spans[antecedent_index])
                # get pair score
                pair_score = self.ffnn(torch.cat([
                    top_scoring_spans[target_index],
                    top_scoring_spans[antecedent_index],
                    multiplied
                ]))
                # add antecedent score to the list
                pairs_scores.append(torch.sum(torch.cat([
                    top_scores[target_index],
                    top_scores[antecedent_index],
                    pair_score
                ])))
                # add the indices of the pair for easier linking
                pairs_indices.append(torch.tensor([target_index, antecedent_index]))
        return torch.stack(pairs_scores), torch.stack(pairs_indices)
