import torch
import ffnn
import utils


class MentionsLinker(torch.nn.Module):
    def __init__(self, mentions_extractor, bert_embedding_size: int):
        super().__init__()
        self.bert_embedding_size = bert_embedding_size
        self.mentions_extractor = mentions_extractor
        self.pair_antecedent_score_model = ffnn.FFNN(3 * 3 * bert_embedding_size)

    def forward(self, encoded_text):
        # Get mentions from mentions extractor model:
        spans_start_indices, spans_end_indices, spans_mention_score, spans_embeddings = self.mentions_extractor(
            encoded_text.to(utils.device))

        num_spans = spans_mention_score.size(0)

        # Create a matrix containing the antecedents' indices of the top scoring spans
        antecedents_distance = torch.tile(torch.arange(0, num_spans) + 1, [num_spans, 1])  # [num_spans ,num_spans]
        raw_antecedents = torch.unsqueeze(torch.arange(num_spans), 1) - antecedents_distance  # [num_spans, num_spans]

        # Create a mask for out of bound indices (it's a lower triangular matrix)
        antecedents_mask = (raw_antecedents >= 0).to(utils.device)  # [num_spans, num_spans]

        # Keep the indices in bound (>= 0), even the useless one, for convenience in future operations
        antecedents = torch.maximum(raw_antecedents, torch.tensor(0)).to(utils.device)  # [num_spans, num_spans]

        # Antecedents is a lower triangular matrix where the line i has numbers i - 1 to 0,
        # followed by (num_spans - i + 1) zeros.
        # It corresponds to the antecedents' indices of span i.
        # Note that only the elements of the lower triangle without the diagonal are relevant, the rest will be
        # "deleted" by the mask.

        # Create a matrix where each element is the sum of the mention scores of the pair
        pairs_mention_score = spans_mention_score.unsqueeze(1) + spans_mention_score[
            antecedents]  # [num_spans, num_spans]
        # Use mask_spans to put out of bound values to -inf
        pairs_mention_score += torch.log(antecedents_mask.float())  # [num_spans, num_spans]

        # Get the antecedents embedding in a matrix
        antecedents_emb = spans_embeddings[antecedents]  # [num_spans, num_spans, emb]

        # Concatenate the targets embeddings, the antecedents embeddings and the multiplication of both
        targets_embeddings = spans_embeddings.unsqueeze(1).tile([1, num_spans, 1])  # [num_spans, 1, emb]
        mult_embeddings = targets_embeddings * spans_embeddings  # [num_spans, num_spans, emb]
        pairs_embeddings = torch.cat([targets_embeddings, antecedents_emb, mult_embeddings],
                                     2)  # [num_spans, num_spans, 3*emb]

        # Get the antecedent score and total pairs score
        pairs_antecedent_score = self.pair_antecedent_score_model(pairs_embeddings).squeeze(2)  # [num_spans, num_spans]
        dummy_score = torch.zeros([num_spans, 1]).to(utils.device)
        pairs_score = torch.cat([pairs_mention_score + pairs_antecedent_score, dummy_score], 1)  # [num_spans, num_spans + 1]

        return spans_start_indices, spans_end_indices, pairs_score, antecedents, antecedents_mask
