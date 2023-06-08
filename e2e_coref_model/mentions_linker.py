import torch
import ffnn
import utils
import config
import mentions_extractor


class MentionsLinker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mentions_extractor = mentions_extractor.MentionsExtractor()
        self.pair_antecedent_score_model = ffnn.FFNN(3 * 3 * config.BERT_EMB_SIZE)

    def forward(self, encoded_text):
        # Get mentions from mentions extractor model:
        spans_start_indices, spans_end_indices, spans_mention_score, spans_embeddings = self.mentions_extractor(
            encoded_text)

        num_spans = spans_mention_score.size(0)
        num_antecedents = min(config.MAX_NUMBER_OF_ANTECEDENTS, num_spans)

        # Create a matrix containing the antecedents' indices of the top scoring spans
        antecedents_distance = torch.tile(torch.arange(0, num_antecedents) + 1, [num_spans, 1])  # [k, num_antecedents]
        raw_antecedents = torch.unsqueeze(torch.arange(num_spans), 1) - antecedents_distance  # [k, num_antecedents]

        # Create a mask for out of bound indices (it's a lower triangular matrix)
        antecedents_mask = (raw_antecedents >= 0).to(utils.device)  # [k, num_antecedents]

        # Keep the indices in bound (>= 0), even the useless one, for convenience in future operations
        antecedents = torch.maximum(raw_antecedents, torch.tensor(0)).to(utils.device)  # [k, num_antecedents]

        # Antecedents is a lower triangular matrix where the line i has numbers i - 1 to 0,
        # followed by (num_spans - i + 1) zeros.
        # It corresponds to the antecedents' indices of span i.
        # Note that only the elements of the lower triangle without the diagonal are relevant, the rest will be
        # "deleted" by the mask.

        # Create a matrix where each element is the sum of the mention scores of the pair
        pairs_mention_score = spans_mention_score.unsqueeze(1) + spans_mention_score[
            antecedents]  # [k, num_antecedents]
        # Use mask_spans to put out of bound values to -inf
        pairs_mention_score += torch.log(antecedents_mask.float())  # [k, num_antecedents]

        # Get the antecedents embedding in a matrix
        antecedents_emb = spans_embeddings[antecedents]  # [k, num_antecedents, emb]

        # Concatenate the targets embeddings, the antecedents embeddings and the multiplication of both
        targets_embeddings = spans_embeddings.unsqueeze(1).tile([1, num_antecedents, 1])  # [k, 1, emb]
        mult_embeddings = targets_embeddings * antecedents_emb  # [k, num_antecedents, emb]
        pairs_embeddings = torch.cat([targets_embeddings, antecedents_emb, mult_embeddings],
                                     2)  # [k, num_antecedents, 3*emb]

        # Get the antecedent score and total pairs score
        pairs_antecedent_score = self.pair_antecedent_score_model(pairs_embeddings).squeeze(2)  # [k, num_antecedents]
        dummy_score = torch.zeros([num_spans, 1]).to(utils.device)
        pairs_score = torch.cat([pairs_mention_score + pairs_antecedent_score, dummy_score], 1)  # [k, num_antecedents + 1]

        return spans_start_indices, spans_end_indices, pairs_score, antecedents, antecedents_mask

    def load_extractor(self, model_path: str):
        self.mentions_extractor.load_state_dict(torch.load(model_path + config.MENTIONS_EXTRACTOR_MODEL_NAME))

