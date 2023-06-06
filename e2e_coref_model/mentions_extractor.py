import torch
import span_attention_embedder
import ffnn
import utils
import config
import bert
from transformers import AutoModel


class MentionsExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.SPANBERT_MODEL)
        self.span_representation_model = span_attention_embedder.SpanAttentionEmbedder(config.BERT_EMB_SIZE)
        self.span_score_model = ffnn.FFNN(3 * config.BERT_EMB_SIZE)

    def forward(self, tokenized_text, word_ids_to_num_bert_tokens):
        # Encode the tokenized text
        token_level_encoding = bert.spanbert_encode(tokenized_text, self.encoder)
        encoded_text = bert.average_spanbert_subwords_tokens(token_level_encoding, word_ids_to_num_bert_tokens)

        num_tokens = encoded_text.size(0)

        # Get all possible start and end indices of the spans up to length max_span_width
        candidates_start_indices = torch.tile(torch.arange(num_tokens).unsqueeze(1).to(utils.device), [1, config.MAX_SPAN_LENGTH]) # [num_tokens, max_span_width]
        candidates_end_indices = candidates_start_indices + torch.arange(config.MAX_SPAN_LENGTH).unsqueeze(0).to(utils.device)  #[num_tokens, max_span_width]

        # Create a mask for all spans whose end is out of bounds
        candidates_mask = torch.reshape(torch.where(candidates_end_indices < num_tokens, True, False), [-1])  # [num_tokens * max_span_width]

        # Get the actual start and end indices of these spans
        candidates_start_indices = torch.masked_select(torch.reshape(candidates_start_indices, [-1]), candidates_mask)  # [num_candidates]
        candidates_end_indices = torch.masked_select(torch.reshape(candidates_end_indices, [-1]), candidates_mask)  # [num_candidates]

        # Get the representation of these spans
        candidates_spans_representation = self.span_representation_model(
            encoded_text, candidates_start_indices, candidates_end_indices, config.MAX_SPAN_LENGTH)  # [num_candidates, span_emb = 3 * bert_emb]

        # Get the mention score of these spans
        candidates_mentions_scores = self.span_score_model(candidates_spans_representation)  # [num_candidates, 1]

        # Keep only the k top scoring spans and relative information
        k = torch.floor(torch.tensor([config.TOP_SPAN_RATIO * num_tokens])).int().item()
        top_spans_scores, top_spans_indices = torch.topk(candidates_mentions_scores, k, 0, True, False)  # [k, 1], [k, 1]
        top_spans_scores = torch.squeeze(top_spans_scores)  # [k]
        top_spans_indices = torch.squeeze(top_spans_indices)  # [k]
        top_spans_start_indices = candidates_start_indices[top_spans_indices]  # [k]
        top_spans_end_indices = candidates_end_indices[top_spans_indices]  # [k]
        top_spans_embeddings = candidates_spans_representation[top_spans_indices]  # [k, span_emb]

        return top_spans_start_indices, top_spans_end_indices, top_spans_scores, top_spans_embeddings, candidates_start_indices, candidates_end_indices, candidates_mentions_scores.squeeze()
