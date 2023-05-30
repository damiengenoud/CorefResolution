import torch
import span_attention_embedder
import ffnn
import utils


class MentionsExtractor(torch.nn.Module):
    def __init__(self, bert_embedding_size: int, top_spans_ratio: float, max_span_width: int):
        super().__init__()
        self.bert_embedding_size = bert_embedding_size
        self.top_spans_ratio = top_spans_ratio
        self.max_span_width = max_span_width
        self.span_representation_model = span_attention_embedder.SpanAttentionEmbedder(bert_embedding_size)
        self.span_score_model = ffnn.FFNN(3 * bert_embedding_size)

    def forward(self, encoded_text: torch.Tensor):
        num_tokens = encoded_text.size(0)

        # Get all possible start and end indices of the spans up to length max_span_width
        candidates_start_indices = torch.tile(torch.arange(num_tokens).unsqueeze(1).to(self.device), [1, self.max_span_width]) # [num_tokens, max_span_width]
        candidates_end_indices = candidates_start_indices + torch.arange(self.max_span_width).unsqueeze(0).to(utils.device)  #[num_tokens, max_span_width]

        # Create a mask for all spans whose end is out of bounds
        candidates_mask = torch.reshape(torch.where(candidates_end_indices < num_tokens, True, False), [-1])  # [num_tokens * max_span_width]

        # Get the actual start and end indices of these spans
        candidates_start_indices = torch.masked_select(torch.reshape(candidates_start_indices, [-1]), candidates_mask)  # [num_candidates]
        candidates_end_indices = torch.masked_select(torch.reshape(candidates_end_indices, [-1]), candidates_mask)  # [num_candidates]

        # Get the representation of these spans
        candidates_spans_representation = self.span_representation_model(
            encoded_text, candidates_start_indices, candidates_end_indices, self.max_span_width)  # [num_candidates, span_emb = 3 * bert_emb]

        # Get the mention score of these spans
        candidates_mentions_scores = self.span_score_model(candidates_spans_representation)  # [num_candidates, 1]

        # Keep only the k top scoring spans and relative information
        k = torch.floor(torch.tensor([self.top_spans_ratio * num_tokens])).int().item()
        top_spans_scores, top_spans_indices = torch.topk(candidates_mentions_scores, k, 0, True, False)  # [k, 1], [k, 1]
        top_spans_scores = torch.squeeze(top_spans_scores)  # [k]
        top_spans_indices = torch.squeeze(top_spans_indices)  # [k]
        top_spans_start_indices = candidates_start_indices[top_spans_indices]  # [k]
        top_spans_end_indices = candidates_end_indices[top_spans_indices]  # [k]
        top_spans_embeddings = candidates_spans_representation[top_spans_indices]  # [k, span_emb]

        return top_spans_start_indices, top_spans_end_indices, top_spans_scores, top_spans_embeddings
