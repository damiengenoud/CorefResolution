import torch
import span_attention_embedder
import top_spans_extractor


class MentionsExtractor(torch.nn.Module):

    def __init__(self, bert_embedding_size: int):
        super().__init__()
        self.span_representation_model = span_attention_embedder.SpanAttentionEmbedder(bert_embedding_size)
        self.mentions_extractor_model = top_spans_extractor.TopSpansExtractor()

    def forward(self, encoded_text: torch.Tensor, max_span_width: int):
        num_tokens = encoded_text.size(0)

        candidates_start_indices = torch.tile(torch.arange(num_tokens).unsqueeze(1), [1, max_span_width])  # [num_tokens, max_span_width]
        candidates_end_indices = candidates_start_indices + torch.arange(max_span_width).unsqueeze(0)  #[num_tokens, max_span_width]

        candidates_mask = torch.reshape(torch.where(candidates_end_indices < num_tokens, True, False), [-1])  # [num_tokens * max_span_width]

        candidates_start_indices = torch.masked_select(torch.reshape(candidates_start_indices, [-1]), candidates_mask)  # [num_candidates]
        candidates_end_indices = torch.masked_select(torch.reshape(candidates_end_indices, [-1]), candidates_mask)  # [num_candidates]

        candidates_spans_representation = self.span_representation_model(encoded_text, candidates_start_indices, candidates_end_indices, max_span_width)  # [num_candidates, span_emb = 3*bert_emb]

        # top_scoring_spans, scores, top_scoring_spans_indices = self.mention_extractor_model(candidates_spans_representation, encoded_text.size(0) // 4)
        return candidates_spans_representation, candidates_start_indices, candidates_end_indices
