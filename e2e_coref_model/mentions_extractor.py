import torch
import span_attention_embedder
import top_spans_extractor


class MentionsExtractor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.span_representation_model = span_attention_embedder.SpanAttentionEmbedder()
        self.mentions_extractor_model = top_spans_extractor.TopSpansExtractor()

    def forward(self, encoded_text: torch.Tensor):
        span_representation, span_representation_boundaries = self.span_representation_model(encoded_text)
        top_scoring_spans, scores, top_scoring_spans_indices = self.mention_extractor_model(span_representation, encoded_text.size(0) // 4)
        return scores, top_scoring_spans_indices, span_representation_boundaries
