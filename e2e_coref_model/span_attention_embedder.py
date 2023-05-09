import torch
import ffnn


class SpanAttentionEmbedder(torch.nn.Module):

    def __init__(self, bert_embedding_size: int):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ffnn = ffnn.FFNN(bert_embedding_size)

    def forward(self, spanbert_out: torch.Tensor, spans_start: torch.Tensor, spans_end: torch.Tensor,
                max_span_width: int) -> torch.Tensor:
        # spanbert_out: [num_tokens]
        # spans_start: [num_candidates]
        # spans_end: [num_candidates]

        num_tokens = spanbert_out.size(0)

        # These are alpha in paper
        tokens_attention_score = self.ffnn(spanbert_out)  # [num_tokens, 1]

        spans_embedding_list = []

        # Get the representations of the beginning and ending tokens of each span (Xstart and Xend in paper)
        spans_start_emb = spanbert_out[spans_start]  # [num_candidates, bert_emb]
        spans_end_emb = spanbert_out[spans_end]  # [num_candidates, bert_emb]

        spans_embedding_list.append(spans_start_emb)
        spans_embedding_list.append(spans_end_emb)

        # Get the indices of each possible span (even if the indices are larger than the number of tokens)
        spans_indices = torch.arange(max_span_width).unsqueeze(0).to(self.device) + spans_start.unsqueeze(
            1)  # [num_candidates, max_span_width]
        spans_indices = torch.where(spans_indices < num_tokens, spans_indices,
                                    num_tokens - 1)  # [num_candidates, max_span_width]

        spans_tokens_representation = spanbert_out[spans_indices]  # [num_candidates, max_span_width, bert_emb]

        # Get the attention scores of each token in the spans (using masking to remove unwanted tokens)
        spans_width = spans_end - spans_start + 1  # [num_candidates]
        spans_tokens_attention = tokens_attention_score[spans_indices]  # [num_candidates, max_span_width, 1]
        spans_mask = torch.lt(torch.arange(max_span_width).unsqueeze(0).to(self.device),
                              spans_width.unsqueeze(1)).unsqueeze(2).float()  # [num_candidates, max_span_width, 1]
        spans_tokens_attention += torch.log(spans_mask)  # [num_candidates, max_span_width, 1]

        # These are beta in paper
        spans_softmax = torch.softmax(spans_tokens_attention, 1)  # [num_candidates, max_span_width, 1]
        # These are x-hat in paper
        spans_representation = torch.sum(spans_softmax * spans_tokens_representation, 1)  # [num_candidates, bert_emb]

        spans_embedding_list.append(spans_representation)
        # These are the g in the paper
        spans_embedding = torch.cat(spans_embedding_list, 1)  # [num_candidates, span_emb]

        return spans_embedding  # [num_candidates, span_emb]
