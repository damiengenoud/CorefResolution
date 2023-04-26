import pyodbc
from transformers import AutoTokenizer, AutoModel
import torch


def spanbert_encode(text: str):
    """
    Takes a text and returns its SpanBERT representation.

    :param text: the string to encode with SpanBERT
    :return: a tensor with the sub-tokens representations [n_sub-words, 768]
    """

    # config parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # TODO change this to large when accessing GPUs
    tokenizer = AutoTokenizer.from_pretrained('bert-large-cased')
    model = AutoModel.from_pretrained("SpanBERT/spanbert-large-cased").to(device)
    max_size = 512
    cls_token_id = 101
    sep_token_id = 102

    # tokenize the text without special tokens, we will add them ourselves.
    tokenized_text = tokenizer.encode_plus(text, add_special_tokens=False, return_tensors='pt')

    # split the text in lists of less than 512 sub-words tokens (required for most BERT-like transformers),
    # that overlap each other. Here 510 to add 2 special tokens.
    window_size = max_size - 2
    stride = int((max_size - 2) / 2)
    size = tokenized_text['input_ids'][0].size(0)

    print("Number of sub-words tokens in input:" + str(len(tokenized_text['input_ids'][0])))

    split_ids = [tokenized_text['input_ids'][0][i:i + window_size] for i in range(0, size, stride)]
    split_masks = [tokenized_text['attention_mask'][0][i:i + window_size] for i in range(0, size, stride)]

    # remove last chunk if multiples exist
    last_chunk_size = len(split_ids[len(split_ids) - 1])
    if len(split_ids) > 1:
        split_ids.pop()
        split_masks.pop()
        last_chunk_size = len(split_ids[len(split_ids) - 1])

    # prepare input for SpanBERT
    for i in range(len(split_ids)):
        # add special tokens CLS and SEP
        split_ids[i] = torch.cat([torch.tensor([cls_token_id]), split_ids[i], torch.tensor([sep_token_id])])

        # add masks for the 2 special tokens
        split_masks[i] = torch.cat([torch.tensor([1]), split_masks[i], torch.tensor([1])])

        # we need to pad if the part is not 512 tokens long (only for last part)
        padding = max_size - len(split_ids[i])
        if padding > 0:
            split_ids[i] = torch.cat([split_ids[i], torch.Tensor([0] * padding)])
            split_masks[i] = torch.cat([split_masks[i], torch.Tensor([0] * padding)])

    # create dictionary for SpanBERT input.
    ids = torch.stack(split_ids).to(device)
    masks = torch.stack(split_masks).to(device)
    data_dict = {'input_ids': ids.long(), 'attention_mask': masks.int()}

    # get SpanBERT encoding
    spanbert_output = model(**data_dict).last_hidden_state  # float32 tensor
    spanbert_output_size = spanbert_output.size(dim=0)

    # create new tensor from overlapping one
    non_overlapping_tensors = []

    if spanbert_output_size > 1:
        # add first tensor
        non_overlapping_tensors.append(torch.narrow(spanbert_output[0], 0, 1, stride + stride // 2))
        # add middle tensors
        for i in range(1, spanbert_output_size - 1):
            non_overlapping_tensors.append(torch.narrow(spanbert_output[i], 0, stride // 2 + 1, stride))
        # add last tensor
        non_overlapping_tensors.append(
            torch.narrow(spanbert_output[spanbert_output_size - 1], 0, stride // 2 + 1,
                         last_chunk_size - stride // 2))
    else:
        non_overlapping_tensors.append(torch.narrow(spanbert_output[0], 0, 1, last_chunk_size))

    output_tensor = torch.cat(non_overlapping_tensors, dim=0)

    print("Size of the encoded text tensor: " + str(output_tensor.size()))

    return output_tensor
