import re
import torch


def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_data(tokenizer, encoder, split):
    ids_folder_path = 'data'
    data_list = []
    print("Preparing " + split + " documents...")
    with open(ids_folder_path + '/' + split + '.txt', mode='r', encoding='utf-8') as file:
        for line in file:
            doc_id = line.rstrip()
            tokenized_text, word_ids_to_num_bert_tokens, doc_entities = get_conll_data(doc_id, tokenizer)
            encoding = spanbert_encode(tokenized_text, encoder)
            averaged_encoding = average_spanbert_subwords_tokens(encoding, word_ids_to_num_bert_tokens)
            data_list.append({'doc_id': doc_id,
                              'encoding': averaged_encoding,
                              'doc_entities': doc_entities,
                              })
    return data_list


def get_conll_data(doc_id: str, tokenizer):
    with open("litbank/coref/conll/" + doc_id, mode='r', encoding='utf-8') as file:
        word_id = 0
        input_ids = torch.Tensor()
        word_ids_to_num_bert_tokens = []
        doc_entities = []
        opened_entities = {}

        for line in file:
            # Start or end of doc, or blank line
            if line.startswith('#begin document') or line.startswith('#end document') or line == '\n':
                continue
            # "Word" line
            else:
                parts = re.split("\s+", line.rstrip())
                word = parts[3]

                # Create SpanBERT input dictionary
                tokens = tokenizer.encode(word, add_special_tokens=False, return_tensors='pt')
                input_ids = torch.cat([input_ids, tokens[0]])

                # Convert to BERT ids to know the number of sub-words of this "word"
                num_bert_tokens = len(tokens)
                word_ids_to_num_bert_tokens.append(num_bert_tokens)

                # Get coref information
                coref_indications = parts[-1].split('|')
                for coref_indication in coref_indications:
                    # Entire entity
                    if coref_indication.startswith('(') and coref_indication.endswith(')'):
                        cluster_id = int(coref_indication[1])
                        doc_entities.append([word_id, word_id, cluster_id])
                    # Start of entity
                    elif coref_indication.startswith('('):
                        cluster_id = int(coref_indication[1])
                        # Remember this entity as opened
                        if cluster_id not in opened_entities:
                            opened_entities[cluster_id] = []
                        opened_entities[cluster_id].append(word_id)
                    # End of entity
                    elif coref_indication.endswith(')'):
                        cluster_id = int(coref_indication[0])
                        # Close and add entity to entities list
                        doc_entities.append([opened_entities[cluster_id].pop(), word_id, cluster_id])
                word_id += 1

    tokenized_text = {'input_ids': input_ids,
                      'token_type_ids': torch.tensor([0] * input_ids.size(0)),
                      'attention_mask': torch.tensor([1] * input_ids.size(0))}
    return tokenized_text, torch.tensor(word_ids_to_num_bert_tokens), torch.tensor(doc_entities)


def spanbert_encode(tokenized_text, encoder):
    with torch.no_grad():
        # config parameters
        max_size = 512
        cls_token_id = 101
        sep_token_id = 102

        # tokenize the text without special tokens, we will add them ourselves.
        # tokenized_text = tokenizer.encode_plus(text, add_special_tokens=False, return_tensors='pt')

        # split the text in lists of less than 512 sub-words tokens (required for most BERT-like transformers),
        # that overlap each other. Here 510 to add 2 special tokens.
        window_size = max_size - 2
        stride = int((max_size - 2) / 2)
        size = tokenized_text['input_ids'].size(0)

        # print("Number of sub-words tokens in input:" + str(len(tokenized_text['input_ids'])))

        split_ids = [tokenized_text['input_ids'][i:i + window_size] for i in range(0, size, stride)]
        split_masks = [tokenized_text['attention_mask'][i:i + window_size] for i in range(0, size, stride)]

        # remove last chunk if multiple chunks exist
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
        ids = torch.stack(split_ids).to(device())
        masks = torch.stack(split_masks).to(device())
        data_dict = {'input_ids': ids.long(), 'attention_mask': masks.int()}

        # get SpanBERT encoding
        spanbert_output = encoder(**data_dict).last_hidden_state  # float32 tensor
        spanbert_output_size = spanbert_output.size(dim=0)

        # create new tensor from overlapping ones
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

        # print("Size of the encoded text tensor: " + str(output_tensor.size()))

    return output_tensor


def average_spanbert_subwords_tokens(encoded_text, word_ids_to_num_bert_tokens):
    bert_index = 0
    averaged_word_embeddings = torch.Tensor().to(device())

    for word_index, num_bert_tokens in enumerate(word_ids_to_num_bert_tokens):
        word_embeddings = torch.Tensor().to(device())

        for i in range(num_bert_tokens):
            word_embeddings = torch.cat([word_embeddings, encoded_text[bert_index].unsqueeze(-1)])
            bert_index += 1

        averaged_word_embedding = torch.mean(word_embeddings, dim=1)
        averaged_word_embeddings = torch.cat([averaged_word_embeddings, averaged_word_embedding.unsqueeze(0)])

    return averaged_word_embeddings


def get_mentions_labels(
        true_entities: torch.Tensor,  # [num_true_mentions, 3]
        predicted_start_indices: torch.Tensor,  # [num_predicted_spans]
        predicted_end_indices: torch.Tensor  # [num_predicted_spans]
) -> torch.Tensor:
    with torch.no_grad():
        predicted_indices = torch.stack([predicted_start_indices, predicted_end_indices], dim=1)
        true_indices = true_entities[:, :2]

        # Expands the tensors to be able to use a sort of "contains" function, see :
        # https://stackoverflow.com/questions/73524338/pytorch-test-each-row-of-the-first-2d-tensor-also-exist-in-the-second-tensor
        predicted_expand = predicted_indices.unsqueeze(1).expand(-1, true_indices.shape[0], predicted_indices.shape[1])
        true_expand = true_indices.unsqueeze(0).expand(predicted_indices.shape[0], -1, predicted_indices.shape[1]).to(
            device()
        )

        # Similar to a "contains" function
        labels = (predicted_expand == true_expand).all(-1).any(-1).float()
        return labels

