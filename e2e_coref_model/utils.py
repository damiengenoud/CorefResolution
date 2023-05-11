import re
import torch


def get_conll_data(doc_id: str, tokenizer):
    with open("litbank/coref/conll/" + doc_id, mode='r', encoding='utf-8') as file:
        # TODO check if sentence id is important (also with CLS and SEP tokens)
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
    return tokenized_text, word_ids_to_num_bert_tokens, doc_entities


def spanbert_encode(tokenized_text, encoder):
    # config parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    print("Number of sub-words tokens in input:" + str(len(tokenized_text['input_ids'])))

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
    ids = torch.stack(split_ids).to(device)
    masks = torch.stack(split_masks).to(device)
    data_dict = {'input_ids': ids.long(), 'attention_mask': masks.int()}

    # get SpanBERT encoding
    spanbert_output = encoder(**data_dict).last_hidden_state  # float32 tensor
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


def prepare_training_data(tokenizer, encoder):
    ids_folder_path = 'data'
    splits = ['train', 'dev', 'test']
    for split in splits:
        with open(ids_folder_path + '/' + split + '.txt', mode='r', encoding='utf-8') as file:
            for line in file:
                doc_id = line.rstrip()
                print('Processing: ' + doc_id)
                tokenized_text, word_ids_to_num_bert_tokens, doc_entities = get_conll_data(doc_id, tokenizer)
                spanbert_encode(tokenized_text, encoder)
                print('Processed: ' + doc_id)



