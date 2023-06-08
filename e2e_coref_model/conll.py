import torch
import re
import os
from collections import defaultdict
import config


def get_conll_data(doc_id: str, tokenizer):
    with open(config.CONLL_LITBANK_FOLDER + doc_id, mode='r', encoding='utf-8') as file:
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
                        cluster_id = int(coref_indication[1]) + 1
                        doc_entities.append([word_id, word_id, cluster_id])
                    # Start of entity
                    elif coref_indication.startswith('('):
                        cluster_id = int(coref_indication[1]) + 1
                        # Remember this entity as opened
                        if cluster_id not in opened_entities:
                            opened_entities[cluster_id] = []
                        opened_entities[cluster_id].append(word_id)
                    # End of entity
                    elif coref_indication.endswith(')'):
                        cluster_id = int(coref_indication[0]) + 1
                        # Close and add entity to entities list
                        doc_entities.append([opened_entities[cluster_id].pop(), word_id, cluster_id])
                word_id += 1
                
    tokenized_text = {'input_ids': input_ids,
                      'token_type_ids': torch.tensor([0] * input_ids.size(0)),
                      'attention_mask': torch.tensor([1] * input_ids.size(0))}
    
    return tokenized_text, torch.tensor(word_ids_to_num_bert_tokens), torch.tensor(doc_entities)


def write_conll(doc_id: str, spans_starts, spans_ends, spans_clusters):
    single_word = defaultdict(lambda: [])
    starts = defaultdict(lambda: [])
    ends = defaultdict(lambda: [])

    for cluster_num, cluster in enumerate(spans_clusters):
        for span_num in cluster:
            start = spans_starts[span_num]
            end = spans_ends[span_num]
            if start == end:
                single_word[start].append(cluster_num)
            else:
                starts[start].append(cluster_num)
                ends[end].append(cluster_num)

    for start, end, cluster_num in zip(spans_starts, spans_ends, spans_clusters):
        if start == end:
            single_word[start].append(cluster_num)
        else:
            starts[start].append(cluster_num)
            ends[end].append(cluster_num)

    print(doc_id)
    os.makedirs(config.CONLL_EVAL_OUTPUT_FOLDER, exist_ok=True)
    with open(config.CONLL_LITBANK_FOLDER + doc_id, mode='r', encoding='utf-8') as litbank_file, \
            open(config.CONLL_EVAL_OUTPUT_FOLDER + doc_id, mode='w', encoding='utf-8') as output_file:
        word_id = 0
        for line in litbank_file:
            # Start or end of doc, or blank line
            if line.startswith('#begin document') or line.startswith('#end document') or line == '\n':
                output_file.write(line)
                continue
            # "Word" line
            else:
                # Get all coref cluster information for the line
                coref_indications = []
                for cluster_num in starts[word_id]:
                    coref_indications.append(f"({cluster_num}")
                for cluster_num in single_word[word_id]:
                    coref_indications.append(f"({cluster_num})")
                for cluster_num in ends[word_id]:
                    coref_indications.append(f"{cluster_num})")
                coref_indication = "|".join(coref_indications) if len(coref_indications) != 0 else "_"

                # Write the new line in the output file
                parts = re.split("\s+", line.rstrip())
                parts[-1] = coref_indication
                output_line = "\t".join(parts)
                output_file.write(output_line + '\n')

                word_id += 1
