import torch
import networkx as nx
import custom_dataset
import bert
import conll

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_dataset(tokenizer, encoder, split):
    with torch.no_grad():
        ids_folder_path = 'data'
        data_list = []
        print("Preparing " + split + " documents...")
        with open(ids_folder_path + '/' + split + '.txt', mode='r', encoding='utf-8') as file:
            for line in file:
                doc_id = line.rstrip()
                tokenized_text, word_ids_to_num_bert_tokens, doc_entities = conll.get_conll_data(doc_id, tokenizer)
                encoding = bert.spanbert_encode(tokenized_text, encoder)
                averaged_encoding = bert.average_spanbert_subwords_tokens(encoding, word_ids_to_num_bert_tokens)
                data_list.append({'doc_id': doc_id,
                                  'encoding': averaged_encoding,
                                  'doc_entities': doc_entities,
                                  })
        return custom_dataset.CustomDataset(data_list)


def clusterize_antecedents(
        pairs_score,  # [k, k + 1]
):
    k = pairs_score.size(0)
    best_antecedent = torch.argmax(pairs_score, 1)  # [k]
    best_antecedent_pair = torch.stack([torch.tensor(range(k)), best_antecedent], 1).numpy()  # [k, 2]
    # Remove singleton mentions and mentions linked to dummy antecedent
    best_antecedent_pair = best_antecedent_pair[best_antecedent_pair[:, 1] != k]  # [num_spans_in_clusters, 2]

    # Use graph algorithm to cluster the pairs of mentions-antecedents
    clusters = []
    graph = nx.Graph()
    graph.add_edges_from(best_antecedent_pair)
    for connected_nodes in nx.connected_components(graph):
        clusters.append(list(connected_nodes))

    return clusters
