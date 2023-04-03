# Uses the corefsonversion conll2jsonlines to convert litbank conll to jsonlines
import os
import json, jsonlines
from  conll2jsonlines import * 

source_folder = "litbank/coref/conll"
dest_folder = "jsonlines/cache"

lb_docs = [f for f in os.listdir(source_folder) if f.endswith(".conll")]
print(len(lb_docs) , "documents")

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

def do_import(): # to disable it if needed
    for filename in lb_docs:
        lb_docpath = os.path.join(source_folder, filename)
        dest_docpath = os.path.join(dest_folder, filename.replace(".conll", ".json"))
        conll2jsonlines(
            lb_docpath, dest_docpath,
            sep=None, token_col=3, speaker_col="vetkje", add_coref=True, par_col=0,
            ignore_double_indices=None,
            skip_empty_documents=False, skip_singletons=True)

do_import()
"""
document_id:    str,                   # document name
cased_words:    List[str]              # words
sent_id:        List[int]              # word id to sent id mapping
part_id:        int.                   # document part id
speaker:        List[str]              # word id to speaker mapping
pos:            List[str]              # word id to POS mapping
deprel:         List[str]              # word id to dependency relation mapping
head:           List[int]              # word id to head word id mapping, None for root
clusters:       List[List[List[int]]]  # list of clusters,
                                       #     each cluster is a list of spans
                                       #         each span is a list of two ints (start and end word ids)

{'document_id': 'bc/cctv/00/cctv_0005', 'cased_words': ['WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD', 'WORD'], 'sent_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7], 'part_id': 0, 'speaker': ['speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'speaker#1', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li', 'Xu_li'], 'pos': [':', 'RB', ',', 'PRP', 'VBD', 'RB', 'VBN', 'IN', 'IN', 'DT', 'JJ', 'JJ', 'NNS', '.', 'TO', 'VB', 'PRP$', 'NN', ',', 'DT', 'JJ', 'NNS', 'JJ', 'NN', 'VBZ', 'DT', 'NN', 'NN', 'IN', 'DT', 'NN', 'WDT', 'VBZ', 'VBN', 'VBN', '.', 'PRP', 'VBZ', 'NN', 'TO', 'VB', 'IN', 'DT', 'NN', 'NN', 'MD', 'RB', 'VB', 'NNS', ',', 'CC', 'IN', 'DT', 'NNS', 'WDT', 'VBP', 'IN', 'DT', 'NN', 'NN', 'MD', 'VB', 'RB', 'VBN', '.', 'NNP', 'NNS', ',', 'DT', 'NNP', 'NNP', 'NN', 'MD', 'VB', 'RB', '.', 'DT', 'VBZ', 'JJ', 'NNP', '.', 'VBP', 'PRP', 'NN', 'IN', 'VBG', '.', 'VBG', 'RP', 'VBZ', 'DT', 'NNP', 'NNP', 'NN', 'VBN', 'IN', 'NNP', 'NNP', '.', 'UH', ',', 'NN', 'NNS', '.'], 'deprel': ['punct', 'advmod', 'punct', 'nsubjpass', 'auxpass', 'advmod', 'root', 'prep', 'prep', 'det', 'amod', 'amod', 'pobj', 'punct', 'aux', 'advcl', 'poss', 'dobj', 'punct', 'det', 'amod', 'nn', 'amod', 'nsubj', 'root', 'det', 'nn', 'dobj', 'prep', 'det', 'pobj', 'nsubjpass', 'aux', 'auxpass', 'rcmod', 'punct', 'nsubj', 'root', 'dobj', 'aux', 'xcomp', 'mark', 'det', 'nn', 'nsubj', 'aux', 'advmod', 'ccomp', 'dobj', 'punct', 'cc', 'mark', 'det', 'nsubjpass', 'nsubj', 'rcmod', 'prep', 'det', 'nn', 'pobj', 'aux', 'auxpass', 'advmod', 'conj', 'punct', 'nn', 'nsubj', 'punct', 'det', 'nn', 'nn', 'nsubj', 'aux', 'root', 'advmod', 'punct', 'nsubj', 'cop', 'amod', 'root', 'punct', 'root', 'iobj', 'dobj', 'prep', 'pcomp', 'punct', 'root', 'prt', 'aux', 'det', 'nn', 'nn', 'nsubj', 'partmod', 'prep', 'nn', 'pobj', 'punct', 'discourse', 'punct', 'nn', 'root', 'punct'], 'head': [6, 6, 6, 6, 6, 6, None, 6, 6, 12, 12, 12, 8, 6, 15, 24, 17, 15, 24, 23, 23, 23, 23, 24, None, 27, 27, 24, 24, 30, 28, 34, 34, 34, 30, 24, 37, None, 37, 40, 37, 47, 44, 44, 47, 47, 47, 40, 47, 47, 47, 63, 53, 63, 55, 53, 55, 59, 59, 56, 63, 63, 63, 47, 37, 66, 73, 73, 71, 70, 71, 73, 73, None, 73, 73, 79, 79, 79, None, 79, None, 81, 81, 81, 84, 81, None, 87, 87, 93, 92, 93, 87, 93, 94, 97, 95, 87, 102, 102, 102, None, 102], 'clusters': [[[16, 17], [19, 24]], [[25, 28], [42, 45], [57, 60]], [[82, 83], [83, 84]]]}                         

Please, note that not every key is used, however. For instance, pos and deprel are never used
"""

# # Inspect the litbank jsonlines data
# with open ("jsonlines/cache/514_little_women_brat.json") as rf:
#     doc = json.loads(rf.readline().strip())
# for key, value in doc.items():
#     print(key, type(value), type(value[0]), type(value[0][0]), type(value[0][0][0]))
# docwords = [t for s in doc["sentences"] for t in s]
# doctext = " ".join(docwords)
# democluster =  [[82, 82], [126, 126], [132, 132], [156, 156], [676, 677], [679, 679], [714, 714], [780, 780], [1217, 1217], [1782, 1783]]
# for start, end in democluster:
#     # print(docwords[start:end+1])
#     pass


# Do the format convertion
source_folder = dest_folder
dest_folder = "wl-coref/litbank"
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)
source_filenames = [f for f in os.listdir(source_folder) if ".json" in f]

wl_formatted = []
for idx, f_name in enumerate(source_filenames):
    with open (os.path.join(source_folder, f_name)) as rf:
        # assert len(rf.readlines()) == 1
        # print(f_name)
        doc = json.loads(rf.readline().strip())
    wl_data = {"document_id": "nw"+doc["doc_key"], 
                "cased_words" : [t for s in doc["sentences"] for t in s],
                "sent_id" : [i for i, s in enumerate(doc["sentences"] )for t in s], 
                "part_id" : idx ,
                "speaker": [sp for s in doc["speakers"] for sp in s],
                "head": [None for s in doc["sentences"] for t in s], # Change this to real head when possible
                "clusters": []
                }
    for cluster in doc["clusters"]: # Add 1 to end word-index
        wl_data["clusters"].append( [[s,e+1] for s,e in cluster] )
    wl_formatted.append(wl_data)


# These should now be similar to the output of convert_to_jsonlines.py in wl-coref


with jsonlines.open(os.path.join(dest_folder,'litbank_all.jsonl'), 'w') as wf:
    wf.write_all(wl_formatted)

train_max = int(len(wl_formatted)*0.7)
dev_max = int(len(wl_formatted)*0.85)
splits = {"train":wl_formatted[:train_max], "development": wl_formatted[train_max:dev_max], "test":wl_formatted[dev_max:]}
for split, data in splits.items():
    print(split,len(data))
    with jsonlines.open(os.path.join(dest_folder,f'litbank_{split}.jsonl'), 'w') as wf:
        wf.write_all(data)

