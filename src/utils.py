import logging
import os
import random

import torch
import numpy as np

from model.model import ViMQModel
from transformers import (
    AutoTokenizer,
    RobertaConfig
)

from seqeval.metrics import f1_score, precision_score, recall_score


MODEL_CLASSES = {
    'vimq_model': (RobertaConfig, ViMQModel, AutoTokenizer)
}

MODEL_PATH_MAP = {
    'vimq_model': "/workspace/vinbrain/vutran/Backbone/phoBERT/"
}

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def load_tokenizer(args):
    print(args.model_name_or_path)
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)

def get_entity_label(args):
    
    entity_set_path = os.path.join(args.data_dir, args.file_name_entity_set)
    with open(entity_set_path, 'r', encoding='utf-8') as f:
        label_set = f.read().splitlines()
        label2index = {w: i for i, w in enumerate(label_set)}
        index2label = {i: w for i, w in enumerate(label_set)}
    return label2index, index2label
        


def compute_metrics(preds, labels):
    pass # convert spacy to iob

    assert len(preds) == len(labels)
    return {
        "precision_score": precision_score(labels, preds),
        "recall_score": recall_score(labels, preds),
        "f1_score": f1_score(labels, preds),
    }

def convert_spacy_to_iob(preds, labels, seq_len):
    assert len(preds) == len(labels) == len(seq_len)
    iob_pred = []
    iob_label = []
    for p, l, s in zip(preds, labels, seq_len):
        iob_pred.append(spacy_to_iob(p, s))
        iob_label.append(spacy_to_iob(l, s))
    return iob_pred, iob_label

def spacy_to_iob(spacy, seq_len):
    iob = ['O'] * seq_len
    if not spacy:
        return iob
    else:
        for i in spacy[::-1]:
            start = i[0]
            end = i[1]
            type_ent = i[2]
            sub_iob = ['I-'+type_ent]*(end-start+1)
            sub_iob[0:1] = ['B-'+type_ent]
            iob[start:end+1] = sub_iob
    return iob
def get_iou_score(tensor_1, tensor_2):
    """
    tensor_1: is tensor prediction 
    e.g:
    tensor([[1, 1, 0, 0] # entity_1
            [0, 1, 1, 0]]) # entity_2
    dim: n.o entity of prediction X Seq_len
    tensor_2: is tensor ground truth
    e.g:
    tensor([[1, 1, 0, 0]
            [1, 1, 0, 0]])
    dim: entity of prediction X Seq_len
    return tensor([[1.0]
                   [0.333]])
    """
    inter_area = torch.sum(tensor_1*tensor_2, dim=-1, keepdim=True)
    pred_area = torch.sum(tensor_1, dim=-1, keepdim=True)
    label_area = torch.sum(tensor_2, dim=-1, keepdim=True)
    
    return inter_area/(label_area + pred_area - inter_area)

def get_IoU(span_label, span_pred):
    left = max(span_label[0], span_pred[0])
    right = min(span_label[1]+1, span_pred[1]+1)

    if right < left:    return 0.0

    intersection_area = (right - left)
    
    label_area = span_label[1] +1 - span_label[0]
    pred_area = span_pred[1] + 1 - span_pred[0]

    iou_score = float(intersection_area/ (label_area + pred_area - intersection_area))
    return iou_score

def major_vote(starts, ends, ents):
    """         #span
             ----------  
    # starts [(1, 4, 5),    |
    #         (1, 3, 6),    | # lamda
    #         (1, 3, 6)]    |
    # ends   [(2, 4, 7),
    #         (2, 4, 7),
    #         (1, 4, 7)]
    # ents ....

    """
    new_starts = [0]*len(starts[0])
    new_ends = [0]*len(ends[0])
    new_ents = [0]*len(ents[0])

    for i in range(len(new_starts)):
        list_start = list(list(zip(*starts))[i])
        new_starts[i] = max(set(list_start), key=list_start.count)

        list_end = list(list(zip(*ends))[i])
        new_ends[i] = max(set(list_end), key=list_end.count)

        list_ent = list(list(zip(*ents))[i])
        new_ents[i] = max(set(list_ent), key=list_ent.count)

    label = []
    for s, e, en in zip(new_starts, new_ends, new_ents):
        label.append([s,e,en])
    
    return label


def get_new_label(pred_dict, lamda, approach=1):
    """
    pred_dict - dictionary:
    { id      value
                        # span
            -----------------------------------------
      1:    [[[1,2,'VT'], [4,4,'QT'], [5,7,'A']],   |
             [[1,2,'VT'], [3,4,'QT'], [6,7,'A']],   |   #lamda = epoch
             [[1,1,'VT'], [3,4,'QT'], [6,7,'A']]],  |
      2: .... 
    }
    """
    new_label = {}
    if approach == 3:
    # approach 3
        for idx, ls in pred_dict.items():
            starts = []
            ends = []
            ents = []
            for l in ls:
                starts.append(list(zip(*l))[0]) # start_index of sample # 1 4 5
                ends.append(list(zip(*l))[1])  # end_index of sample      2 4 7 
                ents.append(list(zip(*l))[2])  # ent of sample            VT QT A

            label = major_vote(starts, ends, ents)
            
            new_label[idx] = label
    elif approach == 1:
    # appoach 1
        for idx, ls in pred_dict.items():
            starts = []
            ends = []
            ents = []
            for l in ls[-lamda:]:
                starts.append(list(zip(*l))[0]) # start_index of sample # 1 4 5
                ends.append(list(zip(*l))[1])  # end_index of sample      2 4 7 
                ents.append(list(zip(*l))[2])  # ent of sample            VT QT A

            label = major_vote(starts, ends, ents)
            
            new_label[idx] = label

    return new_label