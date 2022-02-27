import json
import os
import copy
import logging
import random

import torch
from torch.utils.data import Dataset

import numpy as np

from utils import get_entity_label

ALPHA = [1, -1]

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        seq_label: list. The entity labels of the example.
        sent_label: string. The intent label of the example.
    """
    def __init__(self, guid, words, char_seq, seq_label=None, sent_label=None):
        self.guid = guid
        self.words = words
        self.char_seq = char_seq
        self.seq_label = seq_label
        self.sent_label = sent_label

    def __repr__(self):
        return str(self.to_json_string())
    
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class Processor(object):
    """Processor for the data set """
    def __init__(self, args) -> None:
        self.args = args

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def get_character(self, word, max_char_len):
        word_seq = []
        for i in range(max_char_len):
            try:
                char = word[i]
            except:
                char = self.args.pad_char
            word_seq.append(char)
        return word_seq

    def _creat_examples(self, data, set_type):
        """Creates examples for the training/dev/test sets."""
        examples = []
        for i, sample in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            sentence = sample['sentence'].lower()
            words = sentence.split(' ')
            char_seq = [] # char-level
            for word in words:
                char = self.get_character(word, self.args.max_char_len)
                char_seq.append(char)
            
            seq_label = sample['seq_label'] # entity-label
            sent_label = sample['sent_label'] # intent-label
            examples.append(
                InputExample(
                    guid=guid,
                    words=words,
                    char_seq=char_seq,
                    seq_label=seq_label,
                    sent_label=sent_label
                )
            )
        return examples
    
    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_name = mode + '.json'
        data_path = os.path.join(self.args.data_dir, file_name)
        logger.info("LOOKING AT {}".format(data_path))
        return self._creat_examples(
            data=self._read_file(data_path),
            set_type=mode
        )
# Name entity recognition task
class ViMQ(Dataset):
    def __init__(self, args, tokenizer, mode=None, predictions=None, iteration=None):
        self.args = args

        self.tokenizer = tokenizer
        
        
        self.label_set, _ = get_entity_label(args)
        
        char_vocab_path = os.path.join(args.data_dir, args.file_name_char2index)
        with open(char_vocab_path, 'r', encoding='utf-8') as f:
            self.char_vocab = json.load(f)
        
        self.examples = Processor(args).get_examples(mode)

        if predictions:
            logger.info("Updating new label")
            self.examples = self.update_label(self.examples, predictions)
        
        if iteration > args.iternoise:
            logger.info("Noising model")
            self.examples = self.noise_method(self.examples)
        

    def update_label(self, examples, predictions):
        assert len(examples) == len(predictions)

        new_examples = []
        for example in examples:
            guid = example.guid
            words = example.words
            char_seq = example.char_seq
            seq_label = predictions[guid]
            sent_label = example.sent_label
            new_examples.append(
                InputExample(
                    guid=guid,
                    words=words,
                    char_seq=char_seq,
                    seq_label=seq_label,
                    sent_label=sent_label
                )
            )
        return new_examples
    
    def noise_method(self, examples):
        def add_index(start, end, len_seq):
            new_start = None
            new_end = None
            while True:
                if np.random.randn() > 0: # START
                    new_start = start + random.choice(ALPHA)
                else:
                    new_start = start
                if np.random.randn() > 0: # END
                    new_end = end + random.choice(ALPHA)
                else:
                    new_end = end
                if valid_entity(new_start, new_end, len_seq):
                    break
            return new_start, new_end
        def valid_entity(start, end, len_seq):
            if 0 <= start and start <= end and end < len_seq:
                return True
            else:
                return False
            
        new_examples = []
        for example in examples:
            guid = example.guid
            words = example.words
            char_seq = example.char_seq
            seq_label = example.seq_label
            sent_label = example.sent_label

            len_seq = len(words)

            new_seq_label = []
            if seq_label:
                for entity in seq_label:
                    start = entity[0]
                    end = entity[1]
                    type_entity = entity[2]

                    start, end = add_index(start, end, len_seq)
                    new_seq_label.append([start, end, type_entity])
            new_examples.append(
                InputExample(
                    guid=guid,
                    words=words,
                    char_seq=char_seq,
                    seq_label=new_seq_label,
                    sent_label=sent_label
                )
            )
        return new_examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        
        example = self.examples[index]

        guid = example.guid
        words = example.words
        char_seq = example.char_seq
        seq_label = example.seq_label

        seq_len = len(words)


        input_ids, attention_mask, firstSWindices = self.preprocess(self.tokenizer, words, self.args.max_seq_len)
        char_ids = self.char2id(char_seq, self.args.max_seq_len)

        label = self.span_maxtrix_label(seq_label) # AFFINE
        return input_ids, attention_mask, firstSWindices, torch.tensor([seq_len]), char_ids, label, guid

    def preprocess(self, tokenizer, words, max_seq_len, mask_padding_with_zero=True): # sentence_embedding

        input_ids = [tokenizer.cls_token_id]
        firstSWindices = [len(input_ids)]

        for word in words:
            word_token = tokenizer.encode(word)
            input_ids += word_token[1: (len(word_token) - 1)]
            firstSWindices.append(len(input_ids))

        firstSWindices = firstSWindices[: (len(firstSWindices) - 1)]
        input_ids.append(tokenizer.sep_token_id)

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            attention_mask = attention_mask[:max_seq_len]
            firstSWindices = firstSWindices[:max_seq_len]
        else:
            attention_mask = attention_mask + [0 if mask_padding_with_zero else 1] * (max_seq_len - len(input_ids))
            input_ids = input_ids + [tokenizer.pad_token_id] * (max_seq_len - len(input_ids))
            firstSWindices = firstSWindices + [0]*(max_seq_len - len(firstSWindices))

        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(firstSWindices)

    def char2id(self, char_seq, max_seq_len): # char - embedding
        char_ids = []
        for word in char_seq:
            word_char_ids = []
            for char in word:
                if char not in self.char_vocab:
                    word_char_ids.append(self.char_vocab.get("UNK"))
                else:
                    word_char_ids.append(self.char_vocab.get(char))
            char_ids.append(word_char_ids)
        if len(char_ids) < max_seq_len:
            char_ids += [[self.char_vocab.get("PAD")]*self.args.max_char_len]*(max_seq_len - len(char_ids))
        else:
            char_ids = char_ids[:max_seq_len]
        return torch.tensor(char_ids)

    def span_maxtrix_label(self, label):
        if not label:
            return torch.sparse.FloatTensor(torch.tensor([[0], [0]]), torch.tensor(0), torch.Size([self.args.max_seq_len, self.args.max_seq_len])).to_dense()
        start, end, ent = [], [], []
        for lb in label:
            start.append(lb[0])
            end.append(lb[1])
            ent.append(self.label_set[lb[2]])
        label = torch.sparse.FloatTensor(torch.tensor([start, end]), torch.tensor(ent), torch.Size([self.args.max_seq_len, self.args.max_seq_len])).to_dense()
        return label