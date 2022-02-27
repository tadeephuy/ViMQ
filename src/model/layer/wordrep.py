from model.layer import CharCNN

import torch
from torch import nn

from transformers import AutoModel


class WordRep(nn.Module):
    def __init__(self, args):
        super(WordRep, self).__init__()
        print("build word representation...")
        
        self.roberta = AutoModel.from_pretrained(
            args.model_name_or_path
        )
        print('Loaded Backbone')

        self.num_layer_bert = args.num_layer_bert

        self.use_char = args.use_char

        if self.use_char:
            self.char_feature = CharCNN(hidden_dim=args.char_hidden_dim,
                                        vocab_size=args.char_vocab_size,
                                        embedding_dim=args.char_embedding_dim)

    def forward(self, input_ids, attention_mask, first_subword, char_ids):

        bert_output = self.roberta(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        bert_features = []

        for i in range(-1, -(self.num_layer_bert + 1), -1):
            ber_emb = bert_output[2][i] # B X Seq_len X Hidden_Size
            bert_features.append(ber_emb)
        bert_features = torch.cat(bert_features, dim=-1)

        bert_features = torch.cat([torch.index_select(bert_features[i], 0, first_subword[i]).unsqueeze(0) for i in range(bert_features.size(0))], dim=0)
        
        if self.use_char:
            word_features = []
            char_features = self.char_feature(char_ids)
            word_features.append(char_features)
            word_features = torch.cat(word_features, dim=-1)
            return torch.cat((bert_features, word_features), dim=-1)
        else:
            return bert_features

