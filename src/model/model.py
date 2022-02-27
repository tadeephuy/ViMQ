import torch
from torch import nn
from model.layer import WordRep, FeedforwardLayer, BiaffineLayer


class ViMQModel(nn.Module):
    def __init__(self, config, args):
        super(ViMQModel, self).__init__()

        self.args = args

        self.num_labels = args.num_labels
        self.lstm_input_size = args.num_layer_bert * config.hidden_size
        if args.use_char:
            self.lstm_input_size = self.lstm_input_size + 2 * args.char_hidden_dim

        # Word representation
        self.word_rep = WordRep(args=args)
        # LSTM
        self.bilstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=args.hidden_dim // 2,
                              num_layers=3, bidirectional=True, batch_first=True)
        # FFN 
        self.feedStart = FeedforwardLayer(d_in=args.hidden_dim, d_hid=args.hidden_dim_ffw)
        self.feedEnd = FeedforwardLayer(d_in=args.hidden_dim, d_hid=args.hidden_dim_ffw)
        # BIAFINE
        self.biaffine = BiaffineLayer(inSize1=args.hidden_dim, inSize2=args.hidden_dim, classSize=self.num_labels)

    def forward(self,
                input_ids,
                attention_mask,
                first_subword=None,
                seq_len=None,
                char_ids=None,
                label=None):

        word_features = self.word_rep(
            input_ids=input_ids,
            attention_mask=attention_mask,
            first_subword=first_subword,
            char_ids=char_ids
        )

        x, _ = self.bilstm(word_features)

        start = self.feedStart(x)
        end = self.feedEnd(x)
        
        score = self.biaffine(start, end) # tensor B x max_seq_len x max_seq_len x C

        total_loss = 0
        if label is not None:
            mask = [[1]*seq_len[i]+[0]*(self.args.max_seq_len-seq_len[i]) for i in range(len(seq_len))]
            mask = torch.tensor(mask).to(self.args.device)
            mask = mask.unsqueeze(1).expand(-1, mask.shape[-1], -1)
            mask = torch.triu(mask).reshape(-1)
            
            tmp_out = score.reshape(-1, score.shape[-1])
            tmp_label = label.reshape(-1)
            
            # index select, for gpu speed
            indices = mask.nonzero(as_tuple=False).squeeze(-1)
            tmp_out = tmp_out.index_select(0, indices)
            tmp_label = tmp_label.index_select(0, indices)
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(tmp_out, tmp_label)
            total_loss += loss
#         print(total_loss)

        return score, total_loss
