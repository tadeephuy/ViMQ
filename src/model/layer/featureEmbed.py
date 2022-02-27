import torch.nn as nn
import numpy as np
import torch


class FeatureEmbedding(nn.Module):
    """Part-of-Speech embeddings
    Trained with part-of-speech tag, the embedding layer will be the PoS embeddings
    """

    def __init__(self, vocab_size, embedding_size,  dropout=0.1):
        super(FeatureEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data.copy_ = torch.from_numpy(self.random_embedding(vocab_size, embedding_size))
        self.dropout = nn.Dropout(dropout)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, x):

        x = self.embedding(x)
        x = self.dropout(x)
        return x

