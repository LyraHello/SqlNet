import torch
import torch.nn as nn

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size=300, trainable=True):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        if not trainable:
            self.embedding.weight.requires_grad = False

    def forward(self, x):
        # x: [batch_size, seq_len]
        return self.embedding(x)
