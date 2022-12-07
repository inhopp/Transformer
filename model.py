import torch 
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0] # batch size

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # (N, value_len, embed_size)
        values = self.values(values) 
        keys = self.keys(keys)
        queries = self.queries(query)

        # split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # queries : (N, query_len, heads, head_dim)
        # keys : (N, key_len, heads, head_dim)
        # energy : (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3) # dim=3 means normalize by key_len

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]) # (N, query_len, heads, head_dim)
        out = out.reshape(N, query_len, self.embed_size)
        out = self.fc_out(out)

        return out
