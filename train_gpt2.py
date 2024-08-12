from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------

# multi-head attention mechanism
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # not really a 'bias', more of a mask, but following the OpenAI/NF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # autoregressive mask (upper triangular matrix) for decoder block
        # makes sure tokens only look into the past, not the future
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        # normalize the attention weights
        att = F.softmax(att, dim=-1)

        # data dependent weighted average of values relevant to predicting the next token
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
       
       # output projection
        y = self.c_proj(y)
        return y

# multi-layer perceptron      
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        # smooth version of ReLU that is continuously differentiable and has a non-zero gradient everywhere
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# the transformer block is the core building block of the transformer architecture
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        

    def forward(self, x):
        # x = x + -> represents residual connection
        # this means that there is a direct path for the gradient to flow through from the output to the input
        # this is important for training deep networks
        
        # attention block can be thought of as the communication between vectors (aggregation or pooling function 
        # that tells the model what to focus on when generating each token)
        x = x + self.attn(self.ln_1(x))
        # mlp block can be thought of as the computation on the vectors to make sense of the communication
        # allowing each neuron to think individually about the information it has received
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length for GPT2
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of transformer layers
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # embedding dimension

# GPT model
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
    
    self.transformer = nn.ModuleDict(dict(
        # thin wrapper around a look up table converting tokens and positions to vectors in the embedding space
        wte = nn.Embedding(config.vocab_size, config.n_embd),
        wpe = nn.Embedding(config.block_size, config.n_embd),

        # stack of transformer blocks
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        
        # layer norm to keep everything gaussian
        ln_f = nn.LayerNorm(config.n_embd),
    ))
    # head to convert the output of the stack of transformer blocks into a sequence of logits of the vocabulary
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

