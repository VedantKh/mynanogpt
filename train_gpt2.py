from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------


        
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
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

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