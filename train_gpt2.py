from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import tiktoken
import time
from config import GPTConfig, DataConfig
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
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
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
        
        # # attention (materializes the large (T,T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # # autoregressive mask (upper triangular matrix) for decoder block
        # # makes sure tokens only look into the past, not the future
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        # # normalize the attention weights
        # att = F.softmax(att, dim=-1)

        # # data dependent weighted average of values relevant to predicting the next token
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # flash attention replaces the above code by fusing the operations into one kernel
        # far faster on a gpu
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        self.attn = CausalSelfAttention(config)
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

# GPT model
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
    
        self.transformer = nn.ModuleDict(dict(
            # thin wrapper around a look up table converting tokens and positions to vectors in the embedding space
            # wte = word token embeddings, wpe = positional embeddings
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),

            # stack of transformer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            
            # layer norm to keep everything gaussian
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # head to convert the output of the stack of transformer blocks into a sequence of logits of the vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# -------------------------------------------------------
class DataLoaderLite:
    def __init__(self, config):
        self.config = config
        B, T = config.B, config.T
        self.dataset = config.dataset_name

        # at init load tokens from disk and store them in memory
        with open(self.dataset, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)

        # split the data into train, val, and test sets (80%, 10%, 10%)
        total_tokens = len(tokens)
        train_size = int(0.8 * total_tokens)
        val_size = int(0.1 * total_tokens)
        test_size = total_tokens - train_size - val_size

        self.train_tokens = torch.tensor(tokens[:train_size], dtype=torch.long)
        self.val_tokens = torch.tensor(tokens[train_size:train_size+val_size], dtype=torch.long)
        self.test_tokens = torch.tensor(tokens[train_size+val_size:], dtype=torch.long)

        print(f"Train set: {len(self.train_tokens)} tokens")
        print(f"Validation set: {len(self.val_tokens)} tokens")
        print(f"Test set: {len(self.test_tokens)} tokens")
        print(f"1 epoch = {len(self.train_tokens) // (B * T)} batches")

        # state
        self.current_position = 0
        self.current_set = 'train'

    def next_batch(self, set='train'):
        B, T = self.config.B, self.config.T

        if set == 'train':
            self.tokens = self.train_tokens
        elif set == 'val':
            self.tokens = self.val_tokens
        elif set == 'test':
            self.tokens = self.test_tokens

        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor by exactly B*T, wrapping to the beginning if necessary
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

def load_model(config, device, checkpoint_dir, model_path):
    model = GPT(config)
    model.to(device)
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        model.load_state_dict(torch.load(latest_checkpoint, map_location=device, weights_only=True))
        print(f"Model loaded from {latest_checkpoint}")
    elif os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Model loaded from {model_path}")
    else:
        print("No pre-trained model found, starting from scratch.")
    return model

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint

def train_model(model, train_loader, optimizer, device, checkpoint_dir, model_path):
    for i in range(model.config.training_steps):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (train_loader.config.B * train_loader.config.T) / (t1 - t0)
        print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f} ms, tokens/sec: {tokens_per_sec:.2f}")

        if (i + 1) % 100 == 0:
            save_checkpoint(model, checkpoint_dir, i + 1)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def save_checkpoint(model, checkpoint_dir, step):
    checkpoint_path = os.path.join(checkpoint_dir, f'model_step_{step}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def generate_text(model, device, prompt, num_return_sequences=5, max_length=30):
    model.eval()
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1)
    tokens = tokens.to(device)

    while tokens.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(tokens)
            next_token_logits = logits[:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, num_samples=1)
        xcol = torch.gather(topk_indices, -1, ix)
        tokens = torch.cat((tokens, xcol), dim=1)

    for i in range(num_return_sequences):
        decoded = enc.decode(tokens[i, :max_length].tolist())
        print(">", decoded)

def main():
    # Configuration
    config = GPTConfig()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    checkpoint_dir = 'checkpoints'
    model_path = 'gpt2_model.pth'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load model
    model = load_model(config, device, checkpoint_dir, model_path)

    # Data loader
    train_loader = DataLoaderLite(DataConfig())

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Train model
    train_model(model, train_loader, optimizer, device, checkpoint_dir, model_path)

    # Generate text
    generate_text(model, device, prompt='First')

if __name__ == "__main__":
    main()

# model = GPT(GPTConfig())
# # attempt to autodetect the device
# device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = "mps"
# model.to(device)

# # Load the model if present
# model_path = 'gpt2_model.pth'
# checkpoint_dir = 'checkpoints'

# # Ensure the checkpoint directory exists before accessing it
# os.makedirs(checkpoint_dir, exist_ok=True)

# def get_latest_checkpoint(checkpoint_dir):
#     checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
#     if not checkpoints:
#         return None
#     latest_checkpoint = max(checkpoints, key=os.path.getctime)
#     return latest_checkpoint

# latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
# if latest_checkpoint:
#     model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
#     print(f"Model loaded from {latest_checkpoint}")
# elif os.path.exists(model_path):
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     print(f"Model loaded from {model_path}")
# else:
#     print("No pre-trained model found, starting from scratch.")

# # for gpu, this leads to a speed up
# # model = torch.compile(model)

# print(f"using device: {device}")

# # get a data batch
# train_loader = DataLoaderLite(DataConfig())

# # for gpu, can quantize for speed up
# torch.set_float32_matmul_precision('high')

# # alternative to stochastic gradient descent that works better
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# # optimize!
# checkpoint_dir = 'checkpoints'
# os.makedirs(checkpoint_dir, exist_ok=True)
# checkpoint_list = []

# for i in range(model.config.training_steps):
#     t0 = time.time()
#     x, y = train_loader.next_batch()
#     x, y = x.to(device), y.to(device)
#     optimizer.zero_grad()
#     logits, loss = model(x, y)
#     loss.backward()
#     optimizer.step()

#     # if cuda available, synchronize for accurate timing
#     if device == 'cuda':
#         torch.cuda.synchronize()
#     t1 = time.time()
#     dt = (t1 - t0)*1000 # in milliseconds
#     tokens_per_sec = (train_loader.config.B * train_loader.config.T) / (t1 - t0)
#     print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f} ms, tokens/sec: {tokens_per_sec:.2f}")

#     # Save checkpoint every 100 steps
#     if (i + 1) % 100 == 0:
#         checkpoint_path = os.path.join(checkpoint_dir, f'model_step_{i+1}.pth')
#         torch.save(model.state_dict(), checkpoint_path)
#         checkpoint_list.append(checkpoint_path)
#         print(f"Checkpoint saved to {checkpoint_path}")

#         # Maintain only the 10 most recent checkpoints
#         if len(checkpoint_list) > 10:
#             oldest_checkpoint = checkpoint_list.pop(0)
#             os.remove(oldest_checkpoint)
#             print(f"Deleted old checkpoint {oldest_checkpoint}")

# # Save the final model
# torch.save(model.state_dict(), model_path)
# print(f"Model saved to {model_path}")

# print(loss) # should be (B, T, vocab_size)
# # import sys; sys.exit(0)

# # -------------------------------------------------------
# start = time.time()

# # prefix tokens
# model.eval()
# num_return_sequences = 5
# max_length = 30

# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode('First')
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# x = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
# # x = x.to('cuda')
# print(x.shape)

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# x = x.to(device)

# while x.size(1) < max_length:
#     # forward the model to get the logits
#     with torch.no_grad():
#         logits, _ = model(x)  # Unpack the tuple to get logits
#         next_token_logits = logits[:, -1, :].to(device)  # Now logits is just the tensor
    
#     # get the probabilities
#     probs = F.softmax(next_token_logits, dim=-1).to(device) # (B, vocab_size)
    
#     # do top-k sampling of 50 (huggingface pipeline default)
#     # topk_probs here becomes (5, 50), topk_indices is (5, 50)
#     topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    
#     # sample from the top-k probabilities
#     ix = torch.multinomial(topk_probs, num_samples=1).to(device) # (B, 1)
    
#     # gather the token indices
#     xcol = torch.gather(topk_indices, -1, ix).to(device) # (B, 1)
    
#     # append to the sequence
#     x = torch.cat((x, xcol), dim=1).to(device) # (B, T+1)


# # print the generated text
# for i in range(num_return_sequences):
#     # convert the token indices to text
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)

#     print(">", decoded)

# end = time.time()
# print(f"generation took: {end - start:.4f} seconds")
# print(f"avg = {(end - start) / num_return_sequences:.4f} seconds")