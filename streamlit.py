import streamlit as st
from train_gpt2 import GPT, generate_text, detect_device
from config import GPTConfig
import os
import torch
import tiktoken
from torch.nn import functional as F

# sample from model in train_gpt2.py

# build interface to take in user input
st.title("GPT-2 Text Generation")
st.write("This is a simple interface to generate text using the GPT-2 model.")
st.write("Enter some text below and the model will generate a continuation of it.")
user_input = st.text_input("Enter some text:")

checkpoint_dir = 'checkpoints'
model_path = 'gpt2_model.pth'
os.makedirs(checkpoint_dir, exist_ok=True)

device = detect_device()

# Load model
# Load the model if present
model_path = 'gpt2_model.pth'
checkpoint_dir = 'checkpoints'

# Ensure the checkpoint directory exists before accessing it
os.makedirs(checkpoint_dir, exist_ok=True)

model = GPT(GPTConfig())
def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint

latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
    print(f"Model loaded from {latest_checkpoint}")
elif os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
else:
    print("No pre-trained model found, starting from scratch.")

# prefix tokens
model.eval()
num_return_sequences = 5
max_length = 30

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(user_input)
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
x = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
# x = x.to('cuda')
print(x.shape)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
x = x.to(device)

while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits, _ = model(x)  # Unpack the tuple to get logits
        next_token_logits = logits[:, -1, :].to(device)  # Now logits is just the tensor
    
    # get the probabilities
    probs = F.softmax(next_token_logits, dim=-1).to(device) # (B, vocab_size)
    
    # do top-k sampling of 50 (huggingface pipeline default)
    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    
    # sample from the top-k probabilities
    ix = torch.multinomial(topk_probs, num_samples=1).to(device) # (B, 1)
    
    # gather the token indices
    xcol = torch.gather(topk_indices, -1, ix).to(device) # (B, 1)
    
    # append to the sequence
    x = torch.cat((x, xcol), dim=1).to(device) # (B, T+1)


# print the generated text
for i in range(num_return_sequences):
    # convert the token indices to text
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)

    print(">", decoded)

# return text to user
st.write("Generated text:")
st.write(decoded)


