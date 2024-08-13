from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    training_steps: int = 1000

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    max_steps = 500

@dataclass
class DataConfig:
    dataset_name: str = "input.txt"
    B: int = 4 # number of batches of context
    T: int = 1024 # length of each context