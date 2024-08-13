from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    training_steps: int = 10

@dataclass
class DataConfig:
    dataset_name: str = "input.txt"
    B: int = 4 # number of batches of context
    T: int = 1024 # length of each context