import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tokenizers import Tokenizer

import math
import random
import numpy as np
from glob import glob
from tqdm import tqdm
import time
from dataclasses import dataclass, asdict
from typing import List, Optional
import warnings
import os
import json

from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("evaluation.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# --- Global Paths and Constants  ---
TOKENIZER_PATH = "/sd1/chaithra/lm/BPE_tokenizer/bpe_tokenizer.json"
DATA_DIR = "/sd1/chaithra/lm/new/NEW_TOKENIZED_MMAP"
# CHECKPOINT_DIR = "/sd1/chaithra/lm/new/checkpoints"
TOKEN_DTYPE = np.int32

def set_seed(seed: int = 11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set all seeds to {seed}")

@dataclass
class ModelConfig:
    # Architecture
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 2048

    # Training (for consistency, not used in evaluation script)
    num_epochs: int = 2
    batch_size: int = 4
    max_steps: int = 10000
    
    gradient_accumulation_steps: int = 10
    learning_rate: float = 3e-4

    # Qwen3-like specifics
    n_kv_heads: int = 4
    rms_norm_eps: float = 1e-6

    # Data
    max_seq_len: int = 2048
    vocab_size: int = 64000

    # Evaluation
    eval_every: int = 1000
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_kv_groups = self.n_heads // self.n_kv_heads

class MemoryMappedDataset(Dataset):
    def __init__(self, data_dir: str, sequence_length: int, token_dtype=np.int32):
        super().__init__()
        self.sequence_length = sequence_length
        self.mmap_files = sorted(glob(os.path.join(data_dir, "**", "*.mmap"), recursive=True))
        if not self.mmap_files:
            raise FileNotFoundError(f"FATAL: No .mmap files found in '{data_dir}'")

        self.file_handles = [np.memmap(path, dtype=token_dtype, mode='r') for path in self.mmap_files]
        self.file_lengths = [len(handle) for handle in self.file_handles]
        self.cumulative_lengths = np.cumsum(self.file_lengths)
        self.total_tokens = self.cumulative_lengths[-1]

        self.num_samples = self.total_tokens // self.sequence_length
        print(f" Loaded {len(self.mmap_files)} mmap files with {self.total_tokens:,} total tokens.")
        print(f" Created {self.num_samples:,} training samples of length {sequence_length}.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_token_idx = idx * self.sequence_length
        file_idx = np.searchsorted(self.cumulative_lengths, start_token_idx, side='right')
        local_start_idx = start_token_idx - (self.cumulative_lengths[file_idx - 1] if file_idx > 0 else 0)

        handle = self.file_handles[file_idx]

        if local_start_idx + self.sequence_length + 1 > len(handle):
            remaining_len = len(handle) - local_start_idx
            chunk1 = handle[local_start_idx:]
            needed_from_next = self.sequence_length + 1 - remaining_len
            chunk2 = self.file_handles[file_idx + 1][:needed_from_next]
            tokens = np.concatenate((chunk1, chunk2))
        else:
            tokens = handle[local_start_idx : local_start_idx + self.sequence_length + 1]

        x = torch.from_numpy(tokens[:-1].copy().astype(np.int64))
        y = torch.from_numpy(tokens[1:].copy().astype(np.int64))
        return x, y

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_seq_len
        t = torch.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len_cached:
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()[None, :, None, :]
            sin = emb.sin()[None, :, None, :]
        else:
            cos = self.cos_cached[:, :seq_len, :, :]
            sin = self.sin_cached[:, :seq_len, :, :]

        def rotate_half(tensor):
            x1, x2 = tensor.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        x = x.to(dtype=cos.dtype)
        return (x * cos) + (rotate_half(x) * sin)

class Qwen3Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_k, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_k, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary = Rotary(config.d_k, config.max_seq_len)
        self.q_norm = nn.RMSNorm(config.d_k, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.d_k, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.config.n_heads, self.config.d_k)
        k = self.k_proj(x).view(batch_size, seq_len, self.config.n_kv_heads, self.config.d_k)
        v = self.v_proj(x).view(batch_size, seq_len, self.config.n_kv_heads, self.config.d_k)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rotary(q, seq_len=seq_len)
        k = self.rotary(k, seq_len=seq_len)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k = repeat_kv(k, self.config.n_kv_groups)
        v = repeat_kv(v, self.config.n_kv_groups)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.config.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_model)
        return self.o_proj(attn_output)

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        activated_x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(activated_x))

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Qwen3Attention(config)
        self.feed_forward = SwiGLUFeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class MinimalLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.vocab_size is None:
            raise ValueError("vocab_size must be set in ModelConfig")
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.output_dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight.to(self.lm_head.weight.dtype)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits

@torch.no_grad()
def full_evaluate(model: nn.Module, data_loader: DataLoader, config: ModelConfig, device: torch.device):
    """
    Evaluates the model on the entire dataset provided by the data_loader.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    pbar_eval = tqdm(data_loader, desc="Full Evaluation")

    for x, y in pbar_eval:
        x, y = x.to(device), y.to(device)
        with autocast(enabled=config.use_amp, dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1), ignore_index=-1)
        
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

        predictions = logits.argmax(dim=-1)
        total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'loss': avg_loss, 'accuracy': accuracy, 'perplexity': perplexity}

def validate_data(dataset: MemoryMappedDataset, vocab_size: int):
    print("  Validating data against tokenizer vocabulary...")
    indices_to_check = list(range(min(100, len(dataset)))) + \
                       list(range(max(0, len(dataset)//2 - 50), min(len(dataset), len(dataset)//2 + 50))) + \
                       list(range(max(0, len(dataset)-100), len(dataset)))
    indices_to_check = sorted(list(set(indices_to_check)))

    max_id_found = 0
    min_id_found = float('inf')

    for idx in tqdm(indices_to_check, desc="Scanning samples"):
        x, y = dataset[idx]
        max_in_sample = max(x.max().item(), y.max().item()) if x.numel() > 0 or y.numel() > 0 else -1
        min_in_sample = min(x.min().item(), y.min().item()) if x.numel() > 0 or y.numel() > 0 else float('inf')

        if max_in_sample > max_id_found:
            max_id_found = max_in_sample
        if min_in_sample < min_id_found:
            min_id_found = min_in_sample

    print(f"Highest token ID found in sample scan: {max_id_found}")
    print(f"Lowest token ID found in sample scan: {min_id_found}")

    if max_id_found >= vocab_size:
        raise ValueError(
            f"FATAL: Data validation failed! "
            f"Highest token ID in data ({max_id_found}) is out of bounds for vocab size ({vocab_size}). "
            f"Please re-tokenize your data with the correct tokenizer."
        )
    if min_id_found < 0:
        raise ValueError(
            f"FATAL: Data validation failed! "
            f"Lowest token ID in data ({min_id_found}) is less than 0. "
        )

    print("Data validation successful.")

def main():
    logger.info("Starting final evaluation on the entire dataset.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)

    # 1. Load the tokenizer and get the vocab size
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"FATAL: Tokenizer file not found at '{TOKENIZER_PATH}'")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    actual_vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Tokenizer loaded. Actual vocabulary size: {actual_vocab_size}")

    # 2. Check for checkpoints
    best_ckpts_file = Path(CHECKPOINT_DIR) / "best_checkpoints.json"
    if not best_ckpts_file.exists():
        logger.error("No best_checkpoints.json found. Cannot perform final evaluation.")
        return
    
    best_checkpoints = json.load(open(best_ckpts_file))
    if not best_checkpoints:
        logger.error("No checkpoints listed in best_checkpoints.json. Cannot perform final evaluation.")
        return

    # 3. Find the best checkpoint path
    best_ckpt_path = sorted(best_checkpoints, key=lambda x: x["val_loss"])[0]["path"]

    # 4. Instantiate model and load the state dict
    try:
        config = ModelConfig(vocab_size=actual_vocab_size)
        model = MinimalLLM(config).to(device)
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"Successfully loaded best checkpoint from {best_ckpt_path}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {best_ckpt_path}: {e}")
        return

    # 5. Create a data loader for the entire dataset
    if not os.path.exists(DATA_DIR) or not glob(os.path.join(DATA_DIR, "**", "*.mmap"), recursive=True):
        raise FileNotFoundError(f"FATAL: No .mmap files found in '{DATA_DIR}'.")
        
    full_dataset = MemoryMappedDataset(data_dir=DATA_DIR, sequence_length=config.max_seq_len, token_dtype=TOKEN_DTYPE)
    validate_data(full_dataset, actual_vocab_size)
    full_loader = DataLoader(full_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # 6. Run the full evaluation
    final_metrics = full_evaluate(model, full_loader, config, device)
    
    # 7. Print the final results
    logger.info("--- Final Evaluation Results ---")
    logger.info(f"Loss: {final_metrics['loss']:.4f}")
    logger.info(f"Accuracy: {final_metrics['accuracy']:.4f}")
    logger.info(f"Perplexity: {final_metrics['perplexity']:.2f}")
    logger.info("-------------------------------")

if __name__ == "__main__":
    main()
