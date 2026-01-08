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
        logging.StreamHandler(),  # console
        logging.FileHandler("training.log", mode="a")  # file
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# --- Global Paths and Constants ---
TOKENIZER_PATH = "/sd1/chaithra/lm/BPE_tokenizer/bpe_tokenizer.json"
DATA_DIR = "/sd1/chaithra/lm/new/NEW_TOKENIZED_MMAP" # Search recursively within model-pretraining/
CHECKPOINT_DIR = "/sd1/chaithra/lm/new/checkpoints"
TOKEN_DTYPE = np.int32

def set_seed(seed: int = 11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Only set CUDA seed if CUDA is available
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
    d_ff: int = 2048 # Generally 4 * d_model

    # Training
    num_epochs: int = 2
    batch_size: int = 16
    max_steps: int = 10000
    
    gradient_accumulation_steps: int = 10
    learning_rate: float = 3e-4 # Using a standard learning rate for AdamW

    # Qwen3-like specifics 
    n_kv_heads: int = 4 # For Grouped-Query Attention
    rms_norm_eps: float = 1e-6 # Epsilon for RMSNorm

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
        # Ensure n_heads is divisible by n_kv_heads for GQA
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

        # This calculation is robust and correct
        self.num_samples = self.total_tokens // self.sequence_length
        print(f" Loaded {len(self.mmap_files)} mmap files with {self.total_tokens:,} total tokens.")
        print(f" Created {self.num_samples:,} training samples of length {sequence_length}.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate the global start index of the token sequence
        start_token_idx = idx * self.sequence_length

        # Find which file this index falls into
        file_idx = np.searchsorted(self.cumulative_lengths, start_token_idx, side='right')

        # Find the local start index within that file
        local_start_idx = start_token_idx - (self.cumulative_lengths[file_idx - 1] if file_idx > 0 else 0)

        handle = self.file_handles[file_idx]

        
        # Check if the required sequence (length + 1) spans across the file boundary
        if local_start_idx + self.sequence_length + 1 > len(handle):
            # === BOUNDARY SPAN LOGIC ===
            # 1. Read the remainder from the current file
            remaining_len = len(handle) - local_start_idx
            chunk1 = handle[local_start_idx:]

            # 2. Calculate how much more is needed from the next file
            needed_from_next = self.sequence_length + 1 - remaining_len

            # 3. Read the required chunk from the *next* file
            # This logic correctly assumes the next sample (idx+1) won't also be a boundary case
            chunk2 = self.file_handles[file_idx + 1][:needed_from_next]

            # 4. Concatenate them into a single sequence
            tokens = np.concatenate((chunk1, chunk2))
        else:
            # === NORMAL CASE (sequence is fully within one file) ===
            tokens = handle[local_start_idx : local_start_idx + self.sequence_length + 1]
        # --- END OF FIX ---

        # Create the x, y pair for training
        x = torch.from_numpy(tokens[:-1].copy().astype(np.int64))
        y = torch.from_numpy(tokens[1:].copy().astype(np.int64))
        return x, y

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
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
        # Precompute cos and sin for the maximum sequence length
        t = torch.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Cache cos and sin with shape (1, max_seq_len, 1, dim) for broadcasting
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)

    def forward(self, x, seq_len):
        # x shape is (batch, seq_len, heads, dim)
        if seq_len > self.max_seq_len_cached:
            # Recompute if seq_len is larger than cached
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            # Reshape to (1, seq_len, 1, dim) for broadcasting
            cos = emb.cos()[None, :, None, :]
            sin = emb.sin()[None, :, None, :]
        else:
            # Use cached values, slice up to the current sequence length
            cos = self.cos_cached[:, :seq_len, :, :]
            sin = self.sin_cached[:, :seq_len, :, :]

        # Apply RoPE - rotate_half expects last dimension to be dim
        def rotate_half(tensor):
            x1, x2 = tensor.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        # Apply broadcasting: cos, sin are (1, seq_len, 1, dim), x is (batch, seq_len, heads, dim)
        x = x.to(dtype=cos.dtype) # Ensure dtype compatibility

        return (x * cos) + (rotate_half(x) * sin)

class Qwen3Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_k, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_k, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        # Initialize Rotary with head_dim, not d_model
        self.rotary = Rotary(config.d_k, config.max_seq_len)

        # QK-Normalization layers (as in original Qwen3, applied after projection)
        self.q_norm = nn.RMSNorm(config.d_k, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.d_k, eps=config.rms_norm_eps)


    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.config.n_heads, self.config.d_k) # (batch, seq_len, n_heads, d_k)
        k = self.k_proj(x).view(batch_size, seq_len, self.config.n_kv_heads, self.config.d_k) # (batch, seq_len, n_kv_heads, d_k)
        v = self.v_proj(x).view(batch_size, seq_len, self.config.n_kv_heads, self.config.d_k) # (batch, seq_len, n_kv_heads, d_k)

        # Apply QK-Norm (as in original Qwen3)
        q = self.q_norm(q)
        k = self.k_norm(k)


        # Apply Rotary embedding BEFORE transposing for attention
        q = self.rotary(q, seq_len=seq_len) # (batch, seq_len, n_heads, d_k)
        k = self.rotary(k, seq_len=seq_len) # (batch, seq_len, n_kv_heads, d_k)


        # Transpose for attention calculation
        q = q.transpose(1, 2) # (batch, n_heads, seq_len, d_k)
        k = k.transpose(1, 2) # (batch, n_kv_heads, seq_len, d_k)
        v = v.transpose(1, 2) # (batch, n_kv_heads, seq_len, d_k)

        # Repeat K and V heads for GQA
        k = repeat_kv(k, self.config.n_kv_groups) # (batch, n_heads, seq_len, d_k)
        v = repeat_kv(v, self.config.n_kv_groups) # (batch, n_heads, seq_len, d_k)


        # Use PyTorch's optimized backend which will use Flash Attention if available
        # q, k, v are all (batch, n_heads, seq_len, d_k) after repetition
        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.config.dropout if self.training else 0.0
        ) # (batch, n_heads, seq_len, d_k)


        # Reshape and final projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_model) # (batch, seq_len, d_model)
        return self.o_proj(attn_output)

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Implementation of the SwiGLU activation function
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

        # Ensure vocab_size is set
        if config.vocab_size is None:
             raise ValueError("vocab_size must be set in ModelConfig")

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.output_dropout = nn.Dropout(config.dropout)

        # Tie weights
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Use .to(self.token_embedding.weight.dtype) to ensure dtype compatibility
        self.lm_head.weight = self.token_embedding.weight.to(self.lm_head.weight.dtype)


        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Ensure padding_idx does not get initialized if it exists
            # if module.padding_idx is not None:
            #      with torch.no_grad():
            #           module.weight[module.padding_idx].fill_(0)


    def forward(self, x):
        # x is (batch, seq_len)
        x = self.token_embedding(x) * math.sqrt(self.config.d_model) # (batch, seq_len, d_model)
        x = self.position_dropout(x)

        for block in self.transformer_blocks:
            x = block(x) # (batch, seq_len, d_model)

        x = self.norm(x) # (batch, seq_len, d_model)
        x = self.output_dropout(x) # (batch, seq_len, d_model)
        logits = self.lm_head(x) # (batch, seq_len, vocab_size)

        return logits

@torch.no_grad()
def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig, device: torch.device):
    model.eval()
    total_loss = 0
    total_tokens = 0 # Track total tokens for accurate average loss
    total_correct = 0 # Track total correct predictions for accuracy

    pbar_eval = tqdm(val_loader, desc="Evaluating", leave=False, total=min(config.eval_steps, len(val_loader)))

    for i, (x, y) in enumerate(pbar_eval):
        if i >= config.eval_steps:
            break
        x, y = x.to(device), y.to(device)
        with autocast(enabled=config.use_amp, dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1), ignore_index=-1) # Use ignore_index if padding is used
        total_loss += loss.item() * y.numel() # Accumulate loss weighted by number of tokens
        total_tokens += y.numel()

        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        total_correct += (predictions == y).sum().item()


    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0 # Handle case with no tokens
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': math.exp(min(avg_loss, 20))}

def setup_optimizer(model: nn.Module, config: ModelConfig):
    # Use standard AdamW optimizer
    # Separate parameters for weight decay
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    # decay_params: parameters with 2 or more dimensions (weights of linear layers, embeddings)
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # nodecay_params: parameters with less than 2 dimensions (biases, layernorm weights)
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(0.9, 0.95), fused=torch.cuda.is_available())
    return optimizer

def save_checkpoint(state, file_path):
    """Saves checkpoint to a specific file path."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(state, file_path)

def validate_data(dataset: MemoryMappedDataset, vocab_size: int):
    """
    Scans a small part of the dataset to ensure all token IDs are valid.
    """
    print("  Validating data against tokenizer vocabulary...")
    # Check a few samples from the beginning, middle, and end
    indices_to_check = list(range(min(100, len(dataset)))) + \
                       list(range(max(0, len(dataset)//2 - 50), min(len(dataset), len(dataset)//2 + 50))) + \
                       list(range(max(0, len(dataset)-100), len(dataset)))
    indices_to_check = sorted(list(set(indices_to_check))) # Remove duplicates and sort

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
            f"This means the .mmap files were created with a DIFFERENT tokenizer than the one at '{TOKENIZER_PATH}'. "
            f"Please re-tokenize your data with the correct tokenizer."
        )
    if min_id_found < 0:
         raise ValueError(
             f"FATAL: Data validation failed! "
             f"Lowest token ID in data ({min_id_found}) is less than 0. "
             f"This indicates corrupted data or an issue with data loading."
         )

    print(" Data validation successful.")

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    set_seed(42)

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"FATAL: Tokenizer file not found at '{TOKENIZER_PATH}'")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    actual_vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Tokenizer loaded. Actual vocabulary size: {actual_vocab_size}")

    if not os.path.exists(DATA_DIR) or not glob(os.path.join(DATA_DIR, "**", "*.mmap"), recursive=True):
        raise FileNotFoundError(f"FATAL: No .mmap files found in '{DATA_DIR}'.")

    dataset = MemoryMappedDataset(data_dir=DATA_DIR, sequence_length=512, token_dtype=TOKEN_DTYPE)
    validate_data(dataset, actual_vocab_size)

    config = ModelConfig(vocab_size=actual_vocab_size)
    model = MinimalLLM(config).to(device)

    optimizer = setup_optimizer(model, config)
    scaler = GradScaler(enabled=config.use_amp)

    warmup_steps = config.max_steps // 20
    def lr_lambda(step):
        if step < warmup_steps: return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    val_size = len(dataset) // 20
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    num_batches_per_epoch = len(train_loader)
    steps_for_one_epoch = num_batches_per_epoch // config.gradient_accumulation_steps

    logger.info(f"Training samples: {len(train_dataset):,}, Batch size: {config.batch_size}, Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"Batches per epoch: {num_batches_per_epoch:,}, Optimizer steps per epoch: {steps_for_one_epoch:,}")

    config.max_steps = steps_for_one_epoch

    start_step, best_val_loss = 0, float('inf')
    best_ckpts_file = Path(CHECKPOINT_DIR) / "best_checkpoints.json"
    best_checkpoints = []

    if best_ckpts_file.exists():
        best_checkpoints = json.load(open(best_ckpts_file))
        if best_checkpoints:
            best_ckpt = sorted(best_checkpoints, key=lambda x: x["val_loss"])[0]
            try:
                ckpt = torch.load(best_ckpt["path"], map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                scaler.load_state_dict(ckpt['scaler_state_dict'])
                start_step = ckpt['step'] + 1
                best_val_loss = ckpt.get('best_val_loss', float('inf'))
                logger.info(f"Resumed from best checkpoint {best_ckpt['path']}, step={start_step}, best_val_loss={best_val_loss:.4f}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load best checkpoint {best_ckpt['path']}: {e}. Starting fresh.")

    model.train()
    step = start_step
    pbar = tqdm(total=config.max_steps, desc="Training", initial=start_step)
    train_iter = iter(train_loader)

    while step < config.max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            logger.info("End of dataset epoch, restarting iterator.")
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with autocast(enabled=config.use_amp, dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1), ignore_index=-1)
            loss = loss / config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if step % 10 == 0 or step == start_step:
            current_loss_item = loss.item() * config.gradient_accumulation_steps
            pbar.set_postfix({
                'loss': f'{current_loss_item:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

        pbar.update(1)

        if (step > 0 and step % config.eval_every == 0):
            eval_metrics = evaluate_model(model, val_loader, config, device)
            logger.info(f"Step {step}: Val Loss {eval_metrics['val_loss']:.4f}, PPL {eval_metrics['val_perplexity']:.2f}, Acc {eval_metrics['val_accuracy']:.4f}")

            ckpt_state = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': asdict(config),
                'best_val_loss': best_val_loss
            }

            ckpt_path = Path(CHECKPOINT_DIR) / f"best_checkpoint-{step}.pt"
            save_checkpoint(ckpt_state, ckpt_path)

            best_checkpoints.append({
                "step": step,
                "val_loss": eval_metrics['val_loss'],
                "path": str(ckpt_path)
            })

            # Sort all known checkpoints by val_loss, decide keep vs delete first
            all_ckpts_sorted = sorted(best_checkpoints, key=lambda x: x["val_loss"])
            to_keep = all_ckpts_sorted[:10]
            to_delete = all_ckpts_sorted[10:]

            # Persist only the top-10
            best_checkpoints = to_keep

            # Remove older, worse checkpoints from disk
            for ckpt in to_delete:
                path = ckpt.get("path")
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                        logger.info(f"Removed old checkpoint {path}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {path}: {e}")

            json.dump(best_checkpoints, open(best_ckpts_file, "w"), indent=2)

        step += 1

    pbar.close()
    logger.info("TRAINING COMPLETED!")


if __name__ == "__main__":
    train()




