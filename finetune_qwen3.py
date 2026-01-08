import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# --- Configuration for Your Specific Paths and Files ---
# CHANGE THESE PATHS to match your local file structure
PRETRAINED_CKPT_PATH = r"/sd1/chaithra/lm/new/checkpoints/best_checkpoint-27000.pt" # Direct path to your pre-trained checkpoint
# FINETUNE_DATA_PATH = r"/sd1/chaithra/lm/t1_t2_combined_data.json" # Your combined English/Telugu chat data
TOKENIZER_PATH = r"/sd1/chaithra/lm/BPE_tokenizer/bpe_tokenizer.json"
FINAL_PEFT_OUTPUT_DIR = r"/sd1/chaithra/lm/new/latest_fc"# Separate directory for saving the fine-tuned adapters

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # console
        logging.FileHandler("/sd1/chaithra/lm/new/latest_finetuning.log", mode="a")  # file
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# --- Helper Functions and Classes (from your pre-training code) ---

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
class ModelConfig(dict):
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 2048
    n_kv_heads: int = 4
    rms_norm_eps: float = 1e-6
    max_seq_len: int = 2048 
    vocab_size: int = 64000
    num_epochs: int = 5 
    batch_size: int = 16
    max_steps: int = 10000
    learning_rate: float = 3e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    dropout: float = 0.1
    use_amp: bool = True
    gradient_accumulation_steps: int = 10
    eval_every: int = 1000
    eval_steps: int = 100
    
    # PEFT Specifics
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Add model_type for PEFT compatibility
    model_type: str = "llama"
    tie_word_embeddings: bool = True  # Add this for PEFT compatibility
    
    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        
        # Initialize dict functionality
        super().__init__(**asdict(self))
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
        super().__setitem__(key, value)

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
        # Corrected order: d_model -> d_ff
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        # Corrected order: d_ff -> d_model
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
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
        # Store config as an object attribute for PEFT compatibility
        self.config = config
        # Create a dict version of config for backward compatibility
        self.config_dict = asdict(config)
        
        if self.config.vocab_size is None:
            raise ValueError("vocab_size must be set in ModelConfig")
            
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.position_dropout = nn.Dropout(self.config.dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.config) for _ in range(self.config.n_layers)
        ])
        self.norm = nn.RMSNorm(self.config.d_model, eps=self.config.rms_norm_eps)
        self.output_dropout = nn.Dropout(self.config.dropout)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight.to(self.lm_head.weight.dtype)
        
        # Add model_type for PEFT compatibility
        self.config.model_type = "llama"  # Using 'llama' as it's similar to your architecture
        
        # Add generation config for PEFT compatibility
        self.generation_config = {
            "max_length": self.config.max_seq_len,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Handle both direct tensor input and dictionary input
        if input_ids is None and 'input_ids' in kwargs:
            input_ids = kwargs['input_ids']
        
        # If input_ids is a dictionary, extract the tensor
        if isinstance(input_ids, dict):
            input_ids = input_ids['input_ids']
            
        x = self.token_embedding(input_ids)
        x = self.position_dropout(x)
        
        for block in self.transformer_blocks:
            x = block(x)
            
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        
        # Return a dictionary with logits for PEFT compatibility
        return {"logits": logits}
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        Prepare inputs for generation. This method is required by PEFT.
        """
        model_inputs = {"input_ids": input_ids}
        return model_inputs

# --- NEW: Dataset Class for Fine-tuning ---
class FinetuneDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: Tokenizer, max_seq_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            for entry in raw_data:
                text = entry.get("text")
                if text:
                    self.data.append(text)
            
            if not self.data:
                raise ValueError("JSON file is empty or missing 'text' fields.")

            logger.info(f"Loaded {len(self.data)} samples from '{file_path}'.")
        
        except FileNotFoundError:
            raise FileNotFoundError(f"FATAL: Fine-tuning data file not found at '{file_path}'")
        except json.JSONDecodeError:
            raise ValueError(f"FATAL: Failed to parse JSON file at '{file_path}'. Check if the file is valid.")
        except Exception as e:
            raise Exception(f"An error occurred while loading data: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        text = self.data[idx]
        tokens = self.tokenizer.encode(text).ids
        
        # Truncate if too long
        tokens = tokens[:self.max_seq_len]
        
        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        return {"input_ids": input_ids}

def collate_fn(batch: List[dict]) -> dict:
    """Pads a batch of tensors to the same length."""
    # Find the maximum length in the current batch
    max_len = max(len(item["input_ids"]) for item in batch)
    
    padded_input_ids = []
    
    for item in batch:
        input_ids = item["input_ids"]
        # Pad with the tokenizer's padding token
        padding_needed = max_len - len(input_ids)
        padded_ids = F.pad(input_ids, (0, padding_needed), value=0) # Assuming 0 is the pad_token_id
        padded_input_ids.append(padded_ids)
        
    input_ids = torch.stack(padded_input_ids)
    
    # Correctly create labels for causal language modeling
    # Labels are the same as input_ids but shifted by one position.
    # The last token in each sequence has no 'next token' to predict, so we use -100 as the ignore index.
    labels = torch.cat([input_ids[:, 1:], torch.full((input_ids.shape[0], 1), -100, dtype=torch.long, device=input_ids.device)], dim=-1)
    
    # Create attention mask
    attention_mask = (input_ids != 0).long()
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

# --- Fine-tuning Loop ---

# ------------------ Evaluation Function ------------------
def evaluate_peft(peft_model, tokenizer, test_file, config, device):
    peft_model.eval()
    test_dataset = FinetuneDataset(test_file, tokenizer, config.max_seq_len)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)

            outputs = peft_model(input_ids=x, attention_mask=attn_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs

            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1), ignore_index=-100)
            total_loss += loss.item() * x.size(0)  # multiply by batch size

    avg_loss = total_loss / len(test_dataset)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    logger.info(f"\n=== Final Test Results ===\nTest Loss: {avg_loss:.4f}, Test PPL: {ppl:.2f}")
    print(f"\n=== Final Test Results ===\nTest Loss: {avg_loss:.4f}, Test PPL: {ppl:.2f}")


# ------------------ Modified Fine-tuning Loop ------------------
def train_peft_with_test(train_file, test_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    set_seed(42)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    actual_vocab_size = tokenizer.get_vocab_size()
    
    # Load model and checkpoint
    config = ModelConfig(vocab_size=actual_vocab_size)
    model = MinimalLLM(config)
    ckpt = torch.load(PRETRAINED_CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    # LoRA PEFT
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    )
    peft_model = get_peft_model(model, peft_config, adapter_name="default")
    peft_model.print_trainable_parameters()

    # Load training data
    train_dataset = FinetuneDataset(train_file, tokenizer, config.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=config.learning_rate)
    scaler = GradScaler(enabled=config.use_amp)

    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = total_steps // 20
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / warmup_steps)
    )

    # --- Training Loop ---
    peft_model.train()
    step = 0
    logger.info("Starting fine-tuning...")

    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        total_loss = 0.0

        for batch in pbar:
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)

            with autocast(enabled=config.use_amp, dtype=torch.bfloat16):
                outputs = peft_model(input_ids=x, attention_mask=attn_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1), ignore_index=-100)

            total_loss += loss.item()
            scaler.scale(loss).backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            step += 1
            perplexity = torch.exp(torch.tensor(loss.item())).item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'ppl': f'{perplexity:.2f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        avg_epoch_loss = total_loss / len(train_loader)
        avg_epoch_ppl = torch.exp(torch.tensor(avg_epoch_loss)).item()
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}, Average PPL: {avg_epoch_ppl:.2f}, Time: {epoch_time:.2f}s")

    pbar.close()
    logger.info("FINE-TUNING COMPLETED!")

    # --- Save LoRA adapters ---
    os.makedirs(FINAL_PEFT_OUTPUT_DIR, exist_ok=True)
    peft_model.save_pretrained(FINAL_PEFT_OUTPUT_DIR)
    logger.info(f"Final PEFT adapters saved to '{FINAL_PEFT_OUTPUT_DIR}'")

    # --- Evaluate on Test Set ---
    evaluate_peft(peft_model, tokenizer, test_file, config, device)


# ------------------ Run Training + Test ------------------
if __name__ == "__main__":
    TRAIN_FILE = r"/sd1/chaithra/lm/train.json"
    TEST_FILE = r"/sd1/chaithra/lm/test.json"

    train_peft_with_test(TRAIN_FILE, TEST_FILE)
