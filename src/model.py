import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.nn.functional as F
import math
import json
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from typing import List, Dict
from contextlib import nullcontext
from collections import Counter
from utils import strip_bos_eos_pad, print_val, aggregate_text_metrics, prf, compute_val_loss, \
    sanity_overfit_one_batch


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / 'data'
TOK_DIR     = ROOT / 'tokenizer'
RUNS_DIR    = ROOT / 'runs'
WEIGHTS_DIR = ROOT / 'weights'

for _d in (DATA_DIR, TOK_DIR, RUNS_DIR, WEIGHTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# We define settings for building and training the transformer model
def get_config():
    return {
        'data_dir': str(DATA_DIR),

        # Training parameters
        'batch_size': 8,
        'num_epochs': 80,
        'lr': 1.0e-4 / 25,
        'lr_max': 1.0e-4,
        'div_factor': 25.0,
        'final_div_factor': 1e6,
        'warmup_ratio': 0.05,
        'accum_steps': 1,

        # Transformer parameters
        'N': 4,
        'h': 8,
        'dropout': 0.1,
        'label_smoothing': 0.1,
        'seq_len': 512,
        'd_model': 512,

        # MHA with relative position encoding
        'num_buckets': 32,
        'max_distance': 128,

        # MLA with Decoupled RoPE
        'use_mla': False, # If True, uses MLA instead of MHA
        'rope_d_r': 16,

        # Pattern of Interleaved Attention Layers (True = MLA, False = MHA)
        'interleave_pattern': None,  # [True, False, True, False]
        # If Interleave pattern is set, the model uses the chosen pattern

        # Paths for saving and preload the model
        'model_folder': str(WEIGHTS_DIR),
        'model_basename': 'nanosocrates_',
        'preload': None,

        # Paths for tokenizer and runs
        'tokenizer_path': str(TOK_DIR),
        'experiment_name': str(RUNS_DIR / "nanosocrates"),

        # Setting for performing the sanity check
        'sanity_one_batch': False, # If True, starts a sanity check
        'task_types': ['rdf2text', 'text2rdf', 'rdf_completion_1', 'rdf_completion_2'],

        # Settings to perform a full evaluation every 20 epochs and a fast one on the other epochs.
        # If we want to perform always the full evaluation, we can simply set 'eval_profile'
        # to 'FULL'
        'eval_profile': 'FAST',
        'eval_max_batches': 3,
        'eval_max_text': 100,
        'eval_max_triples': 800,
        'eval_max_new_tokens_fast': 64,
        'eval_max_rc1': 48,
        'eval_skip_meteor_fast': True,
    }


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model  # Embedding vector dimension per token (512)
        self.vocab_size = vocab_size  # Dimension of the vocabulary obtained from the tokenizer
        self.embedding = nn.Embedding(vocab_size,
                                      d_model)  # Pytorch layer that converts indices in dense embeddings

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  # Used to normalise embeddings variance


# RoPE (Rotary Positional Embedding) is a technique used to encode the position rotating query/key channel pairs.
# For each couple of dimensions, we apply a rotation that depends on the position p, calculating theta = p * w_i.
class RoPE(nn.Module):
    def __init__(self, d_rotary: int, base: float = 10000.0, precompute_len: int | None = None):
        super().__init__()
        assert d_rotary % 2 == 0, "d_rotary must be even"  # Because each couple must have 2 elements
        self.d_rotary = d_rotary  # d_rotary defines the number of channels we use, for each head, in the rotary part.
        # So we can also calculate d_c (the part referred to the content) as d_c = d_head - d_r
        self.base = base
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)
        if precompute_len is not None:
            self._build_cache(precompute_len, device=torch.device("cpu"), dtype=torch.float32)

    @torch.no_grad()
    def _build_cache(self, seq_len: int, device, dtype):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.d_rotary, 2, device=device, dtype=dtype) / self.d_rotary))
        # Creates the inv_freq w_i with exponential decay, just like in Positional encoding
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # p * w_i as we said before, or
        # freqs[p, f] = t[p] * inv_freq[f], so the phase associated to position p and frequency f
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _ensure_cache(self, need_len: int, device, dtype):
        if self.cos.numel() == 0 or self.cos.size(2) < need_len or self.cos.device != device:
            # Checks if no cache was built, or if the cache is too short, or if the cache is on another device
            self._build_cache(need_len, device=device, dtype=torch.float32)  # If at least one of the previous
            # conditions is true, build the cache again

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0):
        B, H, S, D = x.shape  # We assume that the input tensor is of dimensions B (Batch), H (number of heads), S (seq_len),
        # D (dimension of the embedding)
        cos = cos[:, :, start_pos:start_pos + S, :]
        sin = sin[:, :, start_pos:start_pos + S, :]
        x_ = x.float().reshape(B, H, S, D // 2, 2)  # D is d_rotary, so we split d in two, making the couples

        x_even, x_odd = x_[..., 0], x_[..., 1]
        # Here we apply the rotation to each even/odd couple
        xr_even = x_even * cos - x_odd * sin
        xr_odd = x_odd * cos + x_even * sin
        xr = torch.stack([xr_even, xr_odd], dim=-1).flatten(
            -2)  # We stack xr_even and xr_odd so that we get back to (B,H,S,D)
        return xr.type_as(x)

    def apply(self, x: torch.Tensor, start_pos: int = 0):

        # We use apply function to actually use RoPE, building up all the function that we described before
        assert x.size(-1) == self.d_rotary, "Last dim of x must be d_rotary"

        self._ensure_cache(start_pos + x.size(2), x.device, x.dtype)
        return self._apply_rope(x, self.cos, self.sin, start_pos)


# Positional Encoding for standard MHA. The idea is that we want to give the model an idea of the positioning
# of the tokens in the sequence, so we sum Positional Encoding to the embeddings of the tokens
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Dimensionality of the model
        self.seq_len = seq_len  # Maximum sequence length
        self.dropout = nn.Dropout(dropout)  # Dropout layer to prevent overfitting

        # Creating a positional encoding matrix of shape (seq_len, d_model) filled with zeros
        pe = torch.zeros(seq_len, d_model)  # First position of the sequence -> embedding associated to that token

        # Creating a tensor representing positions (0 to seq_len - 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1)  # Transforming 'position' into a 2D tensor['seq_len, 1']

        # Creating the division term for the positional encoding formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices in pe
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in pe
        pe[:, 1::2] = torch.cos(position * div_term)

        # Adding an extra dimension at the beginning of pe matrix for batch handling
        pe = pe.unsqueeze(0)

        # Registering 'pe' as buffer. Buffer is a tensor not considered as a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Adding positional encoding to the input tensor X
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)  # Dropout for regularization


# Creating Layer Normalization. Layer normalization is used to normalize the inputs for the subsequent layers,
# so that the model can correctly learn avoiding wasting time trying to correct the normalization
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10 ** -6) -> None:  # We define epsilon as 0.000001 to avoid division by zero
        super().__init__()
        self.eps = eps

        # We define alpha as a trainable parameter and initialize it with ones
        self.alpha = nn.Parameter(torch.ones(1))  # One-dimensional tensor that will be used to scale the input data

        # We define bias as a trainable parameter and initialize it with zeros
        self.bias = nn.Parameter(torch.zeros(1))  # One-dimensional tenso that will be added to the input data

    def forward(self, x):
        mean = x.mean(dim=-1,
                      keepdim=True)  # Computing the mean of the input data. Keeping the number of dimensions unchanged
        std = x.std(dim=-1,
                    keepdim=True)  # Computing the standard deviation of the input data. Keeping the number of dimensions unchanged

        # Returning the normalized input
        return self.alpha * (x - mean) / (std + self.eps) + self.bias



# We can find the FeedForwardBlock both in the Encoder and in the Decoder part of the Transformer after the Attention. It is
# used to further process the embeddings, getting a more meaningful representation.
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # First linear transformation
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 & b1
        self.dropout = nn.Dropout(dropout)  # Dropout to prevent overfitting
        # Second linear transformation
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 & b2

    def forward(self, x):
        # The dimensions of the matrices are (batch, seq_len, d_model) --> after the first linear (batch, seq_len, d_ff)
        # --> after the second linear (batch, seq_len, d_model)
        return self.linear_2(self.dropout(F.gelu(self.linear_1(x))))



# MultiHeadAttention is used to give the model an idea of the context in which the words are used. Each embedding carries
# information about the token it represents, but also about all the other token in the sequence. Based on the score obtained
# by matching the query of the token we are considering and the keys of all the other tokens, we can evaluate the importance
# of each token in the sequence. In this implementation, we consider relative position bias, inspired from the implementation
# of Hugging Face T5. Relative position bias adds a bias dependent from the relative distance j-i, discretized in buckets,
# so the attention knows if we're looking far or close, and if on the left or on the right
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float,
                 num_buckets: int = 32, max_distance: int = 128, bidirectional: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, 'd_model is not divisible by h'
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        # Bias per-bucket per-head (like T5)
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.h) # For each bucket of distance, we create
        # a bias of dimension h (one for each head)

    @staticmethod
    # Maps the relative distance (j-i) in a number in range [0, num_buckets -1]
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        orig_num_buckets = num_buckets
        if bidirectional: # If we can look in both directions. It will be false in case of decoder self-attention.
            num_buckets = num_buckets // 2
            offset = (relative_position > 0).to(torch.long) * num_buckets # Creates an offset that moves the indices of the buckets
            # of positives distances in the second half of the total interval.
            # With relative_position > 0, we get True where j - i > 0, converted in long becomes 0 or 1.
            # Basically, offset = 0 for distances <= 0, while offset = num_buckets for distances > 0.
            n = relative_position.abs()
        else:
            # Only the positions <= 0 are allowed, so we set the positives to 0
            n = (-torch.min(relative_position, torch.zeros_like(relative_position))).to(relative_position.dtype)
            # n returns, element-wise, the minimum between j-i and 0, so:
            # - if j-i <= 0, returns j-i,
            # - j-i > 0, returns 0
            # The - is used to get n = -(j-i) = i - j

        max_exact = max(num_buckets // 2, 1) # Considering each half in case of bidirectional = True
        is_small = n < max_exact # For each direction (if bidirectional), we consider half of the buckets
        # for little distances, and the remaining half for big distances

        # We do this because if max_distance <= max_exact, div = max_distance / max_exact <= 1, so log(div) <= 0
        denom = math.log(max(max_distance / float(max_exact), 1.000001))
        n_float = n.to(torch.float32)
        # We calculate the large part
        n_large = n_float.clamp_min(1.0) # We avoid log(0) in case of n = 0
        val_large = max_exact + (torch.log(n_large / max_exact) / denom) * (num_buckets - max_exact) # We map
        # the resulting buckets [max_exact, ..., num_buckets - 1]
        val_large = val_large.to(torch.long).clamp(max=num_buckets - 1) # clamp allows to saturate everything that exceeds
        # the expected scale in the last logaritmic bucket
        val = torch.where(is_small, n.to(torch.long), val_large)
        # Where is_small = True (n < max_exact), val = n, so it's linear
        # Where is_small == False, we use log, so val = val_large
        if bidirectional:
            val = val + offset
        return val.clamp(0, orig_num_buckets - 1)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout, pos_bias=None):
        d_k = query.shape[-1]
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # (B,H,Q,K)

        if pos_bias is not None:
            # pos_bias: (H,Q,K)
            scores = scores + pos_bias.unsqueeze(0)  # In this way we get the dimensions (B,H,Q,K)

        if mask is not None:
            m = mask if mask.dtype == torch.bool else (mask != 0)
            scores = scores.masked_fill(~m, -1e9)
            # We manage the case Q with no valid K (so, all the row is masked)
            no_valid = ~m.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(no_valid, 0.0)

        # Stable softmax in fp32
        scores_max = scores.amax(dim=-1, keepdim=True) # Calculates the max along the key axis for each row of the
        # attention matrix. The main idea is to avoid overflow or underflow. For example:
        # scores = [1000, 995, 980], if I subtract 1000 I get [0, -5, -20], and doing the softmax I get a correct
        # vector of probabilities
        attn = torch.softmax((scores - scores_max).to(torch.float32), dim=-1).to(scores.dtype)

        if dropout is not None:
            attn = dropout(attn)

        out = attn @ value
        return out, attn

    def _build_rel_pos_bias(self, Q: int, K: int, device, position_offset: int = 0):
        # Here we are building the relative position to pass to _relative_position_bucket
        q_pos = torch.arange(position_offset, position_offset + Q, device=device) # context_position, i
        k_pos = torch.arange(0, K, device=device) # memory_position, j
        # relative = j - i, with shape (Q,K)
        rel = k_pos[None, :] - q_pos[:, None]
        buckets = self._relative_position_bucket(
            rel, bidirectional=self.bidirectional,
            num_buckets=self.num_buckets, max_distance=self.max_distance
        )  # (Q,K)
        # embedding: (Q,K,H), permute to (H,Q,K)
        pos_bias = self.relative_attention_bias(buckets).permute(2, 0, 1).contiguous()
        return pos_bias  # (H,Q,K)

    def forward(self, q, k, v, mask: torch.Tensor | None = None,
                *, use_cache: bool = False, cache: dict | None = None, position_offset: int = 0):
        B, Q, _ = q.shape
        _, K_new, _ = k.shape
        d_k = self.d_k

        # Projections
        qh = self.w_q(q).view(B, Q, self.h, d_k).transpose(1, 2).contiguous()
        k_new = self.w_k(k).view(B, K_new, self.h, d_k).transpose(1, 2).contiguous()
        v_new = self.w_v(v).view(B, K_new, self.h, d_k).transpose(1, 2).contiguous()

        # We use the cache if the flag is true, saving the K an V matrices. If cache is empty, saves k_new, v_new as
        # k_all and v_all
        if use_cache:
            if cache is None:
                k_all, v_all = k_new, v_new
            else:
                k_all = torch.cat([cache['k'], k_new], dim=2)
                v_all = torch.cat([cache['v'], v_new], dim=2)
            new_cache = {'k': k_all, 'v': v_all}
        else:
            k_all, v_all = k_new, v_new
            new_cache = None

        m = None
        if mask is not None:
            m = mask if mask.dtype == torch.bool else (mask != 0)
            if m.dim() == 2:   # (B,K)
                m = m[:, None, None, :]
            elif m.dim() == 3: # (B,Q,K)
                m = m[:, None, :, :]

        # We create the relative position bias using the function we defined before
        pos_bias = self._build_rel_pos_bias(Q, k_all.size(2), q.device, position_offset)

        # We call the attention mechanism
        ctx, _ = self.attention(qh, k_all, v_all, m, self.dropout, pos_bias=pos_bias)

        # We merge the heads and project to have only one representation
        ctx = ctx.transpose(1, 2).contiguous().view(B, Q, self.h * d_k)
        out = self.w_o(ctx)

        return (out, new_cache) if use_cache else out



# Multihead Latent Attention with DecoupledRoPE. It is implemented from the paper "DeepSeek-V2: A Strong, Economical, and Efficient
# Mixture-of-Experts Language Model".
# Multihead Latent Attention is implemented with the idea of surpassing the bottleneck of KV-cache (used in autoregressive decode
# to avoid calculating every time the K-V matrices by storing them) so we basically
# compress and de-compress the K and V matrices in order to have less memory occupation.
# We have a problem with RoPE: if we multiply the rotary matrix when we up-project the input, we obtain a result which is
# dependent by the position. In autoregressive inference, when we add a new token the relation of the position of all context token
# can change, and so we cannot simply up-project the matrix as we should multiply every time by the Rotary matrix.
# The intuition for Decoupled RoPE is to split the space in two parts: content and rotary. For each head h we divide
# the dimensions in d_r (rotary) and d_c (content). In this way, we can apply RoPE on the rotary part, while preserving the
# content.
class MLADecoupledRoPEBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_r: int = 16, d_latent: int = 128,
                 max_seq_len: int = 4096, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert d_r > 0 and d_r % 2 == 0, "d_r must be > 0 and even"
        assert d_latent > 0, "d_latent must be > 0"

        self.d_model = d_model
        self.h = n_heads
        self.d_c = d_model // n_heads  # per-head content
        self.d_r = d_r  # per-head rotary
        self.d_latent = d_latent
        self.scale = 1.0 / math.sqrt(self.d_c + self.d_r)

        # Query content
        self.WqC = nn.Linear(d_model, n_heads * self.d_c, bias=False)

        # Down-projection yields c (compact) from k (d_model -> d_latent)
        self.W_DKV = nn.Linear(d_model, d_latent, bias=False)

        # Up-projection expands the latent to (n_heads * d_c)
        self.W_UK = nn.Linear(d_latent, n_heads * self.d_c, bias=False)
        self.W_UV = nn.Linear(d_latent, n_heads * self.d_c, bias=False)

        # Rotary projections that yield WqR per-head, WkR shared
        self.WqR = nn.Linear(d_model, n_heads * self.d_r, bias=False)
        self.WkR = nn.Linear(d_model, self.d_r, bias=False)  # Shared across heads
        self.rope = RoPE(d_rotary=self.d_r, base=rope_base, precompute_len=max_seq_len)

        # Output projection (only content aggregated)
        self.Wo = nn.Linear(n_heads * self.d_c, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None,
                *, use_cache: bool = False, cache: dict | None = None, position_offset: int = 0):
        B, Q, _ = q.shape
        _, K_new, _ = k.shape

        # Query content per head
        qC = self.WqC(q).view(B, Q, self.h, self.d_c).transpose(1, 2)  # (B,H,Q,d_c)

        c_new = self.W_DKV(k)  # (B, K_new, d_latent)

        # We compute raw kR (B, K_new, d_r)
        kR_new_raw = self.WkR(k)  # (B, K_new, d_r)
        # RoPE.apply expects (B, H, S, D), so we add head-dim H=1, apply RoPE, then remove head-dim to store compactly
        kR_new = self.rope.apply(kR_new_raw.unsqueeze(1), start_pos=position_offset).squeeze(1)  # (B, K_new, d_r)

        if use_cache:
            if cache is None:
                c_all = c_new
                kR_all = kR_new
            else:
                # We store the compact vector obtained from k
                c_all = torch.cat([cache['c'], c_new], dim=1)  # (B, K_all, d_latent)
                kR_all = torch.cat([cache['kR'], kR_new], dim=1)  # (B, K_all, d_r)
            new_cache = {'c': c_all, 'kR': kR_all}

            # We expand c_all to K/V per-head only when is needed
            kC = self.W_UK(c_all).view(B, c_all.size(1), self.h, self.d_c).transpose(1, 2)  # (B,H,K_all,d_c)
            vC = self.W_UV(c_all).view(B, c_all.size(1), self.h, self.d_c).transpose(1, 2)  # (B,H,K_all,d_c)

            # kR_all -> (B,H,K_all,d_r)
            kR = kR_all.unsqueeze(1).expand(-1, self.h, -1, -1)  # (B,H,K_all,d_r)
        else:
            # If we don't use cache, we don't store the matrices K and V compressed
            kC = self.W_UK(c_new).view(B, K_new, self.h, self.d_c).transpose(1, 2)  # (B,H,K_new,d_c)
            vC = self.W_UV(c_new).view(B, K_new, self.h, self.d_c).transpose(1, 2)  # (B,H,K_new,d_c)
            kR = kR_new.unsqueeze(1).expand(-1, self.h, -1, -1)
            new_cache = None

        # We apply Rope to the query
        qR = self.WqR(q).view(B, Q, self.h, self.d_r).transpose(1, 2)  # (B,H,Q,d_r)
        qR = self.rope.apply(qR,
                             start_pos=position_offset)

        # We concatenate the content and rotary part and we compute attention
        Qh = torch.cat([qC, qR], dim=-1)  # (B,H,Q,d_c+d_r)
        Kh = torch.cat([kC, kR], dim=-1)  # (B,H,K_all,d_c+d_r)

        # We calculate the score, just as in standard Multi Head Attention
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) * self.scale
        scores = scores.to(torch.float32)


        no_valid = None
        if mask is not None:
            m = mask
            if m.dtype != torch.bool:
                m = (m != 0)
            if m.dim() == 2:
                m = m[:, None, None, :]  # (B,1,1,K_all)
            elif m.dim() == 3:
                m = m[:, None, :, :]  # (B,1,Q,K_all)
            scores = scores.masked_fill(~m, -1e9)
            no_valid = ~m.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(no_valid, 0.0)

        # We normalise the scores, like in MHA
        maxes = scores.amax(dim=-1, keepdim=True)
        scores = (scores - maxes).to(torch.float32)
        attn = torch.softmax(scores, dim=-1)

        if no_valid is not None:
            attn = attn.masked_fill(no_valid, 0.0)

        attn = attn.to(vC.dtype)
        attn = self.drop(attn)

        # The context is calculated only with the content part
        ctx = torch.matmul(attn, vC)  # (B,H,Q,d_c)
        ctx = ctx.transpose(1, 2).contiguous().view(B, Q, self.h * self.d_c)  # (B,Q,H*d_c)
        out = self.Wo(ctx)  # (B,Q,d_model)

        if use_cache:
            return out, new_cache
        else:
            return out


# Building Residual Connection. Residual connections are used to sum the input to the output of each block.
# This helps address the problem of performance degradation in deep models by preserving and propagating early information.
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # We use a dropout layer to prevent overfitting
        self.norm = LayerNormalization()  # We use a normalization layer

    def forward(self, x, sublayer):
        # We normalize the input and add it to the original input 'x'. This creates the residual connection process.
        return x + self.dropout(sublayer(self.norm(x)))


# Building Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        # Storing the self-attention block and feed-forward block
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)])  # 2 Residual Connections with dropout

    def forward(self, x, src_mask):
        # Applying the first residual connection with the self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x,
                                                                                src_mask))  # Three 'x's corresponding to query, key, and value inputs plus source mask

        # Applying the second residual connection with the feed-forward block
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x  # Output tensor after applying self-attention and feed-forward layers with residual connections.


# Building Encoder
# An Encoder can have several Encoder Blocks
class Encoder(nn.Module):
    # The Encoder takes in instances of 'EncoderBlock'
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # Storing the EncoderBlocks
        self.norm = LayerNormalization()  # Layer for the normalization of the output of the encoder layers

    def forward(self, x, mask):
        # Iterating over each EncoderBlock stored in self.layers
        for layer in self.layers:
            x = layer(x, mask)  # Applying each EncoderBlock to the input tensor 'x'
        return self.norm(x)  # Normalizing output


# Building Decoder Block
class DecoderBlock(nn.Module):
    # The DecoderBlock takes in two MultiHeadAttentionBlock. One is self-attention, while the other is cross-attention,
    # obtaining informations from the Encoder stack.
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)])  # List of three Residual Connections with dropout rate

    def forward(self, x, encoder_output, src_mask, tgt_mask, *, use_cache: bool = False, cache: dict | None = None,
                position_offset: int = 0):

        if not use_cache:
            x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
            x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                     src_mask))
            x = self.residual_connections[2](x, self.feed_forward_block)
            return x
        else:
            normed = self.residual_connections[0].norm(x)
            out_self, new_cache = self.self_attention_block(normed, normed, normed, tgt_mask, use_cache=True,
                                                            cache=cache, position_offset=position_offset)
            x = x + self.residual_connections[0].dropout(out_self)
            # We do this to get new_cache, that is returned from the self_attention_block

            x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                     src_mask))

            x = self.residual_connections[2](x, self.feed_forward_block)
            return x, new_cache


# Building Decoder
class Decoder(nn.Module):
    # The Decoder takes in instances of 'DecoderBlock'
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        # Storing the 'DecoderBlock's
        self.layers = layers
        self.norm = LayerNormalization()  # Layer to normalize the output

    def forward(self, x, encoder_output, src_mask, tgt_mask, *, use_cache: bool = False,
                layer_caches: list | None = None, position_offset: int = 0):

        if not use_cache:
            for layer in self.layers:
                x = layer(x, encoder_output, src_mask, tgt_mask)
            return self.norm(x)
        else:
            new_layer_caches = []
            for i, layer in enumerate(self.layers): # For each layer, I retrieve the cache if it exists
                cache_i = None
                if layer_caches is not None:
                    cache_i = layer_caches[i]
                x, new_cache = layer(x, encoder_output, src_mask, tgt_mask, use_cache=True, cache=cache_i,
                                     position_offset=position_offset) # Informs about how many positions have already been emitted
                new_layer_caches.append(new_cache) # Appends the updated cache
            return self.norm(x), new_layer_caches
# The path for the cache is MultiHeadAttention -> Decoder Block -> Decoder -> Transformer -> build_transformer
# And for the update, build_transformer -> Transformer -> Decoder -> Decoder Block -> MultiHeadAttention.
# It's at the Decoder level that the cache effectively enters the game


# Buiding Linear Layer, it projects the output of the Decoder Stack in an higher dimensional space, allowing to
# process more information.
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:  # Model dimension and the size of the output vocabulary
        super().__init__()
        self.proj = nn.Linear(d_model,
                              vocab_size)  # Linear layer for projecting the feature space of 'd_model' to the output space of 'vocab_size'

    def forward(self, x):
        return self.proj(x)  # Applying the log Softmax function to the output


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder  # Encoder taken from the previous Encoder class
        self.decoder = decoder  # Decoder taken from the previous Decoder class
        self.src_embed = src_embed  # Embeddings of the source
        self.tgt_embed = tgt_embed  # Embeddings of the target
        self.projection_layer = projection_layer

    # Encoder
    def encode(self, src, src_mask):
        src = self.src_embed(src)  # Applying source embeddings to the tokenized input
        return self.encoder(src,
                            src_mask)  # Returning the source embeddings plus a source mask to prevent attention to certain elements

    # Decoder
    def decode(self, encoder_output, src_mask, tgt, tgt_mask, *, use_cache: bool = False,
               layer_caches: list | None = None, position_offset: int = 0):

        tgt = self.tgt_embed(tgt)  # Always calculates the embeddings
        if not use_cache: # If we use chache, we pass the layer_caches and the position_offset
            return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        else:
            return self.decoder(tgt, encoder_output, src_mask, tgt_mask, use_cache=True, layer_caches=layer_caches,
                                position_offset=position_offset)

    # Applying Projection Layer to the Decoder output
    def project(self, x):
        return self.projection_layer(x)


# We use this function to build the transformer using Interleaved Attention Layers. This function is called only if
# an Interleave pattern is defined, otherwise we will use the function "build_transformer"
def build_transformer_interleaved(
        vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 512,
        N: int = 4,
        h: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048,
        rope_d_r: int = 16,
        interleave_pattern: list | tuple = None,  # Must be list/tuple of bool of length N
        num_buckets: int = 32, max_distance: int = 128
) -> Transformer:
    # We check if there is the pattern, and if it is correct
    if interleave_pattern is None:
        raise ValueError("interleave_pattern is required and must be a list/tuple of booleans of length N.")
    if not isinstance(interleave_pattern, (list, tuple)):
        raise TypeError("interleave_pattern must be a list or tuple of booleans.")
    if len(interleave_pattern) != N:
        raise ValueError(f"interleave_pattern length ({len(interleave_pattern)}) must equal N ({N}).")
    # We check that all the elements are bool
    for i, v in enumerate(interleave_pattern):
        if not isinstance(v, (bool, int)):
            raise TypeError(f"interleave_pattern[{i}] must be boolean-like (True/False). Got {type(v)}")

    # Explicit cast to bool list
    pattern = [bool(x) for x in interleave_pattern]

    # We create the embeddings
    src_embed = InputEmbeddings(d_model, vocab_size)
    tgt_embed = InputEmbeddings(d_model, vocab_size)

    # We build the encoder block. Pattern is made up of boolean elements (True or False), so we use this characteristic
    # to check if to use MLA or MHA.
    encoder_blocks = []
    for i in range(N):
        if pattern[i]:  # If True, use MLA
            self_attn = MLADecoupledRoPEBlock(
                d_model=d_model, n_heads=h, d_r=rope_d_r,
                max_seq_len=max(src_seq_len, tgt_seq_len), dropout=dropout
            )
        else:  # If False, use standard MHA
            self_attn = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout,
                                                num_buckets=num_buckets, max_distance=max_distance, bidirectional=True)
        ff = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(self_attn, ff, dropout))

    # We do the same thing for the decoder block. In this case, we set bidirectional to False in the self attention
    # because of the causal mask
    decoder_blocks = []
    for i in range(N):
        if pattern[i]:
            self_attn = MLADecoupledRoPEBlock(
                d_model=d_model, n_heads=h, d_r=rope_d_r,
                max_seq_len=tgt_seq_len, dropout=dropout
            )
            cross_attn = MLADecoupledRoPEBlock(
                d_model=d_model, n_heads=h, d_r=rope_d_r,
                max_seq_len=max(src_seq_len, tgt_seq_len), dropout=dropout
            )
        else:
            self_attn = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout,
                                                num_buckets=num_buckets, max_distance=max_distance, bidirectional=False)
            cross_attn = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout,
                                                 num_buckets=num_buckets, max_distance=max_distance, bidirectional=True)

        ff = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(self_attn, cross_attn, ff, dropout))

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, projection_layer)

    # Xavier initialization og the thetas
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


# This function is used to actually build the transformer using all the previous blocks. The structure of the Transformer
# is composed, in this implementation, with 4 Encoder layers and 4 Decoder layers, as suggested in the Nanosocrates guidelines.
def build_transformer(vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                      d_model: int = 512, N: int = 4, h: int = 8,
                      dropout: float = 0.1, d_ff: int = 2048,
                      use_mla: bool = True, rope_d_r: int = 16,
                      num_buckets: int = 32, max_distance: int = 128) -> Transformer:
    # Embeddings
    src_embed = InputEmbeddings(d_model, vocab_size)
    tgt_embed = InputEmbeddings(d_model, vocab_size)

    # We build the encoder stack, iterating N times and appending each time the Encoder block.
    encoder_blocks = []
    for _ in range(N):
        if use_mla:  # If use_mla is set to True, we pass as attn to the encoder the MLA, otherwise we will use
            # standard MHA
            self_attn = MLADecoupledRoPEBlock(
                d_model=d_model, n_heads=h, d_r=rope_d_r,
                max_seq_len=max(src_seq_len, tgt_seq_len),
                dropout=dropout
            )
        else:
            self_attn = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout,
                                                num_buckets=num_buckets, max_distance=max_distance, bidirectional=True)
        ff = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(self_attn, ff, dropout))

    # We build the decoder stack, iterating N times and appending each time the Decoder block.
    decoder_blocks = []
    for _ in range(N):
        if use_mla:  # Same as Encoder, if we have use_mla set to True, we pass as attn and cross_attn MLA, otherwise
            # we will pass standard MHA
            self_attn = MLADecoupledRoPEBlock(
                d_model=d_model, n_heads=h, d_r=rope_d_r,
                max_seq_len=tgt_seq_len, dropout=dropout
            )
            cross_attn = MLADecoupledRoPEBlock(
                d_model=d_model, n_heads=h, d_r=rope_d_r,
                max_seq_len=max(src_seq_len, tgt_seq_len), dropout=dropout
            )
        else:
            self_attn = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout,
                                                num_buckets=num_buckets, max_distance=max_distance, bidirectional=False)

            cross_attn = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout,
                                                 num_buckets=num_buckets, max_distance=max_distance, bidirectional=True)
        ff = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(self_attn, cross_attn, ff, dropout))

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, projection_layer)

    # We initialise the parameters following Xavier weights initialisation
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer


# Loading the Tokenizer. The tokenizer is trained on the corpus composed of the films selected from DBPedia, with the corresponding
# abstract from Wikipedia.
def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # We just check if all the special tokens are in the vocabulary
    special_tokens = ['<SOT>', '<EOT>', '<SUBJ>', '<PRED>', '<OBJ>',
                      '<RDF2Text>', '<Text2RDF>', '<CONTINUERDF>', '<MASK>']

    # If not, we add them to the vocabulary
    missing_tokens = []
    for token in special_tokens:
        if token not in tokenizer.get_vocab():
            missing_tokens.append(token)

    if missing_tokens:
        print(f"Warning: Missing special tokens: {missing_tokens}")
        tokenizer.add_special_tokens({'additional_special_tokens': missing_tokens})

    return tokenizer


def causal_mask(size):
    # Creating a square matrix of dimensions 'size x size' filled with ones.
    # In this way, we mask all the 'future' tokens for the decoder, so the ones in which j > i
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0  # We make it boolean, with True where mask == 0 and False otherwise


# Class for loading the dataset. The dataset is created from the database_creator.py code, here we load the dataset.jsonl
# file containing all the samples created from the films fetched from DBPedia.
class NanoSocratesDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 tokenizer: PreTrainedTokenizerFast,
                 seq_len: int = 512):

        self.data_dir = Path(data_dir)  # Path of the data
        self.tokenizer = tokenizer  # Tokenizer previously loaded
        self.seq_len = seq_len  # Length of the sequences of tokens of the model

        # Here we load the special tokens for padding, start of sequence and end of sequence.
        self.pad_token = int(tokenizer.convert_tokens_to_ids("[PAD]"))
        self.bos_token = int(tokenizer.convert_tokens_to_ids("[BOS]"))  # Start of sequence
        self.eos_token = int(tokenizer.convert_tokens_to_ids("[EOS]"))  # End of sequence
        ds = 'dataset'  # Name of the jsonl file containing the examples

        self.dataset = []
        self._load_data(ds)

        print(f"Loaded {len(self.dataset)} examples")

    def _load_data(self, ds):
        filename = f"{ds}.jsonl"
        filepath = self.data_dir / filename

        assert filepath.exists(), f"{filepath} does not exist"

        # We open the filepath, checking if there are empty lines and skipping them
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)  # We load the line
                example = {  # From the jsonl line, we take the 'input' and 'target', creating an example for the model
                    'input': data['input'],
                    'target': data['target']
                }

                self.dataset.append(example)  # We create the dataset by appending each example

    # Function used after in the code to get the dimension of the dataset
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):  # Function to get an item from the dataset
        ex = self.dataset[idx]
        src_text = ex.get('input')
        tgt_text = ex.get('target')

        # Here we initialise the ids relative to the special tokens. We do this to use the
        # ids to identify the task of the sample
        id_t2r = self.tokenizer.convert_tokens_to_ids("<Text2RDF>")
        id_r2t = self.tokenizer.convert_tokens_to_ids("<RDF2Text>")
        id_c2 = self.tokenizer.convert_tokens_to_ids("<CONTINUERDF>")
        id_mask = self.tokenizer.convert_tokens_to_ids("<MASK>")

        # Tokenization without special tokens
        enc = self.tokenizer.encode(src_text, add_special_tokens=False)
        dec = self.tokenizer.encode(tgt_text, add_special_tokens=False)

        # We check if the ids exist and if they are in the tokenized source text
        has_t2r = (id_t2r is not None) and (id_t2r in enc)
        has_r2t = (id_r2t is not None) and (id_r2t in enc)
        has_c2 = (id_c2 is not None) and (id_c2 in enc)
        has_mask = (id_mask is not None) and (id_mask in enc)

        # We classify the input based on token ids that are in the input.
        if has_c2:
            task_type = "rdf_completion_2"
        elif has_r2t:
            task_type = "rdf2text"
        elif has_t2r:
            task_type = "text2rdf"
        elif has_mask:
            task_type = "rdf_completion_1"
        else:
            task_type = ""

        # Truncate in case the sequence is too long, saving space for the special tokens
        max_len_enc = self.seq_len - 2
        max_len_dec = self.seq_len - 1

        enc = enc[:max_len_enc]
        dec = dec[:max_len_dec]

        # We now build the encoder input, decoder input and labels

        # Encoder input: [BOS] + enc + [EOS] + PAD...
        encoder_input = [self.bos_token] + enc + [self.eos_token]
        # We use the PAD token to reach the seq_len
        enc_pad = self.seq_len - len(encoder_input)
        if enc_pad > 0:
            encoder_input += [self.pad_token] * enc_pad

        # Decoder input: [BOS] + dec + PAD...
        decoder_input = [self.bos_token] + dec
        dec_pad = self.seq_len - len(decoder_input)
        if dec_pad > 0:
            decoder_input += [self.pad_token] * dec_pad

        # Labels: dec + [EOS] + PAD...
        label = dec + [self.eos_token]
        lab_pad = self.seq_len - len(label)
        if lab_pad > 0:
            label += [self.pad_token] * lab_pad

        # We cast the encoder_input, decoder_input and label to int64 (alias, LongTensor). This because some parts of the net
        # like nn.Embedding, nn.CrossEntropyLoss require int tensors.
        encoder_input = torch.tensor(encoder_input, dtype=torch.int64)
        decoder_input = torch.tensor(decoder_input, dtype=torch.int64)
        label = torch.tensor(label, dtype=torch.int64)

        # This is just a check to verify that the sizes of the encoder input, decoder input and labels are correct
        # so, if we added the right number of padding and special tokens
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'task_type': task_type,
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(
                decoder_input.size(0)),
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }


# We create the Dataloaders for training and test set, by splitting randomly the dataset. We use 90% of the dataset for
# training set and 10% for test set
def create_dataloaders(config):
    tokenizer = load_tokenizer(
        config['tokenizer_path'])  # We get the tokenizer and the size of the batches directly from
    # the config
    batch_size = config['batch_size']

    # We call the class NanoSocratesDataset that we defined before
    full_dataset = NanoSocratesDataset(
        data_dir=config['data_dir'],
        tokenizer=tokenizer,
        seq_len=config['seq_len'],
    )

    # We get the len of the dataset and divide it manually in training set and test set
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(  # Takes random indices from the dataset
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # We set the seed for reproducibility
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False
    )

    return train_loader, test_loader, tokenizer



# greedy_decode is used to generate the output sequence step by step, always conditioning on the tokens produced so far.
# "Greedy" means that at each step the model selects the single most probable next token.
# Other decoding strategies (like top-k sampling, nucleus sampling, or beam search) can also be used to introduce diversity or improve overall sequence quality.
def greedy_decode(model, source, source_mask, tokenizer, max_len, device):
    bos_idx = tokenizer.convert_tokens_to_ids('[BOS]')
    eos_idx = tokenizer.convert_tokens_to_ids('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(bos_idx).type_as(source).to(device)

    # Here we prepare the layer caches (one for each encoder / decoder layer)
    num_layers = len(model.decoder.layers)
    layer_caches = [None] * num_layers
    position = 0  # Absolute position of the first token we will pass (0 for BOS)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # We pass only the last token to the decoder in cache modality
        last_token = decoder_input[:, -1:].to(device)  # (1,1)
        decoder_mask = causal_mask(1).type_as(source_mask).to(device)  

        #  Returns out (B,1,d_model) and updated layer_caches
        out, layer_caches = model.decode(encoder_output, source_mask, last_token, decoder_mask,
                                         use_cache=True, layer_caches=layer_caches, position_offset=position)

        # We project the logits and choose the token with the highest probability
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        # We append the token
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(1).type_as(decoder_input)], dim=1)

        # We increment the absolute position, as we have just added one token
        position += 1

        # We stop only if EOS is the next_token id
        if next_word.item() == eos_idx:
            break

    return decoder_input[0].tolist()


def run_validation(
        model,
        validation_ds,
        tokenizer,
        device,
        max_len: int,
        eval_opts: Dict[str, object] = None
) -> Dict[str, Dict[str, float]]:
    import random

    # We set options for a fast evaluation. Since the evaluation is the most time-consuming, we limit full evaluation every 20 epochs.
    # In this way, we can have an idea of how the training is going in intermediate epochs,
    # and have a full overview every 20 epochs.
    eval_opts = eval_opts or {}
    mode = eval_opts.get('profile',
                         'FULL')  # With FAST, we set the limits, while with 'FULL' we do the full evaluation,
    # without the limits and calculating the METEOR score. It can be set in the config
    max_batches_fast = int(eval_opts.get('max_batches', 0)) if mode == 'FAST' else 0
    max_text_fast = int(eval_opts.get('max_text', 0)) if mode == 'FAST' else 0
    max_triples_fast = int(eval_opts.get('max_triples', 0)) if mode == 'FAST' else 0
    max_rc1_fast = int(eval_opts.get('max_rc1', 0)) if mode == 'FAST' else 0
    max_new_tokens_fast = int(eval_opts.get('max_new_tokens', 0)) if mode == 'FAST' else 0
    skip_meteor_fast = bool(eval_opts.get('skip_meteor_fast', True))

    # Max length for the decoding, only if mode == 'FAST'
    max_decode_len = min(max_len, max_new_tokens_fast) if (mode == 'FAST' and max_new_tokens_fast > 0) else max_len

    model.eval()
    # Lists for the RDF2Text task
    task_preds_text: List[str] = []
    task_refs_text: List[str] = []

    # RDF Completion 1 task counters for the correct tokens
    comp1_tp = comp1_ref_tokens = 0
    comp1_correct = 0
    comp1_total = 0

    # Text2RDF task counters for correct tokens
    t2rdf_tp = t2rdf_pred_tokens = t2rdf_ref_tokens = 0
    t2rdf_total = 0

    # RDF Completion 2 counters for correct tokens
    comp2_tp = comp2_pred_tokens = comp2_ref_tokens = 0
    comp2_total = 0

    # Counters for the FAST mode in evaluation
    used_text = 0
    used_trpl = 0

    # Reservoir sampling variables (to pick 2 random examples across all processed samples)
    reservoir: List[Dict[str, object]] = []
    seen_samples = 0
    reservoir_size = 4

    with torch.no_grad():
        for step, batch in enumerate(validation_ds, 1):
            if mode == 'FAST' and max_batches_fast and step > max_batches_fast:
                break

            encoder_input = batch['encoder_input'].to(device, non_blocking=True)
            encoder_mask = batch['encoder_mask'].to(device, non_blocking=True).bool()

            bsz = encoder_input.size(0)
            for i in range(bsz):
                # We just check if we hit one of the limits
                if mode == 'FAST':
                    all_caps_hit = True
                    if max_text_fast and used_text < max_text_fast:
                        all_caps_hit = False
                    if max_triples_fast and used_trpl < max_triples_fast:
                        all_caps_hit = False
                    if max_rc1_fast and comp1_total < max_rc1_fast:
                        all_caps_hit = False
                    if all_caps_hit:
                        break

                enc_in = encoder_input[i:i + 1]  # encoder_input[i] returns a tensor with the first dimension removed,
                # so with shape (S,), while encoder_input[i:i+1] returns a tensor with shape (1,S), so a batch of dimension 1.
                # We do this because the model expects a shape (batch, ...). In this way, we avoid using encoder_input[i].unsqueeze(0)
                enc_mask = encoder_mask[i:i + 1]

                # Greedy decode with length cap in fast mode in fp16, we do this for memory saving and for faster inference
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out_ids = greedy_decode(model, enc_in, enc_mask, tokenizer, max_decode_len, device)

                # Normalize type and strip BOS/EOS/PAD
                if isinstance(out_ids, torch.Tensor):
                    out_ids = out_ids.squeeze(0).detach().cpu().tolist()
                elif hasattr(out_ids, "tolist"):
                    out_ids = out_ids.tolist()
                out_ids = strip_bos_eos_pad(out_ids, tokenizer)

                # We need also the textual version for the RDF2Text metrics
                pred_text = tokenizer.decode(list(out_ids), skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False)
                ref_text = batch['tgt_text'][i] if isinstance(batch['tgt_text'], (list, tuple)) else batch['tgt_text']
                task = batch['task_type'][i] if isinstance(batch['task_type'], list) else batch['task_type']

                if mode == 'FULL':
                    # If the mode if FULL, we output 4 random samples. So, here we build the candidates
                    try:
                        src_seq = encoder_input[i].detach().cpu().tolist()
                        src_seq_clean = strip_bos_eos_pad(src_seq, tokenizer)
                        src_text = tokenizer.decode(src_seq_clean, skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False)
                    except Exception:
                        src_text = str(encoder_input[i].detach().cpu().tolist())

                    # We use label ids if present, otherwise fallback to ref_text
                    tgt_text = None
                    if 'label' in batch:
                        try:
                            tgt_seq = batch['label'][i].detach().cpu().tolist()
                            tgt_seq_clean = strip_bos_eos_pad(tgt_seq, tokenizer)
                            tgt_text = tokenizer.decode(tgt_seq_clean, skip_special_tokens=False,
                                                        clean_up_tokenization_spaces=False)
                        except Exception:
                            tgt_text = None
                    if tgt_text is None:
                        try:
                            tgt_text = ref_text
                        except Exception:
                            tgt_text = str(ref_text)

                    candidate = {
                        "batch": step,
                        "sample_in_batch": i,
                        "source": src_text,
                        "target": tgt_text,
                        "predicted": pred_text,
                        "task": task
                    }

                    seen_samples += 1
                    if len(reservoir) < reservoir_size:
                        reservoir.append(candidate)
                    else:
                        # We replace an item with probability reservoir_size / seen_samples. This is done
                        # to see different examples at each full validation
                        j = random.randrange(seen_samples)
                        if j < reservoir_size:
                            reservoir[j] = candidate

                # Based on task type, we use the appropriate evaluation metrics
                if task == "rdf2text":
                    if mode == 'FAST' and max_text_fast and used_text >= max_text_fast:
                        pass
                    else:
                        task_preds_text.append(pred_text)
                        task_refs_text.append(ref_text)
                        used_text += 1

                elif task == "text2rdf":
                    if mode == 'FAST' and max_triples_fast and used_trpl >= max_triples_fast:
                        pass
                    else:
                        ref_ids = tokenizer.encode(ref_text, add_special_tokens=False)
                        tp, p_tot, r_tot, p, r, f1 = prf(out_ids,
                                                         ref_ids)  # Here we calculate precision, recall and f1 for
                        # single sample. As evaluation metrics, we will consider precision, recall and f1 on all the tokens.
                        # We anyway keep p, r and f1 also for the single sample, eventually calculating f, p and f1 averaging
                        # the values on all the samples
                        t2rdf_tp += tp
                        t2rdf_pred_tokens += p_tot
                        t2rdf_ref_tokens += r_tot
                        t2rdf_total += 1
                        used_trpl += 1

                elif task == "rdf_completion_1":
                    if mode == 'FAST' and max_rc1_fast and comp1_total >= max_rc1_fast:
                        pass
                    else:
                        ref_ids = tokenizer.encode(ref_text, add_special_tokens=False)
                        tp, p_tot, r_tot, p, r, f1 = prf(out_ids, ref_ids)  # Same consideration done for text2rdf
                        comp1_tp += tp
                        comp1_ref_tokens += r_tot
                        comp1_total += 1
                        comp1_correct += 1 if Counter(out_ids) == Counter(ref_ids) else 0  # Using Counter compares multisets: the order of IDs is ignored,
                        # but the multiplicity (count) of each ID must be identical.



                elif task == "rdf_completion_2":
                    if mode == 'FAST' and max_triples_fast and used_trpl >= max_triples_fast:
                        pass
                    else:
                        ref_ids = tokenizer.encode(ref_text, add_special_tokens=False)
                        tp, p_tot, r_tot, p, r, f1 = prf(out_ids, ref_ids)  # Same consideration done for text2rdf
                        comp2_tp += tp
                        comp2_pred_tokens += p_tot
                        comp2_ref_tokens += r_tot
                        comp2_total += 1
                        used_trpl += 1

    # After the loop, if FULL mode, print the randomly selected examples (from reservoir)
    if mode == 'FULL' and reservoir:
        print("\n" + "=" * 80)
        print(f"Randomly selected {len(reservoir)} example(s) from validation set.")
        for idx, ex in enumerate(reservoir):
            print("-" * 80)
            print(f"EXAMPLE {idx + 1} (batch {ex['batch']}, sample {ex['sample_in_batch']})")
            print(f"TASK:      {ex.get('task', 'N/A')}")
            print(f"SOURCE:    {ex['source']}")
            print(f"TARGET:    {ex['target']}")
            print(f"PREDICTED: {ex['predicted']}")
        print("=" * 80 + "\n")

    # We initialize a vocabulary to aggregate the results
    results: Dict[str, Dict[str, float]] = {}

    # We check if task_preds_text is empty: if it isn't, we calculate the results for RDF2Text
    if task_preds_text:
        refs = task_refs_text
        hyps = task_preds_text
        results["rdf2text"] = aggregate_text_metrics(
            refs, hyps,
            skip_meteor=(mode == 'FAST' and skip_meteor_fast)
        )

    # Text2RDF token-based metrics
    if t2rdf_total > 0:
        p = (t2rdf_tp / t2rdf_pred_tokens) if t2rdf_pred_tokens > 0 else 0.0
        r = (t2rdf_tp / t2rdf_ref_tokens) if t2rdf_ref_tokens > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p > 0 and r > 0) else 0.0
        results["text2rdf"] = {"precision": p, "recall": r, "f1": f1}

    # RC1 accuracy, both for the masked span and the single tokens
    if comp1_total > 0:
        acc_token = (comp1_tp / comp1_ref_tokens) if comp1_ref_tokens > 0 else 0.0
        results["rdf_completion_1"] = {"sample_accuracy": (comp1_correct / comp1_total), "token_accuracy": acc_token}

    # RC2 token-based metrics
    if comp2_total > 0:
        p = (comp2_tp / comp2_pred_tokens) if comp2_pred_tokens > 0 else 0.0
        r = (comp2_tp / comp2_ref_tokens) if comp2_ref_tokens > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p > 0 and r > 0) else 0.0
        results["rdf_completion_2"] = {"precision": p, "recall": r, "f1": f1}

    return results


def get_model(config, vocab_size):
    # Loading model using the 'build_transformer' function.
    if config.get('interleave_pattern') is not None and isinstance(config['interleave_pattern'], (list, tuple)):
        model = build_transformer_interleaved(vocab_size=vocab_size,
                                              src_seq_len=config['seq_len'],
                                              tgt_seq_len=config['seq_len'],
                                              d_model=config['d_model'],
                                              N=config['N'],
                                              h=config['h'],
                                              dropout=config['dropout'],
                                              interleave_pattern=config['interleave_pattern'],
                                              num_buckets=config['num_buckets'],
                                              max_distance=config['max_distance'],
                                              rope_d_r=config['rope_d_r']
                                              )

    else:
        model = build_transformer(vocab_size=vocab_size,
                                  src_seq_len=config['seq_len'],
                                  tgt_seq_len=config['seq_len'],
                                  d_model=config['d_model'],
                                  N=config['N'],
                                  h=config['h'],
                                  dropout=config['dropout'],
                                  use_mla=config['use_mla'],
                                  num_buckets=config['num_buckets'],
                                  max_distance=config['max_distance'],
                                  rope_d_r=config['rope_d_r']
                                  )

    return model


# Function to construct the path for saving and retrieving model weights
def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']  # Extracting model folder from the config
    model_basename = config['model_basename']  # Extracting the base name for model files
    model_filename = f"{model_basename}{epoch}.pt"  # Building filename
    return str(Path(
        '.') / model_folder / model_filename)  # Combining current directory, the model folder, and the model filename


def train_model(config):
    # Device and directories to save the learning curves
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    csv_path = Path(config['experiment_name']) / "learning_curves.csv"
    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(
                "epoch,train_loss,val_loss,text2rdf_f1,text2rdf_precision,text2rdf_recall,rc2_f1,rc2_precision,rc2_recall,rc1_acc_sample,rc1_acc_token,bleu4,rougeL,meteor\n")

    # We create the Dataloaders
    train_dataloader, val_dataloader, tokenizer = create_dataloaders(config)
    vocab_size = len(tokenizer.get_vocab())

    # We initialize the model
    model = get_model(config, vocab_size).to(device)

    # We can do a sanity check by overfitting the model on one batch. The sanity check can be formed by setting
    # True "sanity_one_batch" in the config. By default, it is set to False
    if config.get("sanity_one_batch", False):
        print("Running sanity check (overfit on 1 batch)")
        sanity_overfit_one_batch(model, train_dataloader, tokenizer, device, steps=200, lr=1e-3)
        return

    # We initialize the writer
    writer = SummaryWriter(config['experiment_name'])

    # We initialise the optimiser, in this case AdamW.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-8, betas=(0.90, 0.95))

    # Number of steps for which we accumulate the gradients
    accum_steps = config.get('accum_steps', 1)

    # Here we set the Scheduler parameters, considering also the accum_steps
    steps_per_epoch = len(train_dataloader)
    opt_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)
    total_opt_steps = opt_steps_per_epoch * config['num_epochs']

    # Warmup + Cosine Annealing scheduler parameters
    # Another option is to use OnceCycle scheduler with cosine decay
    warmup_ratio = config.get('warmup_ratio', 0.05)  # Fraction of total optimizer steps for warmup
    warmup_steps = int(total_opt_steps * warmup_ratio)  # Steps used for the warmup

    div_factor = config.get('div_factor',
                            25.0)  # This is the factor that divides the maximum learning rate to get the initial one
    # We set it as 25.0, but we can change it in the config
    lr_max = config.get('lr_max', config['lr'])
    initial_lr = float(lr_max) / float(div_factor)
    final_div = config.get('final_div_factor', 1e5)
    min_lr = float(initial_lr) / float(final_div)

    use_amp = (device.type == 'cuda')  # True if we are using an NVIDIA GPU
    scaler = GradScaler(enabled=use_amp)  # With the scaler, we scale the gradients when using reduced precision types
    # to avoid numerical underflow during backward pass
    amp_ctx = (
        autocast(dtype=torch.bfloat16) if use_amp else nullcontext())  # autocast automatically converts the operations
    # to bfloat16 to save the memory and have faster calculations. nullcontext basically does nothing if we don't have cuda device

    # Parameters to resume from checkpoint
    initial_epoch = 0
    global_step_batches = 0

    # We preload the model, if requested
    saved_scheduler_state = None
    state = None
    if config.get('preload', False):
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = int(state.get('epoch', -1)) + 1
        global_step_batches = int(state.get('global_step', 0))
        saved_scheduler_state = state.get('scheduler_state_dict', None)
        try:
            scaler.load_state_dict(state['scaler_state_dict'])
        except Exception:
            pass

    opt_steps_done = global_step_batches // accum_steps

    # If not resuming from checkpoint, we set optimizer base lr to max_lr so that warmup ramps to max_lr.
    # If resuming, optimizer param_group lr is already restored from checkpoint.
    if not config.get('preload', False):
        for g in optimizer.param_groups:
            g['lr'] = float(lr_max)

    # start_factor: initial_lr = start_factor * base_lr (base_lr == max_lr here)
    start_factor = float(initial_lr) / float(lr_max) if lr_max != 0 else 1.0
    start_factor = max(0.0, min(1.0, start_factor))

    # LinearLR scales the learning rates of the parameters linearly from start_factor * base_lr to end_factor * base_lr
    # in total_iters steps
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps,
                                last_epoch=-1)
    cos_T_max = max(1, total_opt_steps - warmup_steps)
    # CosineAnnealingLR apply a cosine decay to lr from eta_max (the current base_lr) to eta_min in T_max steps
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cos_T_max, eta_min=min_lr, last_epoch=-1)
    # SequentialLR concatenates the schedulers, using the first scheduler until the milestone (in this case, the warmup steps),
    # and after uses the second scheduler. Last_epoch is used to resume the training, so that the scheduler knows in which phase we are.
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps],
                             last_epoch=opt_steps_done - 1)

    # If checkpoint contains scheduler state, prefer to restore it
    if saved_scheduler_state is not None:
        try:
            scheduler.load_state_dict(saved_scheduler_state)
        except Exception:
            # if loading fails, we keep the freshly constructed scheduler (aligned by last_epoch)
            pass

    # Loss (CrossEntropyLoss), ignoring pad tokens when calculating it, as they are used simply to reach seq_len
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.convert_tokens_to_ids('[PAD]'),
        label_smoothing=config.get('label_smoothing', 0.0)
        # label_smoothing is used to assign a little probability ε to distribute uniformly on the wrong classes. This
        # often improves the generalisation and avoids overconfidence of the model
    ).to(device)

    # Training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        model.train()
        epoch_train_loss_sum = 0.0
        epoch_train_steps = 0  # Number of effectives updates (number of optimizer.step() called)

        accum_losses_sum = 0.0  # Temporary sum of the micro-losses in the current update

        # We get the informations about the samples in the batch
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device, non_blocking=True)
            decoder_input = batch['decoder_input'].to(device, non_blocking=True)
            encoder_mask = batch['encoder_mask'].to(device, non_blocking=True).bool()
            decoder_mask = batch['decoder_mask'].to(device, non_blocking=True).bool()
            labels = batch['label'].to(device, non_blocking=True)

            with amp_ctx:
                encoder_output = model.encode(encoder_input,
                                              encoder_mask)  # We encode the input, using the encoder_mask
                # the encoder_mask basically masks the padding tokens
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                              decoder_mask)  # We encode the output,
                # using both the mask for the PAD tokens and the mask for the 'future' tokens
                logits = model.project(decoder_output)  # We project the decoder_output in a higher dimensional space
                loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
                loss = loss / accum_steps  # We get gradients equal to the average of the losses of accum_steps batches

            if not torch.isfinite(logits).all():
                print("[WARN] non-finite logits: step", global_step_batches)
                print(" task:", batch.get('task_type', 'n/a'))
                continue

            # scaler.scale multiplies temporarily the loss for a scale factor to avoid underflow when using reduced precision
            # types (AMP). With .basckward, it calculates the gradients starting from the scaled loss.
            scaler.scale(loss).backward()

            single_loss = loss.item()
            accum_losses_sum += single_loss

            # After having accumulated accum_steps batches, we do the optimizer.step()
            if (global_step_batches + 1) % accum_steps == 0:
                batch_loss = accum_losses_sum
                scaler.unscale_(optimizer)  # Divides the gradients by the scale factor
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Applies the clip by L2 norm of the gradients,
                # avoiding exploding gradients
                scaler.step(optimizer)  # Checks if the gradients are Inf/NaN after the unscale. If they are not, it
                # executes optimizer.step()
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()  # We call scheduler only when the weights are updated
                epoch_train_loss_sum += batch_loss
                epoch_train_steps += 1

                try:
                    writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step_batches)
                except Exception:
                    pass

                accum_losses_sum = 0.0

            global_step_batches += 1

        train_loss_epoch = epoch_train_loss_sum / max(1, epoch_train_steps)
        writer.add_scalar('epoch/train_loss', train_loss_epoch, epoch)

        # Validation
        profile = config.get('eval_profile', 'FAST')
        eval_opts = {
            'profile': profile,
            'max_batches': config.get('eval_max_batches', 0),
            'max_text': config.get('eval_max_text', 0),
            'max_triples': config.get('eval_max_triples', 0),
            'max_new_tokens': config.get('eval_max_new_tokens_fast', 0),
            'skip_meteor_fast': config.get('eval_skip_meteor_fast', True),
            'max_rc1': config.get('eval_max_rc1', 0),
        }

        if (epoch + 1) % 20 == 0:
            eval_opts['profile'] = 'FULL'
            val_results = run_validation(
                model=model,
                validation_ds=val_dataloader,
                tokenizer=tokenizer,
                device=device,
                max_len=config['seq_len'],
                eval_opts=eval_opts
            )
            eval_opts['profile'] = profile
        else:
            val_results = run_validation(
                model=model,
                validation_ds=val_dataloader,
                tokenizer=tokenizer,
                device=device,
                max_len=config['seq_len'],
                eval_opts=eval_opts
            )

        print_val(val_results, epoch)
        t2 = val_results.get('text2rdf', {})
        r2 = val_results.get('rdf_completion_2', {})
        rc1 = val_results.get('rdf_completion_1', {})
        rt = val_results.get('rdf2text', {})

        # Here we write the results to the CSV file
        if (epoch + 1) % 5 == 0:
            val_loss_epoch = compute_val_loss(model, val_dataloader, tokenizer, device, vocab_size, max_batches=50) # We compute the validation loss
            # on 50 batches (around 1/3 of the total batches with the current configuration)
            writer.add_scalar('epoch/val_loss', val_loss_epoch, epoch)
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{train_loss_epoch:.6f},{val_loss_epoch:.6f},"
                        f"{t2.get('f1', 0.0):.6f},{t2.get('precision'):.6f},{t2.get('recall'):.6f},"
                        f"{r2.get('f1', 0.0):.6f},{r2.get('precision', 0.0):.6f},{r2.get('recall', 0.0):.6f},"
                        f"{rc1.get('sample_accuracy', 0.0):.6f},{rc1.get('token_accuracy', 0.0):.6f},"
                        f"{rt.get('BLEU4', 0.0):.6f},{rt.get('ROUGE-L', 0.0):.6f},{rt.get('METEOR', 0.0):.6f}\n")

        # Here we save a checkpoint, every full validation epoch
        if (epoch + 1) % 20 == 0:
            model_filename = get_weights_file_path(config, f'{epoch:02d}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'global_step': global_step_batches
            }, model_filename)

    writer.flush()
    writer.close()



if __name__ == '__main__':
    warnings.filterwarnings('ignore')  # Filtering warnings

    config = get_config()  # Retrieving config settings
    train_model(config)  # Training model with the config arguments
