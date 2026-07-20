"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import copy
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_FILE_ID = "1rXyhzh_9ozSHLu0uPwi_R7U0Tgqoi73n"
DEFAULT_CHECKPOINT_PATH = _MODULE_DIR / "checkpoints" / "best_bleu_checkpoint.pt"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"


@lru_cache(maxsize=1)
def _get_text_inference_assets():
    from dataset import Multi30kDataset

    dataset = Multi30kDataset(split="train")
    return dataset.src_vocab, dataset.tgt_vocab, Multi30kDataset._get_tokenizer("de")


def _checkpoint_candidates() -> list[Path]:
    candidates = [
        DEFAULT_CHECKPOINT_PATH,
        _MODULE_DIR / "checkpoint.pt",
        Path.cwd() / "checkpoints" / "best_bleu_checkpoint.pt",
        Path.cwd() / "checkpoint.pt",
    ]
    unique_candidates: list[Path] = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(path)
    return unique_candidates


def _extract_model_state_dict(checkpoint):
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if not isinstance(state_dict, dict):
        return None
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}
    return state_dict

# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    attn_w = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════


def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    batch_size, tgt_len = tgt.shape
    device = tgt.device

    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]
    causal_mask = (
        torch.triu(
            torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device), diagonal=1
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )  # [1, 1, tgt_len, tgt_len]

    return pad_mask | causal_mask.expand(batch_size, 1, tgt_len, tgt_len)


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # depth per head
        self.use_scaling = use_scaling
        self.last_attention_weights: Optional[torch.Tensor] = None

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        batch_size = query.size(0)

        def project(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
            return (
                linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            )

        q = project(self.w_q, query)
        k = project(self.w_k, key)
        v = project(self.w_v, value)
        if self.use_scaling:
            attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1))
            if mask is not None:
                mask = mask.to(dtype=torch.bool, device=scores.device)
                scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

        self.last_attention_weights = attn_weights.detach()
        attn_output = torch.matmul(self.dropout(attn_weights), v)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        return self.w_o(attn_output)


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]

        """
        x = x + self.pe[:, : x.size(1)].to(dtype=x.dtype, device=x.device)
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings for sequence positions.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.position_embeddings(positions)
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]

        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════


class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
        , use_scaling: bool = True
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER
# ══════════════════════════════════════════════════════════════════════


class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
        , use_scaling: bool = True
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        cross_attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════


class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: int = 7853,
        tgt_vocab_size: int = 5893,
        d_model: int = 256,
        N: int = 3,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        use_learned_positional_encoding: bool = False,
        max_position_embeddings: int = 5000,
        use_attention_scaling: bool = True,
    ) -> None:
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.use_learned_positional_encoding = use_learned_positional_encoding
        self.max_position_embeddings = max_position_embeddings
        self.use_attention_scaling = use_attention_scaling
        self._inference_checkpoint_loaded = False
        self._inference_src_vocab = None
        self._inference_tgt_vocab = None
        self._inference_src_tokenizer = None

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        if use_learned_positional_encoding:
            self.positional_encoding = LearnedPositionalEncoding(
                d_model, dropout, max_position_embeddings
            )
        else:
            self.positional_encoding = PositionalEncoding(d_model, dropout)

        self.encoder = Encoder(
            EncoderLayer(d_model, num_heads, d_ff, dropout, use_attention_scaling), N
        )
        self.decoder = Decoder(
            DecoderLayer(d_model, num_heads, d_ff, dropout, use_attention_scaling), N
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def get_config(self) -> dict:
        return {
            "src_vocab_size": self.src_vocab_size,
            "tgt_vocab_size": self.tgt_vocab_size,
            "d_model": self.d_model,
            "N": self.N,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "use_learned_positional_encoding": self.use_learned_positional_encoding,
            "max_position_embeddings": self.max_position_embeddings,
            "use_attention_scaling": self.use_attention_scaling,
        }

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        decoder_out = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        return self.generator(decoder_out)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def _load_text_inference_assets(self):
        if (
            self._inference_src_vocab is not None
            and self._inference_tgt_vocab is not None
            and self._inference_src_tokenizer is not None
        ):
            return

        try:
            self._inference_src_vocab, self._inference_tgt_vocab, self._inference_src_tokenizer = _get_text_inference_assets()
        except Exception:
            self._inference_src_vocab = None
            self._inference_tgt_vocab = None
            self._inference_src_tokenizer = None

    def _state_dict_is_compatible(self, state_dict: dict) -> bool:
        current_state = self.state_dict()
        if set(state_dict.keys()) != set(current_state.keys()):
            return False

        for key, value in current_state.items():
            candidate = state_dict[key]
            if not torch.is_tensor(candidate) or candidate.shape != value.shape:
                return False
        return True

    def _download_default_checkpoint(
        self,
        checkpoint_path: Path,
        checkpoint_file_id: str = DEFAULT_CHECKPOINT_FILE_ID,
    ) -> None:
        if not checkpoint_file_id or checkpoint_path.exists():
            return

        try:
            import gdown
        except Exception:
            return

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={checkpoint_file_id}"
        try:
            gdown.download(url, str(checkpoint_path), quiet=True)
        except Exception:
            if checkpoint_path.exists() and checkpoint_path.stat().st_size == 0:
                checkpoint_path.unlink()

    def _ensure_checkpoint_loaded(self, checkpoint_file_id: str = DEFAULT_CHECKPOINT_FILE_ID) -> None:
        if self._inference_checkpoint_loaded:
            return

        checkpoint_path = next((path for path in _checkpoint_candidates() if path.exists()), None)
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_CHECKPOINT_PATH
            self._download_default_checkpoint(checkpoint_path, checkpoint_file_id)
            if not checkpoint_path.exists():
                return

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            return

        state_dict = _extract_model_state_dict(checkpoint)
        if state_dict is None or not self._state_dict_is_compatible(state_dict):
            return

        self.load_state_dict(state_dict)
        self._inference_checkpoint_loaded = True

    def _decode_token_ids(self, token_ids: list[int], tgt_vocab) -> str:
        special_tokens = {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN}
        decoded_tokens: list[str] = []
        for token_id in token_ids:
            token = tgt_vocab.lookup_token(int(token_id)) if hasattr(tgt_vocab, "lookup_token") else tgt_vocab.itos[int(token_id)]
            if token == EOS_TOKEN:
                break
            if token not in special_tokens:
                decoded_tokens.append(token)
        return " ".join(decoded_tokens).strip()

    def _infer_text(
        self,
        src_text: str,
        max_len: int,
        start_symbol: int,
        end_symbol: int,
        pad_idx: int,
        device: torch.device,
    ) -> str:
        self._ensure_checkpoint_loaded()
        self._load_text_inference_assets()

        if self._inference_src_vocab is None or self._inference_tgt_vocab is None or self._inference_src_tokenizer is None:
            raise RuntimeError("Unable to initialize translation vocabularies for text inference")

        try:
            src_pad_idx = int(self._inference_src_vocab[PAD_TOKEN])
            tgt_pad_idx = int(self._inference_tgt_vocab[PAD_TOKEN])
            start_symbol = int(self._inference_tgt_vocab[SOS_TOKEN])
            end_symbol = int(self._inference_tgt_vocab[EOS_TOKEN])
        except Exception:
            src_pad_idx = pad_idx
            tgt_pad_idx = pad_idx

        tokens = [token.text.lower() for token in self._inference_src_tokenizer(src_text)]
        src_ids = [self._inference_src_vocab[SOS_TOKEN]]
        src_ids.extend(self._inference_src_vocab.lookup_indices(tokens))
        src_ids.append(self._inference_src_vocab[EOS_TOKEN])

        src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src_tensor, pad_idx=src_pad_idx)

        was_training = self.training
        self.eval()
        try:
            memory = self.encode(src_tensor, src_mask)
            generated = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

            for _ in range(max_len - 1):
                tgt_mask = make_tgt_mask(generated, pad_idx=tgt_pad_idx).to(device)
                logits = self.decode(memory, src_mask, generated, tgt_mask)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                if int(next_token.item()) == end_symbol:
                    break
        finally:
            self.train(was_training)

        return self._decode_token_ids(generated.squeeze(0).tolist(), self._inference_tgt_vocab)

    @torch.no_grad()
    def infer(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        max_len: int = 100,
        start_symbol: int = 2,
        end_symbol: int = 3,
        pad_idx: int = 1,
        device: Optional[torch.device] = None,
    ):
        """
        Greedy decoding helper for autograders and quick inference.

        Accepts either a source tensor [batch, src_len] / [src_len] or an
        iterable of batches that yield source tensors or (src, tgt) pairs.
        """
        if device is None:
            device = next(self.parameters()).device

        if isinstance(src, str):
            return self._infer_text(
                src,
                max_len=max_len,
                start_symbol=start_symbol,
                end_symbol=end_symbol,
                pad_idx=pad_idx,
                device=device,
            )

        if torch.is_tensor(src):
            self._ensure_checkpoint_loaded()
            src_tensor = src.to(device)
            if src_tensor.dim() == 1:
                src_tensor = src_tensor.unsqueeze(0)
            batch_src_mask = src_mask.to(device) if src_mask is not None else make_src_mask(src_tensor, pad_idx=pad_idx)
            was_training = self.training
            self.eval()

            try:
                memory = self.encode(src_tensor, batch_src_mask)
                generated = torch.full(
                    (src_tensor.size(0), 1),
                    start_symbol,
                    dtype=torch.long,
                    device=device,
                )

                for _ in range(max_len - 1):
                    tgt_mask = make_tgt_mask(generated, pad_idx=pad_idx).to(device)
                    logits = self.decode(memory, batch_src_mask, generated, tgt_mask)
                    next_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_tokens], dim=1)
                    if torch.all(next_tokens.squeeze(1) == end_symbol):
                        break
                return generated
            finally:
                self.train(was_training)

        outputs = []
        for batch in src:
            batch_src = batch[0] if isinstance(batch, (tuple, list)) else batch
            batch_mask = make_src_mask(batch_src.to(device), pad_idx=pad_idx)
            outputs.append(
                self.infer(
                    batch_src,
                    src_mask=batch_mask,
                    max_len=max_len,
                    start_symbol=start_symbol,
                    end_symbol=end_symbol,
                    pad_idx=pad_idx,
                    device=device,
                )
            )
        return outputs
