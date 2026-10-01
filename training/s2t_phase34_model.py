"""
Sign2Text — Phase 3 + 4: Model (pose encoder + pretrained T5 decoder)
=====================================================================

    features (B, T, 384) ─► PoseEncoder ─► (B, T/2, d_t5) ─► T5 decoder ─► English

PoseEncoder
    Linear 384 → d  +  LayerNorm  +  GELU
    Conv1d stride 2          halves the sequence (T ≤ 256 → ≤ 128)
    Sinusoidal positions
    Transformer encoder      pre-norm, padding masked
    Linear d → d_t5          matches T5's hidden size

T5 decoder
    Pretrained t5-small. Its text encoder is not used; our PoseEncoder
    replaces it. The decoder already knows English, which matters because
    How2Sign has 16k words and 6.5k of them appear only once.

Bag-of-words helper loss (run 2)
    The encoder output is pooled and must predict which T5 tokens appear in
    the sentence (order ignored). This gives the encoder a direct learning
    signal, so T5 cannot get a good loss by just guessing common sentences.

Notes
    - Kaggle needs Internet ON the first time (Settings → Internet) to
      download t5-small.
    - Train T5 in fp32 (T5 overflows in fp16).

Run locally (no download needed, uses a tiny T5):
    python s2t_phase34_model.py
"""

import math
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

T5_NAME     = 'google-t5/t5-small'
FEATURE_DIM = 384


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — POSE ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 2048) -> None:
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe  = torch.zeros(max_len, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PoseEncoder(nn.Module):
    """
    (B, T, 384) + mask (B, T)  →  (B, ceil(T/2), out_dim) + mask (B, ceil(T/2))
    Padded frames never influence real ones.
    """

    def __init__(
        self,
        in_dim:  int   = FEATURE_DIM,
        d:       int   = 256,
        layers:  int   = 4,
        heads:   int   = 4,
        ff:      int   = 1024,
        dropout: float = 0.1,
        out_dim: int   = 512,
    ) -> None:
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(in_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout),
        )
        self.sub = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)
        self.pos = PositionalEncoding(d)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ff, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.tf   = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out  = nn.Linear(d, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.inp(x) * mask.unsqueeze(-1)                       # zero padded frames
        h = self.sub(h.transpose(1, 2)).transpose(1, 2)            # (B, ceil(T/2), d)
        mask = mask[:, ::2]                                        # same length
        h = self.drop(self.pos(h))
        h = self.tf(h, src_key_padding_mask=~mask)
        h = self.out(self.norm(h))
        return h * mask.unsqueeze(-1), mask


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class Sign2Text(nn.Module):

    def __init__(self, t5: nn.Module, aux_weight: float = 0.0, **enc_kwargs) -> None:
        super().__init__()
        self.t5 = t5
        d_t5 = t5.config.d_model
        self.encoder = PoseEncoder(out_dim=d_t5, **enc_kwargs)
        self.aux_weight = aux_weight
        if aux_weight > 0:
            self.bow_head = nn.Sequential(nn.LayerNorm(d_t5),
                                          nn.Linear(d_t5, t5.config.vocab_size))
        self.parts: dict = {}
        # T5's own text encoder is never called: exclude it from training
        for p in self.t5.encoder.parameters():
            p.requires_grad = False

    def set_decoder_trainable(self, flag: bool) -> None:
        """Freeze / unfreeze everything in T5 except its unused text encoder."""
        for n, p in self.t5.named_parameters():
            if not n.startswith('encoder.'):
                p.requires_grad = flag

    def _bow_loss(self, h: torch.Tensor, m: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pooled = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        logits = self.bow_head(pooled)                                  # (B, V)
        V = logits.size(-1)
        ids = labels.clamp(min=0)
        valid = (labels > 1).float()                                    # skip pad, eos, -100
        target = torch.zeros(labels.size(0), V, device=labels.device)
        target.scatter_add_(1, ids, valid)
        target = target / target.sum(1, keepdim=True).clamp(min=1)
        return -(target * torch.log_softmax(logits, -1)).sum(1).mean()

    def encode(self, features: torch.Tensor, feature_mask: torch.Tensor):
        from transformers.modeling_outputs import BaseModelOutput
        h, m = self.encoder(features, feature_mask)
        return BaseModelOutput(last_hidden_state=h), m

    def forward(self, features: torch.Tensor, feature_mask: torch.Tensor,
                labels: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
        """Returns total loss. Parts are kept in self.parts for logging."""
        enc, m = self.encode(features, feature_mask)
        out = self.t5(encoder_outputs=enc, attention_mask=m.long(), labels=labels)
        if label_smoothing <= 0:
            ce = out.loss
        else:
            ce = nn.functional.cross_entropy(
                out.logits.reshape(-1, out.logits.size(-1)), labels.reshape(-1),
                ignore_index=-100, label_smoothing=label_smoothing,
            )
        self.parts = {'ce': ce.item()}
        if self.aux_weight <= 0:
            return ce
        bow = self._bow_loss(enc.last_hidden_state, m, labels)
        self.parts['bow'] = bow.item()
        return ce + self.aux_weight * bow

    @torch.no_grad()
    def translate(self, features: torch.Tensor, feature_mask: torch.Tensor,
                  tokenizer=None, num_beams: int = 4, max_new_tokens: int = 64,
                  no_repeat_ngram_size: int = 3, repetition_penalty: float = 1.2):
        """
        Returns decoded strings if a tokenizer is given, else token ids.
        no_repeat_ngram_size / repetition_penalty stop loops like
        "put it in the sand and put it in the sand".
        """
        enc, m = self.encode(features, feature_mask)
        ids = self.t5.generate(
            encoder_outputs=enc, attention_mask=m.long(),
            num_beams=num_beams, max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
        )
        return tokenizer.batch_decode(ids, skip_special_tokens=True) if tokenizer else ids

    def param_groups(self, lr_encoder: float = 3e-4, lr_decoder: float = 1e-4,
                     weight_decay: float = 0.01) -> List[dict]:
        """New encoder (+ BoW head) learns fast; pretrained decoder learns gently."""
        enc = [p for n, p in self.named_parameters()
               if (n.startswith('encoder.') or n.startswith('bow_head.')) and p.requires_grad]
        dec = [p for p in self.t5.parameters() if p.requires_grad]
        return [
            {'params': enc, 'lr': lr_encoder, 'weight_decay': weight_decay},
            {'params': dec, 'lr': lr_decoder, 'weight_decay': weight_decay},
        ]

    def count_params(self) -> dict:
        n = lambda ps: sum(p.numel() for p in ps)
        return {
            'pose_encoder_M': round(n(self.encoder.parameters()) / 1e6, 2),
            'bow_head_M': round(n(self.bow_head.parameters()) / 1e6, 2) if self.aux_weight > 0 else 0,
            't5_trainable_M': round(n(p for p in self.t5.parameters() if p.requires_grad) / 1e6, 2),
        }


def build_model(t5_name: str = T5_NAME, aux_weight: float = 0.0, **enc_kwargs):
    """Load pretrained T5 + tokenizer and wrap them. Returns (model, tokenizer)."""
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(t5_name)
    t5  = T5ForConditionalGeneration.from_pretrained(t5_name)
    model = Sign2Text(t5, aux_weight=aux_weight, **enc_kwargs)
    log.info(f"Model built from {t5_name}: {model.count_params()}")
    return model, tok


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SMOKE TEST (tiny random T5, no download)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from transformers import T5Config, T5ForConditionalGeneration

    print("\n" + "=" * 65)
    print("  Sign2Text Phase 3+4 — Smoke Test")
    print("=" * 65 + "\n")
    torch.manual_seed(0)

    cfg = T5Config(vocab_size=100, d_model=64, d_ff=128, d_kv=16, num_heads=4,
                   num_layers=2, num_decoder_layers=2,
                   pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
    model = Sign2Text(T5ForConditionalGeneration(cfg), d=32, layers=2, heads=4, ff=64)
    print("── Params:", model.count_params())

    B, T = 3, 50
    lengths = [50, 31, 8]
    feats = torch.randn(B, T, FEATURE_DIM)
    mask  = torch.zeros(B, T, dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask[i, :L] = True
        feats[i, L:] = 0
    labels = torch.randint(2, 100, (B, 7))
    labels[2, 4:] = -100

    print("── Test 1: Encoder shapes")
    h, m = model.encoder(feats, mask)
    assert h.shape == (B, 25, 64) and m.shape == (B, 25)
    assert m.sum(1).tolist() == [25, 16, 4]
    print(f"  {tuple(feats.shape)} → {tuple(h.shape)}, lengths {m.sum(1).tolist()} ✓")

    print("── Test 2: Padding does not change real outputs")
    model.eval()
    h_batch, _ = model.encoder(feats, mask)
    h_alone, _ = model.encoder(feats[1:2, :31], mask[1:2, :31])
    diff = (h_batch[1, :16] - h_alone[0]).abs().max().item()
    assert diff < 1e-5, diff
    print(f"  max diff {diff:.2e} ✓")

    print("── Test 3: Loss + backward")
    model.train()
    loss = model(feats, mask, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.inp[0].weight.grad is not None
    assert all(p.grad is None for p in model.t5.encoder.parameters())
    print(f"  loss {loss.item():.3f}, encoder gets grads, T5 text encoder frozen ✓")

    print("── Test 4: Optimizer groups")
    opt = torch.optim.AdamW(model.param_groups())
    opt.step()
    print(f"  lrs {[g['lr'] for g in opt.param_groups]} ✓")

    print("── Test 5: BoW helper loss + decoder freeze")
    m2 = Sign2Text(T5ForConditionalGeneration(cfg), aux_weight=0.5, d=32, layers=2, heads=4, ff=64)
    m2.set_decoder_trainable(False)
    loss = m2(feats, mask, labels)
    loss.backward()
    assert 'bow' in m2.parts and torch.isfinite(loss)
    assert m2.bow_head[1].weight.grad is not None and m2.encoder.inp[0].weight.grad is not None
    assert all(p.grad is None for n, p in m2.t5.named_parameters())
    m2.set_decoder_trainable(True)
    assert all(p.requires_grad for n, p in m2.t5.named_parameters() if not n.startswith('encoder.'))
    print(f"  parts {m2.parts}, frozen decoder gets no grads, unfreeze works ✓")

    print("── Test 6: Generation (beam search, no repeats)")
    model.eval()
    ids = model.translate(feats, mask, num_beams=2, max_new_tokens=10)
    assert ids.shape[0] == B
    print(f"  output ids {tuple(ids.shape)} ✓")

    print("\n" + "=" * 65)
    print("  All tests PASSED ✓")
    print("=" * 65)
