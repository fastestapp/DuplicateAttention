"""
legacy_model.py -- ORIGINAL (pre-round-2-review) Transformer architecture, kept ONLY
so Run #1's checkpoint (step_100000.pt, trained before the correctness fixes in
model.py) can still be loaded and evaluated. Do not use this for new training runs --
it deliberately reproduces two bugs described in README "Code review findings and
fixes" #1 and #4:

  1. Embeddings are NOT tied: src_embed, tgt_embed, and the generator's output
     projection are three independent matrices (paper section 3.4 says they should
     share one). model.py's tied-embeddings fix is the one change being kept in the
     current systematic revert; this file intentionally does not have it, because
     Run #1's checkpoint has three separate weight tensors, not one shared one.
  2. LayerNorm is the original hand-rolled version built on torch.std() (unbiased/
     Bessel's-correction variance) with parameters named a_2/b_2, not nn.LayerNorm's
     population-variance weight/bias. model.py (as of the current revert) ALSO still
     has this bug -- both files agree on it for now, deliberately, since it hasn't
     been tested in isolation yet.

Why this file exists at all: model.py now keeps ONE fix (tied embeddings) that
legacy checkpoints don't have, so model.load_state_dict(run1_checkpoint) still fails
with a structural key mismatch against the current model.py. Evaluating Run #1's
actual trained weights requires reconstructing the architecture they were trained
under. See evaluate.py's --legacy_arch flag and README "Phase 1" / "Systematic
re-application log".

Everything below other than LayerNorm/SublayerConnection/Encoder/Decoder/Generator/
make_model is unchanged from model.py (imported directly) -- post-LN placement,
attention, feed-forward, and positional encoding were already correct in Run #1.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    EncoderDecoder,
    Embeddings,
    MultiHeadedAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    clones,
)


class LayerNorm(nn.Module):
    """Original hand-rolled LayerNorm -- torch.std() defaults to the UNBIASED
    (Bessel's-correction) variance estimator, not what LayerNorm is defined to use
    (population variance). This is a bug relative to the paper/standard LayerNorm --
    kept here (and, for now, also still in model.py) because it's what Run #1's
    checkpoint was actually trained with (parameter names a_2/b_2, not weight/bias).
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)  # unbiased=True (default) -- the bug
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))  # post-LN (this was correct)


class Encoder(nn.Module):
    def __init__(self, layer, n):
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, n):
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    """Untied output projection -- independent weight matrix, not shared with the
    embeddings (the bug; see module docstring).
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def make_model(src_vocab, tgt_vocab, n=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Reconstructs Run #1's actual architecture: untied embeddings, biased-variance
    LayerNorm. Uses _LegacyEncoderLayer/_LegacyDecoderLayer (below) rather than
    model.py's EncoderLayer/DecoderLayer, because those construct model.py's
    SublayerConnection internally -- Python binds that name inside model.py's own
    module namespace, so importing EncoderLayer here wouldn't actually pick up this
    file's legacy SublayerConnection.
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(_LegacyEncoderLayer(d_model, c(attn), c(ff), dropout), n),
        Decoder(_LegacyDecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class _LegacyEncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class _LegacyDecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
