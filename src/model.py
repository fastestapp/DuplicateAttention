"""
model.py -- Base Transformer architecture, following "Attention Is All You Need"
(Vaswani et al., 2017), base config: N=6, d_model=512, d_ff=2048, h=8, dropout=0.1.

Structure follows the widely-used "Annotated Transformer" (Harvard NLP) layout.

SYSTEMATIC RE-APPLICATION IN PROGRESS (see README "Systematic re-application log"):
This file has been reverted to the exact architecture Run #1 was trained with
(step_100000.pt), with ONE deliberate exception: tied embeddings (below). Everything
else -- including the LayerNorm bug -- is intentionally back to the original,
unfixed state, so we can test each correction in isolation via real training runs
instead of changing 10 things at once and not knowing which one (if any) mattered.

  - LayerNorm: uses a hand-rolled version built on torch.std(), which defaults to
    the UNBIASED (Bessel's-correction) variance estimator -- not what LayerNorm is
    defined to use (population variance). This is a real bug relative to spec, but
    deliberately NOT fixed here yet -- it's next in the queue, tested on its own.
  - Post-LN placement (paper section 3.1: LayerNorm AFTER the residual add) was
    already correct in Run #1 and was never part of the round-2 fix list, so it's
    unchanged either way.
  - Embeddings ARE tied (paper section 3.4): "we share the same weight matrix
    between the two embedding layers and the pre-softmax linear transformation."
    This is the one change kept in this revert, per discussion -- it's the fix
    judged most likely to actually matter (cuts ~106M params to ~65M, matches what
    the paper's reported 27.3 BLEU was measured with), so it's being tested first
    rather than reverted along with everything else.

count_parameters() (bottom of file) is also kept, deliberately -- it doesn't affect
training or the model's output at all, it only prints a number. It exists solely to
verify the one change above actually landed (~65M tied vs. ~106M if untied), which
matters precisely because we're now testing that specific change in isolation.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, n):
    """Produce n identical (but independently parameterized) layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class LayerNorm(nn.Module):
    """ORIGINAL hand-rolled LayerNorm -- torch.std() defaults to the unbiased
    (Bessel's-correction) variance estimator, not the population variance LayerNorm
    is defined to use. This is a known bug (see module docstring), deliberately left
    unfixed here so it can be tested in its own isolated run later.
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
    """Residual connection followed by layer norm -- post-LN, per the paper (this
    part was already correct in Run #1, unrelated to the LayerNorm variance bug).
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))  # post-LN


class Encoder(nn.Module):
    def __init__(self, layer, n):
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, n):
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
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


def subsequent_mask(size):
    """Mask out subsequent positions so the decoder can't attend to future tokens."""
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    # Part 15:
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class ProjectionCPUGradInput(torch.autograd.Function):
    """Output-projection linear whose INPUT-gradient is computed on the CPU.

    Diagnostic/repair for a specific MPS fault. Forward is the ordinary linear
    (logits = x @ W^T + b). In backward, the gradient w.r.t. the input --
    grad_gen_input = grad_logits @ W, a contraction over the full ~40k-word vocab --
    is computed on the CPU and moved back to the input's device. We isolated this one
    step as the point where MPS diverges from CPU by ~4e-2 despite byte-identical
    inputs (grad_logits and W both match CPU). grad_weight / grad_bias use only a
    small (batch*seq) contraction and show no divergence, so they stay on-device;
    ONLY the faulty large-K matmul is offloaded.
    """

    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return F.linear(x, weight, bias)

    # How to compute grad(gen_input) in backward:
    #   "cpu"   -- offload the faulty matmul to the CPU (known-good, but leaves the GPU)
    #   "chunk" -- stay on-device, but split the ~40k vocab contraction into CHUNK_K
    #              pieces and accumulate the partials in fp32
    MODE = "cpu"
    CHUNK_K = 1024

    @staticmethod
    def backward(ctx, grad_logits):
        x, weight = ctx.saved_tensors
        dev = grad_logits.device
        gl = grad_logits.detach()
        w = weight.detach()

        if ProjectionCPUGradInput.MODE == "chunk":
            # on-device fix: shorter reductions, fp32 accumulation of the partials
            k = w.shape[0]
            step = ProjectionCPUGradInput.CHUNK_K
            grad_x = torch.zeros(*gl.shape[:-1], w.shape[1], dtype=torch.float32, device=dev)
            for start in range(0, k, step):
                stop = min(start + step, k)
                grad_x = grad_x + (gl[..., start:stop].float() @ w[start:stop].float())
        else:
            # the isolated faulty step, done on the CPU where it is correct
            grad_x = (gl.cpu() @ w.cpu()).to(dev)

        # grad_weight / grad_bias: small contraction, left on-device (not the fault).
        gl2 = grad_logits.reshape(-1, grad_logits.shape[-1])
        x2 = x.reshape(-1, x.shape[-1])
        grad_weight = gl2.t() @ x2
        grad_bias = gl2.sum(0) if ctx.has_bias else None
        return grad_x, grad_weight, grad_bias


class Generator(nn.Module):
    """Final linear + log-softmax projection to target vocab.

    If tied_weight is given (the shared embedding table's weight, shape (vocab,
    d_model) -- same shape nn.Linear expects for a (d_model -> vocab) projection),
    self.proj.weight is *replaced* with that exact Parameter object, so gradients to
    the projection and to the embedding lookup accumulate into the same tensor. Paper
    section 3.4. This is the one change being kept from the round-2 fix list -- see
    module docstring.

    cpu_grad_input (default False): when True, route the projection through
    ProjectionCPUGradInput so grad(gen_input) is computed on the CPU. This repairs the
    MPS large-K backward divergence; on CPU/CUDA it is a no-op difference (same math).
    """

    def __init__(self, d_model, vocab, tied_weight=None):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)
        if tied_weight is not None:
            assert tied_weight.shape == self.proj.weight.shape
            self.proj.weight = tied_weight
        self.cpu_grad_input = False

    def forward(self, x):
        if self.cpu_grad_input:
            logits = ProjectionCPUGradInput.apply(x, self.proj.weight, self.proj.bias)
        else:
            logits = self.proj(x)
        return F.log_softmax(logits, dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(src_vocab, tgt_vocab, n=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Base config from the paper: N=6, d_model=512, d_ff=2048, h=8, dropout=0.1.
    Label smoothing (eps=0.1) lives in the loss (see LabelSmoothing below), not here.

    Embeddings are tied (paper section 3.4, kept from the round-2 fix list -- see
    module docstring): one embedding table is shared by the source embedding, target
    embedding, and the generator's output projection. This requires a shared
    vocabulary, which is what our pipeline builds (vocab.shared) -- hence the assert.
    """
    assert src_vocab == tgt_vocab, (
        "tied embeddings (paper section 3.4) require a shared src/tgt vocabulary -- "
        "pass the same vocab size for both, as build_vocab.py's vocab.shared provides"
    )
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    # ONE embedding table (not three) -- reused, not deep-copied, for src_embed and
    # tgt_embed, and its weight is handed to Generator to tie the output projection too.
    shared_embed = Embeddings(d_model, src_vocab)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n),
        nn.Sequential(shared_embed, c(position)),
        nn.Sequential(shared_embed, c(position)),
        Generator(d_model, tgt_vocab, tied_weight=shared_embed.lut.weight),
    )
    # Xavier init, as noted in the paper's reference implementations. model.parameters()
    # de-duplicates by Parameter identity, so the tied embedding/projection matrix is
    # initialized exactly once here, not three times.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # ---- ROUND-2 FIX #1: embedding init. THE single variable under test in Run #2. ----
    # Xavier scales by 1/sqrt(fan_in + fan_out). For the tied embedding table that is
    # (vocab x d_model) = (39,984 x 512), so fan_in+fan_out ~ 40,496 and the resulting
    # std is ~0.0068 -- roughly 6x smaller than the paper's scale of d_model^-0.5
    # (~0.0442). Embeddings are multiplied by sqrt(d_model) in Embeddings.forward and
    # then ADDED to the positional encodings (std ~0.7); at Xavier's scale the token
    # identity is swamped by position. Applied AFTER the Xavier loop so it overrides
    # that loop for this one tensor. Verify at startup: train.py prints the
    # embedding/positional ratio, which should be ~1.4, not ~0.2.
    nn.init.normal_(shared_embed.lut.weight, mean=0.0, std=d_model ** -0.5)

    return model


def count_parameters(model):
    """Total trainable parameters, de-duplicated (tied weights counted once). Kept
    deliberately in this revert (see module docstring) -- it's the verification tool
    for the one change we're keeping: ~65M confirms tying is wired up, ~106M would
    mean it silently isn't. Doesn't affect training or output in any way.
    """
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if p.requires_grad:
            total += p.numel()
    return total


class LabelSmoothing(nn.Module):
    """KL-divergence label smoothing, eps=0.1 in the paper (section 5.4)."""

    def __init__(self, size, padding_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0 and mask.size(0) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist.clone().detach())
