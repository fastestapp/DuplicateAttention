"""
data.py -- Vocabulary, dataset, and batching for the WMT14 En-De BPE data produced by
scripts/prepare_data.sh (or scripts/tokenize_and_bpe.py). Batches are sized by *total
token count* (dynamic batching), matching the paper's ~25,000-tokens/batch convention
for the base model, rather than a fixed number of sentences per batch.

DDP note: build_dataloader() takes rank/world_size and shards the dataset by index
(examples[rank::world_size]) so each of the 2 H100 processes sees a disjoint slice of
the data every epoch -- avoids duplicated work across GPUs.

SYSTEMATIC RE-APPLICATION IN PROGRESS (see README "Systematic re-application log"):
Change #2 being tested in isolation: the TokenBudgetBatchSampler shuffle-then-sort bug
is now fixed (see its docstring) -- everything else remains reverted to Run #1's
original behavior, plus tied embeddings (change #1, already tested -- see README).
"""
import random
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from model import subsequent_mask
from vocab import BOS, EOS, PAD, SPECIALS, UNK, Vocab  # noqa: F401  (re-exported for callers)


class ParallelTextDataset(Dataset):
    """Aligned BPE'd source/target files, one sentence per line."""

    def __init__(self, src_path, tgt_path, vocab, max_len=256, rank=0, world_size=1):
        examples = []
        # newline="\n": don't let a lone "\r" (common in scraped web text, e.g. the
        # CommonCrawl portion of WMT14) be treated as a line break by Python's default
        # universal-newlines mode -- that silently desyncs src/tgt line counts from
        # what wc -l reports and from each other. Match tokenize_and_bpe.py's handling.
        with open(src_path, encoding="utf-8", newline="\n") as fs, \
             open(tgt_path, encoding="utf-8", newline="\n") as ft:
            for s_line, t_line in zip(fs, ft):
                s_toks = s_line.strip().split()
                t_toks = t_line.strip().split()
                if not s_toks or not t_toks:
                    continue
                if len(s_toks) > max_len or len(t_toks) > max_len:
                    continue
                examples.append((s_toks, t_toks))
        # DDP shard: each rank owns a disjoint slice, so 2 GPUs never train on
        # duplicate sentence pairs in the same epoch.
        self.examples = examples[rank::world_size]
        self.vocab = vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        s_toks, t_toks = self.examples[idx]
        src = torch.tensor(self.vocab.encode(s_toks), dtype=torch.long)
        tgt = torch.tensor(
            self.vocab.encode(t_toks) + [self.vocab.stoi[EOS]],
            dtype=torch.long,
        )
        return src, tgt


def collate_pad(batch, pad_idx, pad_to=None):
    srcs, tgts = zip(*batch)
    src_len = max(x.size(0) for x in srcs)
    tgt_len = max(x.size(0) for x in tgts)
    if pad_to is not None:
        src_len = max(src_len, pad_to)
        tgt_len = max(tgt_len, pad_to)
    src_out = torch.full((len(batch), src_len), pad_idx, dtype=torch.long)
    tgt_out = torch.full((len(batch), tgt_len), pad_idx, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_out[i, : s.size(0)] = s
        tgt_out[i, : t.size(0)] = t
    return src_out, tgt_out


def collate_with_pad(batch, pad_idx, pad_to=None):
    """Pickle-safe DataLoader collate entry point for macOS spawn workers."""
    return collate_pad(batch, pad_idx, pad_to)


class Batch:
    """Wraps a padded (src, tgt) pair with the masks model.py's EncoderDecoder expects.
    Labels are ``[y1, ..., EOS]``. Decoder IDs are those labels shifted right with a
    PAD placeholder at position zero; model.py replaces that placeholder's embedding
    with the fixed all-zero GO vector before adding positional encoding.
    """

    def __init__(self, src, tgt, pad_idx):
        self.src = src
        self.src_mask = (src != pad_idx).unsqueeze(-2)
        self.tgt_y = tgt
        self.tgt = torch.full_like(tgt, pad_idx)
        self.tgt[:, 1:] = tgt[:, :-1]
        # The first decoder ID is PAD only as a placeholder for the zero GO vector,
        # so validity must come from the labels rather than decoder token IDs.
        self.tgt_mask = self._make_std_mask(self.tgt_y, pad_idx)
        self.ntokens = (self.tgt_y != pad_idx).sum().item()

    @staticmethod
    def _make_std_mask(labels, pad_idx):
        tgt_mask = (labels != pad_idx).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(labels.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

    def to(self, device):
        self.src = self.src.to(device)
        self.src_mask = self.src_mask.to(device)
        self.tgt = self.tgt.to(device)
        self.tgt_y = self.tgt_y.to(device)
        self.tgt_mask = self.tgt_mask.to(device)
        return self


class TokenBudgetBatchSampler(Sampler):
    """Buckets examples by length (low padding waste), then chunks into batches
    capped by total token count rather than sentence count -- matches the paper's
    ~25,000-tokens/batch convention. With 2 GPUs, pass max_tokens=12500 per rank so
    the *effective* global batch (summed across ranks) lands at ~25,000 tokens.

    FIXED (change #2 in the systematic re-application, see module docstring):
    shuffling the full index list and then immediately sorting ALL of it by length
    used to undo almost the entire shuffle -- batch order ended up nearly identical
    every epoch (deterministic short-to-long order, randomness surviving only among
    ties). Fixed with the standard shuffle -> chunk into pools -> sort-within-pool ->
    shuffle-batch-order pattern: examples can only ever be batched with others from
    the same pool (bounding padding waste), but which pool they land in, and the
    order pools/batches are presented in, is now actually randomized per epoch.
    pool_size is a tunable middle ground -- smaller pools waste less padding but
    shuffle less; larger pools shuffle more thoroughly but waste more padding.
    """

    def __init__(self, dataset, max_tokens=12500, shuffle=True, seed=0, pool_size=200_000):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.pool_size = pool_size
        self.lengths = [max(len(s), len(t)) for s, t in dataset.examples]

    def set_epoch(self, epoch):
        self.epoch = epoch  # call each epoch so shuffling differs across epochs

    def _batches_from_pool(self, pool_indices):
        pool_indices = sorted(pool_indices, key=lambda i: self.lengths[i])
        batches, batch, max_len_in_batch = [], [], 0
        for idx in pool_indices:
            candidate_max = max(max_len_in_batch, self.lengths[idx])
            if batch and candidate_max * (len(batch) + 1) > self.max_tokens:
                batches.append(batch)
                batch, max_len_in_batch = [], 0
            batch.append(idx)
            max_len_in_batch = max(max_len_in_batch, self.lengths[idx])
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        rng = random.Random(self.seed + self.epoch) if self.shuffle else None
        if rng:
            rng.shuffle(indices)  # randomizes which examples land in which pool

        batches = []
        for start in range(0, len(indices), self.pool_size):
            batches.extend(self._batches_from_pool(indices[start:start + self.pool_size]))

        if rng:
            rng.shuffle(batches)  # randomizes batch ORDER across the epoch

        yield from batches

    def __len__(self):
        # NOT total_tokens // max_tokens -- that floor-division estimate ignores
        # padding waste within each batch, the leftover partial batch at the end of
        # every pool, and imperfect packing generally, so it silently undercounts.
        # Materializing the real batch list is the only way to get an exact count;
        # cheap in practice since nothing in this codebase currently calls len() on
        # this sampler during training (verified), so this only runs if something
        # asks for it directly.
        return sum(1 for _ in self)


def build_dataloader(
    src_path, tgt_path, vocab, max_tokens=12500, shuffle=True,
    num_workers=2, rank=0, world_size=1, seed=0, pad_to=None,
):
    ds = ParallelTextDataset(src_path, tgt_path, vocab, rank=rank, world_size=world_size)
    sampler = TokenBudgetBatchSampler(ds, max_tokens=max_tokens, shuffle=shuffle, seed=seed)
    pad_idx = vocab.stoi[PAD]
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=partial(collate_with_pad, pad_idx=pad_idx, pad_to=pad_to),
        num_workers=num_workers,
    )
    return loader, sampler
