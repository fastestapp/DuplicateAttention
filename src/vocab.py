"""
vocab.py -- Shared (joint src+tgt) vocabulary over BPE'd text. Deliberately has zero
non-stdlib dependencies (no torch) so it can run in a plain local venv -- e.g. via
scripts/build_vocab.py -- without needing the full training environment installed.
"""
from collections import Counter

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]


class Vocab:
    def __init__(self, counter, min_freq=1, max_size=None):
        self.itos = list(SPECIALS)
        for tok, freq in counter.most_common(max_size):
            if freq < min_freq:
                continue
            if tok not in SPECIALS:
                self.itos.append(tok)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    @classmethod
    def build(cls, *files, min_freq=2, max_size=None):
        counter = Counter()
        for path in files:
            # newline="\n": don't let a lone "\r" (common in scraped web text, e.g.
            # the CommonCrawl portion of WMT14) be treated as a line break by
            # Python's default universal-newlines mode.
            with open(path, encoding="utf-8", newline="\n") as f:
                for line in f:
                    counter.update(line.strip().split())
        return cls(counter, min_freq=min_freq, max_size=max_size)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.itos))

    @classmethod
    def load(cls, path):
        v = cls.__new__(cls)
        with open(path, encoding="utf-8", newline="\n") as f:
            v.itos = [line.rstrip("\n") for line in f]
        v.stoi = {tok: i for i, tok in enumerate(v.itos)}
        return v
