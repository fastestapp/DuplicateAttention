# `evaluate.py`, annotated

The complete file, in order, with an explanation after each piece.

What the file does overall: it loads a trained model, translates English test sentences
into German one at a time using beam search, and scores those translations against the
known-correct German references using BLEU.

---

## 1. Module docstring

```python
"""
evaluate.py -- Beam search decoding + BLEU scoring against newstest2014, matching the
paper's decoding config (section 6.1): beam size 4, length penalty alpha=0.6.

Usage:
    python evaluate.py --ckpt ../checkpoints/averaged.pt \
        --vocab_path ../data/wmt14_en_de/prepared/vocab.shared \
        --data_dir ../data/wmt14_en_de/prepared

Note: this does simple per-sentence (batch=1) beam search for clarity/correctness first.
It's slow across the full ~3000-sentence newstest2014 set -- fine for a one-off eval; use
--limit for a quick sanity check, and batch the beam search later if you want faster
iteration.

SYSTEMATIC RE-APPLICATION IN PROGRESS (see README "Systematic re-application log"):
Beam search decoding reverted to exactly what Run #1's original eval used (length-
normalized-score pruning at every step, no_repeat_ngram_size defaulting to 3) -- see
beam_search_decode's docstring. This is being tested in isolation later, not fixed
here alongside everything else.

The ONE exception is --legacy_arch, added below: it's not a behavior fix, it's
scaffolding needed because model.py itself has already been partially reverted (kept
tied embeddings, reverted LayerNorm) -- so it no longer exactly matches the fully
untied architecture step_100000.pt was actually trained with. --legacy_arch loads
legacy_model.py instead, which still has that exact original (untied) architecture,
so Run #1's checkpoint can still be loaded and evaluated correctly.
"""
```

**Explanation.** The docstring records three things.

First, the decoding settings match the paper: beam size 4, length penalty 0.6.

Second, the docstring admits the speed problem you hit. The phrase "batch=1" means the
script translates exactly one sentence at a time. The author already knew this would be
slow and left a note suggesting batching later.

Third, the docstring records that beam search was deliberately reverted to Run #1's
version, bugs included, so that the beam-search bug can be tested as its own isolated
change.

---

## 2. Imports and the special-token set

```python
import argparse

import sacrebleu
import torch
from sacremoses import MosesDetokenizer

from data import BOS, EOS, PAD, Vocab
from model import subsequent_mask

SPECIAL_TOKS = {"<bos>", "<eos>", "<pad>"}
```

**Explanation.** `sacrebleu` computes the BLEU score. `MosesDetokenizer` turns
tokenized text back into normal German punctuation and spacing. `subsequent_mask` comes
from your model file and prevents the decoder from looking at future positions.

`SPECIAL_TOKS` lists the three bookkeeping tokens that must be stripped out of the
model's output before scoring, because those tokens are not real German words.

---

## 3. Undoing BPE

```python
def debpe(line):
    """Reverse subword-nmt BPE: 'un@@ believ@@ able' -> 'unbelievable' (as tokens)."""
    return line.replace("@@ ", "").replace("@@", "")
```

**Explanation.** Your training data splits rare words into subword pieces, marking a
split with `@@`. The model therefore produces `Fußgän@@ ger` rather than `Fußgänger`.
This function glues those pieces back together. It must run before scoring, because
BLEU compares whole words.

---

## 4. The length penalty

```python
def length_normalized_score(seq, score, length_penalty):
    # GNMT-style length penalty, as referenced for the paper's beam search config.
    lp = ((5 + seq.size(1)) ** length_penalty) / (5 + 1) ** length_penalty
    return score / lp
```

**Explanation.** A sentence's score is the sum of the log-probabilities of its tokens,
and every added token makes that sum more negative. Without a correction, beam search
would always prefer the shortest possible sentence.

This function divides the raw score by a number that grows with sentence length, which
offsets the penalty for being long. The `length_penalty` argument (0.6, from the paper)
controls how strong that offset is.

**This function is at the centre of the known beam-search bug**, described in section 6.

---

## 5. Blocking repeated n-grams

```python
def banned_next_tokens(seq_ids, no_repeat_ngram_size):
    """Return the set of token ids that would complete an n-gram already present
    earlier in seq_ids -- standard "no-repeat n-gram" beam search guard against
    degenerate repetition loops.
    """
    n = no_repeat_ngram_size
    if n <= 0 or len(seq_ids) < n - 1:
        return set()
    seen = {}
    for i in range(len(seq_ids) - n + 1):
        prefix, last = tuple(seq_ids[i:i + n - 1]), seq_ids[i + n - 1]
        seen.setdefault(prefix, set()).add(last)
    current_prefix = tuple(seq_ids[-(n - 1):])
    return seen.get(current_prefix, set())
```

**Explanation.** Language models sometimes fall into loops, repeating a phrase forever.
This function prevents that. With the default setting of 3, the function looks at the
last two tokens generated, finds every three-token sequence earlier in the sentence that
began with those same two tokens, and forbids the model from producing the token that
would complete such a repeat.

You saw exactly the failure this guards against in your MPS experiments, where the model
produced "der der der der der".

---

## 6. Beam search — the core of the file

```python
def beam_search_decode(
    model, src, src_mask, max_len, bos_idx, eos_idx, beam_size, length_penalty, device,
    no_repeat_ngram_size=3,
):
    """Single-sentence beam search. src: (1, src_len).

    KNOWN ISSUE, deliberately not fixed here yet (see module docstring): candidates
    are pruned by length_normalized_score at EVERY step, not just at final selection.
    This biases the search itself toward short sequences during expansion. Being
    tested in isolation later.
    """
    memory = model.encode(src, src_mask)
    beams = [(torch.tensor([[bos_idx]], device=device), 0.0)]
    finished = []
```

**Explanation of the setup.** `model.encode` reads the English sentence once and
produces `memory`, the encoder's representation of it. Every later decoding step reaches
back into `memory` through cross-attention, and `memory` never changes during this
sentence.

`beams` holds the candidate translations currently under consideration. It starts with
one candidate: a sequence containing only the begin-of-sentence token, with score 0.

`finished` collects candidates that have produced an end-of-sentence token.

```python
    for _ in range(max_len - 1):
        candidates = []
        for seq, score in beams:
            if seq[0, -1].item() == eos_idx:
                finished.append((seq, score))
                continue
            tgt_mask = subsequent_mask(seq.size(1)).to(device)
            out = model.decode(memory, src_mask, seq, tgt_mask)
            log_probs = model.generator(out[:, -1]).clone()
```

**Explanation of the generation step.** The outer loop adds one token per pass, up to
`max_len` (256 by default). The inner loop walks each surviving candidate.

If a candidate already ended, that candidate moves to `finished`.

Otherwise the decoder runs and produces `log_probs`: a score for every one of the 39,984
German vocabulary words, describing how likely each word is to come next.

**This is the first speed problem.** The call `model.decode(memory, src_mask, seq,
tgt_mask)` passes the entire sequence generated so far. Producing token 50 means running
the decoder over all 50 tokens, even though tokens 1 through 49 were already computed on
previous passes. A standard implementation caches those earlier computations. This one
recomputes them every time, which makes the work grow with the square of sentence length.

**This is also the second speed problem.** The inner loop `for seq, score in beams`
processes each candidate in a separate call to the model. With beam size 4, that is four
separate model calls per token, where one call handling all four candidates together
would do.

```python
            banned = banned_next_tokens(seq.squeeze(0).tolist(), no_repeat_ngram_size)
            if banned:
                log_probs[0, list(banned)] = float("-inf")
            topk_logp, topk_idx = log_probs.topk(beam_size, dim=-1)
            for k in range(beam_size):
                next_tok = topk_idx[0, k].view(1, 1)
                new_seq = torch.cat([seq, next_tok], dim=1)
                new_score = score + topk_logp[0, k].item()
                candidates.append((new_seq, new_score))
```

**Explanation.** Any token that would create a repeated three-gram has its score set to
negative infinity, which removes that token from consideration.

`topk` then picks the 4 highest-scoring next words. Each of those 4 words extends the
current candidate into a new candidate, and each new candidate's score is the old score
plus the new word's log-probability.

With 4 candidates each producing 4 extensions, this produces 16 candidates per step.

```python
        if not candidates:
            break

        candidates.sort(key=lambda item: length_normalized_score(*item, length_penalty), reverse=True)
        beams = candidates[:beam_size]
        if all(seq[0, -1].item() == eos_idx for seq, _ in beams):
            finished.extend(beams)
            break
```

**Explanation, and this is where the known bug lives.** The 16 candidates are sorted and
only the best 4 survive into the next step. Discarding the other 12 is normal and
necessary; that pruning is what makes beam search affordable.

The bug is the sorting key. The candidates are ranked by `length_normalized_score`, the
length-corrected score, at every single step. Correct beam search ranks by the **raw**
score during the search and applies the length correction only once, when choosing the
final winner.

Why the difference matters: during the search, all 16 candidates are nearly the same
length, so the length correction adds nothing useful. What it does instead is distort
the comparison, because dividing by a length-dependent number changes the ordering of
candidates that a raw comparison would rank differently. The result is a search biased
toward shorter output.

```python
    finished.extend(beams)
    best_seq, _ = max(finished, key=lambda item: length_normalized_score(*item, length_penalty))
    return best_seq.squeeze(0).tolist()
```

**Explanation.** Every remaining candidate joins the finished pool, the highest
length-corrected score wins, and the winning sequence is returned as a plain list of
token numbers.

Using `length_normalized_score` **here** is correct. Using it in the sort above is not.

---

## 7. Command-line arguments

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="../checkpoints/averaged.pt")
    parser.add_argument("--vocab_path", default="../data/wmt14_en_de/prepared/vocab.shared")
    parser.add_argument("--data_dir", default="../data/wmt14_en_de/prepared")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--length_penalty", type=float, default=0.6)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="debug: only score first N sentences")
    parser.add_argument("--show", type=int, default=0, help="debug: print first N src/hyp/ref triples")
    parser.add_argument(
        "--no_repeat_ngram_size", type=int, default=3,
        help="block repeating an n-gram of this size during beam search (0 disables)",
    )
    parser.add_argument(
        "--legacy_arch", action="store_true",
        help="load the model with legacy_model.py's architecture (fully untied "
             "embeddings) instead of model.py (which now keeps tied embeddings as "
             "the one exception in this revert) -- required for Run #1's "
             "step_100000.pt specifically. See module docstring.",
    )
    args = parser.parse_args()
```

**Explanation of the ones that matter to you.**

`--ckpt` selects which saved model to evaluate. You used this flag to compare
`run2_averaged.pt` against `run2_step100000.pt`.

`--limit` restricts the evaluation to the first N sentences. Your 200-sentence runs used
this flag.

`--max_len 256` caps generation at 256 tokens. A model that fails to emit
end-of-sentence will run all the way to this cap, which is why your un-averaged
checkpoint took 21 seconds per sentence.

`--legacy_arch` exists only for Run #1's checkpoint, which was trained with untied
embeddings. Do not pass this flag for Run #2 checkpoints.

---

## 8. Loading the model

```python
    if args.legacy_arch:
        from legacy_model import make_model
    else:
        from model import make_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(args.vocab_path)
    bos_idx, eos_idx = vocab.stoi[BOS], vocab.stoi[EOS]

    model = make_model(len(vocab), len(vocab)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
```

**Explanation.** The device line checks only for an NVIDIA GPU. On your Mac that check
fails, so everything runs on the CPU. The Apple GPU is never considered, which is a
large part of why local evaluation is slow.

`make_model` builds an untrained model with the right shape, `load_state_dict` pours the
trained weights into it, and `model.eval()` switches off dropout.

---

## 9. Reading the test data

```python
    src_path = f"{args.data_dir}/{args.split}.bpe.en"
    ref_path = f"{args.data_dir}/{args.split}.raw.de"  # untokenized reference for BLEU

    with open(src_path, encoding="utf-8", newline="\n") as f:
        src_lines = [line.strip() for line in f]
    with open(ref_path, encoding="utf-8", newline="\n") as f:
        ref_lines = [line.strip() for line in f]

    if args.limit:
        src_lines = src_lines[: args.limit]
        ref_lines = ref_lines[: args.limit]
```

**Explanation.** Two files are read. The English source is read in BPE form, because
that is what the model expects as input. The German reference is read in raw,
untokenized form, because that is what BLEU should be scored against.

`--limit` truncates both lists to the same length, which keeps sentences paired with
their references.

---

## 10. The translation loop

```python
    detok = MosesDetokenizer(lang="de")
    hyps = []
    with torch.no_grad():
        for i, line in enumerate(src_lines):
            toks = line.split()
            src = torch.tensor([vocab.encode(toks)], device=device)
            src_mask = torch.ones(1, 1, src.size(1), dtype=torch.bool, device=device)
            out_ids = beam_search_decode(
                model, src, src_mask, args.max_len, bos_idx, eos_idx,
                args.beam_size, args.length_penalty, device,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            out_toks_raw = vocab.decode(out_ids)
            out_toks = [t for t in out_toks_raw if t not in SPECIAL_TOKS]
            plain = debpe(" ".join(out_toks))
            hyp = detok.detokenize(plain.split())
            hyps.append(hyp)
```

**Explanation.** `torch.no_grad()` tells PyTorch not to track gradients, since no
training happens here.

For each English sentence, the loop converts words to numbers, builds a mask marking
which positions are real, runs beam search, converts the resulting numbers back to
tokens, strips the special tokens, glues the BPE pieces together, and finally
detokenizes into normal German text.

Note that this loop handles one sentence per iteration. Nothing here is batched. This is
the third speed problem: your Mac could translate several sentences simultaneously, and
a GPU could translate many.

```python
            if i < args.show:
                print(f"--- example {i} ---")
                print(f"src (bpe):   {line}")
                print(f"hyp (raw):   {' '.join(out_toks_raw)}")
                print(f"hyp (final): {hyp}")
                print(f"ref:         {ref_lines[i]}")
                print(f"hyp len: {len(out_toks)}  ref len (approx): {len(ref_lines[i].split())}")
            if (i + 1) % 100 == 0:
                print(f"decoded {i + 1}/{len(src_lines)}")
```

**Explanation.** The first block prints example translations when `--show` is used. The
second block prints a progress line every 100 sentences, which is what you watched
during the long run.

---

## 11. Scoring

```python
    bleu = sacrebleu.corpus_bleu(hyps, [ref_lines])
    print(f"\n{args.split} BLEU: {bleu.score:.2f}  (paper's base-model En-De BLEU: 27.3)")
    print(bleu)


if __name__ == "__main__":
    main()
```

**Explanation.** `sacrebleu.corpus_bleu` compares every translation against its
reference and produces the score you have been reading, for example:

```
BLEU = 24.30 55.5/29.9/18.2/11.6 (BP = 1.000 ratio = 1.038 hyp_len = 65097 ref_len = 62688)
```

The four numbers after the score are the fractions of your 1-word, 2-word, 3-word, and
4-word sequences that appear in the reference. `ratio` is your total output length
divided by the reference length; 1.038 means your translations are 3.8% longer than the
references, which is close to ideal.

One caveat for comparing against the paper: `sacrebleu` is stricter than the
`multi-bleu.perl` tool used for the paper's 27.3, and typically scores 1 to 2 points
lower on this language pair.

---

## Summary of the three speed problems

All three live in `beam_search_decode` and the loop that calls it. None of them change
what the model produces, so fixing them should leave BLEU **exactly unchanged** — which
is how you verify the fix is correct.

1. **No caching.** The decoder reruns over the whole sequence for every new token.
2. **Beams run separately.** Four model calls per token where one would do.
3. **Sentences run one at a time.** No batching across sentences.

## Summary of the one correctness problem

`candidates.sort(...)` ranks by length-normalized score during the search. Correct beam
search ranks by raw score during the search and applies length normalization only at
final selection. Fixing this **will** change BLEU, so it must be tested separately from
the speed fixes.

## One unresolved detail

Line 79 seeds the beam with `bos_idx`, and `vocab.shared` lists `<pad>` first and
`<bos>` second, so position 0 of every output should print as `<bos>`. Your recent runs
printed `<pad>` there instead. This does not affect BLEU, because `SPECIAL_TOKS` strips
both tokens before scoring, but the reason for the discrepancy is not yet understood.
