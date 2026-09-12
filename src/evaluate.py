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
import argparse

import sacrebleu
import torch
from sacremoses import MosesDetokenizer, MosesTokenizer

from data import BOS, EOS, PAD, Vocab
from model import subsequent_mask

SPECIAL_TOKS = {"<bos>", "<eos>", "<pad>"}


def debpe(line):
    """Reverse subword-nmt BPE: 'un@@ believ@@ able' -> 'unbelievable' (as tokens)."""
    return line.replace("@@ ", "").replace("@@", "")


def length_normalized_score(seq, score, length_penalty):
    # GNMT-style length penalty, as referenced for the paper's beam search config.
    lp = ((5 + seq.size(1)) ** length_penalty) / (5 + 1) ** length_penalty
    return score / lp


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


def beam_search_decode(
    model, src, src_mask, max_len, start_idx, eos_idx, beam_size, length_penalty, device,
    no_repeat_ngram_size=3, prune_by_raw_score=False,
):
    """Single-sentence beam search. src: (1, src_len).

    prune_by_raw_score=False (default) reproduces Runs #1-#3: candidates are pruned by
    length_normalized_score at EVERY step, not just at final selection, which biases the
    search toward short sequences during expansion.

    prune_by_raw_score=True is the corrected behaviour: prune on the raw cumulative
    log-probability during the search, and apply the length penalty only when picking
    the final winner.
    """
    memory = model.encode(src, src_mask)
    beams = [(torch.tensor([[start_idx]], device=device), 0.0)]
    finished = []

    for _ in range(max_len - 1):
        candidates = []
        for seq, score in beams:
            if seq[0, -1].item() == eos_idx:
                finished.append((seq, score))
                continue
            tgt_mask = subsequent_mask(seq.size(1)).to(device)
            out = model.decode(memory, src_mask, seq, tgt_mask)
            log_probs = model.generator(out[:, -1]).clone()
            banned = banned_next_tokens(seq.squeeze(0).tolist(), no_repeat_ngram_size)
            if banned:
                log_probs[0, list(banned)] = float("-inf")
            topk_logp, topk_idx = log_probs.topk(beam_size, dim=-1)
            for k in range(beam_size):
                next_tok = topk_idx[0, k].view(1, 1)
                new_seq = torch.cat([seq, next_tok], dim=1)
                new_score = score + topk_logp[0, k].item()
                candidates.append((new_seq, new_score))
        if not candidates:
            break

        # Pruning key. The length penalty exists to stop the FINAL choice favouring short
        # sequences. Applying it during the search instead distorts comparisons between
        # candidates that are all nearly the same length, biasing expansion toward short
        # output. Correct beam search prunes on the RAW cumulative score here and applies
        # the penalty only at final selection (line below the loop, unchanged).
        if prune_by_raw_score:
            candidates.sort(key=lambda item: item[1], reverse=True)
        else:
            candidates.sort(key=lambda item: length_normalized_score(*item, length_penalty), reverse=True)
        beams = candidates[:beam_size]
        if all(seq[0, -1].item() == eos_idx for seq, _ in beams):
            finished.extend(beams)
            break

    finished.extend(beams)
    best_seq, _ = max(finished, key=lambda item: length_normalized_score(*item, length_penalty))
    return best_seq.squeeze(0).tolist()


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
        "--prune_by_raw_score", action="store_true",
        help="prune beam candidates by raw cumulative log-probability during the search, "
             "applying the length penalty only at final selection. Omit to reproduce "
             "Runs #1-#3 exactly.",
    )
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

    if args.legacy_arch:
        from legacy_model import make_model
    else:
        from model import make_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(args.vocab_path)
    eos_idx = vocab.stoi[EOS]
    # New models use PAD only as the ID placeholder for model.py's zero GO vector.
    # The legacy architecture retains its historical learned-BOS start behavior.
    start_idx = vocab.stoi[BOS] if args.legacy_arch else vocab.stoi[PAD]

    model = make_model(len(vocab), len(vocab)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    src_path = f"{args.data_dir}/{args.split}.bpe.en"
    ref_path = f"{args.data_dir}/{args.split}.raw.de"  # untokenized reference for BLEU

    # newline="\n": keep line-splitting consistent with wc -l / tokenize_and_bpe.py
    # (don't let a lone "\r" from scraped web text be treated as a line break).
    with open(src_path, encoding="utf-8", newline="\n") as f:
        src_lines = [line.strip() for line in f]
    with open(ref_path, encoding="utf-8", newline="\n") as f:
        ref_lines = [line.strip() for line in f]

    if args.limit:
        src_lines = src_lines[: args.limit]
        ref_lines = ref_lines[: args.limit]

    detok = MosesDetokenizer(lang="de")
    hyps = []
    with torch.no_grad():
        for i, line in enumerate(src_lines):
            toks = line.split()
            src = torch.tensor([vocab.encode(toks)], device=device)
            src_mask = torch.ones(1, 1, src.size(1), dtype=torch.bool, device=device)
            out_ids = beam_search_decode(
                model, src, src_mask, args.max_len, start_idx, eos_idx,
                args.beam_size, args.length_penalty, device,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                prune_by_raw_score=args.prune_by_raw_score,
            )
            out_toks_raw = vocab.decode(out_ids)
            out_toks = [t for t in out_toks_raw if t not in SPECIAL_TOKS]
            plain = debpe(" ".join(out_toks))
            hyp = detok.detokenize(plain.split())
            hyps.append(hyp)
            if i < args.show:
                print(f"--- example {i} ---")
                print(f"src (bpe):   {line}")
                print(f"hyp (raw):   {' '.join(out_toks_raw)}")
                print(f"hyp (final): {hyp}")
                print(f"ref:         {ref_lines[i]}")
                print(f"hyp len: {len(out_toks)}  ref len (approx): {len(ref_lines[i].split())}")
            if (i + 1) % 100 == 0:
                print(f"decoded {i + 1}/{len(src_lines)}")

    # ---------------------------------------------------------------------------
    # SCORING -- two methods, reported side by side. They are NOT comparable to each
    # other, and the paper used the second one.
    #
    #   sacrebleu: scores DETOKENIZED hypotheses against RAW references, applying its
    #   own internal "13a" tokenization. Modern standard, reproducible across papers.
    #   This is the method that produced Run #2's 22.26 / 24.30.
    #
    #   multi-bleu style: plain 4-gram BLEU over whitespace-separated tokens, with BOTH
    #   sides Moses-tokenized first. Instead of shelling out to multi-bleu.perl, we
    #   Moses-tokenize both sides here and call sacrebleu with tokenize="none" so it
    #   performs no further tokenization -- the same arithmetic multi-bleu.perl does,
    #   with no extra dependency.
    #
    # multi-bleu style typically scores 1-2 points HIGHER on En-De, so that is the
    # number actually comparable to the paper's 27.3.
    # ---------------------------------------------------------------------------

    # --- original method, unchanged: sacrebleu on detokenized text ---
    bleu = sacrebleu.corpus_bleu(hyps, [ref_lines])
    print(f"\n{args.split} BLEU (sacrebleu, detokenized): {bleu.score:.2f}")
    print(bleu)

    # --- paper-comparable method: Moses-tokenized both sides ---
    mt = MosesTokenizer(lang="de")
    hyps_tok = [" ".join(mt.tokenize(h, escape=False)) for h in hyps]
    refs_tok = [" ".join(mt.tokenize(r, escape=False)) for r in ref_lines]
    bleu_mb = sacrebleu.corpus_bleu(hyps_tok, [refs_tok], tokenize="none", force=True)
    print(f"\n{args.split} BLEU (multi-bleu style, tokenized): {bleu_mb.score:.2f}"
          f"  (paper's base-model En-De BLEU: 27.3)")
    print(bleu_mb)


if __name__ == "__main__":
    main()
