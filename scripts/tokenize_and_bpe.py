"""
tokenize_and_bpe.py -- Tokenize + BPE-encode a parallel (en, de) split *line-by-line in
lockstep*, guaranteeing the output has exactly as many lines as the input on both sides.

Why this exists: the sacremoses/subword-nmt CLIs process each language file
independently as a whole-text blob. Web-scraped text (esp. the CommonCrawl portion of
WMT14) occasionally contains exotic line-separator-like characters that get treated as
extra line breaks on one side but not the other (English and German content don't share
the same oddities), silently shifting the two files out of alignment by a handful of
lines -- exactly what happened with train.bpe.en/de (4,590,804 vs 4,590,661 lines,
should both have been 4,590,101). Doing this explicitly, one raw line at a time via the
Python APIs (not the file-level CLIs), makes that class of bug impossible: the loop
itself guarantees 1 input line -> 1 output line, always.

Usage:
    # 1. Tokenize (run once per split, before BPE codes exist):
    python tokenize_and_bpe.py tokenize --src_raw train.raw.en --tgt_raw train.raw.de \
        --src_out train.tok.en --tgt_out train.tok.de

    # 2. Learn BPE the normal way (whole-corpus stats, order doesn't matter here):
    subword-nmt learn-joint-bpe-and-vocab --input train.tok.en train.tok.de -s 37000 \
        -o bpe.codes --write-vocabulary vocab.en vocab.de

    # 3. Apply BPE in lockstep for every split:
    python tokenize_and_bpe.py apply-bpe --src_in train.tok.en --tgt_in train.tok.de \
        --src_out train.bpe.en --tgt_out train.bpe.de \
        --codes bpe.codes --src_vocab vocab.en --tgt_vocab vocab.de
"""
import argparse

from sacremoses import MosesTokenizer
from subword_nmt.apply_bpe import BPE, read_vocabulary


def cmd_tokenize(args):
    tok_src = MosesTokenizer(lang=args.src_lang)
    tok_tgt = MosesTokenizer(lang=args.tgt_lang)

    n = 0
    # newline="\n" disables Python's universal-newlines line splitting, which by
    # default also breaks lines on a lone "\r" -- an artifact common in scraped web
    # text (CommonCrawl) that wc -l doesn't count (it only counts literal "\n" bytes).
    # Without this, Python and wc -l can disagree on line count for the same file.
    with open(args.src_raw, encoding="utf-8", newline="\n") as fs_in, \
         open(args.tgt_raw, encoding="utf-8", newline="\n") as ft_in, \
         open(args.src_out, "w", encoding="utf-8", newline="\n") as fs_out, \
         open(args.tgt_out, "w", encoding="utf-8", newline="\n") as ft_out:
        for s_line, t_line in zip(fs_in, ft_in):
            # .tokenize() takes ONE line, returns tokens for THAT line -- no internal
            # whole-file splitting, so this can't drift out of lockstep.
            s_toks = tok_src.tokenize(s_line.strip(), escape=False, aggressive_dash_splits=True)
            t_toks = tok_tgt.tokenize(t_line.strip(), escape=False, aggressive_dash_splits=True)
            fs_out.write(" ".join(s_toks) + "\n")
            ft_out.write(" ".join(t_toks) + "\n")
            n += 1
            if n % 200_000 == 0:
                print(f"tokenized {n:,} lines")
    print(f"Done: {n:,} lines written to {args.src_out} and {args.tgt_out}")


def cmd_apply_bpe(args):
    with open(args.codes, encoding="utf-8") as codes_f:
        bpe = BPE(codes_f, separator="@@")

    src_vocab = None
    if args.src_vocab:
        with open(args.src_vocab, encoding="utf-8") as vf:
            src_vocab = read_vocabulary(vf, args.vocab_threshold)
    tgt_vocab = None
    if args.tgt_vocab:
        with open(args.tgt_vocab, encoding="utf-8") as vf:
            tgt_vocab = read_vocabulary(vf, args.vocab_threshold)

    n = 0
    with open(args.src_in, encoding="utf-8", newline="\n") as fs_in, \
         open(args.tgt_in, encoding="utf-8", newline="\n") as ft_in, \
         open(args.src_out, "w", encoding="utf-8", newline="\n") as fs_out, \
         open(args.tgt_out, "w", encoding="utf-8", newline="\n") as ft_out:
        for s_line, t_line in zip(fs_in, ft_in):
            bpe.vocab = src_vocab
            fs_out.write(bpe.process_line(s_line.strip()) + "\n")
            bpe.vocab = tgt_vocab
            ft_out.write(bpe.process_line(t_line.strip()) + "\n")
            n += 1
            if n % 500_000 == 0:
                print(f"BPE'd {n:,} lines")
    print(f"Done: {n:,} lines written to {args.src_out} and {args.tgt_out}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_tok = sub.add_parser("tokenize")
    p_tok.add_argument("--src_raw", required=True)
    p_tok.add_argument("--tgt_raw", required=True)
    p_tok.add_argument("--src_out", required=True)
    p_tok.add_argument("--tgt_out", required=True)
    p_tok.add_argument("--src_lang", default="en")
    p_tok.add_argument("--tgt_lang", default="de")
    p_tok.set_defaults(func=cmd_tokenize)

    p_bpe = sub.add_parser("apply-bpe")
    p_bpe.add_argument("--src_in", required=True)
    p_bpe.add_argument("--tgt_in", required=True)
    p_bpe.add_argument("--src_out", required=True)
    p_bpe.add_argument("--tgt_out", required=True)
    p_bpe.add_argument("--codes", required=True)
    p_bpe.add_argument("--src_vocab", default=None)
    p_bpe.add_argument("--tgt_vocab", default=None)
    p_bpe.add_argument("--vocab_threshold", type=int, default=50)
    p_bpe.set_defaults(func=cmd_apply_bpe)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
