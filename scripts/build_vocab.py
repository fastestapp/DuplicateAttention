"""
build_vocab.py -- Build the shared (joint) src+tgt vocab file that train.py and
evaluate.py load, from the BPE'd training data produced by prepare_data.sh /
tokenize_and_bpe.py.

Run this after data prep, before train.py:
    python build_vocab.py --data_dir ../data/wmt14_en_de/prepared
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vocab import Vocab  # noqa: E402  (zero-dependency module, no torch needed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/wmt14_en_de/prepared")
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = args.out or os.path.join(args.data_dir, "vocab.shared")
    vocab = Vocab.build(
        os.path.join(args.data_dir, "train.bpe.en"),
        os.path.join(args.data_dir, "train.bpe.de"),
        min_freq=args.min_freq,
    )
    vocab.save(out_path)
    print(f"Built shared vocab: {len(vocab)} tokens -> {out_path}")


if __name__ == "__main__":
    main()
