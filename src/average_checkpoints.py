"""
average_checkpoints.py -- Average the weights of the last N checkpoints, as done in
"Attention Is All You Need" section 5.3 ("For the base models, we used a single model
obtained by averaging the last 5 checkpoints").

Usage:
    python average_checkpoints.py --ckpt_dir ../checkpoints --n 5 --out ../checkpoints/averaged.pt
"""
import argparse
import glob
import os
import re

import torch


def latest_n_checkpoints(ckpt_dir, n):
    # Match exactly "step_<digits>.pt" -- NOT "step_<digits>_final.pt". train.py writes
    # both a regular checkpoint and a "_final" copy of the same weights at the last
    # step, and without this anchor both would match, share the same step number, and
    # risk being double-counted (dropping an earlier, genuinely distinct checkpoint).
    paths = [
        p for p in glob.glob(os.path.join(ckpt_dir, "step_*.pt"))
        if re.fullmatch(r"step_\d+\.pt", os.path.basename(p))
    ]

    def step_of(path):
        m = re.search(r"step_(\d+)\.pt$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    paths = [p for p in paths if step_of(p) >= 0]
    paths.sort(key=step_of)
    return paths[-n:]


def average_state_dicts(paths):
    avg_state = None
    for path in paths:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt["model"]
        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k].float()
    for k in avg_state:
        avg_state[k] /= len(paths)
    return avg_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="../checkpoints")
    parser.add_argument("--n", type=int, default=5, help="number of trailing checkpoints to average")
    parser.add_argument("--out", default="../checkpoints/averaged.pt")
    args = parser.parse_args()

    paths = latest_n_checkpoints(args.ckpt_dir, args.n)
    if len(paths) < args.n:
        print(f"Warning: only found {len(paths)} checkpoints, averaging what's available.")
    print("Averaging:")
    for p in paths:
        print(f"  {p}")

    avg_state = average_state_dicts(paths)
    torch.save({"model": avg_state, "averaged_from": paths}, args.out)
    print(f"Saved averaged checkpoint to {args.out}")


if __name__ == "__main__":
    main()
