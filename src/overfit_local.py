"""
overfit_local.py -- Lightweight CPU/MPS train+eval harness for FAST LOCAL DEBUGGING.

Purpose: the "overfit test". Train a SMALL model on a SMALL subset of the real data,
on your Mac (CPU or Apple-GPU/MPS -- no CUDA, no rented H100), hard enough that it
should essentially MEMORIZE those specific sentence pairs. A correct encoder-decoder
can drive training loss to ~0 and reproduce its training targets almost perfectly.

  - If it CANNOT overfit (loss stays high / it can't reproduce sentences it trained on),
    there is a real structural bug (masking, loss, teacher forcing, tied generator,
    decode, ...). You found it locally, in minutes, for free.
  - If it CAN overfit (loss -> ~0, high token accuracy, sample decodes match targets),
    then model + training loop + decoder are fundamentally correct, and the poor
    full-scale BLEU (~5-6) is a DIFFERENT class of problem: generalization, data scale,
    or an eval/tokenization mismatch. That also narrows the search enormously.

This reuses the EXACT same model.py / data.py / vocab.py code the real (H100) pipeline
uses -- it only strips out DDP and runs single-process on a laptop-friendly device, so
what you're testing here is the real implementation, not a toy reimplementation.

Typical usage (from the src/ directory, in your venv with torch installed):

    # fastest smoke test (<1 min on MPS, a couple min on CPU):
    python overfit_local.py --n_sentences 100 --steps 300

    # a bit more thorough:
    python overfit_local.py --n_sentences 300 --steps 800 --show 5

    # force CPU if MPS gives trouble:
    python overfit_local.py --device cpu

By default it builds a TINY vocab from just the chosen subset (so the model is small
and training is fast). Pass --shared_vocab to instead use the real 37k vocab.shared
(slower, heavier, but exercises the exact tokenization the H100 runs use).
"""
import argparse
import time

import torch

from data import Batch, collate_pad
from model import LabelSmoothing, count_parameters, make_model, subsequent_mask
from vocab import BOS, EOS, PAD, Vocab
from collections import Counter


def pick_device(explicit):
    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def read_subset(src_path, tgt_path, n, max_len):
    """Read the first n usable (short-enough, non-empty) BPE sentence pairs."""
    pairs = []
    # newline="\n": match data.py / tokenize_and_bpe.py so a lone "\r" in scraped web
    # text isn't treated as a line break (keeps src/tgt aligned).
    with open(src_path, encoding="utf-8", newline="\n") as fs, \
         open(tgt_path, encoding="utf-8", newline="\n") as ft:
        for s_line, t_line in zip(fs, ft):
            s_toks, t_toks = s_line.strip().split(), t_line.strip().split()
            if not s_toks or not t_toks:
                continue
            if len(s_toks) > max_len or len(t_toks) > max_len:
                continue
            pairs.append((s_toks, t_toks))
            if len(pairs) >= n:
                break
    return pairs


def build_tiny_vocab(pairs):
    """Shared src+tgt vocab built from ONLY the subset -- keeps the model small/fast.
    min_freq=1 so every token in the subset is covered (no UNKs on the training set).
    """
    counter = Counter()
    for s_toks, t_toks in pairs:
        counter.update(s_toks)
        counter.update(t_toks)
    return Vocab(counter, min_freq=1)


def make_batches(pairs, vocab, batch_size, device):
    """Encode pairs into (src, tgt) tensors and pack into fixed-size padded batches."""
    pad_idx = vocab.stoi[PAD]
    encoded = []
    for s_toks, t_toks in pairs:
        src = torch.tensor(vocab.encode(s_toks), dtype=torch.long)
        tgt = torch.tensor(
            [vocab.stoi[BOS]] + vocab.encode(t_toks) + [vocab.stoi[EOS]],
            dtype=torch.long,
        )
        encoded.append((src, tgt))
    batches = []
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i:i + batch_size]
        src_out, tgt_out = collate_pad(chunk, pad_idx)
        batches.append(Batch(src_out, tgt_out, pad_idx).to(device))
    return batches


@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len, bos_idx, eos_idx, device):
    """Simple greedy decode (no beam) -- enough to check the model reproduces targets."""
    memory = model.encode(src, src_mask)
    ys = torch.tensor([[bos_idx]], device=device)
    for _ in range(max_len - 1):
        tgt_mask = subsequent_mask(ys.size(1)).to(device)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        logp = model.generator(out[:, -1])
        nxt = int(logp.argmax(dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == eos_idx:
            break
    return ys[0].tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="../data/wmt14_en_de/prepared")
    p.add_argument("--vocab_path", default="../data/wmt14_en_de/prepared/vocab.shared")
    p.add_argument("--n_sentences", type=int, default=200, help="subset size to overfit")
    p.add_argument("--max_len", type=int, default=40, help="skip pairs longer than this (keeps it fast)")
    p.add_argument("--steps", type=int, default=600, help="optimizer steps")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--show", type=int, default=3, help="print this many src/hyp/ref decode examples")
    p.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    p.add_argument("--shared_vocab", action="store_true",
                   help="use the real 37k vocab.shared instead of a tiny subset vocab (slower)")
    p.add_argument("--cpu_grad_input", action="store_true",
                   help="compute the output projection's grad(gen_input) on CPU "
                        "(repairs the MPS large-K backward fault; no-op on cpu/cuda)")
    p.add_argument("--grad_input_mode", choices=["cpu", "chunk"], default="cpu",
                   help="with --cpu_grad_input: 'cpu' offloads the matmul to the CPU; "
                        "'chunk' stays on-device and splits the vocab contraction")
    p.add_argument("--chunk_k", type=int, default=1024,
                   help="vocab-chunk size when --grad_input_mode chunk")
    # Small model by default -- big enough to learn, small enough to train on a laptop.
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"[local] device: {device}")

    pairs = read_subset(
        f"{args.data_dir}/train.bpe.en", f"{args.data_dir}/train.bpe.de",
        args.n_sentences, args.max_len,
    )
    print(f"[local] loaded {len(pairs)} sentence pairs (max_len={args.max_len})")
    if not pairs:
        raise SystemExit("No pairs loaded -- check --data_dir paths.")

    if args.shared_vocab:
        vocab = Vocab.load(args.vocab_path)
        print(f"[local] using shared vocab.shared: {len(vocab)} tokens")
    else:
        vocab = build_tiny_vocab(pairs)
        print(f"[local] built tiny subset vocab: {len(vocab)} tokens "
              f"(pass --shared_vocab to use the full 37k instead)")

    pad_idx = vocab.stoi[PAD]
    bos_idx, eos_idx = vocab.stoi[BOS], vocab.stoi[EOS]

    model = make_model(
        len(vocab), len(vocab),
        n=args.layers, d_model=args.d_model, d_ff=args.d_ff, h=args.heads,
    ).to(device)
    print(f"[local] model params: {count_parameters(model):,} "
          f"(d_model={args.d_model}, layers={args.layers}, heads={args.heads}, d_ff={args.d_ff})")

    if args.cpu_grad_input:
        from model import ProjectionCPUGradInput
        model.generator.cpu_grad_input = True
        ProjectionCPUGradInput.MODE = args.grad_input_mode
        ProjectionCPUGradInput.CHUNK_K = args.chunk_k
        if args.grad_input_mode == "chunk":
            print(f"[local] SPLICE ON (chunk, k={args.chunk_k}): grad(gen_input) "
                  f"computed on-device in chunks")
        else:
            print("[local] SPLICE ON (cpu): projection grad(gen_input) computed on CPU")

    criterion = LabelSmoothing(size=len(vocab), padding_idx=pad_idx, smoothing=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    batches = make_batches(pairs, vocab, args.batch_size, device)

    # --- train: loop over the fixed subset until we hit the step budget ---
    model.train()
    t0 = time.time()
    step = 0
    last_loss = float("nan")
    while step < args.steps:
        for b in batches:
            if step >= args.steps:
                break
            out = model(b.src, b.tgt, b.src_mask, b.tgt_mask)
            logp = model.generator(out)
            loss = criterion(
                logp.contiguous().view(-1, logp.size(-1)),
                b.tgt_y.contiguous().view(-1),
            ) / b.ntokens
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            last_loss = loss.item()
            step += 1
            if step % args.log_every == 0 or step == 1:
                print(f"[local] step {step:>5}/{args.steps} | loss {last_loss:.4f} "
                      f"| {step / max(time.time() - t0, 1e-6):.1f} steps/s")
    print(f"[local] trained {step} steps in {time.time() - t0:.1f}s | final loss {last_loss:.4f}")

    # --- eval: greedy-decode the SAME training sentences, measure reproduction ---
    model.eval()
    total_tok, correct_tok, exact = 0, 0, 0
    shown = 0
    for s_toks, t_toks in pairs:
        src = torch.tensor([vocab.encode(s_toks)], device=device)
        src_mask = torch.ones(1, 1, src.size(1), dtype=torch.bool, device=device)
        out_ids = greedy_decode(src=src, src_mask=src_mask, model=model,
                                max_len=args.max_len + 5, bos_idx=bos_idx,
                                eos_idx=eos_idx, device=device)
        hyp = [vocab.itos[i] for i in out_ids if vocab.itos[i] not in ("<bos>", "<eos>", "<pad>")]
        ref = t_toks
        # token accuracy: fraction of reference positions the hyp gets right (position-wise)
        for j in range(len(ref)):
            total_tok += 1
            if j < len(hyp) and hyp[j] == ref[j]:
                correct_tok += 1
        if hyp == ref:
            exact += 1
        if shown < args.show:
            print(f"\n  src: {' '.join(s_toks)}")
            print(f"  hyp: {' '.join(hyp)}")
            print(f"  ref: {' '.join(ref)}")
            shown += 1

    tok_acc = 100.0 * correct_tok / max(total_tok, 1)
    exact_pct = 100.0 * exact / max(len(pairs), 1)
    print(f"\n[local] RESULT on the {len(pairs)} training sentences:")
    print(f"[local]   position-wise token accuracy: {tok_acc:.1f}%")
    print(f"[local]   exactly-reproduced sentences: {exact}/{len(pairs)} ({exact_pct:.1f}%)")
    print("[local] interpretation:")
    print("[local]   HIGH accuracy (>~85%) + low final loss  => model/loop/decode look CORRECT;")
    print("[local]     the ~5-6 BLEU at full scale is a generalization/data/eval issue, not a code bug.")
    print("[local]   LOW accuracy + stuck loss                => a real structural bug is present;")
    print("[local]     debug it here (cheap) before spending on another H100 run.")


if __name__ == "__main__":
    main()
