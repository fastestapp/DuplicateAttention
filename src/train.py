"""
train.py -- 2x H100 DDP training loop for the base Transformer, following section 5.3
of "Attention Is All You Need": Adam (b1=0.9, b2=0.98, eps=1e-9), Noam warmup + inverse-
sqrt LR schedule (warmup_steps=4000), label smoothing (eps=0.1), ~25,000 effective
tokens/batch, 100,000 steps.

Launch with:
    torchrun --standalone --nproc_per_node=2 train.py [args]

DDP essentials:
  - one process per GPU (LOCAL_RANK env var set by torchrun)
  - dataset sharded per rank in data.py (build_dataloader(rank=..., world_size=...))
  - per-rank token budget = 12,500 -> summed across 2 GPUs = ~25,000 effective tokens,
    matching the paper's base-model batch size (see data.py's TokenBudgetBatchSampler)

SYSTEMATIC RE-APPLICATION IN PROGRESS (see README "Systematic re-application log"):
Reverted to the exact training loop Run #1 used (step_100000.pt), so each round-2
review fix can be tested in its own isolated run rather than all at once. Specifically
still using, unfixed, on purpose:
  - Loss normalized by each rank's own LOCAL token count, not a true cross-rank global
    count -- only approximately correct when ranks' token counts happen to be close.
  - Checkpoint saving evaluated once per epoch (not every step) -- with ~5,954 steps
    per epoch, this produced "last 5" checkpoints spanning ~23% of training when
    averaged, which measurably hurt Run #1's result. Being tested on its own later.
  - No gradient clipping.
  - No validation loss during training, no fp32 toggle -- these were pure add-ons,
    not bug fixes, so they're not part of what we're testing right now either.
"""
import argparse
import contextlib
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import PAD, Batch, Vocab, build_dataloader
from model import LabelSmoothing, count_parameters, make_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class NoamOpt:
    """LR schedule from the paper, section 5.3, eq. 3:
    lrate = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
    """

    def __init__(self, model_size, warmup, optimizer, factor=1.0, max_rate=None):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self.max_rate = max_rate
        self._rate = 0.0

    def rate(self, step=None):
        step = self._step if step is None else step
        step = max(step, 1)
        rate = self.factor * (
            self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )
        return min(rate, self.max_rate) if self.max_rate is not None else rate

    def step(self):
        self._step += 1
        rate = self.rate()
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return {
            "step": self._step,
            "optimizer": self.optimizer.state_dict(),
            "max_rate": self.max_rate,
        }

    def load_state_dict(self, sd):
        self._step = sd["step"]
        self.optimizer.load_state_dict(sd["optimizer"])
        self.max_rate = sd.get("max_rate", self.max_rate)


def setup_runtime(device_name):
    """Choose a local device or initialize CUDA DDP when launched by torchrun.

    DDP is deliberately opt-in through torchrun's WORLD_SIZE/LOCAL_RANK environment.
    A plain ``python train.py`` invocation therefore remains a useful one-process
    CPU, MPS, or CUDA debug path instead of unexpectedly claiming every GPU.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if device_name == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps was requested but MPS is unavailable")

    if distributed:
        if device_name != "cuda":
            raise RuntimeError("multi-process training requires CUDA/NCCL; use torchrun only on CUDA")
        if world_size > torch.cuda.device_count():
            raise RuntimeError(
                f"torchrun requested {world_size} processes but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible"
            )
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank), dist.get_rank(), dist.get_world_size(), True

    return torch.device(device_name), 0, 1, False


def autocast_context(device, precision):
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" else "fp32"
    if precision == "bf16":
        if device.type != "cuda":
            raise RuntimeError("bf16 autocast is only enabled for CUDA in this trainer")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def model_for_checkpoint(model):
    return model.module if isinstance(model, DDP) else model


PROBE_STEPS = (1, 25, 50)   # steps at which to dump full state; change freely


def _snapshot_params(model):
    """Full copy of every weight, moved to CPU so the dump is device-independent."""
    m = model_for_checkpoint(model)
    return {n: p.detach().to("cpu", copy=True) for n, p in m.named_parameters()}


def _snapshot_grads(model):
    """Full copy of every gradient (or None), moved to CPU."""
    m = model_for_checkpoint(model)
    return {n: (p.grad.detach().to("cpu", copy=True) if p.grad is not None else None)
            for n, p in m.named_parameters()}


def require_finite(tensor, name, step, batch):
    """Stop at the first non-finite training value with enough batch context to debug."""
    if torch.isfinite(tensor).all():
        return
    src_lengths = (batch.src != 0).sum(dim=1).tolist()
    tgt_lengths = (batch.tgt_y != 0).sum(dim=1).tolist()
    raise FloatingPointError(
        f"non-finite {name} at step {step}; "
        f"src_lengths={src_lengths}; tgt_lengths={tgt_lengths}"
    )


def run_epoch(
    loader, model, criterion, opt, device, pad_idx, rank, total_steps,
    precision="auto", log_every=50, probe_dir=None, clip_grad=None,
    world_size=1, global_loss_norm=False,
    save_by_step=False, ckpt_dir=None, save_every_steps=None, last_saved_step=0,
    save_after_step=0, t_start=None,
):
    model.train()
    generator = model_for_checkpoint(model).generator
    total_tokens, total_loss, tokens_since_log = 0, 0.0, 0
    start = time.time()
    for i, (src, tgt) in enumerate(loader):
        if opt._step >= total_steps:
            break  # stop mid-epoch once the target step count is hit, not just between epochs
        step_num = opt._step + 1  # the step about to run; matches the printed numbering
        probe = probe_dir is not None and rank == 0 and step_num in PROBE_STEPS
        batch = Batch(src, tgt, pad_idx).to(device)

        weights_before = _snapshot_params(model) if probe else None

        # Loss normalizer.
        #
        #   global_loss_norm=False (Runs #1-#3): divide by THIS RANK's own token count.
        #   DDP then AVERAGES the per-rank gradients, which only equals the true
        #   per-token gradient when both ranks happen to hold the same token count.
        #
        #   global_loss_norm=True: divide by (global_tokens / world_size). Since DDP
        #   averages, dividing by N/W makes each rank contribute loss_r * W / N, and the
        #   average of those is exactly sum(loss_r) / N -- the true global per-token
        #   loss. Costs one small all_reduce per step.
        #
        # When the ranks are balanced the two are identical, which is why the old path
        # was "approximately correct".
        denom = batch.ntokens
        if global_loss_norm and world_size > 1:
            ntok = torch.tensor([float(batch.ntokens)], device=device, dtype=torch.float64)
            dist.all_reduce(ntok, op=dist.ReduceOp.SUM)
            denom = ntok.item() / world_size

        with autocast_context(device, precision):
            gen_input = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            log_probs = generator(gen_input)
            loss = criterion(
                log_probs.contiguous().view(-1, log_probs.size(-1)),
                batch.tgt_y.contiguous().view(-1),
            ) / denom
        require_finite(loss, "loss", opt._step + 1, batch)
        loss.backward()

        # Gradient clipping (round-2 fix, THE single variable under test in Run #3).
        # Placement matters: AFTER backward, because DDP has already all-reduced the
        # gradients by that point so every rank clips identical values; and BEFORE
        # opt.step(), because the clipped gradients are what the optimizer must see.
        # clip_grad_norm_ returns the total L2 norm measured BEFORE clipping, which is
        # what gets logged -- if that number stays below clip_grad the whole run, then
        # clipping never fired and the experiment says nothing.
        grad_norm = float("nan")
        if clip_grad is not None:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad))

        grads = _snapshot_grads(model) if probe else None

        opt.step()
        opt.zero_grad()

        # ---- checkpoint saving, evaluated PER STEP (round-2 fix, Run #7) ------------
        # Runs #1-#6 saved from main(), i.e. only once per epoch, so --save_every_steps
        # could never fire more often than an epoch boundary (~5,878 steps). The "last 5"
        # therefore spanned ~24% of training. The paper averaged checkpoints written 10
        # minutes apart -- roughly the last 7%. Saving here, inside the step loop, is what
        # makes a tight averaging window possible.
        #
        # save_after_step lets you write frequently only near the end, so a tight window
        # costs a handful of checkpoints instead of one per interval for the whole run.
        if (save_by_step and rank == 0 and ckpt_dir and save_every_steps
                and opt._step >= save_after_step
                and opt._step - last_saved_step >= save_every_steps):
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"step_{opt._step}.pt")
            torch.save(
                {"model": model_for_checkpoint(model).state_dict(),
                 "opt": opt.state_dict(), "step": opt._step},
                ckpt_path,
            )
            last_saved_step = opt._step
            elapsed_hr = (time.time() - t_start) / 3600 if t_start else float("nan")
            print(f"[rank0] saved {ckpt_path} | elapsed {elapsed_hr:.2f}h | "
                  f"step {opt._step}")

        if probe:
            os.makedirs(probe_dir, exist_ok=True)
            path = os.path.join(probe_dir, f"probe_step_{step_num:04d}_{device.type}.pt")
            torch.save({
                "step": step_num,
                "device": device.type,
                "loss": float(loss.detach().cpu().item()),
                "weights_before": weights_before,
                "grads": grads,
                "weights_after": _snapshot_params(model),
            }, path)
            print(f"[probe] wrote {path}")

        total_loss += loss.item() * batch.ntokens
        total_tokens += batch.ntokens
        tokens_since_log += batch.ntokens

        if rank == 0 and opt._step % log_every == 0:
            elapsed = time.time() - start
            toks_per_sec = tokens_since_log / max(elapsed, 1e-6)
            print(
                f"step {opt._step:>7} | batch {i:>6} | loss {loss.item():.4f} "
                f"| lr {opt._rate:.2e} | gnorm {grad_norm:.2f} | tok/s {toks_per_sec:,.0f}"
            )
            start = time.time()
            tokens_since_log = 0
    return total_loss / max(total_tokens, 1), last_saved_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "data/wmt14_en_de/prepared"))
    parser.add_argument("--vocab_path", default=str(PROJECT_ROOT / "data/wmt14_en_de/prepared/vocab.shared"))
    parser.add_argument("--ckpt_dir", default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--max_tokens", type=int, default=12500, help="per-GPU token budget")
    parser.add_argument(
        "--pad_to", type=int, default=None,
        help="optionally pad source and target tensors to at least this length",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="random seed for model initialization and dynamic-batch ordering",
    )
    parser.add_argument("--layers", type=int, default=6, help="encoder and decoder layer count")
    parser.add_argument("--d_model", type=int, default=512, help="Transformer hidden width")
    parser.add_argument("--d_ff", type=int, default=2048, help="feed-forward hidden width")
    parser.add_argument("--heads", type=int, default=8, help="attention head count")
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument(
        "--max_lr", type=float, default=None,
        help="optional ceiling on the Noam learning rate (useful for tiny-corpus controls)",
    )
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument(
        "--save_every_steps", type=int, default=5000,
        help="checkpoint interval, evaluated once per epoch (see module docstring "
             "for why this is coarser in practice than the number suggests)",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument(
        "--save_by_step", action="store_true",
        help="evaluate the checkpoint-save condition every STEP instead of once per "
             "epoch. Runs #1-#6 saved per epoch, so --save_every_steps could never fire "
             "more often than an epoch boundary (~5,878 steps) and the 'last 5' spanned "
             "~24%% of training. The paper averaged checkpoints ~10 min apart (~7%%). "
             "Omit to reproduce Runs #1-#6 exactly.",
    )
    parser.add_argument(
        "--save_after_step", type=int, default=0,
        help="with --save_by_step, only start saving once this step is reached. Lets you "
             "checkpoint frequently near the end without writing hundreds of files. "
             "E.g. --save_by_step --save_every_steps 1500 --save_after_step 92000 gives "
             "5 checkpoints over the last 8%% of a 100k-step run.",
    )
    parser.add_argument(
        "--global_loss_norm", action="store_true",
        help="normalize the loss by the CROSS-RANK global token count instead of each "
             "rank's own count. Corrects the DDP gradient when ranks hold unequal token "
             "counts. Omit to reproduce Runs #1-#3 exactly. No effect on 1 GPU.",
    )
    parser.add_argument(
        "--clip_grad", type=float, default=None,
        help="clip gradients to this max L2 norm before the optimizer step. The paper "
             "does not specify clipping; 1.0 is the standard choice. Omit to disable, "
             "which reproduces Run #1 and Run #2 exactly.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Transformer dropout probability (use 0 for a tiny-corpus overfit test)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="DataLoader workers (default: 0 on MPS, 2 otherwise)",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
        help="auto chooses CUDA, then Apple MPS, then CPU. CUDA uses one visible GPU "
             "with python train.py; use torchrun for 2-GPU DDP.",
    )
    parser.add_argument(
        "--precision", default="auto", choices=["auto", "fp32", "bf16"],
        help="auto uses bf16 on CUDA and fp32 on MPS/CPU",
    )
    args = parser.parse_args()
    if args.d_model % args.heads:
        parser.error("--d_model must be divisible by --heads")
    if args.pad_to is not None and args.pad_to <= 0:
        parser.error("--pad_to must be positive")
    if args.max_lr is not None and args.max_lr <= 0:
        parser.error("--max_lr must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device, rank, world_size, distributed = setup_runtime(args.device)
    if rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        if device.type == "cuda":
            print(
                f"[rank0] runtime: CUDA | GPU(s) visible: {torch.cuda.device_count()} | "
                f"device: {torch.cuda.get_device_name(device)} | world size: {world_size}"
            )
            if not distributed and torch.cuda.device_count() > 1:
                print("[rank0] using one GPU; launch with torchrun --nproc_per_node=2 for DDP on two GPUs")
        elif device.type == "mps":
            print("[rank0] runtime: Apple Metal (MPS) | world size: 1 | precision: fp32")
        else:
            print("[rank0] runtime: CPU | world size: 1 | precision: fp32")
        print(f"[rank0] random seed: {args.seed}")
        print(f"[rank0] gradient clipping: "
              f"{'OFF (matches Run #1 and #2)' if args.clip_grad is None else f'max L2 norm {args.clip_grad}'}")
        print(f"[rank0] loss normalization: "
              f"{'GLOBAL cross-rank token count' if args.global_loss_norm else 'per-rank token count (matches Runs #1-#3)'}")
        print(f"[rank0] precision: {args.precision}"
              f"{' (-> bf16 on CUDA)' if args.precision == 'auto' and device.type == 'cuda' else ''}")
        if args.save_by_step:
            print(f"[rank0] checkpointing: every {args.save_every_steps} steps"
                  f"{f' from step {args.save_after_step}' if args.save_after_step else ''}")
        else:
            print(f"[rank0] checkpointing: epoch boundaries only, >= "
                  f"{args.save_every_steps} steps apart (matches Runs #1-#6)")
        if args.max_lr is not None:
            print(f"[rank0] Noam learning-rate ceiling: {args.max_lr:.2e}")
    if distributed:
        dist.barrier()

    vocab = Vocab.load(args.vocab_path)
    pad_idx = vocab.stoi[PAD]
    num_workers = args.num_workers
    if num_workers is None:
        # macOS uses spawn workers, adding substantial setup/pickling cost for a
        # short local smoke test. CUDA production retains the established default.
        num_workers = 0 if device.type == "mps" else 2

    model = make_model(
        len(vocab), len(vocab), n=args.layers, d_model=args.d_model,
        d_ff=args.d_ff, h=args.heads, dropout=args.dropout,
    ).to(device)
    if rank == 0:
        n_params = count_parameters(model)
        print(f"[rank0] model parameters: {n_params:,} (paper's base config: ~65M -- "
              f"a number near 106M here would mean embedding tying isn't actually wired up)")
        emb_scale = (model.src_embed[0].lut.weight.std() * (args.d_model ** 0.5)).item()
        pe_scale = model.src_embed[1].pe[:, :100].std().item()
        print(f"[rank0] embedding scale {emb_scale:.3f} | positional-encoding scale "
              f"{pe_scale:.3f} | ratio {emb_scale / pe_scale:.2f} (paper target ~1.4)")
    if distributed:
        model = DDP(model, device_ids=[device.index])

    criterion = LabelSmoothing(
        size=len(vocab), padding_idx=pad_idx, smoothing=args.label_smoothing
    ).to(device)

    base_opt = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    opt = NoamOpt(
        model_size=args.d_model, warmup=args.warmup_steps, optimizer=base_opt,
        max_rate=args.max_lr,
    )

    train_loader, train_sampler = build_dataloader(
        f"{args.data_dir}/train.bpe.en",
        f"{args.data_dir}/train.bpe.de",
        vocab,
        max_tokens=args.max_tokens,
        shuffle=True,
        num_workers=num_workers,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        pad_to=args.pad_to,
    )

    step, epoch, last_saved_step = 0, 0, 0
    t_start = time.time()
    while step < args.total_steps:
        train_sampler.set_epoch(epoch)
        avg_loss, last_saved_step = run_epoch(
            train_loader, model, criterion, opt, device, pad_idx, rank, args.total_steps,
            precision=args.precision, probe_dir=args.ckpt_dir, clip_grad=args.clip_grad,
            world_size=world_size, global_loss_norm=args.global_loss_norm,
            save_by_step=args.save_by_step, ckpt_dir=args.ckpt_dir,
            save_every_steps=args.save_every_steps, last_saved_step=last_saved_step,
            save_after_step=args.save_after_step, t_start=t_start,
        )
        step = opt._step
        epoch += 1

        # Epoch-boundary saving. Used by Runs #1-#6, and kept as the default so those
        # runs stay reproducible. With --save_by_step the saving happens inside
        # run_epoch instead and this block is skipped.
        if (not args.save_by_step
                and rank == 0 and step - last_saved_step >= args.save_every_steps):
            ckpt_path = os.path.join(args.ckpt_dir, f"step_{step}.pt")
            torch.save(
                {"model": model_for_checkpoint(model).state_dict(), "opt": opt.state_dict(), "step": step},
                ckpt_path,
            )
            last_saved_step = step
            elapsed_hr = (time.time() - t_start) / 3600
            print(f"[rank0] saved {ckpt_path} | elapsed {elapsed_hr:.2f}h | epoch {epoch} | avg_loss {avg_loss:.4f}")

    if rank == 0:
        if last_saved_step != step:
            final_path = os.path.join(args.ckpt_dir, f"step_{step}_final.pt")
            torch.save(
                {"model": model_for_checkpoint(model).state_dict(), "opt": opt.state_dict(), "step": step},
                final_path,
            )
        total_hr = (time.time() - t_start) / 3600
        print(f"[rank0] training done | {step} steps | {total_hr:.2f}h wall-clock | last checkpoint step {last_saved_step}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
