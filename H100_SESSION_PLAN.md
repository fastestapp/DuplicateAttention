# H100 Session Plan — Run #2: embedding-init fix (single variable)

**Hypothesis under test:** Run #1's ~5.5–6.6 BLEU was caused by Xavier init crushing the
tied embedding table (~6x too small at vocab 39,984). Fix: `normal_(std=d_model**-0.5)`,
already in `model.py`.

**Discipline:** ONE change vs Run #1. Everything else stays as Run #1 had it — per-rank
loss normalization, once-per-epoch checkpoint evaluation, no gradient clipping, bf16.
Those are known-imperfect and are deliberately NOT being fixed in this run.

**Budget:** ~$175 remaining of ~$250. Cost control is a first-class goal below.

---

## SECTION 0 — Before you rent anything (on the Mac, free)

- [ ] **0.1 Confirm the fix is in the code you'll deploy**
  ```bash
  grep -n "normal_" src/model.py          # expect: nn.init.normal_(shared_embed.lut.weight, mean=0.0, std=d_model ** -0.5)
  ```

- [ ] **0.2 Confirm the MPS work cannot leak into the CUDA path**
  ```bash
  grep -n "cpu_grad_input\|ProjectionCPUGradInput" src/train.py    # expect: NOTHING
  ```
  (`Generator.cpu_grad_input` defaults to False; only `overfit_local.py` sets it.)

- [x] **0.3 Transfer method — DECIDED: scp everything from the Mac. No GitHub.**
  There is NO persistent storage on the rented instance — code and data must be uploaded
  every session, with `scp -r`. Measure the data size tonight so you know how long the
  upload will take before the meter starts:
  ```bash
  du -sh data/wmt14_en_de/prepared/
  ```

- [x] **0.4 Run #1 baseline — ALREADY EXTRACTED (done 2026-08-09)**

  | quantity | Run #1 value |
  |---|---|
  | hardware | **2 x H100** (Lambda, Texas, Jul 15–16 2025) — confirmed by epoch arithmetic |
  | total steps | 100,000 |
  | wall clock | **2.29 h** |
  | throughput | **~153,000 tok/s** (rank-0) |
  | steps per epoch | **5,954** (~16.8 epochs total) |
  | effective batch | ~25,000 tokens (12,500/GPU x 2) — **matches the paper** |
  | **loss @ step ~19,000** | **~1.72** |
  | loss @ step ~20,400 | ~1.80 |
  | target tokens in corpus | 136,728,300 |
  | final BLEU | ~5.5–6.6 |

  **THE comparison for Run #2: loss at step ~19,000. Run #1 = 1.72.**

  Known gap: `full_train_log.txt` starts at step 19,013 — the early trajectory (steps
  1–19,000), which is exactly where an init fix shows itself, was never captured. Run #2
  MUST use `tee` from step 1 so the next comparison is two-sided.

  Batch size is therefore OFF the suspect list — Run #1 already matched the paper's ~25k.

- [x] **0.5 Run length — DECIDED: truncated 20,000 first, then full 100,000 if it looks good.**

- [ ] **0.6 Data plan.** Decide: re-prepare on the instance, restore from a persistent
  volume, or upload from the Mac. See Section 3.2. This is the single biggest time sink.

---

## SECTION 1 — Instance and cost

### 1.1 Instance
- **2x H100 (80GB) — must match Run #1** or the comparison is invalid (it would change
  the effective batch size from ~25,000 tokens to ~12,500).
- Lambda, Ubuntu (used for Run #1; the Texas instances worked, a Georgia one had a bug).
- Rate ~$6/hr for 2x H100.
- **No persistent storage.** Every session starts empty: upload code + data each time,
  and pull results off before terminating (Section 8).

### 1.2 RUN PLAN — DECIDED: truncated first, then full

Grounded in Run #1's measured 2.29 h for 100,000 steps (~0.0000229 h/step):

| stage | steps | time | cost @ ~$6/hr | purpose |
|---|---:|---:|---:|---|
| **Stage 1 — truncated** | **20,000** | **~28 min** | **~$3** | Compare loss at step ~19,000 against Run #1's **1.72**. Warmup ends at 4,000, so 20k is well clear of it. |
| **Stage 2 — full** | 100,000 | ~2.3 h | ~$14 | Only if Stage 1 beats Run #1. Produces the BLEU number. |

Both stages together ≈ **$17 plus setup time** — comfortably inside the ~$175 remaining.
Budget is not the binding constraint; instance availability and your time are.

**Decision rule after Stage 1:**
- Loss at step ~19,000 **clearly below 1.72** → fix is working. Run Stage 2.
- Loss **at or above ~1.72** → embedding init was NOT the problem. Stop, keep the log,
  terminate. A cheap negative result, and a genuine finding for the article.

Judgement note: 20,000 steps is only ~3.4 epochs, so treat a *small* improvement with
caution — early-training noise is real. A meaningful fix should show a clear gap, not a
rounding-error one.

### 1.3 Hard cost rules
- Set a wall-clock alarm on your phone (~35 min for Stage 1, ~2.5 h for Stage 2).
- Never leave an instance running after training ends — pull checkpoints off first
  (Section 8), then terminate.
- If anything is broken and unfixable in 15 minutes, terminate and debug on the Mac.
- Consider running Stage 1 and Stage 2 in ONE rental: if Stage 1 looks good, relaunch
  immediately at 100,000 steps rather than paying setup cost twice.

---

## SECTION 2 — After SSH: environment (target: 10 minutes)

- [ ] **2.1 Verify the hardware you're paying for**
  ```bash
  nvidia-smi
  ```
  Confirm: 2 GPUs, H100, 80GB each, ~0% utilization, no other tenants.

- [ ] **2.2 Start a persistent session IMMEDIATELY — before anything else**
  ```bash
  tmux new -s train
  ```
  Without this, an SSH drop kills training and burns the whole run.
  Detach: `Ctrl-b` then `d`. Reattach: `tmux attach -t train`.

- [ ] **2.3 Verify PyTorch sees both GPUs**
  ```bash
  python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
  ```
  Expect device_count() == 2.

- [ ] **2.4 Dependencies**
  ```bash
  pip install -r requirements.txt
  pip install sacrebleu          # if not in requirements
  ```

---

## SECTION 3 — Code and data

**Transfer method: scp from the Mac. No GitHub.**

### 3.1 Create the directory on the instance
```bash
# on the INSTANCE:
mkdir -p ~/attention_reproduction/data/wmt14_en_de
```

### 3.2 Push the code (from the Mac, ~1 MB, seconds)
```bash
# from the Mac, in the project root:
cd ~/attention_reproduction

scp -r src scripts requirements.txt user@INSTANCE:~/attention_reproduction/
```
Small enough that copying all of `src/` (including the MPS diagnostic scripts) is
harmless — they're inert on CUDA and cost nothing to carry.

### 3.3 Push the data — the slow part, START IT FIRST
`data/wmt14_en_de/prepared/` holds ~137M target tokens; expect **1–3 GB** and a transfer
time set entirely by your home uplink. Check the size before you rent anything:

```bash
# on the Mac, BEFORE renting:
du -sh data/wmt14_en_de/prepared/
```

At a typical ~20 Mbps upload, 2 GB is roughly 15 minutes — **while the instance bills**.

```bash
# from the Mac:
scp -r data/wmt14_en_de/prepared user@INSTANCE:~/attention_reproduction/data/wmt14_en_de/
```

**Overlap it with setup.** Start the transfer in one terminal and do Section 2 (tmux,
nvidia-smi, pip install) in another while it runs — don't sit and watch the progress bar
at $6/hr.

**If it drops partway**, resume rather than restarting from zero:
```bash
rsync -avP data/wmt14_en_de/prepared/ user@INSTANCE:~/attention_reproduction/data/wmt14_en_de/prepared/
```

### 3.4 Verify the transfer landed intact — DO NOT SKIP
```bash
# on the INSTANCE:
cd ~/attention_reproduction

ls -la data/wmt14_en_de/prepared/
wc -l data/wmt14_en_de/prepared/train.bpe.en data/wmt14_en_de/prepared/train.bpe.de
```
The two line counts **must match exactly** — a truncated scp misaligns source and target
and silently destroys training.

```bash
wc -l data/wmt14_en_de/prepared/vocab.shared     # expect ~39,984
awk '{n+=NF} END {print n}' data/wmt14_en_de/prepared/train.bpe.de   # expect 136,728,300
```
That last number is the exact token count measured on the Mac — if it matches, the corpus
transferred byte-complete.

```bash
grep -n "normal_" src/model.py     # CONFIRM the embedding fix travelled
```

---

## SECTION 4 — Pre-flight checks (target: 15 minutes, ~$1). DO NOT SKIP.

These are cheap and each one can save an entire wasted run.

- [ ] **4.1 CUDA overfit test — also tests H100 candidate #2 for free**
  ```bash
  cd src
  python3 overfit_local.py --shared_vocab --device cuda --n_sentences 100 --steps 2000 \
    --log_every 200 --show 3
  ```
  **Expect: loss → ~0.03, 100% token accuracy, 100/100 exact** (this is what CPU and
  fixed-MPS both achieved).
  - PASSES → the CUDA path is numerically sound; the large-K backward problem found on
    MPS does **not** affect CUDA. That substantially clears candidate #2 (bf16/precision)
    for ~$1 and without a separate experiment.
  - FAILS → **stop.** You've found a CUDA analogue of the MPS bug, and that is almost
    certainly the real cause of Run #1. Do not start the full run; investigate instead
    (first move: rerun with `--device cpu` on the same instance to confirm it's CUDA-specific).

- [ ] **4.2 Embedding-scale check — verifies the fix is live**
  Start training for ~30 seconds and read the startup banner. `train.py` prints:
  ```
  [rank0] embedding scale X.XXX | positional-encoding scale Y.YYY | ratio Z.ZZ (paper target ~1.4)
  ```
  **The ratio must be near 1.4.** Run #1 with Xavier would have shown roughly 0.2–0.3.
  If the ratio is still low, the fix is not active — stop and fix before spending hours.

- [ ] **4.3 Parameter count**
  Same banner prints total parameters. **Expect ~65M.** A number near 106M means embedding
  tying is not wired up.

- [ ] **4.4 Smoke run — 200 steps, both GPUs**
  ```bash
  torchrun --standalone --nproc_per_node=2 train.py --total_steps 200 --ckpt_dir ../checkpoints_smoke
  ```
  Watch for: loss falling from ~10, `tok/s` in a sane range (compare to Run #1's), both
  GPUs busy in `nvidia-smi`, no NaN/inf crash from `require_finite`.

---

## SECTION 5 — Launch

### STAGE 1 — truncated (20,000 steps, ~28 min, ~$3)

```bash
cd src
tmux attach -t train      # make sure you are INSIDE tmux

torchrun --standalone --nproc_per_node=2 train.py \
  --total_steps 20000 \
  --ckpt_dir ../checkpoints_run2_s1 \
  2>&1 | tee ../run2_stage1_log.txt
```

Notes:
- Everything else is left at defaults, which already match Run #1 and the paper:
  `--max_tokens 12500` per GPU (~25k effective), `--warmup_steps 4000`,
  `--label_smoothing 0.1`, `--dropout 0.1`, 6 layers, d_model 512, d_ff 2048, 8 heads,
  `--precision auto` (= bf16 on CUDA), `--save_every_steps 5000`, `--seed 0`.
  **Do not change any of them** — embedding init is the single variable.
- `tee` is not optional. Run #1's log lost steps 1–19,000 and that's exactly the region
  where an init fix reveals itself. Capture from step 1 this time.
- Detach with `Ctrl-b` `d`; the run survives an SSH drop.

**Then apply the Section 1.2 decision rule** (loss at ~step 19,000 vs Run #1's 1.72)
before spending anything more.

### STAGE 2 — full (100,000 steps, ~2.3 h, ~$14) — only if Stage 1 wins

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --total_steps 100000 \
  --ckpt_dir ../checkpoints_run2 \
  2>&1 | tee ../run2_full_log.txt
```

Relaunch in the same rental if possible — you've already paid the setup cost.

---

## SECTION 6 — Monitoring

Reattach any time with `tmux attach -t train`.

- [ ] **6.1 The comparison that matters.**
  ```bash
  grep "^step  19" ../run2_stage1_log.txt      # Run #1 was ~1.72 here
  ```
  - **Clearly below 1.72 → the fix is working.** Proceed to Stage 2.
  - **At or above 1.72 → embedding init was not the problem.** Stop, save the log,
    terminate. A cheap, valid, publishable negative result.

  Also worth capturing (Run #1 has no data here, so it's new information regardless):
  loss at steps 1,000 / 4,000 (end of warmup) / 10,000.

- [ ] **6.1b Throughput sanity.** `tok/s` should be near Run #1's **153,000**. Far below
  means a setup problem (dataloader, wrong GPU count) and invalidates the comparison.

- [ ] **6.2 GPU utilization** (separate pane: `Ctrl-b` `"`)
  ```bash
  watch -n 5 nvidia-smi
  ```
  Both GPUs should sit high. Persistent low utilization = dataloader bottleneck
  (consider `--num_workers 4`).

- [ ] **6.3 Disk**
  ```bash
  df -h .        # checkpoints are ~250MB each at 65M params
  ```

---

## SECTION 7 — Evaluation

- [ ] **7.1 Average the trailing checkpoints** (paper section 5.4 does this)
  ```bash
  cd src
  python3 average_checkpoints.py --ckpt_dir ../checkpoints_run2 --n 5 --out ../checkpoints_run2/averaged.pt
  ```
  Caveat to record: with once-per-epoch saving, the "last 5" may span a wide slice of
  training — this is a known Run #1 defect being carried forward deliberately.

- [ ] **7.2 BLEU on the test set**
  ```bash
  python3 evaluate.py --ckpt ../checkpoints_run2/averaged.pt --split test \
    --beam_size 4 --length_penalty 0.6 2>&1 | tee ../run2_eval_test.txt
  ```
  Do **not** pass `--legacy_arch` (that flag is only for Run #1's `step_100000.pt`).

- [ ] **7.3 Also evaluate the final checkpoint alone**, to separate "model quality" from
  "averaging quality":
  ```bash
  python3 evaluate.py --ckpt ../checkpoints_run2/step_20000.pt --split test | tee ../run2_eval_final.txt
  ```

- [ ] **7.4 Record BLEU methodology.** Note the exact tool/version and whether it's
  tokenized BLEU or sacrebleu — this alone can move the number by several points and
  matters for any claim about matching 27.3.

---

## SECTION 8 — Get the results off the instance, THEN terminate

**Do this before touching the terminate button.** Everything on the instance dies with it.

From the Mac:
```bash
scp user@INSTANCE:~/AttentionDuplicated/run2_train_log.txt  ./
scp user@INSTANCE:~/AttentionDuplicated/run2_eval_test.txt  ./
scp user@INSTANCE:~/AttentionDuplicated/run2_eval_final.txt ./
scp user@INSTANCE:~/AttentionDuplicated/checkpoints_run2/averaged.pt ./checkpoints_backup/
```

- [ ] Verify each file arrived and is non-empty.
- [ ] **Then terminate the instance.** Confirm in the provider console that billing stopped.

---

## Appendix — Failure modes and responses

| symptom | likely cause | response |
|---|---|---|
| Overfit test (4.1) fails on CUDA | CUDA analogue of the MPS large-K bug | STOP. Do not run. Investigate — this would be the real cause. |
| Embedding ratio not ~1.4 | fix not in deployed code | STOP. Check you cloned the right commit. |
| Param count ~106M | embedding tying not wired | STOP. Wrong model file / wrong branch. |
| `FloatingPointError: non-finite loss` | instability; no grad clipping (deliberate) | Note the step and batch lengths; this is itself a finding. |
| Loss stuck ~9 after 1000 steps | LR schedule or data misalignment | Check `wc -l` on train.bpe.en/de match; check warmup. |
| Low GPU utilization | dataloader bound | `--num_workers 4` |
| SSH drops | — | Training survives if inside tmux. Reattach. |
| OOM | batch too large for the config | lower `--max_tokens` (but this breaks the A/B — record it) |

---

## Quick reference — the whole session in one column

```
# --- on the Mac, BEFORE renting ---
du -sh data/wmt14_en_de/prepared/            # know your upload time

# --- on the instance: first minute ---
nvidia-smi                                   # verify 2x H100
tmux new -s train                            # BEFORE anything else
mkdir -p ~/attention_reproduction/data/wmt14_en_de

# --- from the Mac: start the big upload, then work in parallel ---
scp -r src scripts requirements.txt user@INSTANCE:~/attention_reproduction/
scp -r data/wmt14_en_de/prepared user@INSTANCE:~/attention_reproduction/data/wmt14_en_de/

# --- on the instance, while that uploads ---
python3 -c "import torch; print(torch.cuda.device_count())"     # expect 2
cd ~/attention_reproduction && pip install -r requirements.txt

# --- once the upload finishes: verify it landed intact ---
wc -l data/wmt14_en_de/prepared/train.bpe.en data/wmt14_en_de/prepared/train.bpe.de  # MUST match
awk '{n+=NF} END {print n}' data/wmt14_en_de/prepared/train.bpe.de   # expect 136,728,300
grep -n "normal_" src/model.py               # CONFIRM the fix travelled
cd src

# PRE-FLIGHT (~15 min, ~$2)
python3 overfit_local.py --shared_vocab --device cuda --n_sentences 100 --steps 2000
torchrun --standalone --nproc_per_node=2 train.py --total_steps 200 --ckpt_dir ../checkpoints_smoke
#   -> check banner: params ~65M, embedding/positional ratio ~1.4

# STAGE 1 (~28 min, ~$3)
torchrun --standalone --nproc_per_node=2 train.py --total_steps 20000 \
  --ckpt_dir ../checkpoints_run2_s1 2>&1 | tee ../run2_stage1_log.txt
grep "^step  19" ../run2_stage1_log.txt      # vs Run #1's 1.72  -> GO / NO-GO

# STAGE 2 (~2.3 h, ~$14) -- only on GO
torchrun --standalone --nproc_per_node=2 train.py --total_steps 100000 \
  --ckpt_dir ../checkpoints_run2 2>&1 | tee ../run2_full_log.txt
python3 average_checkpoints.py --ckpt_dir ../checkpoints_run2 --n 5 --out ../checkpoints_run2/averaged.pt
python3 evaluate.py --ckpt ../checkpoints_run2/averaged.pt --split test | tee ../run2_eval_test.txt

# scp everything off, THEN terminate
```
