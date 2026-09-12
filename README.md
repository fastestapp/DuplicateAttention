# Attention Is All You Need — Base Transformer Reproduction

A from-scratch PyTorch reproduction of the base Transformer described in
[Attention Is All You Need](https://arxiv.org/abs/1706.03762), trained for English-to-German translation on WMT14.

The project documents both the implementation and the experimental process: training runs, failed hypotheses, debugging controls, architecture corrections, and comparisons with the paper’s reported result.

## Model configuration

- 6 encoder layers
- 6 decoder layers
- Model dimension: 512
- Feed-forward dimension: 2,048
- 8 attention heads
- Shared English–German vocabulary
- 37,000 joint BPE merge operations
- Tied encoder, decoder, and output embeddings
- Adam optimizer with the paper’s Noam learning-rate schedule
- Label smoothing: 0.1
- Beam size: 4
- Length-penalty exponent: 0.6
- Approximately 65 million trainable parameters

## Repository structure

- `src/model.py` — Transformer architecture
- `src/train.py` — training loop, distributed training, checkpointing, and optimization
- `src/evaluate.py` — beam-search decoding and BLEU evaluation
- `src/data.py` — datasets, dynamic token-budget batching, and masks
- `src/vocab.py` — shared vocabulary handling
- `src/legacy_model.py` — compatibility with early checkpoints
- `src/average_checkpoints.py` — checkpoint averaging
- `src/overfit_local.py` — small-corpus overfitting control
- `scripts/prepare_data.sh` — download and preprocess WMT14 English–German
- `scripts/build_vocab.py` — construct the shared vocabulary
- `scripts/tokenize_and_bpe.py` — tokenization and BPE utilities
- `RESULTS.md` — chronological experimental results and analysis
- `PARAMETER_BREAKDOWN.md` — exact parameter-count breakdown
- `CROSS_ATTENTION_WALKTHROUGH.md` — explanation of cross-attention
- `EVALUATE_ANNOTATED.md` — annotated evaluation walkthrough

Additional diagnostic programs in `src/` investigate numerical behavior, projection operations, and Apple MPS reproducibility.

## Installation

Python 3.10 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch
pip install -r requirements.txt
```

On a managed CUDA system such as Lambda GPU Cloud, use the compatible preinstalled PyTorch build rather than replacing it unnecessarily.

## Prepare the data

The preparation script downloads the WMT14 English–German training corpora, obtains the WMT13 validation and WMT14 test sets, applies Moses tokenization, learns 37,000 joint BPE merge rules, and applies those rules to both languages.

```bash
cd scripts
bash prepare_data.sh
python build_vocab.py
cd ..
```

Generated datasets are intentionally excluded from Git because they are large and reproducible from the preparation scripts.

## Train

### Single device

The training program automatically selects CUDA, Apple MPS, or CPU:

```bash
python3 train.py --device auto
```

A short smoke test can be run with:

```bash
python3 train.py \
    --device auto \
    --max_tokens 256 \
    --total_steps 10
```

### Two GPUs

```bash
torchrun --standalone --nproc_per_node=2 train.py \
    --device cuda \
    --data_dir data/wmt14_en_de/prepared \
    --vocab_path vocab.shared \
    --ckpt_dir checkpoints
```

The default per-device token budget is 12,500, giving an effective budget of approximately 25,000 tokens across two GPUs.

## Average checkpoints

```bash
python3 average_checkpoints.py \
    --ckpt_dir checkpoints \
    --n 5 \
    --out checkpoints/averaged.pt
```

## Evaluate

```bash
python3 evaluate.py \
    --ckpt checkpoints/averaged.pt \
    --vocab_path vocab.shared \
    --data_dir data/wmt14_en_de/prepared \
    --split test
```

For a quick check:

```bash
python3 evaluate.py \
    --ckpt checkpoints/averaged.pt \
    --vocab_path vocab.shared \
    --data_dir data/wmt14_en_de/prepared \
    --split test \
    --limit 20 \
    --show 5
```

The evaluator contains two BLEU variants:

1. Modern sacreBLEU over detokenized output and raw references.
2. A Moses-tokenized, `multi-bleu`-style score intended for comparison with the paper’s reported 27.3 BLEU.

See `RESULTS.md` for exact experimental configurations, limitations, and measured results.

## Important implementation finding

A major issue discovered during reproduction concerned embedding initialization. Xavier initialization is appropriate for ordinary matrix transformations, where its scale is calculated from the number of input and output connections. An embedding table is instead a lookup table: one vocabulary row is selected at a time, rather than all vocabulary rows contributing to a sum.

Applying Xavier directly to a vocabulary-by-model-dimension embedding table therefore allowed vocabulary size to determine the initialization scale. With approximately 40,100 vocabulary entries and a model dimension of 512, this made the token embeddings about 6.5 times smaller than the intended scale. Positional encodings consequently dominated token identity, allowing the model to learn fluent target-language patterns while conditioning too weakly on the English source.

The corrected embedding initialization uses a standard deviation of:

\[
d_{\text{model}}^{-1/2}
\]

while retaining Xavier initialization for conventional multidimensional projection weights.

## Reproducibility notes

This repository intentionally excludes:

- Downloaded and processed datasets
- Model checkpoints
- Training and evaluation logs
- Python virtual environments
- Editor history and caches
- Cloud credentials and private machine configuration

The scripts and documentation needed to reconstruct the data pipeline, model, training procedure, and evaluation are included.

Some source options preserve earlier experimental behavior so that previous runs can be reproduced exactly. Consult the module documentation and `RESULTS.md` before comparing configurations.

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, Ł., and Polosukhin, I. (2017).  
**Attention Is All You Need.**  
*Advances in Neural Information Processing Systems 30.*

```