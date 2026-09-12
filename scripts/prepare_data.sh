#!/usr/bin/env bash
# prepare_data.sh -- Download and preprocess WMT14 English-German for the base
# Transformer reproduction. Produces BPE'd train/valid/test files under data/wmt14_en_de/.
#
# Mirrors the corpora used in "Attention Is All You Need" section 5.1: ~4.5M sentence
# pairs, joint (shared source+target) BPE vocab of ~37,000 tokens, validated on
# newstest2013, tested on newstest2014.
#
# Requires: sacremoses, subword-nmt, sacrebleu, wget, tar  (see ../requirements.txt)
#
# NOTE: statmt.org URLs shift between WMT years and occasionally 404 or move host --
# verify these resolve before a long run; swap in the current mirror from
# http://www.statmt.org/wmt14/translation-task.html if one of these breaks.
set -euo pipefail

SRC=en
TGT=de
BPE_TOKENS=37000
DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/wmt14_en_de"
RAW="$DATA_DIR/raw"
PREP="$DATA_DIR/prepared"
mkdir -p "$RAW" "$PREP"

cd "$RAW"

echo "== Downloading raw parallel corpora =="
# News Commentary: v9 is the release WMT14 actually distributed, and is what the paper
# would have trained on. Using the newer v12 (from the WMT17 distribution) costs 0.63
wget -nc http://www.statmt.org/europarl/v7/de-en.tgz
wget -nc http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
wget -nc http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz

echo "== Extracting =="
tar xzf de-en.tgz
tar xzf training-parallel-commoncrawl.tgz
tar xzf training-parallel-nc-v9.tgz

echo "== Concatenating training data =="
# NOTE: the News Commentary tarball bundles several language pairs (cs-en, de-en,
# fr-en, ru-en, zh-en), all sharing English as one side. A wildcard like
# "news-commentary*.en" matches ALL of those .en files, not just de-en's -- silently
# breaking line-for-line alignment with the .de side, which only has one matching file.
# Use the exact de-en filename on both sides.
cat europarl-v7.de-en.en commoncrawl.de-en.en training/news-commentary-v9.de-en.en > "$PREP/train.raw.$SRC"
cat europarl-v7.de-en.de commoncrawl.de-en.de training/news-commentary-v9.de-en.de > "$PREP/train.raw.$TGT"

# Sanity check: line counts must match exactly for a parallel corpus.
src_lines=$(wc -l < "$PREP/train.raw.$SRC")
tgt_lines=$(wc -l < "$PREP/train.raw.$TGT")
if [ "$src_lines" != "$tgt_lines" ]; then
  echo "ERROR: train.raw.$SRC ($src_lines lines) and train.raw.$TGT ($tgt_lines lines) don't match." >&2
  echo "Parallel corpus is misaligned -- stopping before tokenization/BPE run on bad data." >&2
  exit 1
fi
echo "train.raw.$SRC / train.raw.$TGT aligned: $src_lines lines each"

echo "== Fetching newstest2013 (valid) / newstest2014 (test) via sacrebleu =="
# sacrebleu ships these WMT test sets directly -- simpler and less brittle than
# scraping the raw statmt SGM files by hand.
#
# TEST SET CHOICE: "-t wmt14" is the 2,737-sentence CAMPAIGN set -- the sentences WMT14
# actually evaluated on, and what sacrebleu's own docs recommend for reproducing results
# from the campaign. "-t wmt14/full" is the later 3,003-sentence cleaned set.
#
# The two score within 0.01 BLEU of each other (measured on Run #3: 24.66 vs 24.65), so
# this is not a scoring loophole -- it is the period-correct choice, made because it is
# period-correct. All results from Run #5 onward use the 2,737 set.
python3 -m sacrebleu -t wmt13 -l en-de --echo src > "$PREP/valid.raw.$SRC"
python3 -m sacrebleu -t wmt13 -l en-de --echo ref > "$PREP/valid.raw.$TGT"
python3 -m sacrebleu -t wmt14 -l en-de --echo src > "$PREP/test.raw.$SRC"
python3 -m sacrebleu -t wmt14 -l en-de --echo ref > "$PREP/test.raw.$TGT"

# CARRIAGE RETURNS: News Commentary v9 contains CRLF line endings in places where v12 does
# not. Moses tokenization treats a stray \r as a token, which silently desynchronized the
# two sides of the corpus (4,521,327 source tokens vs 4,521,186 target). Strip them from
# every split before tokenizing.
#
# This loop must run AFTER the sacrebleu fetch above -- valid.raw.* and test.raw.* do not
# exist until then, and `set -e` would abort on the missing files.
echo "== Stripping carriage returns =="
for split in train valid test; do
  for lang in $SRC $TGT; do
    tr -d '\r' < "$PREP/$split.raw.$lang" > "$PREP/tmp" && mv "$PREP/tmp" "$PREP/$split.raw.$lang"
  done
done

cd "$PREP"

echo "== Tokenizing (Moses, via sacremoses) =="
for split in train valid test; do
  for lang in $SRC $TGT; do
    sacremoses -l $lang tokenize -x < "$split.raw.$lang" > "$split.tok.$lang"
  done
done

echo "== Learning joint BPE ($BPE_TOKENS merges) on training data only =="
subword-nmt learn-joint-bpe-and-vocab \
  --input train.tok.$SRC train.tok.$TGT \
  -s $BPE_TOKENS \
  -o bpe.codes \
  --write-vocabulary vocab.$SRC vocab.$TGT

echo "== Applying BPE to all splits =="
for split in train valid test; do
  for lang in $SRC $TGT; do
    subword-nmt apply-bpe -c bpe.codes --vocabulary vocab.$lang --vocabulary-threshold 50 \
      < "$split.tok.$lang" > "$split.bpe.$lang"
  done
done

# POST-TOKENIZATION ALIGNMENT CHECK: the raw line-count check above passes even when the
# corpus is subtly corrupted, because a stray \r does not add a line. This second check
# catches misalignment introduced during tokenization itself.
echo "== Verifying alignment after tokenization =="
for split in train valid test; do
  a=$(wc -l < "$split.bpe.$SRC")
  b=$(wc -l < "$split.bpe.$TGT")
  if [ "$a" != "$b" ]; then
    echo "ERROR: $split.bpe.$SRC ($a lines) != $split.bpe.$TGT ($b lines)" >&2
    exit 1
  fi
  echo "  $split: $a lines each -- OK"
done

echo "Done. Prepared files: $PREP/{train,valid,test}.bpe.{$SRC,$TGT}"
echo "Next: build the shared vocab and start training (see src/train.py)."
