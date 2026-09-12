# Where the 64,711,844 parameters are

Reference for the article. Every number below is derived from the architecture
(`d_model=512`, `d_ff=2048`, `h=8`, 6 encoder + 6 decoder layers, vocab 40,100) and sums
**exactly** to the count the training log reports. Nothing here is estimated.

---

## The building blocks

**One attention module — 1,050,624**

Four `nn.Linear(512, 512)` with bias: W_Q, W_K, W_V, W_O.

```
4 x (512 x 512 + 512) = 4 x 262,656 = 1,050,624
```

**Note what is absent: the head count.** All 8 heads share these same four matrices —
a head is a 64-column slice of the 512, not a matrix of its own. Changing `h` from 8 to 16
would not change the parameter count by one. Heads are a reshape, not more weights.

**One feed-forward block — 2,099,712**

```
512 x 2048 + 2048  = 1,050,624
2048 x 512 + 512   = 1,049,088
                     ---------
                     2,099,712
```

**One LayerNorm — 1,024** (a 512 scale and a 512 shift).

---

## One layer of each

| encoder layer | |
|---|---:|
| self-attention | 1,050,624 |
| feed-forward | 2,099,712 |
| 2 LayerNorms | 2,048 |
| **total** | **3,152,384** |

| decoder layer | |
|---|---:|
| masked self-attention | 1,050,624 |
| **cross-attention** | **1,050,624** |
| feed-forward | 2,099,712 |
| 3 LayerNorms | 3,072 |
| **total** | **4,204,032** |

**A decoder layer is 1.33x an encoder layer**, and the entire difference is the
cross-attention sublayer plus its LayerNorm. That is the structural asymmetry of the
architecture expressed as a number: the decoder costs more because it does one extra
thing — look at the encoder's output.

---

## The whole model

| | parameters | share |
|---|---:|---:|
| encoder x 6 | 18,914,304 | 29.2% |
| decoder x 6 | 25,224,192 | 39.0% |
| final LayerNorms (2) | 2,048 | 0.0% |
| embedding table (tied) | 20,531,200 | 31.7% |
| output projection bias | 40,100 | 0.1% |
| **TOTAL** | **64,711,844** | 100% |

Matches the reported count exactly.

---

## The same model cut a different way — and the surprise

Grouping by *what the parameters do* rather than where they sit:

| | parameters | share |
|---|---:|---:|
| **feed-forward** (12 blocks) | 25,196,544 | **38.9%** |
| **embedding** (one tied table) | 20,531,200 | **31.7%** |
| **attention** (18 modules) | 18,911,232 | **29.2%** |
| LayerNorms + output bias | 72,868 | 0.1% |

**Attention is the smallest of the three major blocks.** In a paper called *Attention Is
All You Need*, the attention mechanism holds under 30% of the weights — less than the
plain feed-forward layers, and less than the lookup table.


- **Why the embedding init mattered so much.** That one table is 31.7% of the model — the
  largest single tensor by far. Mis-scaling it by 6.5x mis-scaled a third of the network.
- **Why vocabulary size moves the total.** Embedding is the only block whose size depends
  on the vocabulary; the other 44.1M is fixed by the architecture alone.

---

## Reproducing this table

```python
V, d, dff, L = 40100, 512, 2048, 6
attn = 4 * (d*d + d)
ffn  = (d*dff + dff) + (dff*d + d)
ln   = 2 * d

encoder = L * (attn + ffn + 2*ln)
decoder = L * (2*attn + ffn + 3*ln)
total   = encoder + decoder + 2*ln + V*d + V
assert total == 64_711_844
```

The `V*d` term is the tied embedding (§3.4) counted **once** — it serves encoder input,
decoder input, and the output projection. The trailing `V` is the output projection's bias,
which is not tied to anything.
