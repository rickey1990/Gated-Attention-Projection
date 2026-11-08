# GAP20 — Gated Attention Projection

**A small, configurable Transformer experiment with 100 million scored WikiText-103 training targets per model.** The project compares a conventional four-block Transformer with a fused-gate GAP20 block, and includes a separate PyTorch `torch.compile` speed benchmark. **The standard model achieved better language-model accuracy**, although the widened GAP20 model used fewer parameters.

**[Read the eight-page research paper](GAP20_100M_Research_Paper.pdf)** · **Contact:** rickeyernest1990@gmail.com

## Main findings

| Completed 100M-token run | Standard Transformer | Widened GAP20 |
|---|---:|---:|
| Trainable parameters | 337,280 | 270,128 (**19.91% fewer**) |
| Held-out test perplexity (lower is better) | **40.05** | 45.62 |
| Held-out next-token accuracy (higher is better) | **30.34%** | 28.61% |

Both models used WikiText-103, a 2,048-piece training-only SentencePiece BPE tokenizer, four layers, four attention heads and a 96-token context. Training used one seed and 100 million distinct scored next-token targets per model. The 40M checkpoints are **intermediate milestones of this 100M run**, not a separate final experiment.

A separate, single extended CPU benchmark reported **19.93% higher steady-state training throughput for compiled GAP20 versus eager GAP20**. This is **not** a measured speed advantage of GAP20 over the standard model; the benchmark was noisy and the result is not independently replicated here. The synthetic 2/4/8-fact, update and two-hop tests did **not** show reliable zero-shot fact binding for either model.

## Configurations and original design

The GAP20 block is defined by

```text
z = LN2(x + Attention(LN1(x)))
y = x + W_O(z * SiLU(W_G(z)))
```

| GAP20 configuration | Width | Q/K per head | Parameters | Relative to standard | 100M-trained? |
|---|---:|---:|---:|---:|---|
| `paper100m` | 72 | 22 | 270,128 | 19.91% fewer | **Yes** |
| `compact34` | 64 | 16 | 221,568 | 34.31% fewer | **No; parameter calculation only** |

`models.py` makes the width and attention dimensions configurable without changing the defining block equation. Changing parameter sizes alone does not establish a new architecture. See [NOTICE.md](NOTICE.md) for authorship and attribution.

## The six public files

| File | Purpose |
|---|---|
| `GAP20_100M_Research_Paper.pdf` | Mathematics, training setup, learning curves, 100M results, factual-recall tests, speed measurement and limitations |
| `models.py` | Runnable, parameter-configurable GAP20 and standard Transformer architecture |
| `benchmark_compile.py` | Runnable CPU eager vs `torch.compile` training benchmark; optional checkpoint loading |
| `README.md` | This explanation, results and reproduction boundaries |
| `LICENSE` | PolyForm Noncommercial License 1.0.0 for project software |
| `NOTICE.md` | Original copyright, attribution and third-party dataset clarification |

This lightweight public repository **does not contain** the raw WikiText-103 text, tokenized `.bin` files, pretrained checkpoints, full evaluation logs, tokenizer, or separate factual-recall test scripts. The PDF reports measurements from the original supplied experiment archive; **the six-file repo alone cannot independently reproduce the reported 100M training/test results or factual-recall evaluations**. It does allow inspection of the architecture and running new speed experiments.

## Run the code

Python and PyTorch are required. On a system with PyTorch 2.x and `torch.compile` support:

```bash
pip install torch
python models.py  # prints parameter counts for paper100m and compact34

# Independent measurements using randomly initialized GAP20 models.
# Run each in a fresh process; compilation can take time.
python benchmark_compile.py --mode eager --variant paper100m --groups 3 --steps 10
python benchmark_compile.py --mode compiled --variant paper100m --groups 3 --steps 10

# Compare a different size (not a claim about its quality):
python benchmark_compile.py --mode eager --variant compact34 --groups 3 --steps 10
```

The benchmark normally uses **randomly initialized weights**. To use separately obtained matching training weights, add `--checkpoint /path/to/checkpoint.pt` to both runs; the file must contain the expected `model` state dictionary and compatible dimensions. Neither a random-weights run nor changed hardware settings reproduces the original 19.93% result. Record your device, thread count, first-compile overhead and run-to-run variation.

## Source data and licence

The original training dataset was **WikiText-103**. It is described in the paper, but it is **not redistributed here**. Data and Wikipedia-derived content have their own third-party licensing and attribution requirements, separate from this project's software licence.

**Software licence:** [PolyForm Noncommercial License 1.0.0](LICENSE). Required Notice: Copyright (c) 2026 GAP20 original author, rickeyernest1990@gmail.com. The research PDF has its own copyright notice. The licence does not provide exclusive rights over generic attention/gating mathematics, independent implementations or ideas.

**Scope:** One-seed exploratory model comparison. No multi-seed generalization, hardware-independent performance claim or independent reproduction of 100M training is asserted.