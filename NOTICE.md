# Copyright, attribution and provenance

**Required Notice:** Copyright (c) 2026 GAP20 original author, rickeyernest1990@gmail.com.

The project software in this repository is offered under the [PolyForm Noncommercial License 1.0.0](LICENSE). For commercial permissions, attribution questions or corrections, contact **rickeyernest1990@gmail.com**. The report PDF carries its own copyright notice.

GAP20 is a configurable fused-gate Transformer block described by:

```text
z = LN2(x + Attention(LN1(x)))
y = x + W_O(z * SiLU(W_G(z)))
```

The published 100M-token experiment trained the width-72 / QK-head-22 configuration (270,128 trainable parameters). The width-64 / QK-head-16 variant (221,568 parameters) is a parameter-count calculation, **not** a separately tested 100M model. Adjusting model widths, head sizes or parameter counts while retaining this block is a configuration change; this notice asks that direct adaptations and the original experimental provenance be credited accurately. It does not claim that copyright reserves general equations, abstract methods or independently written implementations.

**WikiText-103 is a third-party dataset** based on Wikipedia text; dataset rights and any required attribution or share-alike obligations are separate from the PolyForm licence for this project's original software. The dataset is not included in this repository.

The PDF's numerical claims come from the original supplied 100M model/evaluation/benchmark archives. This minimal repository publishes the report and example code, **not** the training corpus, 100M trained weights, evaluation logs, tokenizer or factual-recall test scripts, so full independent reproduction from this repository alone is not claimed.