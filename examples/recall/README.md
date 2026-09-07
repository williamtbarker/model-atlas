# Recall-2: an actual tiny trained transformer

This asset supplies real weights and recorded arithmetic for a browser
visualization. It is deliberately small enough that every coordinate can be
displayed: **4,777 learned parameters**, two pre-normalized causal transformer
blocks, model width 16, two attention heads of width 8, and MLP width 32.

The model learns a synthetic lookup task. For example:

```text
C 4 A 2 B 1 ? C  →  4
```

Each input contains three distinct keys from `A B C D`, their values from
`1 2 3 4`, a question marker, and one of those keys. Values may repeat. The target
is the value paired with the query. No attention routes or lookup algorithm are
hard-coded. All weights are learned by final-position cross-entropy. Earlier
positions' logits are computed, but are not supervised or scored by this task.

This is **not a pretrained language model**, and these recordings establish no
claim about induction heads or human-readable feature meanings.

## Results

| Measurement | Accuracy | Mean target probability |
|---|---:|---:|
| Training: 3,684 queries | 100% | 99.967% |
| Held out: 924 queries | 100% | 99.965% |
| Held out, head L0H0 zeroed | 61.69% | 61.27% |
| Held out, head L0H1 zeroed | 54.00% | 52.39% |
| Held out, head L1H0 zeroed | 47.94% | 47.81% |
| Held out, head L1H1 zeroed | 99.89% | 98.66% |

The split is made over entire dictionaries before their three query variants are
generated. Thus variants of the same dictionary cannot occur in both sets. The
finite task universe has 1,536 dictionaries and 4,608 queries. The held-out set
was evaluated during training, but not used for gradients or early stopping;
the supplied run uses the preset 3,000 training steps.

The five visual examples were intentionally selected from correctly answered
held-out inputs for large intervention effects. They are **demonstrations, not a
representative evaluation sample**. Aggregate results above cover all 924
held-out queries, including the examples used in the visual demonstration.

An ablation sets one head's `attention @ V` output to zero at every token position
and then reruns all later computations, including later attention layers. It is
an intervention outside the training distribution; its effect is not a unique
or complete causal attribution. The original attention probabilities remain
inspectable for the disabled head, while its effective contributions are zero.

## Run locally

The browser needs only the supplied trace asset. Python is an optional workflow
for recreating the training and recordings. No CUDA, PyTorch, model download,
LaTeX installation, or Manim installation is involved.

From the repository root:

```bash
cd examples/recall
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_trace.py --output generated
python -m unittest test_trace.py
```

Training took approximately 43 seconds on the development environment; laptop
timing will vary. Seeds are `20260907` for initialization, seed + 1 for the split,
and seed + 2 for training batches. The implementation uses binary64 NumPy arrays,
Adam, batches of 64, a cosine learning-rate schedule, and gradient clipping.
Floating-point details can vary slightly with NumPy and BLAS implementations.

To reconstruct recordings from the supplied small checkpoint without training:

```bash
python train_trace.py --weights recall-weights.npz --output regenerated
```

That command marks training-history metadata as loaded rather than pretending to
reproduce it. Weights and forward computation are unchanged.

## Numerical conventions

- Arrays use row vectors: `X @ W`.
- `embedding` contains the sum of token and learned positional embeddings.
- `q`, `k`, `v`, and `headOutputs`: `[head, token, head_dimension]`.
- `scores` and `attention`: `[head, destination_token, source_token]`.
- Future scores are JSON `null`, representing the causal `−∞` mask, not zero.
- Softmax is over source positions. No dropout is present.
- `sourceWrites`: `[head, source_token, model_dimension]`, equal to each source
  `V` multiplied by that head's slice of the shared output projection `W_O`.
- `sourceContributions`: `[head, destination, source, model_dimension]`, equal
  to the attention probability times that source write. Disabled heads have
  zero contributions.
- Summing source contributions over source tokens yields `projectedHeads`.
- Summing `projectedHeads` over heads yields `attentionOutput`.
- `residual1 = input + attentionOutput`.
- `output = residual1 + mlpDown`.
- Layer normalization uses population variance and epsilon `1e-5`.
- The MLP activation is the explicitly implemented tanh approximation to GELU.
- JSON retains full round-trip binary64 precision. It is not a quantized trace.

The implementation checks directional derivatives of embeddings, attention
weights, an MLP weight, and final normalization against finite differences
before training. `test_trace.py` separately checks serialized contributions,
causal masking, residual additions, actual ablation reruns, and reproducibility
from the checkpoint.

All trainer code and numerical assets were independently created for Model
Atlas. No visualization scene implementation or model weights were copied from
3Blue1Brown, Bycroft, or another project. This code and these learned weights may
be distributed under the parent project's MIT license.
