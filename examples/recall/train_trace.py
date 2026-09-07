#!/usr/bin/env python3
"""Train a tiny transformer on key/value recall and export actual computations.

Independent implementation: NumPy is the only non-standard dependency. No model
weights are downloaded. The task is synthetic symbolic recall, not language.
Every attention head is ordinary causal softmax attention; no routing is coded in.
Row-vector convention throughout: projected = input @ weight.
"""

import argparse
import gzip
import itertools
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

SEED = 20260907
VOCAB = ["A", "B", "C", "D", "1", "2", "3", "4", "?"]
D, H, DH, FF, LAYERS, T = 16, 2, 8, 32, 2, 8
EPS = 1e-5


def initialize(seed=SEED):
    rng = np.random.default_rng(seed)
    p = {}
    p["embedding"] = rng.normal(0, 0.18, (len(VOCAB), D))
    p["position"] = rng.normal(0, 0.12, (T, D))
    for l in range(LAYERS):
        n = f"blocks.{l}."
        for name in ["wq", "wk", "wv", "wo"]:
            p[n + name] = rng.normal(0, 1 / np.sqrt(D), (D, D))
        for norm in ["ln1", "ln2"]:
            p[n + norm + ".gamma"] = np.ones(D)
            p[n + norm + ".beta"] = np.zeros(D)
        p[n + "w1"] = rng.normal(0, 1 / np.sqrt(D), (D, FF))
        p[n + "b1"] = np.zeros(FF)
        p[n + "w2"] = rng.normal(0, 1 / np.sqrt(FF), (FF, D))
        p[n + "b2"] = np.zeros(D)
    p["finalNorm.gamma"] = np.ones(D)
    p["finalNorm.beta"] = np.zeros(D)
    p["unembedding"] = rng.normal(0, 1 / np.sqrt(D), (D, len(VOCAB)))
    p["outputBias"] = np.zeros(len(VOCAB))
    return p


def layer_norm(x, gamma, beta):
    centered = x - x.mean(axis=-1, keepdims=True)
    inv = 1 / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + EPS)
    z = centered * inv
    return z * gamma + beta, (z, inv, gamma)


def norm_backward(dy, cache):
    z, inv, gamma = cache
    dz = dy * gamma
    dx = inv * (dz - dz.mean(axis=-1, keepdims=True)
                - z * (dz * z).mean(axis=-1, keepdims=True))
    return dx, np.sum(dy * z, axis=(0, 1)), dy.sum(axis=(0, 1))


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_derivative(x):
    c = np.sqrt(2 / np.pi)
    t = np.tanh(c * (x + 0.044715 * x ** 3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x)


def softmax(x):
    z = np.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def forward(p, ids, ablate=None, record=False):
    b, t = ids.shape
    x = p["embedding"][ids] + p["position"][:t]
    cache = {"ids": ids, "blocks": []}
    trace = {"embedding": x.copy(), "blocks": []} if record else None
    for l in range(LAYERS):
        n = f"blocks.{l}."
        inp = x
        norm1, nc1 = layer_norm(x, p[n + "ln1.gamma"], p[n + "ln1.beta"])
        q, k, v = [
            (norm1 @ p[n + name]).reshape(b, t, H, DH).transpose(0, 2, 1, 3)
            for name in ["wq", "wk", "wv"]
        ]
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(DH)
        mask = np.triu(np.ones((t, t), dtype=bool), k=1)
        scores[:, :, mask] = -np.inf
        a = softmax(scores)
        heads = a @ v
        if ablate is not None and ablate[0] == l:
            heads[:, ablate[1]] = 0
        merged = heads.transpose(0, 2, 1, 3).reshape(b, t, D)
        attn_out = merged @ p[n + "wo"]
        residual1 = inp + attn_out
        norm2, nc2 = layer_norm(residual1, p[n + "ln2.gamma"], p[n + "ln2.beta"])
        up = norm2 @ p[n + "w1"] + p[n + "b1"]
        act = gelu(up)
        down = act @ p[n + "w2"] + p[n + "b2"]
        x = residual1 + down
        cache["blocks"].append(dict(norm1=norm1, nc1=nc1, q=q, k=k, v=v, a=a,
            merged=merged, norm2=norm2, nc2=nc2, up=up, act=act))
        if record:
            # A head's slice of W_O maps each source V into residual coordinates.
            source_writes = np.einsum("bhtk,hkd->bhtd", v, p[n + "wo"].reshape(H, DH, D))
            contributions = a[..., None] * source_writes[:, :, None, :, :]
            if ablate is not None and ablate[0] == l:
                contributions[:, ablate[1]] = 0
            projected = contributions.sum(axis=3)
            assert np.allclose(projected.sum(axis=1), attn_out, atol=1e-10)
            trace["blocks"].append(dict(input=inp, norm1=norm1, q=q, k=k, v=v,
                scores=scores, attention=a, headOutputs=heads, sourceWrites=source_writes,
                sourceContributions=contributions, projectedHeads=projected,
                attentionOutput=attn_out, residual1=residual1, norm2=norm2,
                mlpUp=up, mlpAct=act, mlpDown=down, output=x))
    final, ncf = layer_norm(x, p["finalNorm.gamma"], p["finalNorm.beta"])
    logits = final @ p["unembedding"] + p["outputBias"]
    prob = softmax(logits)
    cache["final"] = final
    cache["ncf"] = ncf
    if record:
        trace.update(finalNorm=final, logits=logits, probabilities=prob)
    return prob, cache, trace


def loss_and_grad(p, ids, target):
    prob, c, _ = forward(p, ids)
    b = ids.shape[0]
    loss = -np.log(np.maximum(prob[np.arange(b), -1, target], 1e-300)).mean()
    g = {k: np.zeros_like(v) for k, v in p.items()}
    dl = np.zeros_like(prob)
    dl[:, -1] = prob[:, -1]
    dl[np.arange(b), -1, target] -= 1
    dl /= b
    g["unembedding"] = np.einsum("btd,btv->dv", c["final"], dl)
    g["outputBias"] = dl.sum(axis=(0, 1))
    dx, g["finalNorm.gamma"], g["finalNorm.beta"] = norm_backward(dl @ p["unembedding"].T, c["ncf"])
    for l in reversed(range(LAYERS)):
        n, bc = f"blocks.{l}.", c["blocks"][l]
        g[n + "w2"] = np.einsum("btf,btd->fd", bc["act"], dx)
        g[n + "b2"] = dx.sum(axis=(0, 1))
        dup = (dx @ p[n + "w2"].T) * gelu_derivative(bc["up"])
        g[n + "w1"] = np.einsum("btd,btf->df", bc["norm2"], dup)
        g[n + "b1"] = dup.sum(axis=(0, 1))
        dn2, g[n + "ln2.gamma"], g[n + "ln2.beta"] = norm_backward(dup @ p[n + "w1"].T, bc["nc2"])
        dr = dx + dn2
        g[n + "wo"] = np.einsum("bti,btj->ij", bc["merged"], dr)
        dhead = (dr @ p[n + "wo"].T).reshape(b, T, H, DH).transpose(0, 2, 1, 3)
        da = dhead @ bc["v"].transpose(0, 1, 3, 2)
        dv = bc["a"].transpose(0, 1, 3, 2) @ dhead
        ds = bc["a"] * (da - (da * bc["a"]).sum(axis=-1, keepdims=True)) / np.sqrt(DH)
        dq = ds @ bc["k"]
        dk = ds.transpose(0, 1, 3, 2) @ bc["q"]
        dn = np.zeros_like(dr)
        for name, dp in zip(["wq", "wk", "wv"], [dq, dk, dv]):
            dp = dp.transpose(0, 2, 1, 3).reshape(b, T, D)
            g[n + name] = np.einsum("bti,btj->ij", bc["norm1"], dp)
            dn += dp @ p[n + name].T
        dn1, g[n + "ln1.gamma"], g[n + "ln1.beta"] = norm_backward(dn, bc["nc1"])
        dx = dr + dn1
    np.add.at(g["embedding"], ids.reshape(-1), dx.reshape(-1, D))
    g["position"] = dx.sum(axis=0)
    return loss, g


def dataset():
    examples, targets = [], []
    # Split by the entire six-token dictionary, preventing query variants of the
    # same dictionary from straddling training and test sets.
    dictionaries = list(itertools.product(itertools.permutations(range(4), 3),
                                          itertools.product(range(4, 8), repeat=3)))
    rng = np.random.default_rng(SEED + 1)
    order = rng.permutation(len(dictionaries))
    split = int(len(dictionaries) * 0.8)
    outputs = []
    for selected in [order[:split], order[split:]]:
        examples, targets = [], []
        for idx in selected:
            keys, vals = dictionaries[idx]
            for query in range(3):
                examples.append([keys[0], vals[0], keys[1], vals[1], keys[2], vals[2], 8, keys[query]])
                targets.append(vals[query])
        outputs.extend([np.asarray(examples, dtype=np.int64), np.asarray(targets, dtype=np.int64)])
    return outputs


def evaluate(p, x, y, ablate=None):
    correct, loss, conf = 0, 0.0, 0.0
    for start in range(0, len(x), 128):
        ids, target = x[start:start + 128], y[start:start + 128]
        prob, _, _ = forward(p, ids, ablate=ablate)
        final = prob[:, -1]
        correct += int(np.sum(final.argmax(axis=-1) == target))
        selected = final[np.arange(len(ids)), target]
        loss -= float(np.log(np.maximum(selected, 1e-300)).sum())
        conf += float(selected.sum())
    return {"examples": len(x), "accuracy": correct / len(x), "crossEntropy": loss / len(x),
            "meanCorrectProbability": conf / len(x)}


def gradient_check(p, x, y):
    _, g = loss_and_grad(p, x[:2], y[:2])
    rng = np.random.default_rng(17)
    for name in ["embedding", "blocks.0.wq", "blocks.0.wv", "blocks.1.w1", "finalNorm.gamma"]:
        direction = rng.normal(size=p[name].shape)
        direction /= np.linalg.norm(direction)
        original = p[name].copy()
        step = 1e-5
        p[name] = original + step * direction
        a, _ = loss_and_grad(p, x[:2], y[:2])
        p[name] = original - step * direction
        b, _ = loss_and_grad(p, x[:2], y[:2])
        p[name] = original
        numerical = (a - b) / (2 * step)
        analytical = np.sum(g[name] * direction)
        assert abs(numerical - analytical) < 2e-6, (name, numerical, analytical)
    print("Directional gradient checks passed", flush=True)


def train(p, x, y, xt, yt, steps):
    rng = np.random.default_rng(SEED + 2)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    history, started = [], time.time()
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(x), size=64)
        loss, g = loss_and_grad(p, x[idx], y[idx])
        norm = np.sqrt(sum(np.sum(value * value) for value in g.values()))
        clip = min(1.0, 1.0 / max(norm, 1e-12))
        lr = 0.003 * min(step / 100, 1) * (0.15 + 0.85 * (1 + np.cos(np.pi * step / steps)) / 2)
        for name in p:
            grad = g[name] * clip
            m[name] = 0.9 * m[name] + 0.1 * grad
            v[name] = 0.999 * v[name] + 0.001 * grad * grad
            mh, vh = m[name] / (1 - 0.9 ** step), v[name] / (1 - 0.999 ** step)
            p[name] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if step % 250 == 0 or step == steps:
            metrics = evaluate(p, xt, yt)
            history.append({"step": step, "batchLoss": loss, **metrics})
            print(f"step {step:5d} loss {loss:.5f} test {metrics['accuracy']:.4%} elapsed {time.time()-started:.1f}s", flush=True)
    return history


def json_value(x):
    if isinstance(x, np.ndarray):
        return json_value(x.tolist())
    if isinstance(x, (float, np.floating)):
        # Keep full binary64-to-JSON round-trip precision, masked logits as null.
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, dict):
        return {k: json_value(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_value(v) for v in x]
    return x


def unbatch(trace):
    return {name: [{k: v[0] for k, v in block.items()} for block in value]
            if name == "blocks" else value[0] for name, value in trace.items()}


def export(p, x, y, xt, yt, history, directory):
    baseline = evaluate(p, xt, yt)
    interventions = {f"L{l}H{h}": evaluate(p, xt, yt, (l, h))
                     for l in range(LAYERS) for h in range(H)}
    # Pick held-out examples by a documented, deterministic display criterion:
    # baseline correct, high baseline confidence, large single-head probability drop.
    prob, _, _ = forward(p, xt)
    base = prob[np.arange(len(xt)), -1, yt]
    correct = prob[:, -1].argmax(-1) == yt
    drops = np.zeros((LAYERS * H, len(xt)))
    for l in range(LAYERS):
        for h in range(H):
            ap, _, _ = forward(p, xt, (l, h))
            drops[l * H + h] = base - ap[np.arange(len(xt)), -1, yt]
    chosen = []
    # At least one example from each query slot; remaining examples maximize impact.
    for qpos in range(3):
        matches = xt[:, -1] == xt[:, qpos * 2]
        candidates = np.where(matches & correct & (base > 0.9))[0]
        if len(candidates):
            chosen.append(int(candidates[np.argmax(drops[:, candidates].max(axis=0))]))
    ranked = np.argsort(-(drops.max(axis=0) * correct))
    for idx in ranked:
        if int(idx) not in chosen and correct[idx] and base[idx] > 0.9:
            chosen.append(int(idx))
        if len(chosen) >= 5:
            break
    examples = []
    for i, idx in enumerate(chosen):
        ids = xt[idx:idx + 1]
        runs = {}
        for intervention in [None] + [(l, h) for l in range(LAYERS) for h in range(H)]:
            _, _, trace = forward(p, ids, intervention, record=True)
            name = "baseline" if intervention is None else f"L{intervention[0]}H{intervention[1]}"
            runs[name] = unbatch(trace)
        examples.append(dict(id=f"recall-{i+1}", label=" ".join(VOCAB[z] for z in ids[0]),
            tokens=[VOCAB[z] for z in ids[0]], inputIds=ids[0], targetId=int(yt[idx]),
            target=VOCAB[yt[idx]], split="held-out", heldOutIndex=int(idx),
            strongestAblation=f"L{int(np.argmax(drops[:,idx]))//H}H{int(np.argmax(drops[:,idx]))%H}",
            runs=runs))
    package = dict(format="model-atlas-trace", version=1, model=dict(
        name="Recall-2", description="A tiny trained causal transformer for symbolic key/value recall.",
        dModel=D, nHeads=H, dHead=DH, dFF=FF, nLayers=LAYERS, maxSequenceLength=T,
        vocab=VOCAB, parameterCount=sum(v.size for v in p.values()), seed=SEED,
        task="Read three distinct key/value pairs and return the value belonging to the final query key.",
        layerNormEpsilon=EPS, activation="gelu_tanh", precision="float64", convention="row_vectors",
        metrics=dict(training=evaluate(p, x, y), heldOut=baseline, headAblations=interventions),
        provenance=dict(kind="trained_synthetic", trainingExamples=len(x), heldOutExamples=len(xt),
            split="80/20 split by full dictionary, all three query variants stay together",
            trainingSteps=history[-1]["step"], objective="Final-token cross-entropy across all 9 vocabulary items",
            recording="Full deterministic forward pass, including every head and source contribution",
            exampleSelection="Held-out correct examples with >90% target probability, chosen for large head-ablation effects; not representative sampling",
            ablation="Zero one attention head after A @ V, then rerun all subsequent operations and layers",
            limitations=["Synthetic lookup task, not a pretrained language model", "No claim of discovered induction heads",
                "Ablation is an out-of-distribution intervention; effect is not a unique attribution",
                "Early-position logits are unsupervised; the task is scored only at the last input position"])),
        weights=p, examples=examples, trainingHistory=history)
    data = json.dumps(json_value(package), separators=(",", ":"), allow_nan=False).encode()
    (directory / "recall-trace.json").write_bytes(data)
    (directory / "recall-trace.json.gz").write_bytes(gzip.compress(data, mtime=0))
    np.savez(directory / "recall-weights.npz", **p)
    print(json.dumps({"parameters": package["model"]["parameterCount"], "bytes": len(data),
                      "metrics": package["model"]["metrics"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument("--weights", type=Path, help="Reuse an existing .npz checkpoint; skip training")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x, y, xt, yt = dataset()
    if args.weights:
        p = dict(np.load(args.weights))
        history = [{"step": 0, "note": "Loaded existing checkpoint; training history is not reconstructed"}]
    else:
        p = initialize()
        gradient_check(p, x, y)
        history = train(p, x, y, xt, yt, args.steps)
    export(p, x, y, xt, yt, history, args.output)


if __name__ == "__main__":
    main()
