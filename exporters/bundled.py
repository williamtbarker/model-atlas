"""Independent architecture adapters. The browser never imports this module."""

from __future__ import annotations
import json
import math
import re
from pathlib import Path
from .model_ir import Builder, parameter_count, validate
from .deepseek4_shapes import derive_catalog

ROOT = Path(__file__).resolve().parent.parent


def gpt2(config=None):
    c = config or dict(
        model_type="gpt2",
        n_layer=12,
        n_embd=768,
        n_head=12,
        n_positions=1024,
        vocab_size=50257,
        tie_word_embeddings=True,
    )
    if (
        c.get("add_cross_attention")
        or c.get("scale_attn_by_inverse_layer_idx")
        or c.get("reorder_and_upcast_attn")
        or c.get("activation_function", "gelu_new") != "gelu_new"
    ):
        raise ValueError("Unsupported GPT-2 variant")
    n, d, h, v, ctx = [
        c[k] for k in ["n_layer", "n_embd", "n_head", "vocab_size", "n_positions"]
    ]
    if (
        not all(isinstance(x, int) and x > 0 for x in [n, d, h, v, ctx])
        or d % h
        or n > 256
        or h > 256
    ):
        raise ValueError("Invalid GPT-2 dimensions")
    f = c.get("n_inner") or 4 * d
    b = Builder(
        "GPT-2 · 124M",
        {
            "kind": "config + verified adapter",
            "evidence": "derived",
            "notes": "Inference topology derived from public GPT-2 implementation. No weights, tokenizer or activations loaded. Dropout is omitted for inference.",
            "urls": [
                "https://huggingface.co/openai-community/gpt2/blob/1a43f7d38310adb1a52f267670dc678fafaa3db2/config.json",
                "https://github.com/openai/gpt-2/blob/master/src/model.py",
            ],
        },
    )
    b.ir["metadata"] = dict(
        layers=n,
        hiddenWidth=d,
        queryHeads=h,
        kvHeads=h,
        headDim=d // h,
        context=ctx,
        architecture="Dense causal decoder · validation reference",
        initialScope="model.layers",
    )
    b.node(
        "model.embed",
        "model",
        "EmbeddingTable",
        "Token embedding",
        {"formula": "xₜ = W_E[token_idₜ]"},
    )
    b.tensor("transformer.wte.weight", "model.embed", [v, d])
    b.node("model.position", "model", "EmbeddingTable", "Position embedding")
    b.tensor("transformer.wpe.weight", "model.position", [ctx, d])
    b.node(
        "model.embed_add",
        "model",
        "ResidualBus",
        "Add positions",
        {"formula": "xₜ + pₜ"},
    )
    b.node(
        "model.layers",
        "model",
        "RepeatedModuleArray",
        f"{n} decoder blocks",
        {"count": n},
    )
    b.edge("model.embed", "model.embed_add", "token vectors")
    b.edge("model.position", "model.embed_add", "position vectors")
    b.edge("model.embed_add", "model.layers", "residual stream")
    for i in range(n):
        p = f"model.layers.{i}"
        b.node(p, "model.layers", "TransformerBlock", f"Block {i}", {"index": i})
        if i:
            b.edge(f"model.layers.{i - 1}", p, "residual stream")
        b.node(p + ".input", p, "TokenStream", "Residual input", {"shape": ["T", d]})
        b.norm(p + ".ln1", p, d, "Layer norm 1")
        a = p + ".attn"
        b.node(
            a,
            p,
            "AttentionHead",
            "Multi-head attention",
            {"queryHeads": h, "headDim": d // h},
        )
        b.projection(a + ".qkv", a, 3 * d, d, "Fused Q / K / V", True, True)
        b.node(
            a + ".heads", a, "RepeatedModuleArray", f"{h} attention heads", {"count": h}
        )
        for j in range(h):
            head = f"{a}.heads.{j}"
            b.node(
                head,
                a + ".heads",
                "AttentionHead",
                f"Head {j}",
                {
                    "headIndex": j,
                    "headDim": d // h,
                    "formula": "softmax(causal(QKᵀ / √dₖ)) V",
                    "parameterView": "Slices of fused QKV and shared output matrix; no new storage.",
                },
            )
            b.node(
                head + ".scores",
                head,
                "AttentionGrid",
                "Attention coefficients",
                {
                    "rowAxis": "query position",
                    "columnAxis": "key position",
                    "shape": ["T", "T"],
                    "values": "unavailable without a runtime trace",
                },
            )
        b.projection(a + ".out", a, d, d, "Output projection", True, True)
        b.edge(a + ".qkv", a + ".heads", "split into head views")
        b.edge(a + ".heads", a + ".out", "concatenate head outputs")
        b.node(
            p + ".add1",
            p,
            "ResidualBus",
            "Residual add 1",
            {"formula": "x + attention(norm(x))"},
        )
        b.norm(p + ".ln2", p, d, "Layer norm 2")
        m = p + ".mlp"
        b.node(
            m,
            p,
            "ActivationLayer",
            "Feedforward",
            {"formula": "W_down GELU(W_up x + b_up) + b_down"},
        )
        b.projection(m + ".up", m, f, d, "Expand", True, True)
        b.node(
            m + ".gelu",
            m,
            "ActivationLayer",
            "GELU",
            {"formula": "gelu_new: tanh approximation"},
        )
        b.projection(m + ".down", m, d, f, "Contract", True, True)
        b.edge(m + ".up", m + ".gelu", "pre-activation")
        b.edge(m + ".gelu", m + ".down", "activated features")
        b.node(
            p + ".add2",
            p,
            "ResidualBus",
            "Residual add 2",
            {"formula": "x′ + MLP(norm(x′))"},
        )
        for source, target, label, kind in [
            ("input", "ln1", "x", "dataflow"),
            ("ln1", "attn", "normalized x", "dataflow"),
            ("attn", "add1", "attention output", "dataflow"),
            ("input", "add1", "skip connection", "residual"),
            ("add1", "ln2", "x′", "dataflow"),
            ("ln2", "mlp", "normalized x′", "dataflow"),
            ("mlp", "add2", "MLP output", "dataflow"),
            ("add1", "add2", "skip connection", "residual"),
        ]:
            b.edge(p + "." + source, p + "." + target, label, kind)
    b.norm("model.norm", "model", d, "Final layer norm")
    head = b.node(
        "model.output",
        "model",
        "VocabularyPlane",
        "Vocabulary logits",
        {"formula": "logits = x W_Eᵀ"},
    )
    if c.get("tie_word_embeddings", True):
        head["tensorRefs"] = ["transformer.wte.weight"]
        b.edge(
            "model.embed", "model.output", "tied parameter storage", "parameter_share"
        )
    else:
        b.tensor("lm_head.weight", "model.output", [v, d])
    b.edge("model.layers", "model.norm", "final residual")
    b.edge("model.norm", "model.output", "normalized states")
    return validate(b.ir)


def deepseek4():
    spec = json.loads((ROOT / "references/deepseek4-shapes.json").read_text())
    c = spec["config"]
    catalog = derive_catalog(c)
    d = c["hidden_size"]
    n = c["num_hidden_layers"]
    b = Builder(
        "DeepSeek-V4-Pro · 1.6T",
        {
            "kind": "pinned config + checkpoint metadata + verified adapter",
            "evidence": "derived",
            "notes": "April 2026 open-weight model. Full checkpoint names and packed byte count reconciled; 7,030 header entries checked. Remaining shapes follow verified constructors. No weights or runtime trace loaded. Logical parameters include the MTP module; quantization scales and routing-table integers are counted separately.",
            "urls": [
                f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/{spec['revision']}",
                spec["source_urls"]["inference/model.py"],
            ],
        },
    )
    b.ir["metadata"] = dict(
        layers=n,
        hiddenWidth=d,
        queryHeads=c["num_attention_heads"],
        headDim=c["head_dim"],
        experts=c["n_routed_experts"],
        activeExperts=c["num_experts_per_tok"],
        context=c["max_position_embeddings"],
        architecture="2026 · Compressed attention / MoE / mHC",
        initialScope="layers",
        revision=spec["revision"],
        reconciliation=spec["reconciliation"],
        reportedActiveParameters="49B (publisher estimate, input/routing dependent)",
    )
    b.node(
        "embed",
        "model",
        "EmbeddingTable",
        "Token embedding",
        {"vocabulary": c["vocab_size"]},
    )
    b.node(
        "layers",
        "model",
        "RepeatedModuleArray",
        f"{n} core decoder blocks",
        {"count": n, "HCA": 31, "CSA": 30},
    )
    for start in range(0, n, 8):
        end = min(start + 8, n)
        b.node(
            f"group:{start}",
            "layers",
            "RepeatedModuleArray",
            f"Blocks {start}–{end - 1}",
            {"firstBlock": start, "lastBlock": end - 1, "count": end - start},
        )
        if start:
            b.edge(f"group:{start - 8}", f"group:{start}", "four residual streams")
    b.node(
        "hc_head",
        "model",
        "HyperConnectionMix",
        "Collapse residual streams",
        {"streams": c["hc_mult"], "formula": "learned final stream weighting"},
    )
    b.node("norm", "model", "NormalizationPlane", "Final RMSNorm")
    b.node(
        "head",
        "model",
        "VocabularyPlane",
        "Vocabulary logits",
        {"vocabulary": c["vocab_size"]},
    )
    b.node(
        "mtp",
        "model",
        "RepeatedModuleArray",
        "Multi-token prediction",
        {
            "count": c["num_nextn_predict_layers"],
            "role": "Auxiliary prediction module; separate from 61 core blocks.",
        },
    )
    b.edge("embed", "layers", "expand to four residual streams")
    b.edge("layers", "hc_head", "four final streams")
    b.edge("hc_head", "norm", "combined hidden state")
    b.edge("norm", "head", "normalized state")

    for lid in range(n + c["num_nextn_predict_layers"]):
        is_mtp = lid >= n
        p = f"mtp.{lid - n}" if is_mtp else f"layers.{lid}"
        ratio = c["compress_ratios"][lid]
        mode = (
            "Uncompressed MTP attention"
            if ratio == 0
            else "CSA · compression 4"
            if ratio == 4
            else "HCA · compression 128"
        )
        b.node(
            p,
            "mtp" if is_mtp else f"group:{lid // 8 * 8}",
            "TransformerBlock",
            f"MTP block {lid - n}"
            if is_mtp
            else f"Block {lid} · {'CSA' if ratio == 4 else 'HCA'}",
            {"index": lid, "compressionRatio": ratio, "residualStreams": c["hc_mult"]},
        )
        if not is_mtp and lid % 8:
            b.edge(f"layers.{lid - 1}", p, "four residual streams")
        b.node(
            p + ".input", p, "TokenStream", "Four-stream input", {"shape": ["T", 4, d]}
        )
        for branch, label in [("attn", "Attention"), ("ffn", "Feedforward")]:
            b.node(
                p + f".hc_{branch}",
                p,
                "HyperConnectionMix",
                f"{label} hyper-connection",
                {
                    "streams": 4,
                    "formula": "learned pre / post / residual mixing; Sinkhorn-normalized residual map",
                    "sinkhornIterations": c["hc_sinkhorn_iters"],
                },
            )
            b.node(
                p + f".{branch}_norm",
                p,
                "NormalizationPlane",
                f"{label} RMSNorm",
                {"width": d},
            )
        a = p + ".attn"
        b.node(
            a,
            p,
            "CompressedAttention",
            mode,
            {
                "compressionRatio": ratio,
                "localWindow": c["sliding_window"],
                "heads": c["num_attention_heads"],
                "headDim": c["head_dim"],
                "KV": "single shared 512-dimensional latent",
                "formula": "attention over local and eligible compressed KV, with per-head sink",
            },
        )
        b.node(
            a + ".input",
            a,
            "TokenStream",
            "Normalized hidden states",
            {"shape": ["T", d]},
        )
        for key, label, attrs in [
            (
                "wq_a",
                "Q down projection",
                {"inputWidth": d, "outputWidth": c["q_lora_rank"]},
            ),
            ("q_norm", "Q latent norm", {}),
            ("wq_b", "Q head projection", {"headWidth": 512, "heads": 128}),
            ("wkv", "Shared KV projection", {"outputWidth": 512}),
            ("kv_norm", "KV normalization", {}),
            ("wo_a", "Grouped output projection", {"groups": 16, "rank": 1024}),
            ("wo_b", "Output expansion", {"outputWidth": d}),
        ]:
            b.node(
                a + "." + key,
                a,
                "NormalizationPlane" if "norm" in key else "ProjectionBlock",
                label,
                attrs,
            )
        b.node(
            a + ".rotary",
            a,
            "ProjectionBlock",
            "Rotary positions",
            {
                "rotaryDimensions": 64,
                "formula": "YaRN-scaled rotary position transform; partial head coordinates",
            },
        )
        b.node(
            a + ".q_head_norm",
            a,
            "NormalizationPlane",
            "Per-head Q normalization",
            {"formula": "q / sqrt(mean(q²) + ε); no learned scale"},
        )
        b.node(
            a + ".inverse_rotary",
            a,
            "ProjectionBlock",
            "Inverse output rotary",
            {
                "rotaryDimensions": 64,
                "formula": "inverse rotary transform of output positional coordinates",
            },
        )
        b.node(
            a + ".mix",
            a,
            "AttentionHead",
            "Attention mixing",
            {
                "heads": 128,
                "scores": "unavailable without a trace",
                "formula": "selected KV mixing + attention sink",
            },
        )
        for h in range(c["num_attention_heads"]):
            b.node(
                f"{a}.mix.{h}",
                a + ".mix",
                "AttentionHead",
                f"Head {h}",
                {
                    "headIndex": h,
                    "headDim": 512,
                    "parameterView": "Q/output slices and shared KV; no independent head storage.",
                },
            )
        b.edge(a + ".input", a + ".wq_a", "hidden states")
        b.edge(a + ".input", a + ".wkv", "hidden states")
        b.edge(a + ".wq_a", a + ".q_norm", "latent Q")
        b.edge(a + ".q_norm", a + ".wq_b", "normalized latent")
        b.edge(a + ".wq_b", a + ".q_head_norm", "Q heads")
        b.edge(a + ".q_head_norm", a + ".rotary", "normalized Q heads")
        b.edge(a + ".wkv", a + ".kv_norm", "shared KV")
        b.edge(a + ".kv_norm", a + ".rotary", "normalized KV")
        b.edge(a + ".rotary", a + ".mix", "positioned Q and KV")
        b.edge(a + ".mix", a + ".inverse_rotary", "head outputs")
        b.edge(a + ".inverse_rotary", a + ".wo_a", "inverse-positioned outputs")
        b.edge(a + ".wo_a", a + ".wo_b", "grouped low-rank output")
        if ratio:
            comp = a + ".compressor"
            b.node(
                comp,
                a,
                "Compressor",
                "Compressed KV",
                {
                    "compressionRatio": ratio,
                    "formula": "learned gated pooling / normalization / rotary positions",
                },
            )
            b.edge(a + ".input", comp, "original normalized hidden states")
            if ratio == 4:
                ix = a + ".indexer"
                b.node(
                    ix,
                    a,
                    "SparseIndexer",
                    "Sparse position indexer",
                    {
                        "topK": c["index_topk"],
                        "indexHeads": c["index_n_heads"],
                        "headDim": c["index_head_dim"],
                        "formula": "own compressed keys; query projection from latent Q; select compressed positions",
                    },
                )
                b.node(
                    ix + ".compressor",
                    ix,
                    "Compressor",
                    "Indexer compression",
                    {"compressionRatio": 4},
                )
                b.edge(
                    a + ".input",
                    ix,
                    "hidden states for independent compression and score weights",
                )
                b.edge(a + ".q_norm", ix, "latent query")
                b.edge(ix, a + ".mix", "selected position indices")
                b.edge(comp, a + ".mix", "compressed KV payload")
            else:
                b.edge(comp, a + ".mix", "compressed KV positions")
        f = p + ".ffn"
        b.node(
            f,
            p,
            "ExpertCluster",
            "Routed + shared experts",
            {"routedExperts": 384, "activeRoutedExperts": 6, "sharedExperts": 1},
        )
        b.node(
            f + ".input",
            f,
            "TokenStream",
            "Normalized hidden states",
            {"shape": ["T", d]},
        )
        gate = f + ".gate"
        b.node(
            gate,
            f,
            "HashRouter" if lid < c["num_hash_layers"] else "ScoreRouter",
            "Token-ID hash router"
            if lid < c["num_hash_layers"]
            else "Corrected-score router",
            {
                "formula": "IDs from token table; weights from original sqrt(softplus(logits))"
                if lid < c["num_hash_layers"]
                else "sqrt(softplus(logits)) + correction for selection; weights use original selected scores",
                "topK": 6,
                "weightRule": "normalize selected original scores; multiply by 2.5",
            },
        )
        b.node(
            f + ".experts",
            f,
            "ExpertCluster",
            "384 routed experts",
            {"count": 384, "activePerToken": 6},
        )
        for j in range(384):
            b.node(
                f + f".experts.{j}",
                f + ".experts",
                "ActivationLayer",
                f"Expert {j}",
                {
                    "expertIndex": j,
                    "formula": "w2(SiLU(w1 x) ⊙ w3 x), with configured activation clamp",
                    "intermediateWidth": 3072,
                    "values": "route and activations unavailable",
                },
            )
        b.node(
            f + ".shared_experts",
            f,
            "SharedExpert",
            "Shared expert",
            {"formula": "same gated feedforward; not selected by the top-k router"},
        )
        b.node(
            f + ".merge",
            f,
            "ResidualBus",
            "Combine expert outputs",
            {"formula": "Σ selected route_weightₑ × expertₑ(x) + shared(x)"},
        )
        b.edge(f + ".input", gate, "hidden states (+ token IDs for hash routing)")
        b.edge(f + ".input", f + ".experts", "same hidden states to selected experts")
        b.edge(f + ".input", f + ".shared_experts", "same hidden states")
        b.edge(gate, f + ".experts", "selected IDs + weights (conditional)")
        b.edge(f + ".experts", f + ".merge", "weighted routed outputs")
        b.edge(f + ".shared_experts", f + ".merge", "shared output")
        # A hyper-connection is decomposed into pre-mix, branch evaluation, post-mix.
        b.node(
            p + ".post_attn",
            p,
            "HyperConnectionMix",
            "Attention post-mix",
            {
                "formula": "post weights distribute branch output; residual matrix mixes four streams"
            },
        )
        b.node(
            p + ".post_ffn",
            p,
            "HyperConnectionMix",
            "Feedforward post-mix",
            {
                "formula": "post weights distribute branch output; residual matrix mixes four streams"
            },
        )
        for source, target, label, kind in [
            ("input", "hc_attn", "stream mixture", "dataflow"),
            ("hc_attn", "attn_norm", "pre-mixed input", "dataflow"),
            ("attn_norm", "attn", "normalized state", "dataflow"),
            ("attn", "post_attn", "attention result", "dataflow"),
            ("hc_attn", "post_attn", "residual / post weights + streams", "residual"),
            ("post_attn", "hc_ffn", "updated streams", "dataflow"),
            ("hc_ffn", "ffn_norm", "pre-mixed input", "dataflow"),
            ("ffn_norm", "ffn", "normalized state", "dataflow"),
            ("ffn", "post_ffn", "expert result", "dataflow"),
            ("hc_ffn", "post_ffn", "residual / post weights + streams", "residual"),
        ]:
            b.edge(p + "." + source, p + "." + target, label, kind)
        if is_mtp:
            b.entities[p]["tensorRefs"] = ["embed.weight", "head.weight"]
            for key, label in [
                ("e_proj", "Token projection"),
                ("h_proj", "Hidden projection"),
                ("enorm", "Token RMSNorm"),
                ("hnorm", "Hidden RMSNorm"),
                ("norm", "Output RMSNorm"),
                ("hc_head", "MTP final stream mixing"),
            ]:
                b.node(
                    p + "." + key,
                    p,
                    "HyperConnectionMix"
                    if key == "hc_head"
                    else "NormalizationPlane"
                    if "norm" in key
                    else "ProjectionBlock",
                    label,
                )
    for name, t in catalog.items():
        if t["category"] == "quantization_scale":
            continue
        parts = name.split(".")
        parent = ".".join(parts[:-1])
        if name.startswith("hc_head_"):
            parent = "hc_head"
        elif re.search(r"\.hc_(attn|ffn)_(fn|base|scale)$", name):
            parent = re.sub(r"_(fn|base|scale)$", "", name)
        elif re.search(r"\.hc_head_(fn|base|scale)$", name):
            parent = name.rsplit("_", 1)[0]
        elif ".experts." in name or ".shared_experts." in name:
            parent = ".".join(parts[:-2])
        if not parent:
            parent = "model"
        if parent not in b.entities:
            grand = parent.rsplit(".", 1)[0]
            if grand not in b.entities:
                raise ValueError(f"Unclassified tensor owner: {name}")
            b.node(parent, grand, "ProjectionBlock", parent.rsplit(".", 1)[-1])
        storage = {
            "shape": t["stored_shape"],
            "dtype": t["stored_dtype"],
            "encoding": "packed FP4: two values per byte"
            if t["stored_shape"] != t["logical_shape"]
            else "checkpoint values",
        }
        scale = (
            catalog.get(name.removesuffix("weight") + "scale")
            if name.endswith("weight")
            else None
        )
        if scale:
            storage["scale"] = {
                "id": name.removesuffix("weight") + "scale",
                "shape": scale["stored_shape"],
                "dtype": scale["stored_dtype"],
            }
        b.tensor(
            name,
            parent,
            t["logical_shape"],
            "FP4 (logical)"
            if t["stored_shape"] != t["logical_shape"]
            else t["stored_dtype"],
            role="buffer"
            if t["category"] == "nontrainable_routing_table"
            else "parameter",
            storage=storage,
            label=".".join(parts[-2:]) if "experts" in parent else parts[-1],
        )
    expected = spec["reconciliation"]["counts"]["architecture_parameter"]
    if parameter_count(b.ir) != expected:
        raise ValueError("Hero parameter reconciliation failed")
    return validate(b.ir)


def tiny_example():
    x = [[0.8, -0.2, 0.4], [0.1, 0.9, -0.3], [-0.5, 0.4, 0.7], [0.6, 0.3, -0.8]]
    wq = [[0.6, -0.2, 0.4], [0.1, 0.7, -0.3]]
    wk = [[0.4, 0.3, -0.5], [-0.2, 0.8, 0.1]]
    wv = [[0.5, 0.2, 0.1], [-0.1, 0.3, 0.7], [0.4, -0.5, 0.2]]
    project = lambda w: [
        [sum(a * z for a, z in zip(row, weight)) for weight in w] for row in x
    ]
    q, k, v = project(wq), project(wk), project(wv)
    weights = []
    for i, row in enumerate(q):
        scores = [
            sum(a * z for a, z in zip(row, col)) / math.sqrt(2) for col in k[: i + 1]
        ]
        exps = [math.exp(z - max(scores)) for z in scores]
        weights.append([z / sum(exps) for z in exps] + [0] * (3 - i))
    output = [
        [sum(weights[i][j] * v[j][d] for j in range(4)) for d in range(3)]
        for i in range(4)
    ]
    b = Builder(
        "Computed attention example",
        {
            "kind": "computed miniature",
            "evidence": "illustrative",
            "notes": "Hand-chosen small operands and computed values. Not a trained model or a trace of the hero model.",
        },
    )
    for key, label, kind, data in [
        ("x", "Input vectors", "TokenStream", x),
        ("q", "Queries", "VectorColumn", q),
        ("k", "Keys", "VectorColumn", k),
        ("v", "Values", "VectorColumn", v),
        ("weights", "Attention coefficients", "AttentionGrid", weights),
        ("output", "Weighted value sum", "TokenStream", output),
    ]:
        b.node(key, "model", kind, label, evidence="illustrative")
        b.tensor(
            key + ".data",
            key,
            [len(data), len(data[0])],
            "F64",
            "activation",
            "illustrative",
            values=[z for row in data for z in row],
        )
    for source, target, label in [
        ("x", "q", "query projection"),
        ("x", "k", "key projection"),
        ("x", "v", "value projection"),
        ("q", "weights", "scaled QKᵀ + mask + softmax"),
        ("k", "weights", "key comparison"),
        ("weights", "output", "mixing coefficients"),
        ("v", "output", "vectors being mixed"),
    ]:
        b.edge(source, target, label, evidence="illustrative")
    return validate(b.ir)
