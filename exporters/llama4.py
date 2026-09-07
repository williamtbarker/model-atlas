"""Independent Llama 4 Maverick structural adapter; Python standard library only.

Facts are pinned to public Meta constructors. No weights or gated files are read.
Routed expert matrices are disjoint logical partitions of the fused HF storage;
their storage mappings retain the exact packed tensor identity and coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path

from .model_ir import Builder, parameter_count, validate, write_package

ROOT = Path(__file__).resolve().parent.parent


def llama4_maverick():
    spec = json.loads((ROOT / "references/llama4-maverick-facts.json").read_text())
    c, v = spec["text"], spec["vision"]
    d, f = c["hidden_size"], c["intermediate_size"]
    n, h, kv, hd = (
        c[k]
        for k in (
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        )
    )
    b = Builder(
        "Llama 4 Maverick · 402B",
        {
            "kind": "public Meta constructors + HF aggregate reconciliation",
            "evidence": "derived",
            "notes": "Full decoder, vision encoder and multimodal projector. Shapes are derived from pinned public constructors; their sum matches the publisher repository's public safetensors aggregate. No individual checkpoint headers, weights, gated config or runtime activations were accessed. Expert matrices are disjoint logical slices of packed storage. The optional NoPE temperature setting remains unverified.",
            "urls": list(spec["sources"].values()),
        },
    )
    b.ir["metadata"] = {
        "layers": n,
        "hiddenWidth": d,
        "queryHeads": h,
        "kvHeads": kv,
        "headDim": hd,
        "experts": c["num_local_experts"],
        "activeExperts": c["num_experts_per_tok"],
        "context": c["max_position_embeddings"],
        "architecture": "2025 · Alternating dense / MoE · grouped-query attention · early-fusion vision",
        "initialScope": "language_model.model.layers",
        "revision": spec["checkpoint_revision"],
        "scope": "full_multimodal_architecture",
        "scopeLabel": "Full multimodal architecture",
        "scopeNote": "48 decoder blocks plus a 34-block vision encoder and multimodal projector. Vision occupies its own branch; image features enter the token stream before decoding.",
        "parameterScope": "All declared learned scalars, including vision and the projector; constructor-derived and reconciled to the public aggregate.",
        "excludedComponents": [],
        "reportedActiveParameters": "17B (publisher estimate; path and input dependent)",
        "reconciliation": {
            "method": "constructor inventory versus public HF aggregate; no checkpoint headers",
            "expectedParameters": str(spec["publisher_parameter_total"]),
            "decoderParameters": "400711848960",
            "visionAndProjectorParameters": "871932416",
        },
    }

    def tensor(name, parent, shape, storage=None, label=None):
        return b.tensor(name, parent, shape, dtype="BF16", storage=storage, label=label)

    def projection(id, parent, rows, cols, label, bias=False, storage=None):
        b.node(
            id,
            parent,
            "ProjectionBlock",
            label,
            {
                "inputWidth": cols,
                "outputWidth": rows,
                "formula": "y = Wx + b" if bias else "y = Wx",
                "storageOrientation": "[output, input]",
            },
        )
        tensor(id + ".weight", id, [rows, cols], storage)
        if bias:
            tensor(id + ".bias", id, [rows])

    def norm(id, parent, width, label, rms=True):
        b.node(
            id,
            parent,
            "NormalizationPlane",
            label,
            {
                "formula": (
                    "x / sqrt(mean(x²) + ε) ⊙ γ"
                    if rms
                    else "(x − mean(x)) / sqrt(var(x) + ε) ⊙ γ + β"
                ),
                "epsilon": 1e-5,
            },
        )
        tensor(id + ".weight", id, [width])
        if not rms:
            tensor(id + ".bias", id, [width])

    def gated_mlp(
        id,
        parent,
        width,
        intermediate,
        label,
        kind="ActivationLayer",
        attrs=None,
        packed=None,
    ):
        details = {
            "formula": "W_down [SiLU(W_gate x) ⊙ (W_up x)]",
            "hiddenWidth": width,
            "intermediateWidth": intermediate,
            "activation": "silu",
            "gated": True,
        }
        details.update(attrs or {})
        b.node(id, parent, kind, label, details)
        for part in ("gate", "up", "down"):
            rows, cols = (
                (width, intermediate) if part == "down" else (intermediate, width)
            )
            storage = None
            if packed is not None:
                source, expert_index, count = packed
                storage = {
                    "kind": "disjoint_logical_partition",
                    "sourceTensor": source
                    + (".down_proj" if part == "down" else ".gate_up_proj"),
                    "sourceShape": (
                        [count, intermediate, width]
                        if part == "down"
                        else [count, width, 2 * intermediate]
                    ),
                    "expertIndex": expert_index,
                    "mapping": (
                        "source[e, column, row]"
                        if part == "down"
                        else "source[e, column, row + offset]"
                    ),
                    "offset": intermediate if part == "up" else 0,
                    "notes": "Logical [output,input] coordinates; no duplicate copy of the packed source tensor is counted.",
                }
            projection(
                id + f".{part}_proj",
                id,
                rows,
                cols,
                part.title() + " projection",
                storage=storage,
            )
        b.node(
            id + ".gate_product",
            id,
            "ActivationLayer",
            "SiLU × up",
            {
                "formula": "SiLU(gate) ⊙ up",
                "activation": "silu",
                "gated": True,
            },
        )
        b.edge(id + ".gate_proj", id + ".gate_product", "gate pre-activation")
        b.edge(id + ".up_proj", id + ".gate_product", "parallel up features")
        b.edge(id + ".gate_product", id + ".down_proj", "gated features")

    def attention(
        id, parent, width, heads, kv_heads, head_dim, *, index=None, vision=False
    ):
        chunked = not vision and (index + 1) % c["no_rope_layer_interval"] != 0
        attrs = {
            "queryHeads": heads,
            "kvHeads": kv_heads,
            "headDim": head_dim,
            "attentionType": "multi-head" if heads == kv_heads else "grouped-query",
            "queriesPerKVHead": heads // kv_heads,
            "causal": not vision,
            "positionEncoding": "2D RoPE" if vision else "RoPE" if chunked else "NoPE",
            "queryKeyNormalization": "none",
            "scoreScaling": f"1 / sqrt({head_dim})",
            "attentionPattern": (
                "bidirectional patch attention"
                if vision
                else "chunked causal" if chunked else "global causal"
            ),
            "values": "unavailable without a runtime trace",
        }
        if chunked:
            attrs["chunkSize"] = c["attention_chunk_size"]
            attrs["ropeTheta"] = c["rope_theta"]
        if not vision and not chunked:
            attrs["queryPreprocessing"] = (
                "No positional rotation; optional runtime temperature scaling is not established for this checkpoint."
            )
            attrs["temperatureSettingEvidence"] = "unavailable"
            attrs["optionalTemperatureFormula"] = spec["runtime_uncertainties"][
                "optional_temperature_formula"
            ]
        b.node(
            id,
            parent,
            "AttentionHead",
            (
                "Patch attention"
                if vision
                else ("Chunked GQA · RoPE" if chunked else "Global GQA · NoPE")
            ),
            attrs,
        )
        b.node(
            id + ".input",
            id,
            "TokenStream",
            "Normalized states",
            {"shape": ["patches" if vision else "T", width]},
        )
        for part, out in (
            ("q", heads * head_dim),
            ("k", kv_heads * head_dim),
            ("v", kv_heads * head_dim),
        ):
            projection(
                id + f".{part}_proj",
                id,
                out,
                width,
                part.upper() + " projection",
                bias=vision,
            )
            b.edge(id + ".input", id + f".{part}_proj", "same normalized states")
        qsource, ksource = id + ".q_proj", id + ".k_proj"
        if chunked or vision:
            b.node(
                id + ".rotary",
                id,
                "RotaryEmbedding",
                "2D position rotation" if vision else "Rotary position",
                {
                    "learnedParameters": 0,
                    "formula": "rotate Q and K by position-dependent orthogonal 2D rotations",
                },
            )
            b.edge(qsource, id + ".rotary", "queries")
            b.edge(ksource, id + ".rotary", "keys")
            qsource = ksource = id + ".rotary"
        b.node(
            id + ".heads",
            id,
            "RepeatedModuleArray",
            f"{heads} query heads · {kv_heads} KV groups",
            {
                "count": heads,
                "queryHeads": heads,
                "kvHeads": kv_heads,
                "headDim": head_dim,
            },
        )
        for j in range(heads):
            head = id + f".heads.{j}"
            b.node(
                head,
                id + ".heads",
                "AttentionHead",
                f"Query head {j} · KV {j // (heads // kv_heads)}",
                {
                    "headIndex": j,
                    "kvGroupIndex": j // (heads // kv_heads),
                    "headDim": head_dim,
                    "formula": "softmax(mask(QKᵀ / √dₕ)) V",
                    "causal": not vision,
                    "parameterView": "Slices of shared Q/K/V/output projections; no independent storage.",
                },
            )
            b.node(
                head + ".scores",
                head,
                "AttentionGrid",
                "Attention coefficients",
                {
                    "shape": ["P", "P"] if vision else ["T", "T"],
                    "values": "unavailable",
                    "rowAxis": "query position",
                    "columnAxis": "key position",
                },
            )
        for src, label in (
            (qsource, "queries"),
            (ksource, "keys shared within KV group"),
            (id + ".v_proj", "values shared within KV group"),
        ):
            b.edge(src, id + ".heads", label)
        projection(
            id + ".o_proj",
            id,
            width,
            heads * head_dim,
            "Output projection",
            bias=vision,
        )
        b.edge(id + ".heads", id + ".o_proj", "concatenate head results")

    b.node("language_model", "model", "ModelStage", "Text / fused-image decoder")
    b.node("language_model.model", "language_model", "ModelStage", "Decoder backbone")
    embed = "language_model.model.embed_tokens"
    b.node(
        embed,
        "language_model.model",
        "EmbeddingTable",
        "Token embeddings",
        {"formula": "xₜ = W_E[token_idₜ]"},
    )
    tensor(embed + ".weight", embed, [c["vocab_size"], d])
    b.node(
        "fusion",
        "model",
        "TokenStream",
        "Early image / text fusion",
        {
            "formula": "replace image placeholder token embeddings with projected image features",
            "shape": ["T", d],
            "modality": "image + text",
        },
    )
    b.edge(embed, "fusion", "text embeddings with image placeholders")
    layers = "language_model.model.layers"
    b.node(
        layers,
        "language_model.model",
        "RepeatedModuleArray",
        f"{n} decoder blocks",
        {"count": n},
    )
    b.edge("fusion", layers, "fused residual stream")
    for i in range(n):
        p = f"{layers}.{i}"
        is_moe = (i + 1) % c["interleave_moe_layer_step"] == 0
        b.node(
            p,
            layers,
            "TransformerBlock",
            f"Block {i} · {'MoE' if is_moe else 'dense'}",
            {
                "index": i,
                "residualStreams": 1,
                "feedForwardType": "moe" if is_moe else "dense",
                "modality": "fused image / text",
            },
        )
        if i:
            b.edge(f"{layers}.{i-1}", p, "residual stream")
        b.node(p + ".input", p, "TokenStream", "Residual input", {"shape": ["T", d]})
        norm(p + ".input_layernorm", p, d, "RMS norm · attention")
        attention(p + ".self_attn", p, d, h, kv, hd, index=i)
        b.node(
            p + ".add1",
            p,
            "ResidualBus",
            "Attention residual add",
            {"formula": "x + attention(RMSNorm(x))"},
        )
        norm(p + ".post_attention_layernorm", p, d, "RMS norm · feedforward")
        ff = p + ".feed_forward"
        if is_moe:
            count = c["num_local_experts"]
            b.node(
                ff,
                p,
                "ExpertCluster",
                "Routed + shared feedforward",
                {
                    "routedExperts": count,
                    "activeRoutedExperts": c["num_experts_per_tok"],
                    "sharedExperts": 1,
                    "routingWeightApplication": "expert input",
                },
            )
            b.node(ff + ".input", ff, "TokenStream", "Normalized hidden states")
            b.node(
                ff + ".router",
                ff,
                "ScoreRouter",
                "Top-1 sigmoid router",
                {
                    "count": count,
                    "topK": c["num_experts_per_tok"],
                    "formula": "select argmax(W_router x); s = sigmoid(selected logit); selected expert receives s·x",
                    "values": "selected expert and routing score unavailable without a trace",
                },
            )
            tensor(ff + ".router.weight", ff + ".router", [count, d])
            b.node(
                ff + ".experts",
                ff,
                "ExpertCluster",
                f"{count} routed experts",
                {
                    "count": count,
                    "activeExperts": c["num_experts_per_tok"],
                    "topK": c["num_experts_per_tok"],
                    "intermediateWidth": f,
                },
            )
            for j in range(count):
                gated_mlp(
                    ff + f".experts.{j}",
                    ff + ".experts",
                    d,
                    f,
                    f"Expert {j}",
                    attrs={
                        "expertIndex": j,
                        "input": "sigmoid(selected router logit) × x",
                    },
                    packed=(ff + ".experts", j, count),
                )
            gated_mlp(
                ff + ".shared_expert",
                ff,
                d,
                f,
                "Always-on shared expert",
                kind="SharedExpert",
                attrs={"alwaysActive": True},
            )
            b.node(
                ff + ".merge",
                ff,
                "ResidualBus",
                "Sum routed + shared",
                {"formula": "selected_expert(s·x) + shared_expert(x)"},
            )
            for src, dst, label in (
                ("input", "router", "hidden states"),
                ("input", "experts", "hidden states to selected expert"),
                ("router", "experts", "selected ID + input scale"),
                ("input", "shared_expert", "unscaled hidden states"),
                ("experts", "merge", "routed output"),
                ("shared_expert", "merge", "shared output"),
            ):
                b.edge(ff + "." + src, ff + "." + dst, label)
        else:
            gated_mlp(ff, p, d, c["intermediate_size_mlp"], "Dense SwiGLU feedforward")
        b.node(
            p + ".add2",
            p,
            "ResidualBus",
            "Feedforward residual add",
            {"formula": "x′ + feedforward(RMSNorm(x′))"},
        )
        for src, dst, label, kind in (
            ("input", "input_layernorm", "x", "dataflow"),
            ("input_layernorm", "self_attn", "normalized x", "dataflow"),
            ("self_attn", "add1", "attention output", "dataflow"),
            ("input", "add1", "skip connection", "residual"),
            ("add1", "post_attention_layernorm", "x′", "dataflow"),
            ("post_attention_layernorm", "feed_forward", "normalized x′", "dataflow"),
            ("feed_forward", "add2", "feedforward output", "dataflow"),
            ("add1", "add2", "skip connection", "residual"),
        ):
            b.edge(p + "." + src, p + "." + dst, label, kind)
    norm("language_model.model.norm", "language_model.model", d, "Final RMS norm")
    b.node(
        "language_model.lm_head",
        "language_model",
        "VocabularyPlane",
        "Vocabulary logits",
        {"formula": "logits = W_out x", "tiedEmbeddings": False},
    )
    tensor(
        "language_model.lm_head.weight", "language_model.lm_head", [c["vocab_size"], d]
    )
    b.edge(layers, "language_model.model.norm", "final residual stream")
    b.edge("language_model.model.norm", "language_model.lm_head", "normalized states")

    # The vision encoder is an upstream branch, never extra decoder blocks.
    vd, vf = v["hidden_size"], v["intermediate_size"]
    b.node(
        "vision_model",
        "model",
        "VisionEncoder",
        "Vision encoder · 34 blocks",
        {
            "modality": "image",
            "count": v["num_hidden_layers"],
            "imageSize": v["image_size"],
            "patchSize": v["patch_size"],
            "hiddenWidth": vd,
            "parameterScope": "Vision encoder, pixel-shuffle adapter and multimodal projector.",
        },
    )
    b.node(
        "vision_model.image",
        "vision_model",
        "TokenStream",
        "Image tiles",
        {"shape": ["tiles", 3, v["image_size"], v["image_size"]]},
    )
    projection(
        "vision_model.patch_embedding.linear",
        "vision_model",
        vd,
        3 * v["patch_size"] ** 2,
        "Patch embedding",
    )
    b.node(
        "vision_model.position",
        "vision_model",
        "EmbeddingTable",
        "Class + patch position embeddings",
        {
            "patchesPerTile": (v["image_size"] // v["patch_size"]) ** 2,
            "classTokenAppended": True,
        },
    )
    tensor(
        "vision_model.class_embedding",
        "vision_model.position",
        [vd],
        label="Class token",
    )
    tensor(
        "vision_model.positional_embedding_vlm",
        "vision_model.position",
        [(v["image_size"] // v["patch_size"]) ** 2 + 1, vd],
        label="Patch + class positions",
    )
    b.node(
        "vision_model.position_add",
        "vision_model",
        "ResidualBus",
        "Append class token; add positions",
        {"formula": "concat(patch_vectors, class_vector) + learned_positions"},
    )
    norm(
        "vision_model.layernorm_pre",
        "vision_model",
        vd,
        "Vision input layer norm",
        rms=False,
    )
    b.node(
        "vision_model.model",
        "vision_model",
        "RepeatedModuleArray",
        "34 vision transformer blocks",
        {"count": v["num_hidden_layers"], "modality": "image"},
    )
    for src, dst, label in (
        ("image", "patch_embedding.linear", "flatten 14×14 RGB patches"),
        ("patch_embedding.linear", "position_add", "patch vectors"),
        ("position", "position_add", "class token and positions"),
        ("position_add", "layernorm_pre", "positioned image sequence"),
        ("layernorm_pre", "model", "normalized patch states"),
    ):
        b.edge("vision_model." + src, "vision_model." + dst, label)
    for i in range(v["num_hidden_layers"]):
        p = f"vision_model.model.layers.{i}"
        b.node(
            p,
            "vision_model.model",
            "VisionBlock",
            f"Vision block {i}",
            {"index": i, "modality": "image", "causal": False},
        )
        if i:
            b.edge(f"vision_model.model.layers.{i-1}", p, "patch residual stream")
        b.node(p + ".input", p, "TokenStream", "Patch states")
        norm(p + ".input_layernorm", p, vd, "Attention layer norm", rms=False)
        attention(
            p + ".self_attn",
            p,
            vd,
            v["num_attention_heads"],
            v["num_attention_heads"],
            vd // v["num_attention_heads"],
            vision=True,
        )
        b.node(p + ".add1", p, "ResidualBus", "Attention residual add")
        norm(p + ".post_attention_layernorm", p, vd, "MLP layer norm", rms=False)
        b.node(
            p + ".mlp",
            p,
            "ActivationLayer",
            "Vision GELU feedforward",
            {
                "formula": "W₂ GELU(W₁x + b₁) + b₂",
                "activation": "gelu",
                "intermediateWidth": vf,
            },
        )
        projection(
            p + ".mlp.fc1", p + ".mlp", vf, vd, "Expand patch features", bias=True
        )
        b.node(
            p + ".mlp.gelu",
            p + ".mlp",
            "ActivationLayer",
            "GELU",
            {"activation": "gelu"},
        )
        projection(
            p + ".mlp.fc2", p + ".mlp", vd, vf, "Contract patch features", bias=True
        )
        b.edge(p + ".mlp.fc1", p + ".mlp.gelu", "pre-activation")
        b.edge(p + ".mlp.gelu", p + ".mlp.fc2", "activated features")
        b.node(p + ".add2", p, "ResidualBus", "MLP residual add")
        for src, dst, kind in (
            ("input", "input_layernorm", "dataflow"),
            ("input_layernorm", "self_attn", "dataflow"),
            ("self_attn", "add1", "dataflow"),
            ("input", "add1", "residual"),
            ("add1", "post_attention_layernorm", "dataflow"),
            ("post_attention_layernorm", "mlp", "dataflow"),
            ("mlp", "add2", "dataflow"),
            ("add1", "add2", "residual"),
        ):
            b.edge(
                p + "." + src,
                p + "." + dst,
                "patch states" if kind == "dataflow" else "skip connection",
                kind,
            )
    norm(
        "vision_model.layernorm_post",
        "vision_model",
        vd,
        "Vision output layer norm",
        rms=False,
    )
    b.edge(
        "vision_model.model", "vision_model.layernorm_post", "encoded image sequence"
    )
    b.node(
        "vision_model.vision_adapter",
        "vision_model",
        "ActivationLayer",
        "Pixel shuffle + projection",
        {
            "formula": "2×2 neighboring patch positions → one position with 4× channels; GELU(W₂ GELU(W₁x))",
            "pixelShuffleRatio": 0.5,
            "inputWidth": vd,
            "shuffledWidth": 4 * vd,
            "outputWidth": v["projector_output_dim"],
        },
    )
    b.node(
        "vision_model.vision_adapter.shuffle",
        "vision_model.vision_adapter",
        "TensorReshape",
        "Drop class token; gather 2×2 patches",
        {"formula": "[H,W,C] → [H/2,W/2,4C]", "learnedParameters": 0},
    )
    projection(
        "vision_model.vision_adapter.mlp.fc1",
        "vision_model.vision_adapter",
        v["projector_input_dim"],
        4 * vd,
        "Image adapter · expand",
    )
    b.node(
        "vision_model.vision_adapter.gelu1",
        "vision_model.vision_adapter",
        "ActivationLayer",
        "GELU",
        {"activation": "gelu"},
    )
    projection(
        "vision_model.vision_adapter.mlp.fc2",
        "vision_model.vision_adapter",
        v["projector_output_dim"],
        v["projector_output_dim"],
        "Image adapter · transform",
    )
    b.node(
        "vision_model.vision_adapter.gelu2",
        "vision_model.vision_adapter",
        "ActivationLayer",
        "GELU",
        {"activation": "gelu"},
    )
    b.edge(
        "vision_model.layernorm_post",
        "vision_model.vision_adapter",
        "encoded patches; discard class token",
    )
    for src, dst, label in (
        ("shuffle", "mlp.fc1", "gathered channels"),
        ("mlp.fc1", "gelu1", "pre-activation"),
        ("gelu1", "mlp.fc2", "activated features"),
        ("mlp.fc2", "gelu2", "pre-activation"),
    ):
        b.edge(
            "vision_model.vision_adapter." + src,
            "vision_model.vision_adapter." + dst,
            label,
        )
    projection(
        "multi_modal_projector.linear_1",
        "vision_model",
        d,
        v["vision_output_dim"],
        "Vision → token width",
    )
    b.edge(
        "vision_model.vision_adapter",
        "multi_modal_projector.linear_1",
        "adapted image features",
    )
    b.edge(
        "multi_modal_projector.linear_1",
        "fusion",
        "projected image features replace placeholders",
    )
    total = parameter_count(b.ir)
    if total != spec["publisher_parameter_total"]:
        raise ValueError(
            f"Llama 4 inventory mismatch: {total} != {spec['publisher_parameter_total']}"
        )
    return validate(b.ir)


if __name__ == "__main__":
    print(
        json.dumps(
            write_package(
                llama4_maverick(), ROOT / "public/models/llama4-maverick.atlas.json.gz"
            ),
            indent=2,
        )
    )
