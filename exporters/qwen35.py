"""Qwen3.5 text-decoder adapter. No torch, weights, inference, or network required.

This independent adapter derives dimensions from a pinned public configuration
and reconciles serialized names against the pinned checkpoint index. It does
not claim safetensors-header or numerical-value verification. Fused experts are
partitioned into disjoint logical matrix views, never counted twice.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

from .model_ir import Builder, parameter_count, validate, write_package

ROOT = Path(__file__).resolve().parent.parent


def load_references(root=ROOT):
    refs = Path(root) / "references"
    config = json.loads((refs / "qwen35-config.json").read_text())
    provenance = json.loads((refs / "qwen35-provenance.json").read_text())
    index = json.loads(gzip.decompress((refs / "qwen35-index.json.gz").read_bytes()))
    return config, provenance, index


def derive_catalog(config):
    """Serialized parameter shapes for text decoder, embedding and LM head only."""
    c = config["text_config"]
    if config.get("model_type") != "qwen3_5_moe":
        raise ValueError("Expected the supported Qwen3.5 MoE config")
    if config.get("tie_word_embeddings") or c.get("attention_bias"):
        raise ValueError("Unsupported tied or biased Qwen3.5 variant")
    if not c.get("attn_output_gate", True) or c.get("mlp_only_layers"):
        raise ValueError("Unsupported output-gate or MLP-only variant")
    if len(c["layer_types"]) != c["num_hidden_layers"]:
        raise ValueError("Incomplete explicit layer schedule")
    if any(x not in {"linear_attention", "full_attention"} for x in c["layer_types"]):
        raise ValueError("Unsupported token mixer")
    d, f, e = c["hidden_size"], c["moe_intermediate_size"], c["num_experts"]
    h, kv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
    kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
    if h % kv or vh % kh:
        raise ValueError("Nonintegral head grouping")
    catalog = {"model.language_model.embed_tokens.weight": [c["vocab_size"], d],
               "model.language_model.norm.weight": [d],
               "lm_head.weight": [c["vocab_size"], d]}
    for i, kind in enumerate(c["layer_types"]):
        p = f"model.language_model.layers.{i}"
        catalog[p + ".input_layernorm.weight"] = [d]
        catalog[p + ".post_attention_layernorm.weight"] = [d]
        catalog[p + ".mlp.gate.weight"] = [e, d]
        catalog[p + ".mlp.experts.gate_up_proj"] = [e, 2 * f, d]
        catalog[p + ".mlp.experts.down_proj"] = [e, d, f]
        sf = c["shared_expert_intermediate_size"]
        for key, shape in {"gate_proj": [sf, d], "up_proj": [sf, d], "down_proj": [d, sf]}.items():
            catalog[p + ".mlp.shared_expert." + key + ".weight"] = shape
        catalog[p + ".mlp.shared_expert_gate.weight"] = [1, d]
        if kind == "full_attention":
            for key, shape in {"q_proj": [2 * h * hd, d], "k_proj": [kv * hd, d],
                               "v_proj": [kv * hd, d], "o_proj": [d, h * hd],
                               "q_norm": [hd], "k_norm": [hd]}.items():
                catalog[p + ".self_attn." + key + ".weight"] = shape
        else:
            cd = 2 * kh * kd + vh * vd
            for key, shape in {"in_proj_qkv": [cd, d], "in_proj_z": [vh * vd, d],
                               "in_proj_b": [vh, d], "in_proj_a": [vh, d],
                               "conv1d": [cd, 1, c["linear_conv_kernel_dim"]],
                               "norm": [vd], "out_proj": [d, vh * vd]}.items():
                catalog[p + ".linear_attn." + key + ".weight"] = shape
            catalog[p + ".linear_attn.dt_bias"] = [vh]
            catalog[p + ".linear_attn.A_log"] = [vh]
    return catalog


def qwen35(root=ROOT):
    config, provenance, checkpoint = load_references(root)
    c = config["text_config"]
    catalog = derive_catalog(config)
    checkpoint_names = {name for name in checkpoint["weight_map"]
                        if name.startswith("model.language_model.") or name == "lm_head.weight"}
    if set(catalog) != checkpoint_names:
        raise ValueError(f"Text inventory mismatch: missing={sorted(checkpoint_names - catalog.keys())[:6]}, extra={sorted(catalog.keys() - checkpoint_names)[:6]}")
    d, f, e, top = c["hidden_size"], c["moe_intermediate_size"], c["num_experts"], c["num_experts_per_tok"]
    n = c["num_hidden_layers"]
    excluded = ["Vision encoder and visual patch merger", "Auxiliary multi-token prediction module (MTP)"]
    scope_note = "Text decoder, token embedding, final normalization and untied language-model head. Vision encoder and auxiliary MTP are excluded."
    b = Builder("Qwen3.5-397B-A17B · text decoder", {
        "kind": "pinned config + checkpoint-name index + constructor-derived logical shapes",
        "evidence": "derived",
        "notes": scope_note + " Fused expert storage is partitioned into disjoint logical views. Names match the included checkpoint index; tensor headers, values and runtime activity are not loaded.",
        "urls": [provenance["configUrl"], provenance["indexUrl"], provenance["modelingUrl"]],
    })
    b.ir["metadata"] = {
        "layers": n, "hiddenWidth": d, "experts": e, "activeExperts": top,
        "context": c["max_position_embeddings"], "queryHeads": c["num_attention_heads"],
        "kvHeads": c["num_key_value_heads"], "headDim": c["head_dim"],
        "linearAttentionLayers": c["layer_types"].count("linear_attention"),
        "fullAttentionLayers": c["layer_types"].count("full_attention"),
        "architecture": "Hybrid GatedDeltaNet / gated GQA / sparse MoE",
        "initialScope": "layers", "revision": provenance["revision"],
        "scope": "Text decoder", "scopeLabel": "Text decoder only", "scopeNote": scope_note,
        "parameterScope": "Included text decoder + embedding + final norm + LM head; excludes vision and MTP",
        "excludedComponents": excluded,
        "reportedParameters": "397B total / 17B active (publisher's model-level rounded figures, not this scoped exact count)",
        "reconciliation": {"method": "Serialized names matched; shapes derived from pinned constructors", "includedCheckpointTensors": len(catalog),
                           "excludedCheckpointTensors": len(checkpoint["weight_map"]) - len(catalog), "headersVerified": False},
    }
    consumed = set()

    def tensor(name, parent, label=None):
        if name in consumed:
            raise ValueError("Repeated serialized parameter")
        consumed.add(name)
        return b.tensor(name, parent, catalog[name], dtype="BF16", label=label)

    def node(id, parent, kind, label, attrs=None):
        return b.node(id, parent, kind, label, attrs)

    def projection(id, parent, name, label, attrs=None):
        shape = catalog[name]
        node(id, parent, "ProjectionBlock", label,
             {"formula": "y = Wx", "inputWidth": shape[-1], "outputWidth": shape[-2], **(attrs or {})})
        tensor(name, id, "Weight")

    def norm(id, parent, name, label, gated=False):
        node(id, parent, "NormalizationPlane", label,
             {"width": catalog[name][0], "epsilon": c["rms_norm_eps"],
              "formula": "RMSNorm(x) ⊙ γ ⊙ SiLU(z)" if gated else "x / sqrt(mean(x²) + ε) ⊙ (1 + γ)"})
        tensor(name, id, "Learned RMS scale")

    def edge(source, target, label, kind="dataflow"):
        b.edge(source, target, label, kind)

    node("embed", "model", "EmbeddingTable", "Token embedding", {"vocabulary": c["vocab_size"], "formula": "xₜ = embedding[token_idₜ]"})
    tensor("model.language_model.embed_tokens.weight", "embed")
    node("layers", "model", "RepeatedModuleArray", f"{n} text decoder blocks", {"count": n})
    norm("norm", "model", "model.language_model.norm.weight", "Final RMS normalization")
    node("head", "model", "VocabularyPlane", "Vocabulary logits", {"vocabulary": c["vocab_size"], "formula": "logits = W_head x; output weights are untied"})
    tensor("lm_head.weight", "head")
    edge("embed", "layers", "token embeddings")
    edge("layers", "norm", "final decoder states")
    edge("norm", "head", "normalized states")

    for i, kind in enumerate(c["layer_types"]):
        p, stored = f"layers.{i}", f"model.language_model.layers.{i}"
        short = "GatedDeltaNet" if kind == "linear_attention" else "Gated GQA"
        node(p, "layers", "TransformerBlock", f"Block {i} · {short}",
             {"index": i, "residualStreams": 1, "tokenMixer": kind})
        if i:
            edge(f"layers.{i - 1}", p, "residual stream")
        node(p + ".input", p, "TokenStream", "Residual input", {"shape": ["T", d]})
        norm(p + ".attn_norm", p, stored + ".input_layernorm.weight", "Token mixer RMSNorm")
        norm(p + ".ffn_norm", p, stored + ".post_attention_layernorm.weight", "Feedforward RMSNorm")
        a = p + ".attn"
        if kind == "linear_attention":
            kh, vh, kd, vd = [c[key] for key in ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim")]
            ac = stored + ".linear_attn"
            recurrence = "S̄ₜ = αₜ Sₜ₋₁; Sₜ = S̄ₜ + βₜ kₜ(vₜ − kₜᵀS̄ₜ)ᵀ; oₜ = qₜᵀSₜ / √dₖ"
            node(a, p, "LinearAttention", "Gated DeltaNet", {
                "algorithm": "GatedDeltaNet", "keyHeads": kh, "valueHeads": vh,
                "keyHeadDim": kd, "valueHeadDim": vd, "stateShape": [vh, kd, vd],
                "convKernel": c["linear_conv_kernel_dim"], "formula": recurrence,
                "description": "Each head updates a fixed-size recurrent key/value state. There is no softmax attention probability matrix. Q/K heads repeat across value-head groups; Q/K vectors are L2-normalized.",
                "values": "State, decay, update gates and activations require a runtime trace.",
            })
            node(a + ".input", a, "TokenStream", "Normalized hidden states", {"shape": ["T", d]})
            for key, label in [("in_proj_qkv", "Q / K / V projection"), ("in_proj_z", "Output gate projection"),
                               ("in_proj_b", "Update-strength projection"), ("in_proj_a", "Decay projection"), ("out_proj", "Output projection")]:
                projection(a + "." + key, a, ac + "." + key + ".weight", label)
            node(a + ".conv1d", a, "CausalConvolution", "Depthwise causal convolution + SiLU", {
                "kernelSize": c["linear_conv_kernel_dim"], "channels": 2 * kh * kd + vh * vd,
                "formula": "QKVₜ = SiLU(Σⱼ depthwise_kernelⱼ ⊙ projected_QKVₜ₋ⱼ)",
                "stateShape": [2 * kh * kd + vh * vd, c["linear_conv_kernel_dim"]],
            })
            tensor(ac + ".conv1d.weight", a + ".conv1d")
            node(a + ".qk_norm", a, "NormalizationPlane", "Q/K L2 normalization + head grouping", {
                "formula": "q̂ = q / sqrt(Σq² + ε), k̂ = k / sqrt(Σk² + ε); repeat Q/K for V-head groups",
                "keyHeads": kh, "valueHeads": vh, "learnedScale": False, "valueBranch": "V passes through unchanged",
            })
            node(a + ".decay", a, "Gate", "Recurrent decay", {"formula": "αₜ = exp(−exp(A_log) ⊙ softplus(aₜ + dt_bias))", "shape": [vh]})
            tensor(ac + ".A_log", a + ".decay")
            tensor(ac + ".dt_bias", a + ".decay")
            node(a + ".beta", a, "Gate", "Delta update strength", {"formula": "βₜ = sigmoid(bₜ)", "shape": [vh]})
            node(a + ".previous_state", a, "RecurrentState", "Previous state Sₜ₋₁", {"stateShape": [vh, kd, vd], "timeIndex": "t−1", "role": "runtime cache, not learned parameters"})
            node(a + ".state", a, "RecurrentState", "Delta update + state read", {"stateShape": [vh, kd, vd], "formula": recurrence, "timeIndex": "t", "temporalRecurrence": True,
                "role": "runtime cache, not learned parameters", "stateScalarsPerSequence": vh * kd * vd})
            norm(a + ".norm", a, ac + ".norm.weight", "Gated output RMSNorm", gated=True)
            for source, target, label in [("input", "in_proj_qkv", "hidden states"), ("input", "in_proj_z", "hidden states"), ("input", "in_proj_a", "hidden states"),
                                          ("input", "in_proj_b", "hidden states"), ("in_proj_qkv", "conv1d", "projected Q/K/V"), ("conv1d", "qk_norm", "convolved Q/K/V"),
                                          ("qk_norm", "state", "normalized/grouped Q/K and unmodified V"), ("in_proj_a", "decay", "aₜ"), ("in_proj_b", "beta", "bₜ"),
                                          ("decay", "state", "decay αₜ"), ("beta", "state", "update βₜ"), ("previous_state", "state", "cached Sₜ₋₁"),
                                          ("state", "norm", "per-head readout oₜ"), ("in_proj_z", "norm", "output gate zₜ"), ("norm", "out_proj", "normalized and gated readout")]:
                edge(a + "." + source, a + "." + target, label)
        else:
            h, kv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
            ac = stored + ".self_attn"
            node(a, p, "AttentionHead", "Gated grouped-query attention", {
                "queryHeads": h, "kvHeads": kv, "headDim": hd, "outputGate": True,
                "formula": "W_O[softmax(causal(QKᵀ/√dₖ))V ⊙ sigmoid(gate)]",
                "KV": f"{kv} K/V groups, each shared by {h // kv} query heads",
                "rotaryDimensions": int(hd * c["rope_parameters"]["partial_rotary_factor"]),
            })
            node(a + ".input", a, "TokenStream", "Normalized hidden states", {"shape": ["T", d]})
            for key, label in [("q_proj", "Query + output gate projection"), ("k_proj", "Key projection"), ("v_proj", "Value projection"), ("o_proj", "Output projection")]:
                projection(a + "." + key, a, ac + "." + key + ".weight", label)
            for key, label in [("q_norm", "Per-head Q RMSNorm"), ("k_norm", "Per-head K RMSNorm")]:
                norm(a + "." + key, a, ac + "." + key + ".weight", label)
            node(a + ".rotary", a, "ProjectionBlock", "Partial rotary positions", {"formula": "Apply partial rotary positions to Q and K; no learned parameters", "rotaryDimensions": int(hd * c["rope_parameters"]["partial_rotary_factor"])})
            node(a + ".heads", a, "AttentionHead", "Grouped causal attention", {"queryHeads": h, "kvHeads": kv, "headDim": hd, "formula": "softmax(causal(QKᵀ/√dₖ))V"})
            for j in range(h):
                node(a + f".heads.{j}", a + ".heads", "AttentionHead", f"Query head {j}", {"headIndex": j, "kvGroup": j // (h // kv), "headDim": hd, "parameterView": "Slices of shared projection weights; no new storage"})
            node(a + ".gate", a, "Gate", "Attention output gate", {"formula": "attention_output ⊙ sigmoid(gate)", "gateSource": "Second per-head half of q_proj output; not an independent Q projection"})
            for source, target, label in [("input", "q_proj", "hidden states"), ("input", "k_proj", "hidden states"), ("input", "v_proj", "hidden states"),
                                          ("q_proj", "q_norm", "Q half of each head's fused projection"), ("q_proj", "gate", "output-gate half of each head's projection"),
                                          ("k_proj", "k_norm", "key heads"), ("q_norm", "rotary", "normalized Q"), ("k_norm", "rotary", "normalized K"),
                                          ("rotary", "heads", "positioned Q/K"), ("v_proj", "heads", "shared-group values"), ("heads", "gate", "causal attention outputs"), ("gate", "o_proj", "gated concatenated outputs")]:
                edge(a + "." + source, a + "." + target, label)

        m, ms = p + ".ffn", stored + ".mlp"
        node(m, p, "ExpertCluster", "Sparse + shared feedforward", {"routedExperts": e, "activeRoutedExperts": top, "sharedExperts": 1})
        node(m + ".input", m, "TokenStream", "Normalized hidden states", {"shape": ["T", d]})
        node(m + ".gate", m, "ScoreRouter", "Softmax top-k router", {"topK": top, "formula": "p = softmax(W_router x); select top-k; renormalize selected p to sum to one"})
        tensor(ms + ".gate.weight", m + ".gate", "Router weight")
        node(m + ".experts", m, "ExpertCluster", f"{e} routed experts", {"count": e, "activePerToken": top})
        for fused in (ms + ".experts.gate_up_proj", ms + ".experts.down_proj"):
            consumed.add(fused)
        for j in range(e):
            expert = m + f".experts.{j}"
            node(expert, m + ".experts", "ActivationLayer", f"Expert {j}", {"expertIndex": j, "intermediateWidth": f,
                 "formula": "W_down(SiLU(W_gate x) ⊙ W_up x)", "values": "unavailable"})
            for key, shape, fused, offset in [
                ("gate_proj", [f, d], "gate_up_proj", j * 2 * f * d),
                ("up_proj", [f, d], "gate_up_proj", j * 2 * f * d + f * d),
                ("down_proj", [d, f], "down_proj", j * d * f),
            ]:
                checkpoint_name = ms + ".experts." + fused
                tid = expert + "." + key + ".weight"
                b.tensor(tid, expert, shape, dtype="BF16", label=key, storage={
                    "checkpointTensor": checkpoint_name,
                    "checkpointShape": catalog[checkpoint_name], "offsetElements": str(offset),
                })
        node(m + ".shared_expert", m, "SharedExpert", "Shared gated feedforward", {"intermediateWidth": c["shared_expert_intermediate_size"], "formula": "W_down(SiLU(W_gate x) ⊙ W_up x)", "routing": "Always evaluated, then multiplied by a learned scalar sigmoid gate"})
        for key, label in [("gate_proj", "Shared gate matrix"), ("up_proj", "Shared up matrix"), ("down_proj", "Shared down matrix")]:
            tensor(ms + ".shared_expert." + key + ".weight", m + ".shared_expert", label)
        node(m + ".shared_gate", m, "Gate", "Shared-expert output gate", {"formula": "sigmoid(w_shared x) × shared_expert(x)"})
        tensor(ms + ".shared_expert_gate.weight", m + ".shared_gate")
        node(m + ".merge", m, "ResidualBus", "Combine feedforward outputs", {"formula": "Σ selected pₑ expertₑ(x) + sigmoid(w_shared x) shared(x)"})
        for source, target, label in [("input", "gate", "router input"), ("input", "experts", "same states to selected experts"), ("gate", "experts", "selected IDs and normalized weights"),
                                      ("input", "shared_expert", "shared branch input"), ("input", "shared_gate", "scalar-gate input"), ("shared_expert", "shared_gate", "shared output"),
                                      ("shared_gate", "merge", "gated shared output"), ("experts", "merge", "weighted routed outputs")]:
            edge(m + "." + source, m + "." + target, label)
        node(p + ".add1", p, "ResidualBus", "Token mixer residual add", {"formula": "x′ = x + mixer(norm(x))"})
        node(p + ".add2", p, "ResidualBus", "Feedforward residual add", {"formula": "x″ = x′ + MoE(norm(x′))"})
        for source, target, label, kind in [("input", "attn_norm", "x", "dataflow"), ("attn_norm", "attn", "normalized x", "dataflow"), ("attn", "add1", "mixer output", "dataflow"),
            ("input", "add1", "skip connection", "residual"), ("add1", "ffn_norm", "x′", "dataflow"), ("ffn_norm", "ffn", "normalized x′", "dataflow"),
            ("ffn", "add2", "feedforward output", "dataflow"), ("add1", "add2", "skip connection", "residual")]:
            edge(p + "." + source, p + "." + target, label, kind)
    if consumed != set(catalog):
        raise ValueError(f"Unrepresented serialized parameters: {sorted(set(catalog) - consumed)}")
    expected = sum(math.prod(shape) for shape in catalog.values())
    if parameter_count(b.ir) != expected:
        raise ValueError("Logical views do not reconcile with serialized constructor shapes")
    b.ir["metadata"]["includedParameters"] = str(expected)
    b.ir["metadata"]["reconciliation"]["logicalParameterCount"] = str(expected)
    b.ir["metadata"]["reconciliation"]["logicalTensorViews"] = len(b.ir["tensors"])
    return validate(b.ir)


if __name__ == "__main__":
    print(json.dumps(write_package(qwen35(), ROOT / "public/models/qwen35-397b.atlas.json.gz"), indent=2))
