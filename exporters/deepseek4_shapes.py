"""Independent logical/storage tensor derivation for the pinned DeepSeek V4 release.

No model implementation is imported or executed. No weights are read.
See references/deepseek4-shapes.json and docs/deepseek4-provenance.md.
"""

import math


def derive_catalog(config):
    C = config
    D = C["hidden_size"]
    H = C["num_attention_heads"]
    A = C["head_dim"]
    R = C["q_lora_rank"]
    G = C["o_groups"]
    O = C["o_lora_rank"]
    E = C["n_routed_experts"]
    I = C["moe_intermediate_size"]
    V = C["vocab_size"]
    K = C["num_experts_per_tok"]
    HC = C["hc_mult"]
    M = (2 + HC) * HC
    HD = HC * D
    IH = C["index_n_heads"]
    IA = C["index_head_dim"]
    expected = {}

    def put(name, shape, dtype="BF16", category="architecture_parameter", quant=None):
        logical = list(shape)
        stored = logical.copy()
        if quant == "fp4":
            stored[-1] //= 2
            dtype = "I8"
        if quant == "fp8":
            dtype = "F8_E4M3"
        expected[name] = {
            "logical_shape": logical,
            "stored_shape": stored,
            "stored_dtype": dtype,
            "category": category,
            "logical_elements": math.prod(logical),
            "stored_elements": math.prod(stored),
        }
        if quant:
            scale_shape = (
                [logical[0], logical[1] // 32]
                if quant == "fp4"
                else [(logical[0] + 127) // 128, (logical[1] + 127) // 128]
            )
            put(
                name.removesuffix("weight") + "scale",
                scale_shape,
                "F8_E8M0",
                "quantization_scale",
            )

    def comp(prefix, ratio, width):
        c = 2 if ratio == 4 else 1
        put(prefix + ".ape", [ratio, c * width], "F32")
        put(prefix + ".wkv.weight", [c * width, D])
        put(prefix + ".wgate.weight", [c * width, D])
        put(prefix + ".norm.weight", [width])

    def hc_head(prefix):
        put(prefix + "hc_head_fn", [HC, HD], "F32")
        put(prefix + "hc_head_base", [HC], "F32")
        put(prefix + "hc_head_scale", [1], "F32")

    def block(prefix, lid, ratio):
        for branch in ["attn", "ffn"]:
            put(prefix + f"hc_{branch}_fn", [M, HD], "F32")
            put(prefix + f"hc_{branch}_base", [M], "F32")
            put(prefix + f"hc_{branch}_scale", [3], "F32")
            put(prefix + branch + "_norm.weight", [D])
        ap = prefix + "attn."
        put(ap + "attn_sink", [H], "F32")
        for name, shape in [
            ("wq_a", [R, D]),
            ("wq_b", [H * A, R]),
            ("wkv", [A, D]),
            ("wo_a", [G * O, H * A // G]),
            ("wo_b", [D, G * O]),
        ]:
            put(ap + name + ".weight", shape, quant="fp8")
        put(ap + "q_norm.weight", [R])
        put(ap + "kv_norm.weight", [A])
        if ratio:
            comp(ap + "compressor", ratio, A)
            if ratio == 4:
                put(ap + "indexer.wq_b.weight", [IH * IA, R], quant="fp8")
                put(ap + "indexer.weights_proj.weight", [IH, D])
                comp(ap + "indexer.compressor", ratio, IA)
        fp = prefix + "ffn."
        put(fp + "gate.weight", [E, D])
        if lid < C["num_hash_layers"]:
            put(fp + "gate.tid2eid", [V, K], "I64", "nontrainable_routing_table")
        else:
            put(fp + "gate.bias", [E], "F32")
        for name, shape in [("w1", [I, D]), ("w2", [D, I]), ("w3", [I, D])]:
            put(fp + "shared_experts." + name + ".weight", shape, quant="fp8")
            for ei in range(E):
                put(fp + f"experts.{ei}." + name + ".weight", shape, quant="fp4")

    put("embed.weight", [V, D])
    put("head.weight", [V, D])
    put("norm.weight", [D])
    hc_head("")
    for lid in range(C["num_hidden_layers"]):
        block(f"layers.{lid}.", lid, C["compress_ratios"][lid])
    for mid in range(C["num_nextn_predict_layers"]):
        lid = C["num_hidden_layers"] + mid
        prefix = f"mtp.{mid}."
        block(prefix, lid, C["compress_ratios"][lid])
        hc_head(prefix)
        for name in ["e_proj", "h_proj"]:
            put(prefix + name + ".weight", [D, D], quant="fp8")
        for name in ["enorm", "hnorm", "norm"]:
            put(prefix + name + ".weight", [D])
    return expected
