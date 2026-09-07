# Llama 4 Maverick architecture package

The bundled option is **Llama 4 Maverick 17B-128E Instruct**, released by Meta in April 2025. It is the released Maverick architecture, not the separately previewed Behemoth or a claim about Meta's newest non-Llama models.

The package includes all declared learned scalars in the text decoder, vision encoder, image adapter, and multimodal projector. It contains no model weights, tokenizer, activations, or inference runtime. All shapes are **constructor-derived**. The aggregate matches the public Hugging Face repository metadata exactly; this does **not** mean individual checkpoint headers were checked.

| Inventory | Learned scalars |
| --- | ---: |
| Decoder, embeddings and output | 400,711,848,960 |
| Vision encoder, adapter and projector | 871,932,416 |
| Total | **401,583,781,376** |

The publisher's “17B active” label is a separate rounded estimate. It is not the total stored parameter count or a measured route in this explorer.

## Structure that should remain visible

- **48 decoder blocks:** 24 dense SwiGLU blocks alternate with 24 MoE blocks. Dense layers use an intermediate width of 16,384; expert layers use 8,192.
- **Grouped-query attention:** 40 query heads use 8 key/value groups, with 5 queries per KV group and head width 128. K/V groups are not one shared latent vector and are not 40 independent K/V projections.
- **Position and mask pattern:** three chunked causal RoPE layers precede each global causal NoPE layer. The chunk size is 8,192. Chunking partitions the sequence; it is not a sliding window.
- **Routing:** each MoE bank has 128 routed experts, selects one, and also evaluates a shared expert. The reference implementation applies the selected sigmoid routing score to the expert input. Scaling an expert's nonlinear input is not generally equivalent to scaling its output.
- **Vision branch:** 34 bidirectional transformer blocks process 336×336 image tiles using 14×14 patches and width 1,408. A learned class token and patch positions are included. After encoding, the class token is discarded and each 2×2 patch neighborhood is gathered into a wider vector before projection to decoder width.
- **Early fusion:** projected image features replace image-placeholder embeddings before the decoder. Vision is not appended to the end of the text network.

## Pinned evidence

`references/llama4-maverick-facts.json` records source URLs, hashes, architectural facts, and the official aggregate. The inspected sources are:

1. [Meta torchtune model builders, commit bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1](https://github.com/meta-pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/models/llama4/_model_builders.py): Maverick's exact decoder and vision dimensions.
2. [Meta torchtune component builders, same commit](https://github.com/meta-pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/models/llama4/_component_builders.py): attention and MoE patterns, projection head, biases, default selection count, and parameter construction.
3. [Hugging Face Transformers v4.51.3 implementation](https://github.com/huggingface/transformers/blob/5f4ecf2d9f867a1255131d2461d75793c0cf1db2/src/transformers/models/llama4/modeling_llama4.py): an independent constructor cross-check, packed expert tensor orientation, router scaling, vision class/position parameters, and normalization details.
4. [Official model repository, revision 73d14711bcc77c16df3470856949c3764056b617](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct/tree/73d14711bcc77c16df3470856949c3764056b617): the public API reports 401,583,781,376 BF16 scalar parameters.

The official `config.json` download required gated access. It was not accessed, and no mirror was used to obtain it. The parameter inventory was instead independently derived from Meta's openly published constructors. No source implementation is copied into the renderer or extractor.

[Meta’s pinned MoE source](https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama4/moe.py) independently confirms that sigmoid routing scores multiply expert inputs. The public Meta Maverick builder explicitly disables Q/K normalization. One nonparametric runtime option remains unresolved: optional query-temperature scaling in NoPE layers. Meta and the reviewed Transformers defaults differ, and the gated checkpoint setting was not obtained. NoPE components record this uncertainty and the optional formula. The explorer does not claim that scaling is enabled or display fabricated temperature values; this does not affect the reconciled learned-parameter inventory.

## Expert storage and exact addresses

The Transformers implementation packs each bank into `gate_up_proj[128,5120,16384]` and `down_proj[128,8192,5120]`. ModelIR partitions these into three logical matrices per expert, in conventional `[output,input]` orientation. Each learned scalar is counted exactly once. These logical matrix IDs are **not** a claim that the checkpoint stores three separate tensors per expert.

For expert `e`, logical row `r`, column `c`:

| Logical matrix | Shape | Packed source coordinate |
| --- | --- | --- |
| Gate | 8,192 × 5,120 | `gate_up_proj[e,c,r]` |
| Up | 8,192 × 5,120 | `gate_up_proj[e,c,r+8192]` |
| Down | 5,120 × 8,192 | `down_proj[e,c,r]` |

Every partition includes `storage.sourceTensor`, `storage.sourceShape`, `storage.expertIndex`, `storage.offset`, and a coordinate mapping. Tests verify that the partitions cover the source arrays with no gaps or overlaps. Exact values remain unavailable because the package has no weights.

## Rebuild and validation

From the repository root, run `python3 -m exporters.llama4`. This uses the checked-in facts and the Python standard library only, with no network access needed. It creates `public/models/llama4-maverick.atlas.json.gz` reproducibly.

`python3 -m unittest discover -s tests -p test_llama4.py` checks independent parameter arithmetic, dense/MoE placement, query/KV grouping, expert storage partitions, upstream image fusion, and deterministic export.

The landscape summarizes the vision encoder as its own upstream component. Its individual blocks and tensors remain addressable through the canonical inventory; it does not pretend that a collapsed summary displays every vision operation simultaneously. Nonpersistent RoPE frequencies, inference-time dropout, and runtime activations are not learned parameter inventory.

## Acknowledgements

Llama 4 is a Meta model. The inspected Meta torchtune implementation carries its BSD-style project license; the inspected Transformers implementation carries Apache-2.0. This package distributes independently written architecture-description code and factual metadata, with no trained model, tokenizer, gated configuration, or third-party inference implementation. The model itself is governed by Meta's [Llama 4 Community License](https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama4/LICENSE), not this repository's software license.
