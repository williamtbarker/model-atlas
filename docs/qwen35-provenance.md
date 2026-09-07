# Qwen3.5-397B-A17B: scoped text-decoder package

This package covers the **60-block text decoder, token embedding, final RMS normalization, and untied vocabulary head**. It excludes the vision encoder/patch merger and auxiliary multi-token prediction module. The exact included count is **396,346,350,336 learned scalar parameters**. The publisher's rounded 397B total and 17B activated model-level figures are not substituted for this scoped count.

The contrast with DeepSeek is computational: 45 blocks use a fixed-size Gated DeltaNet recurrent state and 15 use gated grouped-query softmax attention. Every block has 512 routed experts, selects 10 per token, and adds a separately gated shared expert. Actual selected experts, gates, states and attention probabilities are unavailable without a runtime trace.

## Evidence

- Official checkpoint revision: [`8472618112abcbd45acbcdc58436aff4233c23f7`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/tree/8472618112abcbd45acbcdc58436aff4233c23f7).
- Official [configuration](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/8472618112abcbd45acbcdc58436aff4233c23f7/config.json) and [checkpoint name index](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/8472618112abcbd45acbcdc58436aff4233c23f7/model.safetensors.index.json) are retained locally.
- Constructor and forward-flow review: [Qwen3.5 MoE implementation in Transformers](https://github.com/huggingface/transformers/blob/26600f6bb24199fa57eb72c811add2cc1f25f78c/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py), pinned at `26600f6bb24199fa57eb72c811add2cc1f25f78c`.
- `references/qwen35-provenance.json` records these revisions, reviewed classes and SHA-256 digests. The package includes metadata and independently written adapter code; it includes no upstream Python implementation or model weights.

All **1,038 serialized tensor names in scope** match the checkpoint index. Dimensions are derived from the pinned constructors and configuration. **Safetensors headers and tensor payloads have not been fetched or verified.** BF16 is the configuration-declared parameter dtype; individual checkpoint storage dtypes have not been observed. Another 1,886 checkpoint entries belong to the excluded vision/MTP scope.

## Exact parameter addresses and fused expert storage

The checkpoint stores each block's routed experts in two fused tensors: `gate_up_proj[expert, 2 × intermediate, hidden]` and `down_proj[expert, hidden, intermediate]`. The explorer exports three disjoint logical matrix views for each expert. It retains the original tensor name, full checkpoint shape and flat element offset in each view's `storage` metadata. These logical IDs are explicitly not serialized checkpoint tensor names.

The full fused tensors are not added again as parameter records. Sorted view intervals must cover each fused tensor exactly once without gaps or overlap. Every expert therefore has the correct 12,582,912 parameters and an inspectable final scalar address. Current browser scalar addresses are local to the logical view; no numerical value is bundled.

## Visual semantics

- **GatedDeltaNet state:** `[64, 128, 128]` per sequence, independent of sequence length. Sixteen normalized Q/K heads are repeated into 64 value-head groups. The recurrent state is runtime memory, not learned parameter storage and not a softmax attention grid.
- **Recurrence:** first decay the previous state by α, then write the β-scaled difference between the current value and the state prediction along the normalized key. Read using the normalized query scaled by `1 / sqrt(key_head_dim)`. The previous-state and updated-state entities are distinct time views, with a directed dependency; no same-token algebraic feedback loop is implied.
- **Convolution:** a depthwise causal kernel of width four plus SiLU precedes the Q/K normalization and recurrent update. The V branch is not L2-normalized.
- **Gated full attention:** 32 Q heads share two K/V groups. The Q projection's output width is doubled because it also produces an output gate. The gate is applied after softmax-weighted value mixing and before the output projection.
- **MoE:** 10 of 512 experts are selected by softmax/top-k routing with selected weights renormalized. The shared expert is always evaluated and multiplied by its separate sigmoid output gate. Structural expert cells do not assert which experts activate.
- **RMS normalization:** ordinary decoder/Q/K norms use a `1 + weight` scale parameterization. The recurrent output norm uses a direct learned scale followed by SiLU gating. Their learned vectors remain separately addressable.

## Reproduction and limits

Run `python3 -m exporters.qwen35` from the repository root. Generation uses only Python's standard library and the bundled pinned metadata. Run `python3 -m unittest discover -s tests -p 'test_qwen35.py'` for scoped inventory, fused-view partition, hybrid schedule and provenance checks.

The package has 125,858 entities and 93,078 logical parameter tensors. Most are metadata addresses; the renderer's visible-object budgets still apply. The gzip package is approximately 2.1 MB, expanding to approximately 54.2 MB of JSON. This is a complete inventory of the declared **text scope**, not a complete visualization of the entire multimodal checkpoint. Geometry illustrates operations and grouping; it does not encode learned numerical values or empirical execution time.

The official Qwen model repository and Transformers implementation are distributed under Apache-2.0. See `references/QWEN-APACHE-2.0.txt` for the upstream license text. Model Atlas's implementation is independent; no Manim scene code is involved.
