# DeepSeek-V4-Pro shape provenance

Model revision: `b5968e9190ef611bbf34a7229255be88a0e937c1`. The live Hugging Face API identified this revision; pinned config and inference source downloads were byte-for-byte identical to the reviewed copies. Their SHA-256 digests are in `references/deepseek4-shapes.json`.

The independent derivation script reconstructs logical unsharded tensor shapes from the publisher's configuration and constructors. It does not import model code, load model weights, or run inference.

Read-only downloads consumed 12,178,016 bytes of response bodies (approximately 12.178 MB), excluding HTTP headers and redirect bodies. This includes the full 11,305,684-byte tensor-name index, source/config/API metadata, five shard-header length prefixes and JSON headers, and two repeated tiny source/config downloads used to verify the pinned revision. Tensor value payload downloaded: **0 bytes**.

Five shard headers: 1 (embedding), 2 (entire HCA block 0 with hash routing), 4 (entire CSA block 2 with hash routing), 63 (root output/norm/mHC head), and 64 (complete MTP block 0). Their JSON headers total 773,792 bytes. All 7,030 entries matched reconstructed checkpoint shapes and dtypes exactly. Other layer shapes were derived from verified constructors; their names were checked against the complete index.

All 145,116 index names reconcile with no missing or extra entries: 73,161 architectural parameter tensors, 71,952 quantization-scale tensors, and 3 nontrainable hash tables. Inferred packed tensor bytes equal the publisher's index total_size exactly: 864,704,792,696.

Architectural parameters including MTP: 1,598,837,347,742. Core model excluding MTP: 1,572,997,201,763. MTP contributes 25,840,145,979 additional parameters and shares root embedding/output-head storage. Quantization-scale elements: 49,150,268,416. Hash-table integer elements: 2,327,040. HF's 1,598,839,674,782 total includes the hash tables and correctly counts packed FP4 logical elements, but excludes scales.

These are metadata-derived architecture counts, not a claim that every declared parameter was optimized by ordinary gradient descent. Quantization scales and integer routing tables are excluded from architecture-parameter totals. Runtime caches, frequency tables, masks and compressor state are nonpersistent buffers.

Checkpoint details matter: FP4 weight arrays use I8 containers packing two logical values per byte; hash tables are I64 on disk although the reference runtime creates int32; wo_a is FP8 plus scale on disk although the reference converter dequantizes it to BF16.

Source files and full immutable URLs are recorded in `references/deepseek4-shapes.json`. The original model implementation is linked but not included. The compact checkpoint index and sampled headers are bundled under `references/` so the reconciliation can be repeated offline.
