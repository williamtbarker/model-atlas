# Review and MacBook testing

This document describes the structural architecture page at `/architecture.html`.
For the numerical microscope, see [its design](microscope-design.md) and
[visual semantics](microscope-visual-semantics.md).

This release is a working prototype for human inspection. Automated numerical, extraction, schema and build checks are included. **No MacBook GPU benchmark or browser interaction test was performed in the build environment.** Local graphics behavior, camera feel, Safari rendering and accessibility need human testing.

Start with `npm install` and `npm run dev`. The first model is DeepSeek-V4-Pro. The left index, central scene and right inspector should be usable immediately after its roughly 1.55 MB package loads. If WebGL2 cannot initialize, the component index and tensor inspector remain available.

Suggested review route:

1. Enter Blocks 0–7, then Block 2 (CSA), then the attention group. Confirm that compressor and sparse indexer are distinct and inspect their declared inputs. Compare Block 0 (HCA), which has no CSA indexer.
2. Open the block's expert group. Page through all 384 experts or search for `layers.60.ffn.experts.383`. Open `w2.weight`; its logical shape is 7,168×3,072, and storage is packed FP4. Inspect the final coordinate `[7167, 3071]`; its address exists and its value correctly remains unavailable.
3. Compare Block 0's hash router with Block 3's corrected-score router. Check that the dataflow does not claim the router's selection is a recorded event.
4. Return to whole model. Locate the separate MTP module and its shared root embedding/output references. The 61 core blocks must not become 62 ordinary blocks.
5. Switch to GPT-2. The total must be 124,439,808. The output projection references the embedding tensor rather than counting a second copy.
6. Open the arithmetic workbench. Step through a projection, attention row, ordinary residual sum and router. Read the persistent note separating this example from the selected model. Change token and output coordinate; confirm the results update.
7. Test ordinary navigation, parent navigation, search, page changes, import failure, repeated model switching, reduced motion, narrow windows and 200% text enlargement. Look for stable memory after repeated navigation and no stuck loading overlay.

## Implemented versus remaining

| Capability | Status |
|---|---|
| TypeScript + Three.js/WebGL2 viewer, Vite local workflow | Implemented; production build checked |
| Python canonical package generation | Implemented; bundled generation reproducible offline |
| GPT-2 and a relevant 2026 trillion-scale hero | Implemented |
| Exact logical parameter identities/counts and coordinates | Implemented for bundled packages |
| Verified complete checkpoint tensor-name reconciliation | Implemented for pinned hero |
| Semantic child disclosure and scope transitions | Implemented; camera feel needs device review |
| Bounded objects, labels, tiles and scalar cache | Implemented |
| Arithmetic playback | Implemented on a separate computed miniature |
| Arbitrary model runtime trace import/playback | Not implemented |
| All metadata loaded lazily from subtree shards | Not implemented; full metadata is indexed after decompression |
| Seamless global coordinates across every semantic level | Not implemented; deterministic layouts within scopes |
| Trained hero scalar values without downloading weights | Unavailable; requires a future explicit value provider |
| Dequantized FP4/FP8 scalar reading | Not implemented |
| Detailed training/optimizer/KV-cache state visualization | Not implemented |
| Embedding PCA/UMAP scatter plots | Design reviewed; not implemented |
| Full operator-level tracing of all model forward paths | Not claimed; current hero graph is a semantic decomposition |

## Highest-value next work

1. Record a small real model run into a typed trace package, with actual input IDs, Q/K/V slices, masks, attention coefficients, projected outputs and residual operands. Link these to the same canonical operation IDs.
2. Let selecting an output coordinate highlight its exact input-coordinate contributions in the 3D scene. Preserve omitted contribution totals when displaying a bounded subset.
3. Add operation-specific playback for the hero's compressed attention, sqrt-softplus routing, hash lookup, and mHC. Test against recorded outputs before displaying them as model behavior.
4. Profile an actual MacBook before introducing workers, a more compact binary format, or Rust/WASM. Move metadata indexing into a worker and shard subtrees if measured load time or memory warrants it.
5. Add visual regression and browser interaction tests once the intended camera behavior and laptop browser baseline have been approved through use.

For GitHub, use a fresh repository and commit this source tree after local review. Keep weights and local trace data outside Git. The supplied `.gitignore` excludes common checkpoint formats. Add screenshots or a short recording from the actual working viewer; generated mockups would overstate the implementation.
