# Model and scene architecture

This document describes the structural architecture page at `/architecture.html`.
For the numerical microscope, see [its design](microscope-design.md) and
[visual semantics](microscope-visual-semantics.md).

## Evidence boundary

A config establishes declarations. A checkpoint header establishes storage names, shapes, types, and byte ranges. Module traversal establishes registered containment. An exported graph establishes a captured computational graph. A runtime trace establishes values and paths for a specific execution. These are complementary sources, not interchangeable degrees of certainty.

Each entity, tensor and relation has an evidence tag: `declared`, `derived`, `observed`, `illustrative`, or `unavailable`. The hero's complete name reconciliation does not make every inferred operator-level connection an observed runtime event. Its operation graph is a documented semantic decomposition; it is not a complete kernel trace.

## Canonical ModelIR 1.0

The types are in `src/types.ts`; runtime validation is in `src/lib/model-ir.js`, and the producer is in `exporters/model_ir.py`.

```json
{
  "version": "1.0",
  "name": "Example",
  "rootId": "model",
  "source": {"kind": "module traversal", "evidence": "observed"},
  "entities": [
    {"id": "model", "parentId": null, "kind": "Model", "label": "Example", "evidence": "observed"},
    {"id": "tensor:w", "parentId": "model", "kind": "MatrixPlane", "label": "weight", "tensorId": "w", "evidence": "observed"}
  ],
  "tensors": [
    {"id": "w", "shape": [4, 8], "dtype": "F32", "role": "parameter", "evidence": "observed"}
  ],
  "edges": []
}
```

Entities use stable path IDs and exactly one containment parent. Tensor storage identities are separate from entities. `tensorId` exposes one tensor; `tensorRefs` links multiple use sites to existing tensor identities. Unknown operation kinds fall back to a generic module. Tensor shape may contain nonnegative integer dimensions or symbolic axis names. Rank zero is a scalar and zero-length axes describe empty tensors.

Tensor roles are `parameter`, `buffer`, `activation`, and `unknown`. A header-only catalog uses `unknown` because names do not prove learned-parameter status; the corresponding parameter count stays unknown. The hero's logical FP4 tensor shape is distinct from its packed I8 storage shape. Scale metadata is associated with its parent tensor and excluded from learned-parameter totals.

Relations:

| Kind | Meaning | Current display |
|---|---|---|
| `dataflow` | Explicit declared computational dependency | Solid direction-marked line |
| `residual` | Explicit residual or hyper-connection dependency | Raised amber direction-marked line |
| `parameter_share` | Shared parameter identity | Dashed line without direction |
| `view` | Alias/slice/conceptual view | Metadata only; not drawn as dataflow |

No graph relation is generated from containment. A scope only shows relations whose endpoints are both in that visible scope. Cross-scope relations remain available in the inspector; current rendering does not synthesize boundary ports or bundle them into invented junctions.

## Package and memory model

Packages are deterministic gzip-compressed ModelIR JSON. The hero compresses from about 46.8 MB of metadata to 1.55 MB. Gzip encodes repeated text efficiently; there is no claim that this is a fully lazy metadata database. On load, the browser decompresses and indexes the complete metadata. The renderer then instantiates only the current scope and bounded detail. Metadata memory and GPU-scene complexity are deliberately separate metrics.

Import limits: 64 MB input file, 128 MB decompressed package, 200,000 entities, 100,000 tensor records, and 400,000 relations. The limits bound the current prototype and can reject larger catalogs even when their actual model weights would be representable by a future sharded package.

Next iteration: a small manifest plus compressed subtree chunks; subtree parameter summaries in the manifest; a worker for validation, indexing and decompression; stable IDs resolved through a directory; optional content-addressed tensor tiles fetched from an explicit provider. Fetch cancellation, cache budgets and version agreement must be defined before a remote value provider is added. No remote provider is implemented in v0.1.

## Semantic zoom contract

Whole model → stages → layer groups → blocks → operation groups → experts/heads/projections → tensor → tile → scalar.

The hero explicitly includes layer groups; GPT-2's twelve blocks fit without that extra tier. The hierarchy can differ between models. A primitive never creates an attention head or expert because a model name suggests one should exist.

Within a scope, graph dependencies establish layout ranks; deterministic row/column layout handles edge-free scopes and cyclic graphs. Anchors do not rearrange when selecting a component. At 160 projected pixels, a selected entity shows real immediate child marks; these retract below 125 pixels. At close range, navigation enters that selected entity. Zooming sufficiently out returns to the parent. Explicit buttons provide an accessible equivalent. Scope transitions deliberately fit a new local layout; seamless global-coordinate refinement is future work.

Rendering uses instancing by primitive kind and standard Three.js batch frustum culling. Labels have separate frustum, projected-size, overlap and count gates. The renderer does not perform per-instance occlusion culling. It runs on demand, caps updates near 30 Hz, pauses offscreen, caps device pixel ratio at 1.6, and disposes instance buffers when scopes change.

## Numeric inspection

An exact scalar address is `(tensorId, indices[])`. Flat offsets are calculated using BigInt, avoiding the JavaScript number precision limit for trillion-scale indexing. Only the final two axes map onto the tile; leading axes remain explicit fixed coordinates. A 12×12 tile contains up to 144 original coordinates, with no averaging or invented fill values.

Values can come from small inline tensors or an optional local safetensors file. The local reader implements F64/F32/F16/BF16, integer and Boolean types; 64-bit integers remain decimal strings. Quantized logical FP4/FP8 decoding is not implemented. Packed values from a raw header catalog are storage entries, not dequantized learned coefficients. The normal hero package supplies no values.

The scalar cache holds at most 2,048 entries and is tied to a file-session generation. An old asynchronous file read cannot overwrite the cache for a newer file. Source switches invalidate visible tile requests.

## Arithmetic and trace roadmap

The workbench computes a tiny attention example from explicit operands. Its matrix-vector, row-softmax, weighted-value and residual calculations are independently checked against the Python-exported fixture. Animation reveals completed terms and an accumulated numerical sum. Its timing represents explanatory steps, not hardware latency.

A future trace package should contain model/package revision, execution ID, input/tokenizer identity, operation ID, operand tensor slices, output slices, head/layer/token coordinates, mask, route IDs/weights and optional timings. The viewer must check operation and shape compatibility before enabling trace playback. Dense attention, compressed attention, hash routing and mHC need distinct mathematical operators; a generic softmax animation must never silently stand in for all of them.
