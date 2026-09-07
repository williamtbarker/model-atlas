# Visual semantics audit

This document describes the structural architecture page at `/architecture.html`.
For the numerical microscope, see [its design](microscope-design.md) and
[visual semantics](microscope-visual-semantics.md).

All geometry is schematic. World-space dimensions organize the diagram; they are not tensor axes, execution time, importance or physical circuitry. Exact shapes and counts live in the inspector. Colors identify mathematical/structural roles, with a separate signed scale for actual numeric tiles.

## Seven questions, applied before accepting an encoding

1. **Property:** name the specific encoded property—identity, hierarchy, operation family, tensor coordinate, explicit dependency, sign or value.
2. **Consistency:** use the same interpretation across packages. Only adapters may decide whether an entity exists.
3. **Compression:** state omitted ranges, visible/total counts and any sampling or aggregation. Never replace missing values with plausible noise.
4. **Exaggeration:** declare minimum thickness, equal-sized containers, diagram spacing and explanatory animation timing.
5. **Inspection:** expose identity, source/evidence, dimensions, role and count, plus exact values when a value source exists.
6. **False inference:** distinguish containment, dataflow, parameter sharing and conceptual views. Label axes. Do not make rank seven look like rank three.
7. **Semantic refinement:** zoom must reveal children or coordinates with stable identities. Enlarging the same solid object does not suffice.

## Primitive vocabulary

Every registry entry below accepts ModelIR entities. Related primitives intentionally reuse geometry while retaining distinct semantics and inspectable metadata. This is a reusable vocabulary, not a claim that every primitive has a bespoke animated renderer in v0.1.

| Primitive | Property encoded | Compressed / exaggerated | Exact inspection and refinement | Misinterpretation prevented |
|---|---|---|---|---|
| TensorVolume | Indexed rank-N tensor | Schematic cuboid; not literal rank three | Full shape, leading-axis coordinates, final-axis tile | Geometry is not learned semantic space |
| MatrixPlane | Rank-two storage | Uniform footprint, visible minimum thickness | Both dimensions, coordinate tile, scalar | Box area does not imply parameter count |
| VectorColumn | Rank-one storage | Uniform column height | Length and scalar coordinates | Column height is not vector magnitude |
| ScalarCell | One tensor coordinate | Minimum visible cell area | Exact coordinate/value or unavailable | Missing is not zero |
| TensorTile | Bounded original-coordinate slice | Maximum 12×12, no averaging | Origin and each coordinate | A tile is not the entire matrix |
| TokenStream | Token-indexed activation or input | No real token values in structural packages | Declared shape and provenance | A schematic stream is not a runtime trace |
| DataflowTube | Explicit dependency | Line thickness and length are fixed for readability | Endpoint IDs, relation type and label | Speed/distance are not latency |
| ResidualBus | Declared addition/weighted merge | Raised amber path | Formula and operands' operation IDs | Color mixing does not compute addition |
| HyperConnectionMix | Multi-stream learned mixing | Four streams summarized by one operation group | Stream count, pre/post/residual mixing attributes | mHC is not an ordinary skip-add |
| AttentionGrid | Query-key coefficient indexing | Structural grid has no values; workbench is 4×4 | Query rows, key columns, mask and row softmax | Attention is not causal importance |
| AttentionHead | Head/attention operation identity | Heads summarized at coarse levels | Head ID, width, KV sharing/view metadata | Head views do not create new stored weights |
| ProjectionBlock | Declared transformation | Uniform box dimensions | Formula, input/output widths, stored matrix orientation | Conceptual projection differs from stored tensor shape |
| NormalizationPlane | Normalization operation | Thin plane exaggerated for visibility | Norm type, dimensions and formula where documented | No implication that values were observed |
| ActivationLayer | Nonlinear operation or feedforward group | Uniform geometry | Operation formula; expert tensor children | A visible expert need not be active |
| ParameterVolume | Parameter container | No proportional-volume encoding | Exact deduplicated count | Volume is not a logarithm or count |
| RepeatedModuleArray | Declared instances | Bounded children, explicit counts/page ranges | Named layer/head IDs | Repetition does not imply parameter sharing |
| ExpertCluster | Available experts | 128 visible per page out of the declared total | All expert IDs via pages/search | All displayed experts do not execute for each token |
| RouterJunction | Selection/weighting operation | No synthetic routes in model view | Top-k, normalization and rule attributes | Different router rules are not interchangeable |
| HashRouter | Token-ID routing table | Table contents unavailable in architecture package | Table shape and routing role | Hash lookup is not a score top-k |
| ScoreRouter | Corrected-score selection | Scores unavailable without trace | Transform, correction and output-weight rule | Selection scores can differ from mixture weights |
| SharedExpert | Expert outside routed selection | Same readable expert geometry | Distinct identity and parameters | Shared expert is not part of selected top-k count |
| Compressor | Sequence compression | Compressed context summarized structurally | Compression ratio, tensor children | No invented dense attention matrix |
| SparseIndexer | Compressed-position selection | Top-k positions unavailable without trace | Indexer dimensions, own compression and dependencies | Indexer keys differ from attention's KV payload |
| EmbeddingTable | Token-to-vector lookup | Table not materialized at full vocabulary size | Vocabulary, width and tensor coordinates | Rows' 3D arrangement is not semantic proximity |
| VocabularyPlane | Output vocabulary projection | Vocabulary compressed | Shape and tied/untied storage identity | Tied weights counted once |

## Numeric colors and motion

Numeric tile colors encode **sign and absolute magnitude relative to the visible tile**: teal positive, amber negative, faint neutral zero, dots for unavailable. They do not encode a globally comparable cross-tensor scale. Integer values retain exact decimal inspection; colors are omitted for 64-bit integer strings to avoid precision loss.

The workbench attention heatmap encodes row-softmax coefficients. Future positions use a hatched masked state. The computed example stores causal zeros numerically; only the workbench has the mask context needed to distinguish them visually. General tensor tiles make no mask claim.

Projection and residual playback reveal actual arithmetic terms and a running sum. The routing example's bar length encodes the actual probability, and gold identifies selected experts. Playback is an explanatory schedule, not a hardware simulation. It has pause/step/scrub controls and never starts automatically.

## Known visual limitations

Only selected-component child marks are refined by projected size in this prototype. The scene then moves to a new scope. A fully continuous, globally anchored multi-level landscape remains future work. The same is true of linked 3D scalar-contribution animation and recorded-model trace playback. These limits are exposed in the review guide rather than hidden behind decorative token pulses.
