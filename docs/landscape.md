# Continuous architecture landscape

The landscape opens at `/`. It unfolds all 61 DeepSeek-V4-Pro core decoder blocks
into a persistent serpentine arrangement. The auxiliary prediction block occupies
a separate island. Zooming replaces summary glyphs with operations at fixed
coordinates; it does not replace the model with a new row of containers.

The bundled structure is derived from the same pinned checkpoint metadata as
v0.1.0. See [provenance](deepseek4-provenance.md). No additional DeepSeek weights,
execution traces or inference service were introduced.

## Reading the geometry

| Geometry | Mathematical or structural meaning | Compression and exact inspection |
| --- | --- | --- |
| Parallel residual ribbons | Declared residual-stream count | One ribbon per stream; line width does not encode values or bandwidth. |
| Residual crossbar | Declared mHC stream mixing | Permitted mixing, not measured coefficients. Ordinary addition uses a plus junction instead. |
| Attention filaments | Declared query-head count | Shared base only when shared KV is declared. At most256 marks; full count stays in metadata. Filaments are schematic, not per-head execution traces. |
| Matrix sheets | Learned tensor storage and logical shape | Guide grid capped at 16×16; it is not a fabricated weight heatmap. Tensor dimensions and any scalar address can be inspected. |
| Expert field | Available experts in a declared bank | One selectable mark per expert in the bundled 384-member banks. Marks use instancing. No experts are claimed active. |
| Expanded expert sheets | The selected expert's actual parameter tensors | Shapes come from ModelIR. The three sheets are storage, not a sequential computation diagram. Formula and connections are inspected separately. |
| Router diamond | Declared routing operation and top-k ports | Ports are capped at 12. They show the routing rule, not an observed selection. Hash and score rules are available in the component lens. |
| Gathering memory lines | Sequence-position compression | At most 32 illustrative input ticks. Exact compression ratio is in the lens; feature width is not compressed by this glyph. |
| Indexed memory slots | Candidate-position selection by a sparse indexer | No slot is highlighted as selected without a trace. This is distinct from memory pooling. |
| Normalization comb | A normalization operation | Coordinate count is preserved; comb teeth are a symbolic guide. |
| Nonlinear curve | A nonlinear operator | Symbolic transfer-curve icon; not an activation distribution or a measurement. Exact declared formula is available in the lens. |
| Connection curves | Supplied dataflow/residual edges projected through LOD summaries | Every curve retains its source edge IDs. Containment never creates computation edges. No speed, timing or magnitude is encoded. |

World positions, lengths, colors and physical separation do not encode model
accuracy, execution latency, latent geometry, hardware wiring or unique causal
importance. Parameter counts are exact and deduplicated in the metadata index;
geometry is not a volume-proportional parameter-count chart.

Selecting a component exposes its declared input/output dependencies. Following
one moves the camera to the connected operator in the same model. The exact
recorded arithmetic for the teaching model is at `/microscope.html`. Its values
are not DeepSeek values.

## Implementation

- `layout.ts` compiles canonical ModelIR into persistent spatial anchors. It
  recognizes semantic kinds; it does not dispatch on a model name.
- `renderer.ts` creates independent Three.js glyphs, frustum culls and applies
  screen-space semantic detail with hysteresis. Distant operation summaries
  expand as the camera approaches.
- Expert marks and repeated strips are instanced. Connections share a single
  batched line geometry. Inactive glyph geometry is disposed and labels removed.
- Visible glyphs are capped at 650, DOM labels at 42, instanced marks at 48,000 and
  primitive draw calls at 1,800. Connection geometry and the small selection lens
  add a bounded overhead. Device pixel ratio is capped at 1.7. Rendering pauses
  when the camera and scene are idle; reduced-motion settings remove camera tweening.
- The full metadata index is loaded into CPU memory. Only visible glyphs become
  graphics objects. This version does not stream arbitrary architecture metadata
  or out-of-core tensor values.
- Parameter entities without their own glyph resolve to their nearest visible
  semantic ancestor. Exact tensor addresses remain available in the inspector;
  this version does not build billions of scalar meshes.

## Review status

TypeScript, model accounting, graph provenance and numerical fixture checks can
be validated automatically. Browser access was blocked by the development
environment's URL security policy, so this branch has **not** been visually
verified in a browser. No rendered screenshots, MacBook performance measurements
or completed interaction-QA claim accompanies this preview.

Review locally before merging: fit the whole model; approach blocks 0,2,3 and60;
compare pooling and sparse indexing; select experts 0 and 383; open their tensor
coordinates; inspect the separate auxiliary block; switch to GPT-2; verify the
microscope link. Judge the visual quality directly. Passing tests does not settle
that question.
