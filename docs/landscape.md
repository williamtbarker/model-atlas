# Continuous architecture landscape

The landscape opens at `/`. It unfolds all 61 DeepSeek-V4-Pro core decoder blocks
into a persistent serpentine arrangement. The auxiliary prediction block occupies
a separate island. Zooming replaces summary glyphs with operations at fixed
coordinates; it does not replace the model with a new row of containers.

The model selector also includes GPT-2, Llama 4 Maverick and Qwen3.5-397B-A17B.
Maverick includes its decoder and vision parameters, with the vision branch shown
as a collapsed upstream landmark. Qwen's package covers the text decoder only;
vision and multi-token prediction are excluded. Each package declares its scope,
evidence and omissions in the component lens. See the pinned provenance for
[DeepSeek](deepseek4-provenance.md), [Maverick](llama4-maverick-provenance.md)
and [Qwen](qwen35-provenance.md). No model weights or inference service are needed.

## Reading the geometry

| Geometry | Mathematical or structural meaning | Compression and exact inspection |
| --- | --- | --- |
| Parallel residual ribbons | Declared residual-stream count | One ribbon per stream; line width does not encode values or bandwidth. |
| Residual crossbar | Declared mHC stream mixing | Permitted mixing, not measured coefficients. Ordinary addition uses a plus junction instead. |
| Attention filaments | Query heads assigned to their declared KV groups | At most 256 sampled query marks; each connects to its actual KV group. Counts and group membership come from ModelIR. Filaments are schematic, not per-head execution traces. |
| State plane and return arc | A recurrent state with an explicit previous-time input | Plane dimensions describe state shape; the arc denotes temporal recurrence, not an algebraic graph cycle. No state values or update timing are invented. |
| Short parallel convolution strips | Declared causal convolution kernel | At most 12 kernel positions; the exact size and formula remain inspectable. Length does not imply sequence length or an activation trace. |
| Multiplication junction | A declared gate applied to its connected operand | Symbolic multiplication, without an invented gate value. Shared-expert gates and attention output gates remain separate operations. |
| Paired rotation arcs | Declared rotary position transformation | Symbolic paired-coordinate rotation; exact dimensionality and position rules remain in the lens. No angle or phase is measured. |
| Vision patch lattice | An upstream image encoder | Compressed landmark, not every patch or vision block. Internal entities and parameter addresses remain searchable; the declared fusion edge locates entry into the text stream. |
| Matrix sheets | Learned tensor storage and logical shape | Guide grid capped at 16×16; it is not a fabricated weight heatmap. Tensor dimensions and any scalar address can be inspected. |
| Expert field | Available experts in a declared bank | One selectable mark per expert in the bundled banks, up to 512 here. Marks use instancing. No experts are claimed active. |
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
one moves the camera to the connected operator in the same model. **Copy view
link** preserves the model, selected entity and view mode; **Back** retraces recent
selections. `/` focuses component search. Switching models cancels the old request
and prevents stale data or search results from entering the new scene. The exact
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

The maintainer tried the original continuous landscape ZIP locally, liked the
result and authorized this expanded version for `main`. The new models and
navigation changes passed source review, TypeScript compilation, accounting,
graph, layout and numerical checks. Browser access remains blocked in the
development environment; the new glyphs have no fresh browser visual QA or
MacBook performance benchmark here.

For local review, switch through all four models, compare a Maverick dense block
with its neighboring expert block, and compare Qwen blocks 0 and 3 for recurrence
versus full attention. Follow the shared-expert gate, inspect a logical expert
tensor's source mapping, copy a component link, and switch models during loading.
Visual quality and graphics performance still require direct human judgment.
