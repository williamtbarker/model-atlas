# Computation microscope, public v0.1.0

This release concentrates on a small, completely recorded computation before
attempting a new visual language for a frontier model. Its meaningful capability
is the connection between a selected scalar, its operands in the same 3D space,
and the consequences of a measured intervention.

## Source of truth

The NumPy trainer exports row-vector weights (`X @ W`) and binary64 recordings.
The adapter transposes activation arrays for display: rows are feature
coordinates, columns are token positions. Effective matrix planes display `Wᵀ`
so an output coordinate selects one visual row. The parameter inventory preserves
checkpoint orientation. The UI labels these conventions.

ModelIR contains a model root, block entities, numerical tensor views, explicit
dataflow and residual edges, and a canonical inventory of stored parameters.
Computational slices have role `unknown` in parameter accounting; only full
stored tensors have role `parameter`. This prevents head views from multiplying
the parameter count.

`SceneIR` adds layout anchors and operation references to this hierarchy. The
renderer accepts these data; it contains no model-name branch. The current layout
and chapter navigation target pre-norm transformer computations, not arbitrary
new architecture families. The older architecture page remains available for
structural introspection of GPT-2 and the large bundled architecture.

## Actual computation and interventions

The baseline and four single-head interventions are complete forward recordings.
Weights are unchanged. A head is disabled after `A @ V`, before its output matrix
slice. The original attention grid and pre-ablation mixture remain available;
the explicit head gate shows the zeroed effective output. Subsequent attention,
normalization, MLPs, logits and probabilities come from the intervention run.

The browser performs selected-coordinate arithmetic from supplied operands. It
does not execute the full model or support arbitrary new prompts. The five
examples are intentionally selected demonstrations. Forward activity, ablation
effect, and a unique causal explanation are not treated as equivalent.

## Persistent rendering

The world uses computation depth along X and separates heads along Y. Coordinate
arrays lie in XY planes. Weights sit behind their operations on Z. The scene is
a schematic spatial layout; no axis claims to represent learned meaning.

Changing a recording updates existing meshes and keeps the anchors. Interventions
preserve the camera; changing examples returns to the selected stage.
Stage navigation animates the camera through the same world. Per-plane projected
cell size determines detail: summary plate → scalar cells → rounded numerical
text. Entry/exit thresholds differ to reduce flicker. Frustum checks precede
materialization. Instance buffers, numerical textures and generated focus
geometry are disposed when replaced.

Budgets: 18,000 cell instances, 10 numerical texture planes, 36 labels, at most
64 selected contribution paths; pixel ratio capped at 1.7. The actual current
arithmetic has at most 32 contributing terms. Background topology is fixed and
small. Frames are demand-driven except during camera motion or explicit playback,
with a roughly 40 FPS cap. Reduced-motion settings shorten camera flights;
the arithmetic scrubber remains available.

## What comes next

1. Human MacBook review of readability, picking, camera behavior, and arithmetic
   comprehension. Revise the visual vocabulary from observed failures.
2. Add direct per-source residual-space writes `A[i,j] * V[j] * WO_head` as an
   alternative algebraic view linked to the current value-mix/output sequence.
3. Add a small recorded pretrained GPT-2 computation using the same schema, with
   exact adapter validation and provenance. The viewer still needs no full weights.
4. Export independently addressable trace chunks and camera-local hierarchy
   refinement before loading large activations or many model blocks.
5. Add measured circuits and precomputed interventions only where evidence exists.

No claim of novelty is made for attention attribution, 3D tensor geography, or
semantic zoom individually. The design aim is their integration into one
continuous, inspectable computation.
