# Visual semantics audit

The following answers the seven-question audit for the new microscope. Every
value is from a recorded trained-model run or an explicitly recomputed scalar
identity using its operands. No random values decorate the scene.

| Encoding                       | Mathematical meaning                                                               | Compression / exaggeration                                                                                       | Exact inspection and limits                                                                                          |
| ------------------------------ | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Activation grid                | Rows are coordinates; columns are token positions                                  | Spatial gaps and depth are chosen for readability                                                                | Click or use row/column selectors. Grid position is not semantic embedding geometry.                                 |
| Weight plane                   | Matrix coefficients; effective operator views show Wᵀ for XW checkpoint convention | All small-model coefficients are available; numbers appear only nearby                                           | Full checkpoint orientation is available under Weights. Per-head slices reuse storage.                               |
| Positive teal / negative coral | Sign; intensity encodes magnitude divided by that baseline tensor's maximum        | Square-root intensity exaggerates small values; magnitude saturates at the baseline maximum                      | Inspector exposes exact float64 and baseline delta. Colors from different tensor scales are not directly comparable. |
| Cell depth                     | Absolute normalized magnitude                                                      | Depth is intentionally shallow and exaggerated                                                                   | Same caveat as intensity; it does not add a tensor dimension.                                                        |
| QK score grid                  | Scaled dot products, destination queries by source keys                            | Layout chosen for reading                                                                                        | Selected entry shows every coordinate product and division by √8.                                                    |
| Softmax grid                   | Probabilities over source positions, one row per query                             | None numerically; display decimals round                                                                         | Full denominator, exponentials and per-key normalized terms are available.                                           |
| Causal upper triangle          | Unavailable future routes                                                          | Dark cells; hatching at numeric detail                                                                           | Raw score null represents a −∞ mask; probability is zero. Playback is disabled for masked routes.                    |
| Residual bus                   | Unnormalized input bypasses the branch and adds at its output                      | Raised path makes the bypass recognizable                                                                        | Both attention and MLP additions have separate, exact coordinate sums.                                               |
| Contribution path              | A selected operand participates in the selected computation                        | Paths rise above the scene; moving-dot radius scales with magnitude for visibility                                          | Path direction is dependency direction. Signed values can be negative without reversing the dependency arrow.        |
| Moving contribution            | A term approaches the operation's selected output                                  | Time is explanatory order, not measured latency; sequential display does not imply sequential hardware execution | Scrubber and term list share arrival thresholds. Zero terms do not contribute to the accumulator.                    |
| Signed bars and accumulator    | Terms share a zero baseline; summation uses actual signed values                   | Height normalized to the largest term, partial sum, or result in this calculation                                | This scale is shared within the selected calculation. Norm/GELU/softmax stages are not represented as additive sums. |
| Head gate                      | Identity for baseline; zero after value mixing for the selected ablation           | Gate is a visible intervention boundary, not a trained model parameter                                           | Whole downstream recording changes. Baseline attention remains visible at a disabled head.                           |
| Changes-only colors            | Recorded intervention value minus baseline value                                   | Uses the same baseline magnitude scale, so large deltas can saturate                                             | Inspector always states actual and baseline values plus delta.                                                       |
| Camera zoom                    | Reveals a different level of numerical detail                                      | Far planes summarize; nearby planes gain cells and then numbers                                                  | Anchors and selected addresses persist. It is semantic refinement of the same tensor, not a newly invented entity.   |

## Seven checks applied to every encoding

1. The mathematical property is stated in the table and the relevant operation.
2. Its meaning is independent of a model name. Architecture interpretation is in
   the adapter; these recorded operations must actually exist in the source.
3. The small model has all coordinates present in the package. Only visual detail
   is compressed with distance. The large architecture page does not imply known
   values for unavailable weights.
4. Spatial separation, intensity, cell depth, moving-dot radius, and animation time are
   explicitly chosen for readability and are not additional model measurements.
5. Exact values are addressable through the inspector. Rounded numerical textures
   are not used as computational inputs.
6. Geometry does not imply that coordinates have named semantic meanings, that
   attention equals causal importance, or that heads are independent explanations.
7. Zoom resolves the same tensor into scalar cells and exact arithmetic while
   maintaining neighboring structure and camera continuity.

## Human review sequence

- Open at your normal MacBook resolution and check that selected cells and their
  operands are readable after Approach; test Face on as well as free orbit.
- Change block/head and directly select a cell in another block; confirm the
  Disable button names the context actually being inspected.
- Select masked and unmasked attention entries; verify that only allowed routes
  can animate and that each softmax row sums to one.
- Scrub a residual sum with opposing signs; compare the 3D accumulator, term list,
  and exact sum.
- Disable Block 1 / Head 1 on Example 1: prediction should change from 4 to 2.
- Check the later block while disabled and then restore; values must change
  without moving the tensors to different locations.
- Enter Weights, approach a matrix, and inspect its corner coordinates.

Browser visual QA and MacBook GPU performance have not been measured in the
development environment. The numerical and build checks are documented separately.
