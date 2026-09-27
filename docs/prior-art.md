# Prior art and independent design

Reviewed on 7 September 2026. The aim was to learn a visual language for interactive inspection, not reproduce an authored video or adopt its scene architecture.

## 3Blue1Brown: source-level review

The reviewed `3b1b/videos` snapshot is `674b966fbb6cf0307590d27744d186165e8b6a76`. It contains the 2024 transformer material; it is not asserted to be the exact engine/scene revision used to render the original releases.

| Source and methods inspected | Useful principle | Independent implementation decision |
|---|---|---|
| [`network_flow.py`](https://github.com/3b1b/videos/blob/674b966fbb6cf0307590d27744d186165e8b6a76/_2024/transformers/network_flow.py): `HighLevelNetworkFlow`, `get_embedding_array`, `get_next_layer_array`, `progress_through_attention_block`, `progress_through_mlp_block`, `mention_repetitions` | Persistent token positions, translucent computational regions, attention communication versus position-wise feedforward, repeated-layer compression | Stable entity IDs, operation-family colors, visible stage/group boundaries, user-controlled navigation. Authored camera coordinates and randomized illustrative flow are not imported. |
| [`helpers.py`](https://github.com/3b1b/videos/blob/674b966fbb6cf0307590d27744d186165e8b6a76/_2024/transformers/helpers.py): `WeightMatrix`, `NumericEmbedding`, `EmbeddingArray`, `ContextAnimation`, `show_matrix_vector_product`, `matrix_row_vector_product`, `value_to_color` | Numeric vectors, sign-sensitive colors, omissions, row/vector alignment, accumulation of products | A tile inspector with exact coordinates and a computed contribution table. Helpers that default to random values or strengths are not used. Missing data is a separate state. |
| [`embedding.py`](https://github.com/3b1b/videos/blob/674b966fbb6cf0307590d27744d186165e8b6a76/_2024/transformers/embedding.py): `IntroduceEmbeddingMatrix`, `get_principle_components`, `Word2VecScene.get_basis`, tokenization helpers | Link a token to its table entry; display a comprehensible subset while preserving full dimensions; use projection to make vector relationships visible | Architecture position is not embedding position. A future embedding projection must name its model, tokenizer, sample, basis and method. The source's GloVe illustrations must not be presented as GPT embeddings. |
| [`attention.py`](https://github.com/3b1b/videos/blob/674b966fbb6cf0307590d27744d186165e8b6a76/_2024/transformers/attention.py): `QueryMap`, `KeyMap`, `DescribeAttentionEquation`, `ShowMasking`, `IntroduceValueMatrix`, `CountMatrixParameters`, `LowRankTransformation`, `MultiHeadedAttention`, `OutputMatrix` | Q/K/V roles; score grids; causal masking; low-rank transformations; parallel heads; dimension-based parameter accounting | Separate attention operations, explicit axes, storage identity independent of conceptual head slices, and original-operand arithmetic. No invented per-head output parameters. |

The [1 April 2024 transformer lesson](https://www.3blue1brown.com/lessons/gpt/) distinguishes fixed learned weights from input-dependent vectors and moves between full dimensions and small readable subsets. The [7 April attention lesson](https://www.3blue1brown.com/lessons/attention/) builds the attention equation in stages. It uses a grid orientation different from many software conventions, reinforcing the need to label query/key axes and the softmax direction explicitly.

The attention lesson's conceptual per-head value-up maps are ordinarily represented by the combined output matrix. Drawing those as independent stored tensors would double-count parameters. Model Atlas keeps parameter identity separate from head views. The workbench's rows are query positions; columns are key positions; row softmax runs across keys.

The [Manim interaction implementation](https://github.com/3b1b/manim/blob/master/manimlib/scene/interactive_scene.py) was also inspected, including `toggle_selection_mode`, `regenerate_selection_search_set`, and `refresh_selection_scope`. Parent/child selection scopes and recoverable camera state are useful concepts. The implementation architecture remains independent Three.js/WebGL2.

## Licensing boundary

The [videos repository](https://github.com/3b1b/videos/blob/master/README.md) specifies **CC BY-NC-SA 4.0**. The [Manim engine](https://github.com/3b1b/manim/blob/master/LICENSE.md) is **MIT**. A permissive engine license does not relicense the scene code or media made with it.

No scene implementation, scene helpers, video assets, narration, camera choreography or extracted screenshots are included or adapted. Only general mathematical ideas and explanatory principles informed independent implementation. Research copies remained outside the deliverable. Any future scene-code reuse requires its own explicit licensing review.

## Additional interactive prior art

| Project / primary source | Finding | Application here |
|---|---|---|
| [Brendan Bycroft, llm-viz](https://github.com/bbycroft/llm-viz), [license](https://github.com/bbycroft/llm-viz/blob/main/LICENSE) | Stable 3D tensor geography and scalar-level inspection; the current repository is MIT with stated personal/third-party exclusions. Its example architecture is GPT-oriented. | Keep inspectable coordinates and spatial orientation. Do not equate configurable model size with universal architecture support. No code copied. |
| [Transformer Explainer](https://github.com/poloclub/transformer-explainer), [paper](https://arxiv.org/html/2408.04619v2) | Interactive token-centered explanation and expandable operations with browser GPT-2 execution; MIT project | Coordinate overview and arithmetic detail. This project's normal viewer separates inspection from inference. |
| [BertViz](https://github.com/jessevig/bertviz), [paper](https://arxiv.org/abs/1904.02679) | Coordinated model/head/neuron views; deeper views require Q/K access and have narrower model support; Apache-2.0 | Selected-head detail must declare which trace fields exist. Attention is a mixing coefficient, not a causal explanation of the prediction. |
| [Netron](https://github.com/lutzroeder/netron) | Broad format adapters and graph/tensor metadata; MIT | Put source-format interpretation in adapters. Preserve unknown operations rather than drawing a convenient transformer template. |
| [TensorSpace](https://github.com/tensorspace-team/tensorspace) | Three.js/TensorFlow.js layer visualization with intermediate outputs; Apache-2.0 | Make intermediate tensor values first-class, while recognizing that a layer-rendering API does not solve arbitrary introspection. |

## Computation microscope: September 2026 extension

The paper [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) supplies the most useful change in emphasis: treat the residual stream as shared communication, and distinguish an attention head's **where-to-read** calculation from **what-to-write** transformation. The paper's analysis emphasizes simplified attention-only transformers; its circuit interpretations are not automatically established for this two-block model with MLPs.

The microscope makes scaled QK scores and row softmax separate inspectable planes, then shows value mixing, the output projection, additive head writes, and the unchanged residual bypass. Signed contributions can reinforce or cancel. Head ablations use complete precomputed downstream runs, so the display can show an intervention's effect without confusing attention strength with causal importance. A future direct source-to-residual view can expose the combined OV transformation; this release follows its two constituent operations instead.

Anthropic's [HeadVis](https://github.com/anthropics/headvis) also demonstrates a useful static-viewer/precomputed-data split. Its Apache-2.0 frontend and published pipeline outline were reviewed as architectural prior art; no implementation or assets were incorporated. Here the trainer and trace exporter are separate from the browser, and the bundled small fixture includes every learned weight and intermediate value.

The extension also strengthens the earlier 3Blue1Brown-inspired matrix language: cell color carries sign, selected products visibly accumulate, future attention scores have an explicit masked state, and semantic zoom exposes numbers in the same persistent tensor positions. These encodings and their deliberate compressions are audited in [microscope-visual-semantics.md](microscope-visual-semantics.md).

## Scale and dimension reduction

[Shneiderman's visual information-seeking paper](https://www.cs.umd.edu/~ben/papers/Shneiderman1996eyes.pdf) supports overview, zoom/filter and details on demand. [Pad++](https://www.cs.umd.edu/projects/hcil/pad%2B%2B/papers/jvlc-96-pad/jvlc-96-pad.pdf) supplies foundational zoomable-interface ideas. [OGC 3D Tiles](https://docs.ogc.org/cs/18-053r2/18-053r2.html) illustrates hierarchical refinement based on screen-space error; Model Atlas borrows the general refinement principle, not the geospatial format.

[Three.js LOD](https://threejs.org/docs/pages/LOD.html) exposes hysteresis, and [InstancedMesh](https://threejs.org/docs/pages/InstancedMesh.html) reduces repeated draw calls. The design combines these general techniques with semantic disclosure: the next level contains new inspectable children, not merely additional triangles.

The [Distill investigation of t-SNE](https://distill.pub/2016/misread-tsne/) demonstrates why cluster sizes and distances in projected space can mislead. A future PCA/UMAP/t-SNE panel must disclose method, input representation, preprocessing, seed, sample selection and, for PCA, explained variance. This release does not generate embedding scatter plots; it avoids implying that an architecture's 3D world coordinates are learned semantic coordinates.

## Hero selection

GPT-2 was retained for independently checkable accounting. Qwen3-235B-A22B-Instruct-2507 was considered and its GQA/MoE structure reviewed, but its 2025 release did not meet the 2026 release scope selected for the showcase. Qwen3.5, Kimi K3 and DeepSeek V4 were then checked against public primary metadata.

[DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) satisfies both the 2026 and trillion-scale criteria, with a public implementation and MIT license. This is a useful inspection target, not a claim of current-best performance or frontier parity. Its compressed attention and hyper-connections make architectural independence testable: relabeling a conventional GPT diagram would be wrong. The exact pinned sources and accounting are recorded separately.
