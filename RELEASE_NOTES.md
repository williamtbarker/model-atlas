# Model Atlas v0.2.0

GitHub currently supplies source archives without a prebuilt viewer asset. Follow
the [README build instructions](README.md#serve-a-local-production-build).
Historical references below to a ready-built ZIP describe the prepared bundle;
they do not mean a binary asset is attached to the public release.

The landing page is now a continuous architecture landscape. Components retain
spatial context while zoom resolves attention, residual paths, routing, expert
banks and parameter matrices. The numerical microscope remains available for
exact recorded calculations in Recall-2.

## Four bundled models

- **DeepSeek-V4-Pro:** 61 core blocks plus its separate auxiliary prediction block.
- **Llama 4 Maverick:** all 48 decoder blocks and the complete learned vision,
  adapter and projector inventory. The vision branch is summarized spatially;
  its constituent blocks and tensors remain addressable.
- **Qwen3.5-397B-A17B:** 60-block **text decoder only**, including embeddings,
  final norm and vocabulary head; vision and auxiliary prediction are excluded.
- **GPT-2:** the complete reference architecture with tied-storage accounting.

All large-model packages are compact structural metadata. Llama and Qwen shapes
are constructor-derived, with explicit provenance and logical expert-to-storage
mappings. No large-model weights or runtime activations are included.

## Interaction and semantics

- Query-to-KV grouping is visible in attention fans.
- Recurrent state, causal convolution, gates, rotary operations and the vision
  branch have distinct geometric representations.
- Model/component view links, Back navigation and a search shortcut improve
  exploration. Links preserve view mode but not exact camera orbit.
- Obsolete model requests are cancelled; stale successes and errors cannot
  replace the current view. Search is cleared safely during model changes.
- Source clones and source archives use `npm ci` followed by `npm run dev`, or
  `npm run build` and `node serve.mjs`. Python extraction remains a separate,
  optional workflow.

## Validation

The previous landscape received positive local review from the project owner.
The new models and navigation changes received automated and source validation,
including parameter reconciliation, expert storage partitions, grouped-query
mapping, recurrent dependencies, semantic layout and navigation state. No fresh
browser visual QA or MacBook graphics benchmark was completed in this environment.

The original `v0.1.0` tag is preserved. Its historical release notes follow.

---

# Model Atlas v0.1.0

First public experimental release of a browser-based transformer visualization
tool. This checkpoint preserves the working computation microscope and the
structural model explorer as the basis for further visual development.

## Included

- A trained 4,777-parameter transformer with full numerical recordings.
- Inspectable Q/K/V projections, attention scores, softmax, value mixing, MLPs,
  normalization, and residual additions.
- Signed contribution animation and recorded single-head interventions.
- GPT-2 and DeepSeek-V4-Pro architecture packages with parameter accounting.
- Python extraction/training workflows, TypeScript and Three.js renderer,
  numerical checks, design notes, and prior-art acknowledgements.
- A ready-built release ZIP served with Node's standard library.

The normal viewer requires no CUDA, Python inference, or full-model download.
Source clones use `npm ci` and `npm run dev`; the release ZIP also supports
`node serve.mjs` without installing dependencies.

## Limits at v0.1.0

The numerical microscope uses the small Recall-2 model. DeepSeek is represented
structurally, without its trained scalar values or full execution traces.
The two views are not yet a unified, continuous landscape. Visual design,
MacBook performance, and browser interaction need further human testing.

The production build, 13 JavaScript/TypeScript tests, and four NumPy trace tests
passed during preparation of the computation checkpoint. No browser performance
benchmark or completed visual QA is claimed.

Original code and the small learned fixture are MIT licensed. Third-party
notices and model-metadata provenance are included.
