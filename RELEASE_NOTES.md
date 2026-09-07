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

## Current limits

The numerical microscope uses the small Recall-2 model. DeepSeek is represented
structurally, without its trained scalar values or full execution traces.
The two views are not yet a unified, continuous landscape. Visual design,
MacBook performance, and browser interaction need further human testing.

The production build, 13 JavaScript/TypeScript tests, and four NumPy trace tests
passed during preparation of the computation checkpoint. No browser performance
benchmark or completed visual QA is claimed.

Original code and the small learned fixture are MIT licensed. Third-party
notices and model-metadata provenance are included.
