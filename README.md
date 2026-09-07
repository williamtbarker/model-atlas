# Model Atlas

Explore a transformer as one continuous landscape: residual ribbons, attention
heads, compressed memory, routers, expert fields and parameter matrices. Approach
an operation, follow its declared connections, and inspect exact tensor dimensions
without downloading the model's weights.

**0.2.0-alpha.1 — continuous landscape review build.** The landing page now opens
DeepSeek-V4-Pro's 61 core blocks together. An auxiliary prediction block remains
separate. GPT-2 is included as a reference model. Camera zoom reveals operations
inside the same spatial layout.

This is a structural map. It does not contain DeepSeek's trained parameter values
or runtime activations. The separate [numerical microscope](microscope.html)
contains exact recorded arithmetic from the small trained Recall-2 model.

**Visual review is still required.** Browser access was blocked in the development
environment. The build and mathematical checks are verified separately; no claim
of completed browser visual QA or measured MacBook graphics performance is made.
See [review notes](LANDSCAPE_REVIEW.md) and [visual semantics](docs/landscape.md).

## Run from GitHub

With Node.js 22.12 or newer:

```bash
git clone --branch feat/continuous-landscape https://github.com/williamtbarker/model-atlas.git
cd model-atlas
npm ci
npm run dev
```

Open the address Vite prints, normally http://127.0.0.1:5173. The landscape opens
DeepSeek immediately. Use the model selector for GPT-2. The `main` branch and
`v0.1.0` tag retain the prior public release until this preview is reviewed.

`dist/` is generated and is not committed to Git. To use `node serve.mjs` from a
clone, first run `npm run build`. The downloadable release ZIP already includes
the build.

## Open the supplied ZIP

Install [Node.js](https://nodejs.org/) 22.12+ or 24, unzip the bundle, and open a
terminal in its `model-atlas` folder.

The ZIP includes a built viewer, so the quickest route needs **no npm install**:

```bash
node serve.mjs
```

Open **http://127.0.0.1:4173** in a current browser. Keep the terminal open; Ctrl+C
stops the server. If the port is occupied, use `ATLAS_PORT=4174 node serve.mjs`.
Double-clicking `dist/index.html` will not work because browsers restrict local
module and data requests.

To edit the code:

```bash
npm install
npm run dev
```

Open the local address Vite prints, normally http://127.0.0.1:5173. Changes to
source reload automatically. `npm run build` refreshes the supplied static build.

The browser requires no Python, CUDA, PyTorch, LaTeX, Manim, inference endpoint,
account, or large model download. Its only runtime graphics dependency is Three.js.
The ZIP omits `node_modules`. Compressed architecture metadata, locally bundled
fonts and application assets are sufficient for the landscape; the numerical
recordings load only when you open the microscope.

## Explore the landscape

1. **Whole model** fits the complete decoder. The bottom strip addresses every
   core block; the component lens also links to the auxiliary block.
2. **Block** approaches the selected block. **Attention** and **Experts** move
   closer; scrolling preserves the same spatial anchors.
3. Click an operation to see dimensions, formulas and declared connections.
   Follow an input or output to inspect the connected operator.
4. Click any expert mark, or choose an expert in the component lens. Approach it
   to reveal its own parameter sheets, then enter a valid scalar coordinate.
   The address is exact; absent trained values are explicitly unavailable.
5. Use **Numerical microscope** for recorded calculations, signed contributions
   and head ablations in Recall-2. The earlier metadata catalog is linked under
   **Reading this map**.

## Numerical microscope

Open `/microscope.html` for these recorded calculations:

1. The initial prompt is `C 4 A 2 B 1 ? C`. The learned task is to return `4`.
2. Choose **Project**, click a Q coordinate, and press **Approach**. Its input and
   weight cells are framed together. Press **Follow contributions**, or scrub
   manually. Products arrive in order and a signed accumulator grows or cancels.
3. Choose **Attend**. Inspect the **QK scores / √d** plane, then the separate
   **Softmax** plane. Rows are queries; columns are keys. Future positions are masked.
4. Choose **Write**. Inspect the actual weighted values, output projection, and
   writes into the residual. A large attention weight does not guarantee a large write.
5. Choose **Add** and pick a coordinate. The original residual bypasses the
   normalization branch and adds to the attention update. Positive and negative
   terms use the same zero baseline in the selected calculation.
6. With Block 1 / Head 1 selected, press **Disable B1/H1**. The first example's
   prediction changes from `4` to `2`. **Changes only** colors differences from
   the baseline. All later operations use the complete recorded intervention run.
7. Choose **Weights** to navigate all **4,777** learned scalar parameters. Head
   slices in the computation refer to this same storage and are not extra parameters.

Drag to orbit, scroll to zoom, click a cell to select, double-click to approach.
Use the operation, row, and column selectors for keyboard access to the same values.
**Face on** provides a planar camera. Tensor positions persist across examples and interventions. The camera stays
in place during interventions; changing examples returns to the selected stage. The computation timeline explains arithmetic order;
it does not reproduce hardware timing.

## What model is this?

**Recall-2** is an independently trained, decoder-only transformer with two
pre-normalized blocks, two heads per block, residual width 16, head width 8, and
MLP width 32. It learns a finite symbolic key/value lookup task.

It is a teaching model, **not GPT-2 or a pretrained language model**. It achieved
100% accuracy on 924 held-out queries in the supplied training run. Dictionaries
were split before query variants were generated. The five displayed examples
were deliberately selected for large ablation effects; they are demonstrations,
not a representative performance sample. No induction-head or semantic-feature
interpretation is asserted.

The 1.68 MB compressed package contains all learned weights, intermediate tensors,
and five full runs for each of five examples: baseline plus each individual head
disabled. Ablations zero a head after value mixing and rerun everything downstream.
The explicit head gate distinguishes the pre-ablation mixture from the effective
head output. Effects are intervention results, not unique causal attributions.

See [training and provenance](examples/recall/README.md) for metrics, limitations,
and the optional NumPy-only reproduction workflow.

## Code and data

| Location                       | Responsibility                                                                                           |
| ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `src/landscape/` | Continuous architecture layout, semantic glyphs, camera and connection lens |
| `src/microscope/adapter.ts`    | Recorded transformer trace → canonical ModelIR and numerical scene views                                 |
| `src/microscope/renderer.ts`   | Persistent Three.js scene, numerical planes, coordinate picking, semantic detail, contribution transport |
| `src/microscope/arithmetic.ts` | Exact selected-coordinate decomposition                                                                  |
| `src/microscope/app.ts`        | Example, stage, head, token, and intervention interaction                                                |
| `public/traces/recall.json.gz` | Complete numerical evidence for the teaching example                                                     |
| `examples/recall/`             | Independent trainer, small checkpoint, and numerical verification                                        |
| `src/lib/` / `exporters/`      | Earlier generic architecture explorer and Python inspection adapters                                     |

Rendering has no model-name conditionals. The microscope's adapter currently
supports this recorded pre-norm transformer schema. Adapting another architecture
requires exporting its actual operations and traces; changing a model label does
not make it supported.

The small scene uses persistent tensor anchors. Distant planes collapse to
summaries, nearer planes instantiate cells, and close planes gain numerical text.
GPU cell and label budgets are explicit. This is a visual prototype, not yet a
streaming renderer for arbitrarily large recorded computations.

## Verification

```bash
npm test
npm run build
```

The new numerical tests reconcile more than 100,000 displayed scalar calculations
across the recordings, verify unique parameter accounting, and confirm that
ablations change downstream attention while preserving scene positions.

To run the optional Python checks:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/recall/requirements.txt
npm run test:trace
```

TypeScript compilation and numerical checks were run in the development
environment. **MacBook graphics performance and browser interaction still require
human review.** This release does not claim completed browser visual QA.

## Design, prior art, and license

- [Continuous landscape semantics and limits](docs/landscape.md)
- [Landscape review build](LANDSCAPE_REVIEW.md)
- [Development roadmap](ROADMAP.md)
- [v0.1.0 release notes](RELEASE_NOTES.md)
- [Visual semantics and review checklist](docs/microscope-visual-semantics.md)
- [Implementation limits and scaling path](docs/microscope-design.md)
- [3Blue1Brown / Manim and other prior art](docs/prior-art.md)
- [Earlier architecture explorer guide](docs/architecture-explorer.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

The new renderer, trainer, and learned fixture were independently implemented.
3Blue1Brown's scene code is CC BY-NC-SA; Manim is MIT. No scene implementation or
assets were copied. Bycroft and Transformer Circuits informed the visual design.
The original code and trained numerical fixture in this project are **MIT licensed**.
