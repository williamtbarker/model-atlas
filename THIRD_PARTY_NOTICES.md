# Third-party notices

- **Three.js 0.180.0** — MIT, copyright its authors. Installed from the pinned npm dependency; its license is retained by npm and included in `references/THREE-LICENSE.txt` for the production source archive.
- **DeepSeek-V4-Pro metadata** — MIT, DeepSeek. The exact config, derived accounting, checkpoint index and small header samples are included for reproducibility. See `references/DEEPSEEK-LICENSE.txt` and the pinned source URLs in `references/deepseek4-shapes.json`. No trained weight-value payloads are bundled.
- **GPT-2 architecture declarations** — public OpenAI GPT-2 implementation/configuration, MIT. Architecture quantities and mathematical facts are independently encoded; no pretrained weights are included.
- **3Blue1Brown videos scenes** — CC BY-NC-SA 4.0; **not included or adapted**. The prior-art review links the original scene code and lessons. The independently implemented mathematical visual concepts do not import the scene implementation, media or camera choreography.
- **3Blue1Brown Manim engine** — MIT; reviewed, not used as a dependency.
- **Brendan Bycroft, llm-viz** — MIT with repository-specific exclusions; reviewed for persistent tensor geography and inspection. No code or assets copied.
- **Transformer Circuits / HeadVis** — mathematical framework and precomputed-data viewer reviewed; HeadVis is Apache-2.0. No code, figures, or assets copied.
- **Recall-2 numerical fixture** — independently trained for this project using the included NumPy trainer. The checkpoint and recorded calculations are distributed under this project's MIT license.
- Vite, TypeScript, tsx, type definitions and their transitive dependencies retain the licenses distributed with their npm packages. `package-lock.json` pins the dependency tree.

Acknowledgement: Visual explanations of embeddings, matrix transformations, token flow, and attention were informed by Grant Sanderson's 2024 3Blue1Brown transformer lessons. This project independently implements interactive model visualization and incorporates no scene implementation or media assets from the 3Blue1Brown videos repository.

- **DM Sans** — SIL Open Font License 1.1. Local font files are redistributed with `public/fonts/dmsans-OFL.txt`.
- **IBM Plex Mono** — SIL Open Font License 1.1. Local font files are redistributed with `public/fonts/ibmplexmono-OFL.txt`.
  The landscape loads these fonts locally; it makes no Google Fonts request at runtime.
