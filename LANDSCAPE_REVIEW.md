# Model Atlas 0.2.0-alpha.1

The landing view has been replaced with a continuous architecture landscape.
The 61 core DeepSeek blocks retain their positions while attention, memory,
residual mixing, routing and expert details resolve with zoom. Exact IR-backed
dependencies and tensor coordinates are inspectable. The earlier metadata catalog
and Recall-2 numerical microscope remain separate pages.

This is a review build. Browser visual QA was blocked by the development
environment, and MacBook rendering performance has not been measured. The changes
are not presented as a finished replacement for the original visual ambition.

The ZIP includes a fresh static build. In its `model-atlas` directory:

```bash
node serve.mjs
```

Open http://127.0.0.1:4173. The page title should be **Model Atlas — Inside the
architecture**, with Whole model / Block / Attention / Experts controls. If you
see the old landing page, check which directory is serving and reload.

For source development, run `npm install` and `npm run dev`. Node 22.12+ is
required; no Python or model-weight download is needed to view the package.

The prior public v0.1.0 tag is preserved. This preview does not create or move a
release tag and does not replace `main` automatically.
