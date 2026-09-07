# Model Atlas on GitHub

The public repository is https://github.com/williamtbarker/model-atlas.
Version 0.2.0 brings the continuous landscape and four-model gallery to `main`.
The original `v0.1.0` tag remains the preserved public checkpoint; do not move it
or rerun the old repository-creation script.

## Fresh clone

```bash
git clone https://github.com/williamtbarker/model-atlas.git
cd model-atlas
npm ci
npm run dev
```

For an existing checkout with no uncommitted changes:

```bash
git switch main
git pull --ff-only
npm ci
npm run dev
```

Preserve local edits before changing branches. No Git initialization, repository
creation, forced push or retagging is required.

The supplied ZIP includes a production build and starts with `node serve.mjs`.
A source clone needs `npm run build` before that command. Node 22.12+ is required.

## Release history and validation

The project owner reviewed the previous landscape locally and authorized moving
forward with the new model options. The additional packages and navigation
changes have automated and source validation. They have not received fresh
browser visual QA or a MacBook performance benchmark in this environment.
`LANDSCAPE_REVIEW.md` records that distinction.

Use a new version tag for a new release. Keep the original `v0.1.0` history
intact; normal pull-request merging is sufficient to update `main`.
