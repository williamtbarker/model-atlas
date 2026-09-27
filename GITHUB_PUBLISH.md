# Model Atlas on GitHub

The public repository is https://github.com/williamtbarker/model-atlas.
Version 0.2.0 brings the continuous landscape and four-model gallery to `main`.
The original `v0.1.0` tag remains the preserved public checkpoint; keep it intact.

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

GitHub's source archives do not include a production build. Run `npm ci` and
`npm run build` before `node serve.mjs`, whether using a clone or an extracted
source archive. Node 22.12+ is required.

## Release history and validation

The previous landscape received local review. The additional packages and
navigation changes have automated and source validation. They have not received fresh
browser visual QA or a MacBook performance benchmark in this environment.
`LANDSCAPE_REVIEW.md` records that distinction.

Use a new version tag for a new release. Keep the original `v0.1.0` history
intact; normal pull-request merging is sufficient to update `main`.
