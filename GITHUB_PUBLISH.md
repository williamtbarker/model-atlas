# Review the new landscape on GitHub

The public repository already exists at
https://github.com/williamtbarker/model-atlas. The original `v0.1.0` tag is a
preserved checkpoint. Do not rerun the old repository-creation script or move that tag.

The continuous landscape is developed on `feat/continuous-landscape` as
`0.2.0-alpha.1`. To try the published review branch in a separate directory:

```bash
git clone --branch feat/continuous-landscape \
  https://github.com/williamtbarker/model-atlas.git model-atlas-landscape
cd model-atlas-landscape
npm install
npm run dev
```

Review the new landing page and the limits in `LANDSCAPE_REVIEW.md` before merging
into `main`. Browser visual QA and MacBook graphics performance remain unverified.
The supplied preview ZIP includes a static build and starts with `node serve.mjs`.

If you are already working in the repository, preserve any local edits before
switching branches. No Git initialization, repository creation, forced push or
retagging is required to try this version.

After human review, merge the pull request normally. A later release can use a
new version tag; keep `v0.1.0` as the original public checkpoint.
