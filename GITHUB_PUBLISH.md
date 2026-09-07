# Preserve this checkpoint on GitHub

These instructions create a new public repository at
`williamtbarker/model-atlas` and preserve this source as the annotated tag
`v0.1.0`. They do not change an existing remote repository. Run them from a
freshly extracted `model-atlas` folder.

Git, Node.js 22.12+, and GitHub CLI (`gh`) are required. Check `gh auth status`;
use `gh auth login` if needed. The commands below assume the authenticated account
can create repositories under `williamtbarker`.

```bash
bash <<'BASH'
set -euo pipefail

test -f package.json
test ! -e .git
gh auth status

npm ci
npm test
npm run build

git init -b main
git add .
git commit -m "Release Model Atlas v0.1.0"

gh repo create williamtbarker/model-atlas \
  --public \
  --source=. \
  --remote=origin \
  --push \
  --disable-wiki \
  --description "Interactive transformer computation and architecture explorer with numerical traces and semantic zoom."

git tag -a v0.1.0 -m "First public experimental release"
git push origin v0.1.0

gh release create v0.1.0 \
  --repo williamtbarker/model-atlas \
  --verify-tag \
  --prerelease \
  --title "Model Atlas v0.1.0" \
  --notes-file RELEASE_NOTES.md
BASH
```

The existing `.gitignore` keeps `node_modules/`, `dist/`, virtual environments,
and local deployment files out of Git. The source, fixtures, metadata packages,
tests, and documentation are committed.

Attach the supplied ZIP to the GitHub release so people can use the prebuilt
viewer. If the download is in your Downloads folder:

```bash
gh release upload v0.1.0 "$HOME/Downloads/model-atlas-v0.1.0-github.zip" \
  --repo williamtbarker/model-atlas
```

The command has no overwrite option; it preserves an existing release asset.
Alternatively, upload the ZIP using the release's Edit page.

Keep the `v0.1.0` tag fixed. Continue development on `main` or a new branch:

```bash
git switch -c feat/visual-design
```

The public version begins at 0.1.0. Earlier version labels in the validation
record identify internal development checkpoints, not earlier GitHub releases.

Official command references:

- https://cli.github.com/manual/gh_repo_create
- https://cli.github.com/manual/gh_release_create
- https://cli.github.com/manual/gh_release_upload
