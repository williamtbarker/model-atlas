# Contributing

Report a focused issue or propose a small change with its expected behavior.
For visual problems, include the browser, selected model/component, reproduction
steps, and a screenshot where useful. Do not include private model weights or data.

## Development setup

Use Node.js 22.12 or newer and Python 3.12 for the Python checks used in CI.
From a clone of this repository:

```bash
npm ci
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/recall/requirements.txt
npm run dev
```

## Verification

Run the checks defined in `.github/workflows/ci.yml`:

```bash
npm test
npm run test:python
npm run test:trace
npm run build
```

The Python exporter checks use the standard library. Recall-2 trace checks require
NumPy from the pinned requirements file. These tests do not measure browser
rendering quality or graphics performance. For visual changes, also exercise the
relevant interactions in a browser and record what you actually reviewed; use
[the review guide](docs/review-guide.md) and
[landscape review notes](LANDSCAPE_REVIEW.md) for scope.

## Model and evidence boundaries

- Keep parameter accounting, declared architecture scope, and tensor storage
  identity independently testable. A displayed address does not imply a trained
  value is available.
- Keep recorded numerical calculations distinct from structural metadata.
- Preserve pinned source provenance and third-party license notices. Describe
  constructor-derived shapes separately from checkpoint-header verification.
- Add a meaningful regression check for changed behavior and retain honest
  unsupported cases, uncertainty, and validation limits.
- Commit source and necessary distributable fixtures; omit `node_modules`, `dist`,
  virtual environments, local deployment configuration, and downloaded model weights.

Include the change's purpose and observed verification results in the pull request.
Update release documentation when the supported scope or behavior changes.
