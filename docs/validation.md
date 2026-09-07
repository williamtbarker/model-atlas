# Validation record — 7 September 2026

The first public release, **v0.1.0**, combines both internal checkpoints below.
The public release preparation changes documentation, package metadata, and CI
coverage; renderer and numerical source are unchanged from the validated build.

## Computation microscope (internal checkpoint 0.2.0)

The final release build passed TypeScript checking and Vite production bundling.
Both the microscope and the retained architecture page are included. The shared
Three.js/OrbitControls chunk is about 514 KB uncompressed (130 KB gzip); the
microscope JavaScript is about 43 KB (16 KB gzip). The numerical package is
1,677,661 bytes, already gzip-compressed. No dependency installation is needed
to serve the supplied build with `node serve.mjs`.

All 13 TypeScript/JavaScript tests passed, including the ten architecture tests
listed below and three new microscope tests. The new tests independently
recompute more than 100,000 displayed scalar results, reconcile all 4,777 stored
parameters, and check intervention effects and stable scene positions. All four
NumPy trace tests passed against the supplied compressed package and checkpoint.
The dependency-free server passed Node's syntax check. These are source,
numerical, and build checks; the server was not exercised in a browser.

Read-only review also corrected inconsistent layer camera offsets, head and
token selection state, masked-score operand paths, arithmetic animation timing,
operation framing, the visible intervention gate, and duplicate animation-frame
scheduling. Camera and contribution-geometry claims were checked against code.

**Browser interaction, visual quality, GPU frame rate, and MacBook compatibility
remain for human review.** No screenshots, end-to-end browser run, or hardware
performance benchmark are claimed.

## Earlier architecture checkpoint (internal 0.1.0)

The build environment used Node 24.19.0. The documented local target is Node 22 LTS or newer. A standard Vite static build completed with TypeScript checking. A single 552 KB JavaScript bundle (about 144 KB gzip) contains the viewer and Three.js. Architecture packages remain separate assets.

Ten TypeScript/JavaScript tests passed:

- GPT-2 exact total and tied output storage.
- Hero exact total, last-layer/expert addressability, paging and distinct router/compression types.
- Rejection of missing provenance, invalid roles, invalid tensor references, cycles and dangling edges.
- Safe fallback for unknown operations, including inherited JavaScript property names.
- Deterministic scope layout.
- Causal attention, normalized coefficients, independently computed Python fixture, and top-k renormalization.
- BigInt scalar offsets and arbitrary-rank coordinates.
- FP16 boundaries and real local F32 file reads.
- Empty tensors, missing values and actual zeros.
- File-switch race regression: stale asynchronous reads cannot contaminate the new file's cache.

Four Python tests passed:

- GPT-2 parameter count from an independent formula.
- All 145,116 checkpoint names, 7,030 sampled header entries, and complete packed-byte reconciliation.
- Nonexecuting module traversal and tied-Parameter identity reconciliation using a small controlled module fixture.
- Deterministic gzip package serialization.

Read-only code review found and fixed GPU instance-buffer disposal, stale scalar cache writes, incomplete import validation, unsafe primitive fallback, incorrect rank-to-primitive choices, a file-import race, and parameter certainty in raw tensor catalogs. Slice/view relations are no longer presented as computational arrows.

The tests do not establish browser frame rate, Safari compatibility, graphical layout quality or complete third-party adapter coverage. The PyTorch traversal test uses a controlled fixture; ONNX requires its optional dependency and has not been exercised against a broad model corpus. No browser interaction test or MacBook hardware run was performed. The review guide identifies those next checks.
