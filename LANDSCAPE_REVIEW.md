# Model Atlas v0.2.0: validation and local review

The continuous landscape keeps model components in place as semantic zoom
reveals their operations. DeepSeek-V4-Pro and GPT-2 are joined by Llama 4 Maverick
and Qwen3.5-397B-A17B. The earlier metadata catalog and Recall-2 numerical
microscope remain separate pages.

## What changed

- **Maverick:** all 48 text blocks, alternating dense and expert feed-forward
  branches, grouped-query attention, and the upstream vision/fusion inventory.
  The vision encoder is a summary in the landscape, with its blocks and tensors
  addressable through the inventory.
- **Qwen3.5:** the complete declared **text-decoder scope**, with 45 recurrent
  attention blocks, 15 gated softmax-attention blocks and expert routing. Vision
  and auxiliary prediction are excluded and are not counted in this package.
- Grouped-query fans now encode the actual query-to-KV assignment. Recurrent
  state, causal convolution, gates and rotary operations have distinct glyphs.
- Shareable model/component links, a Back control and search shortcut were added.
  Model switching cancels obsolete requests and prevents stale results or errors
  from replacing the current view.

The packages contain metadata, not large-model weights or recorded execution.
Exact scalar addresses do not imply that their trained values are available.
The [Maverick](docs/llama4-maverick-provenance.md) and
[Qwen](docs/qwen35-provenance.md) provenance notes identify constructor-derived
shapes, checkpoint-name evidence, storage partitions and unresolved options.

## Evidence and remaining limits

The project owner downloaded the preceding landscape build, ran it locally on a
MacBook, and reported liking the result. This is positive review of that earlier
landscape, not a claim that the newly added models have been visually tested.

The new additions received automated tests and source review, including parameter
accounting, scope, expert partitions, grouped-query assignments, recurrent
connections, bounded layout and navigation state. Browser access was blocked in
the development environment. **No fresh browser visual QA or graphics performance
benchmark is claimed for this round.**

## Open the supplied build

In the unzipped `model-atlas` directory, run:

```bash
node serve.mjs
```

Open http://127.0.0.1:4173. Choose any of the four models, approach an attention
branch and a feed-forward branch, then try an exact tensor address. Copy a view
link and reopen it to return to the same model/component; camera orbit is not
part of the link. Back traverses prior selections across models.

For source development, use `npm ci` and `npm run dev`. Node 22.12+ is required;
Python and model-weight downloads are not needed to view the bundled packages.
The original public `v0.1.0` tag is preserved.
