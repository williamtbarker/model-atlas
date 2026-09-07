import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import zlib from "node:zlib";
import {
  compileLandscape,
  projectLandscapeLinks,
} from "../src/landscape/layout";
import type { ModelIR } from "../src/types";
function load(name: string): ModelIR {
  return JSON.parse(
    zlib
      .gunzipSync(
        fs.readFileSync(
          new URL(`../public/models/${name}.atlas.json.gz`, import.meta.url),
        ),
      )
      .toString(),
  );
}
test("Maverick keeps nested text landmarks and image fusion outside the complete decoder chain", () => {
  const ir = load("llama4-maverick"),
    scene = compileLandscape(ir);
  assert.equal(scene.index.parameterCount(ir.rootId), 401583781376n);
  assert.equal(scene.blocks.filter((b) => !b.auxiliary).length, 48);
  const glyph = new Map(scene.glyphs.map((g) => [g.id, g]));
  for (const id of [
    "language_model.model.embed_tokens",
    "language_model.model.norm",
    "language_model.lm_head",
    "fusion",
    "vision_model",
  ])
    assert.ok(glyph.has(id), `Missing real landmark ${id}`);
  assert.ok(
    !glyph.has("language_model"),
    "A containing stage must not hide its real operators",
  );
  assert.ok(
    glyph.get("vision_model")!.position[0] < scene.blocks[0].position[0],
  );
  assert.ok(
    scene.links.some((e) => e.from === "vision_model" && e.to === "fusion"),
  );
  assert.ok(
    scene.blocks.filter(
      (b) => b.expertsId && glyph.get(b.expertsId)?.kind === "experts",
    ).length === 24,
  );
  assert.ok(scene.blocks.every((b) => b.glyphIds.length <= 35));
});
test("Qwen hybrid states, output gates and exact scoped inventory remain explicit across semantic zoom", () => {
  const ir = load("qwen35-397b"),
    scene = compileLandscape(ir);
  assert.equal(scene.index.parameterCount(ir.rootId), 396346350336n);
  assert.equal(scene.blocks.length, 60);
  const glyph = new Map(scene.glyphs.map((g) => [g.id, g]));
  assert.deepEqual(glyph.get("layers.0.attn")?.shape, [64, 128, 128]);
  assert.equal(
    scene.blocks.filter((b) => glyph.get(b.attentionId!)?.kind === "state")
      .length,
    45,
  );
  assert.equal(
    scene.blocks.filter((b) => glyph.get(b.attentionId!)?.kind === "heads")
      .length,
    15,
  );
  assert.equal(
    scene.glyphs.filter(
      (g) => g.kind === "gate" && g.id.endsWith(".shared_gate"),
    ).length,
    60,
  );
  const visible = new Set(
    scene.glyphs.filter((g) => !g.attrs?.semanticSummary).map((g) => g.id),
  );
  const edges = projectLandscapeLinks(scene, visible);
  assert.ok(
    edges.some(
      (e) =>
        e.from === "layers.0.attn.previous_state" &&
        e.to === "layers.0.attn.state",
    ),
  );
  assert.ok(
    edges.some(
      (e) =>
        e.from === "layers.0.ffn.shared_gate" && e.to === "layers.0.ffn.merge",
    ),
  );
  assert.ok(scene.blocks.every((b) => b.glyphIds.length <= 35));
  assert.ok(String(ir.metadata?.scopeLabel).includes("only"));
  assert.ok(scene.glyphs.every((g) => g.position.every(Number.isFinite)));
});
