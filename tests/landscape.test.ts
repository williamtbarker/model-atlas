import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import zlib from "node:zlib";
import {
  compileLandscape,
  projectLandscapeLinks,
  type Landscape,
} from "../src/landscape/layout";
import type { ModelIR } from "../src/types";

const fixture = (key: string): ModelIR =>
  JSON.parse(
    zlib
      .gunzipSync(
        fs.readFileSync(
          new URL(`../public/models/${key}.atlas.json.gz`, import.meta.url),
        ),
      )
      .toString(),
  );
const heroIR = fixture("deepseek4");
const hero = compileLandscape(heroIR);
const glyphMap = new Map(hero.glyphs.map((g) => [g.id, g]));
const positionSnapshot = (scene: Landscape) =>
  scene.glyphs.map((g) => [g.id, [...g.position]]);

test("landscape keeps every core block in a bounded persistent scene and MTP in a separate island", () => {
  const core = hero.blocks.filter((b) => !b.auxiliary),
    auxiliary = hero.blocks.filter((b) => b.auxiliary);
  assert.equal(core.length, 61);
  assert.deepEqual(
    core.map((b) => b.index),
    Array.from({ length: 61 }, (_, i) => i),
  );
  assert.equal(auxiliary.length, 1);
  assert.equal(auxiliary[0].id, "mtp.0");
  assert.equal(new Set(core.map((b) => b.position.join(","))).size, 61);
  assert.equal(
    glyphMap.size,
    hero.glyphs.length,
    "glyph IDs must remain unique",
  );
  for (const b of hero.blocks) {
    assert.ok(b.glyphIds.length <= 35, `${b.id} expanded too many objects`);
    assert.ok(b.glyphIds.every((id) => glyphMap.has(id)));
    assert.ok(
      glyphMap.get(b.id)?.position.every((v, i) => v === b.position[i]),
    );
  }
  // Actual expert weights stay addressable through the index; no weight-sized scene.
  assert.equal(hero.index.childList("layers.60.ffn.experts").length, 384);
  assert.equal(hero.index.tensors.size, heroIR.tensors.length);
  assert.ok(hero.glyphs.length < 2500);
  assert.equal(hero.glyphs.filter((g) => g.id.startsWith("tensor:")).length, 0);
  assert.ok(
    auxiliary[0].position[0] > Math.max(...core.map((b) => b.position[0])),
  );
});

test("landscape links preserve real IR dependencies across group boundaries without connecting MTP into the decoder chain", () => {
  const sourceEdges = new Map(heroIR.edges.map((e) => [e.id, e]));
  const covered = new Set<string>();
  for (const link of hero.links) {
    assert.ok(glyphMap.has(link.from) && glyphMap.has(link.to));
    assert.notEqual(link.from, link.to);
    assert.ok(link.kind === "dataflow" || link.kind === "residual");
    assert.ok(link.sourceEdgeIds.length > 0);
    for (const id of link.sourceEdgeIds) {
      const source = sourceEdges.get(id);
      assert.ok(source, `${id} is not an original IR edge`);
      assert.equal(source.kind, link.kind);
      assert.equal(source.evidence, link.evidence);
      covered.add(id);
    }
    const from = glyphMap.get(link.from)!,
      to = glyphMap.get(link.to)!;
    if (from.blockId === "mtp.0" || to.blockId === "mtp.0")
      assert.equal(from.blockId, to.blockId);
  }
  for (let i = 0; i < 60; i++)
    assert.ok(
      hero.links.some(
        (e) => e.from === `layers.${i}` && e.to === `layers.${i + 1}`,
      ),
      `missing block boundary ${i}`,
    );
  // The bundled hero's declared computation edges are all represented, including
  // collection edges that become last-block -> first-block when unfolded.
  assert.deepEqual(
    covered,
    new Set(
      heroIR.edges
        .filter((e) => e.kind === "dataflow" || e.kind === "residual")
        .map((e) => e.id),
    ),
  );
});

test("semantic zoom resolves summary inputs and outputs while retaining fixed coordinates", () => {
  const before = positionSnapshot(hero);
  const far = new Set(
    hero.glyphs.filter((g) => g.level === 1).map((g) => g.id),
  );
  const close = new Set(
    hero.glyphs
      .filter((g) => g.level <= 2 && !g.attrs?.semanticSummary)
      .map((g) => g.id),
  );
  const mixed = new Set(
    hero.glyphs
      .filter((g) =>
        g.blockId === "layers.7"
          ? g.level <= 2 && !g.attrs?.semanticSummary
          : g.level === 1,
      )
      .map((g) => g.id),
  );
  for (const visible of [far, close, mixed]) {
    const links = projectLandscapeLinks(hero, visible);
    assert.ok(links.every((e) => visible.has(e.from) && visible.has(e.to)));
    assert.ok(links.some((e) => e.from === "layers.7" && e.to === "layers.8"));
    assert.equal(
      new Set(links.map((e) => `${e.kind}/${e.from}/${e.to}/${e.evidence}`))
        .size,
      links.length,
    );
  }
  const detailed = projectLandscapeLinks(hero, close);
  assert.ok(
    detailed.some(
      (e) => e.from === "layers.0.attn.wo_b" && e.to === "layers.0.post_attn",
    ),
  );
  assert.ok(
    detailed.some(
      (e) => e.from === "layers.0.ffn.merge" && e.to === "layers.0.post_ffn",
    ),
  );
  assert.ok(
    detailed.some(
      (e) => e.from === "layers.0.attn_norm" && e.to === "layers.0.attn.input",
    ),
  );
  assert.deepEqual(
    positionSnapshot(hero),
    before,
    "changing semantic visibility must not relayout the world",
  );
});

test("landscape preserves logical tensor shapes and expert parameter inventory independently of packed storage", () => {
  assert.deepEqual(glyphMap.get("layers.0.attn.wq_a")?.shape, [1536, 7168]);
  assert.deepEqual(glyphMap.get("layers.0.attn.wq_b")?.shape, [65536, 1536]);
  const tensor = hero.index.tensors.get("layers.60.ffn.experts.383.w2.weight");
  assert.deepEqual(tensor.shape, [7168, 3072]);
  assert.deepEqual(tensor.storage.shape, [7168, 1536]);
  assert.equal(
    hero.index.parameterCount("layers.60.ffn.experts.383"),
    66060288n,
  );
  assert.equal(hero.index.parameterCount(), 1598837347742n);
  assert.equal(hero.blocks[0].expertsId, "layers.0.ffn.experts");
  assert.equal(hero.blocks[0].routerId, "layers.0.ffn.gate");
  assert.equal(
    hero.index.entities.get(hero.blocks[0].routerId)?.kind,
    "HashRouter",
  );
  assert.equal(
    hero.index.entities.get(hero.blocks[3].routerId)?.kind,
    "ScoreRouter",
  );
  assert.ok(!glyphMap.has("layers.0.attn.indexer"));
  assert.equal(glyphMap.get("layers.2.attn.indexer")?.attrs?.topK, 1024);
});

test("containment creates no computational links and semantic layouts do not depend on the model name", () => {
  const ir = fixture("gpt2");
  const original = compileLandscape(ir);
  const renamed = compileLandscape({
    ...ir,
    name: "An unrelated display name",
  });
  assert.deepEqual(renamed.glyphs, original.glyphs);
  assert.deepEqual(renamed.links, original.links);
  const disconnected = compileLandscape({ ...ir, edges: [] });
  assert.equal(disconnected.links.length, 0);
  assert.equal(
    projectLandscapeLinks(
      disconnected,
      new Set(disconnected.glyphs.map((g) => g.id)),
    ).length,
    0,
  );
  assert.deepEqual(
    disconnected.blocks.map((b) => [b.id, b.position]),
    original.blocks.map((b) => [b.id, b.position]),
  );
});
