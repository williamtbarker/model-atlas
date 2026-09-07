import test from "node:test";
import assert from "node:assert/strict";
import { headGrouping } from "../src/landscape/visual-encoding";
test("GQA geometry preserves every declared query-to-KV assignment", () => {
  for (const [q, kv] of [
    [40, 8],
    [32, 2],
    [128, 1],
    [12, 12],
  ]) {
    const result = headGrouping(q, kv);
    assert.equal(result.heads.length, q);
    assert.equal(result.keys.length, kv);
    for (let i = 0; i < kv; i++)
      assert.equal(result.heads.filter((h) => h.kvIndex === i).length, q / kv);
    assert.ok(
      result.heads.every(
        (h) => h.kvX === result.keys.find((k) => k.index === h.kvIndex)!.x,
      ),
    );
  }
});
test("Head sampling stays bounded without inventing or changing group identities", () => {
  const result = headGrouping(1024, 32, 64);
  assert.equal(result.heads.length, 64);
  assert.equal(result.sampled, true);
  assert.equal(new Set(result.heads.map((h) => h.index)).size, 64);
  assert.ok(result.heads.every((h) => h.kvIndex === Math.floor(h.index / 32)));
  for (const pair of [
    [0, 1],
    [10, 3],
    [4, 8],
    [4, NaN],
  ])
    assert.throws(() => headGrouping(...(pair as [number, number])));
});
