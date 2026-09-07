import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import zlib from "node:zlib";
import {
  ModelIndex,
  validateIR,
  numel,
  layoutScope,
} from "../src/lib/model-ir.js";
import { computedExample, attention, route } from "../src/lib/math.js";
import { TensorReader, decodeHalf, flatIndex } from "../src/lib/tensors.ts";
import { primitiveFor } from "../src/lib/primitives.ts";
import type { ModelIR, Tensor, Entity } from "../src/types.ts";

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
const near = (a: number, b: number) =>
  assert.ok(Math.abs(a - b) < 1e-10, `${a} != ${b}`);

test("GPT-2 reconciles 124,439,808 unique parameters and tied output storage", () => {
  const ir = fixture("gpt2"),
    index = new ModelIndex(ir);
  assert.equal(index.parameterCount(), 124439808n);
  assert.deepEqual(index.entities.get("model.output")?.tensorRefs, [
    "transformer.wte.weight",
  ]);
  assert.equal(index.parameterCount("model.output"), 50257n * 768n);
});
test("trillion-class hero is complete, addressable and rendering pages remain bounded", () => {
  const ir = fixture("deepseek4"),
    index = new ModelIndex(ir);
  assert.equal(index.parameterCount(), 1598837347742n);
  assert.equal(
    index.parameterCount("layers"),
    1572997201763n - 2n * 129280n * 7168n - 7168n - 4n * 28672n - 5n,
  ); // excludes root embedding/head/final norm
  assert.equal(index.childList("layers.60.ffn.experts").length, 384);
  assert.equal(index.page("layers.60.ffn.experts").items.length, 128);
  assert.equal(
    index.page("layers.60.ffn.experts", 2).items.at(-1)?.id,
    "layers.60.ffn.experts.383",
  );
  const t = index.tensors.get("layers.60.ffn.experts.383.w2.weight")!;
  assert.deepEqual(t.shape, [7168, 3072]);
  assert.equal(numel(t.shape), 22020096n);
  assert.equal(index.childList("mtp").length, 1);
  assert.equal(index.entities.get("layers.0.ffn.gate")?.kind, "HashRouter");
  assert.equal(index.entities.get("layers.3.ffn.gate")?.kind, "ScoreRouter");
  assert.equal(index.entities.get("layers.2.attn")?.attrs.compressionRatio, 4);
});
test("provenance, roles, dangling edges and cycles are rejected before commit", () => {
  const base = fixture("tiny");
  for (const mutate of [
    (x: any) => delete x.source,
    (x: any) => delete x.tensors[0].dtype,
    (x: any) => delete x.tensors[0].role,
    (x: any) => (x.entities[1].tensorRefs = "bad"),
    (x: any) => (x.entities[1].parentId = x.entities[1].id),
    (x: any) => (x.edges[0].to = "missing"),
  ]) {
    const ir = structuredClone(base);
    mutate(ir);
    assert.throws(() => validateIR(ir));
  }
});
test("unknown operation names have a safe generic primitive", () => {
  for (const kind of ["constructor", "__proto__", "FutureOperator"])
    assert.equal(primitiveFor({ kind } as Entity).family, "structure");
});
test("module layout is deterministic and containment alone implies no dependency", () => {
  const ir = fixture("tiny");
  const items = ir.entities.filter((e) => e.parentId === "model");
  assert.deepEqual(layoutScope(items, ir.edges), layoutScope(items, ir.edges));
  const a = layoutScope(items, []);
  assert.equal(a.size, items.length);
});
test("attention math: causal zeros, normalized rows and the independently exported fixture agree", () => {
  const e = computedExample(),
    ir = fixture("tiny");
  e.weights.forEach((row: number[], i: number) => {
    near(
      row.reduce((a, b) => a + b, 0),
      1,
    );
    row.slice(i + 1).forEach((v) => assert.equal(v, 0));
  });
  const expected = ir.tensors.find((t) => t.id === "output.data")!.values!;
  e.output.flat().forEach((v: number, i: number) => near(v, expected[i]!));
  const shifted = e.k.map((row: number[]) => row.map((x) => x + 0.1));
  assert.equal(attention(e.q, shifted, e.v).output.length, 4);
  const r = route([1, 3, 2], 2);
  assert.deepEqual(
    r.selected.map((x) => x.id),
    [1, 2],
  );
  near(
    r.selected.reduce((s, x) => s + x.weight, 0),
    1,
  );
});
test("scalar flattening preserves huge exact addresses and arbitrary rank", () => {
  assert.equal(
    flatIndex([1000000000, 1000000000], [999999999, 999999999]),
    999999999999999999n,
  );
  assert.equal(flatIndex([], []), 0n);
  assert.equal(flatIndex([2, 3, 4], [1, 2, 3]), 23n);
  assert.throws(() => flatIndex([2, 3], [2, 0]));
  assert.throws(() => flatIndex([2, 3], [0]));
});
test("half precision boundaries and BF16/F32 local values", async () => {
  assert.equal(decodeHalf(0x3c00), 1);
  assert.equal(decodeHalf(0xc000), -2);
  assert.equal(decodeHalf(0x7c00), Infinity);
  assert.ok(Number.isNaN(decodeHalf(0x7e00)));
  const header = { w: { dtype: "F32", shape: [1, 2], data_offsets: [0, 8] } };
  const h = Buffer.from(JSON.stringify(header));
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(h.length));
  const payload = Buffer.alloc(8);
  payload.writeFloatLE(1.25, 0);
  payload.writeFloatLE(-2.5, 4);
  const f = new File([prefix, h, payload], "weights.safetensors");
  const reader = new TensorReader();
  const [tensor] = await reader.openSafetensors(f);
  assert.equal(tensor.role, "unknown");
  assert.equal(await reader.scalar(tensor, [0, 1]), -2.5);
  const tile = await reader.tile(tensor, [0, 0]);
  assert.deepEqual(tile.values, [1.25, -2.5]);
});
test("missing and empty tensor values remain distinct from zero", async () => {
  const r = new TensorReader();
  const tensor: Tensor = {
    id: "t",
    shape: [2, 3],
    role: "parameter",
    dtype: "F32",
    evidence: "derived",
  };
  assert.equal(await r.scalar(tensor, [0, 0]), null);
  assert.deepEqual(
    (await r.tile({ ...tensor, shape: [0, 3] }, [0, 0])).values,
    [],
  );
  assert.deepEqual(
    (await r.tile({ ...tensor, shape: [], values: [0] }, [])).values,
    [0],
  );
});
test("stale asynchronous reads cannot pollute a newly attached file cache", async () => {
  const r = new TensorReader();
  let resolve: (b: ArrayBuffer) => void;
  const old = {
    name: "same",
    slice: () => ({
      arrayBuffer: () =>
        new Promise<ArrayBuffer>((ok) => {
          resolve = ok;
        }),
    }),
  } as unknown as File;
  r.file = old;
  r.dataStart = 0;
  const t: Tensor = {
    id: "x",
    shape: [1],
    dtype: "F32",
    role: "unknown",
    evidence: "observed",
    source: { file: "same", offsets: [0, 4] },
  };
  const pending = r.scalar(t, [0]);
  r.reset();
  const next = new Float32Array([7]);
  r.file = new File([next], "same");
  resolve!(new Float32Array([3]).buffer);
  assert.equal(await pending, 3);
  assert.equal(await r.scalar(t, [0]), 7);
});
