import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { compileTrace } from "../src/microscope/adapter";
import { explain } from "../src/microscope/arithmetic";
import type { TracePackage } from "../src/microscope/types";

const pkg: TracePackage = JSON.parse(
  gunzipSync(
    readFileSync(new URL("../public/traces/recall.json.gz", import.meta.url)),
  ).toString(),
);
const close = (actual: number, expected: number, msg: string) =>
  assert.ok(
    Math.abs(actual - expected) < 2e-10,
    `${msg}: ${actual} vs ${expected}`,
  );

test("every displayed scalar computation reconciles with its recorded result, including all interventions", () => {
  let checked = 0;
  for (const example of pkg.examples)
    for (const [name, run] of Object.entries(example.runs)) {
      const scene = compileTrace(pkg, example, run, name),
        byId = new Map(scene.views.map((v) => [v.id, v]));
      for (const v of scene.views) {
        if (!v.op) continue;
        for (let r = 0; r < v.values.length; r++)
          for (let c = 0; c < v.values[0].length; c++) {
            const a = explain(scene, { viewId: v.id, row: r, col: c }),
              op = v.op;
            let result = a.result;
            if (a.mode === "sum")
              result =
                (a.terms.reduce((s, t) => s + t.value, 0) + (a.bias ?? 0)) /
                (a.divide ?? 1);
            else if (op.kind === "attention") {
              if (c > r) {
                assert.equal(a.terms.length, 0);
                assert.equal(a.result, 0);
                continue;
              }
              const scores = op.scores[r],
                finite = scores.filter((x) => x !== null) as number[],
                max = Math.max(...finite);
              const dot = a.terms.reduce((s, t) => s + t.value, 0) / op.scale;
              close(dot, scores[c]!, `${name} ${v.id} qk`);
              result = dot;
            } else if (op.kind === "rowSoftmax") {
              if (c > r) {
                assert.equal(a.terms.length, 0);
                assert.equal(a.result, 0);
                continue;
              }
              const scores = byId.get(op.input)!.values[r],
                max = Math.max(
                  ...(scores.filter((x) => x !== null) as number[]),
                );
              result =
                Math.exp(scores[c]! - max) /
                scores.reduce(
                  (sum, x) => sum + (x === null ? 0 : Math.exp(x - max)),
                  0,
                );
            } else if (op.kind === "norm") {
              const x = byId.get(op.input)!.values.map((row) => row[c]!),
                mean = x.reduce((s, v) => s + v, 0) / x.length,
                variance =
                  x.reduce((s, v) => s + (v - mean) ** 2, 0) / x.length;
              result =
                (op.gamma[r] * (x[r] - mean)) /
                  Math.sqrt(variance + op.epsilon) +
                op.beta[r];
            } else if (op.kind === "gelu") {
              const x = byId.get(op.input)!.values[r][c]!;
              result =
                0.5 *
                x *
                (1 +
                  Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
            } else if (op.kind === "softmax") {
              const x = byId.get(op.input)!.values.map((row) => row[c]!),
                max = Math.max(...x);
              result =
                Math.exp(x[r] - max) /
                x.reduce((s, v) => s + Math.exp(v - max), 0);
            }
            close(result, v.values[r][c]!, `${name} ${v.id}[${r},${c}]`);
            checked++;
          }
      }
    }
  assert.ok(checked > 100000);
});

test("stored parameter inventory counts every learned scalar exactly once", () => {
  const scene = compileTrace(
    pkg,
    pkg.examples[0],
    pkg.examples[0].runs.baseline,
  );
  const params = scene.model.tensors.filter((t) => t.role === "parameter");
  assert.equal(params.length, Object.keys(pkg.weights).length);
  assert.equal(
    params.reduce((s, t) => s + t.values!.length, 0),
    4777,
  );
  const ids = new Set(scene.model.entities.map((e) => e.id));
  assert.equal(ids.size, scene.model.entities.length);
  for (const edge of scene.model.edges) {
    assert.ok(ids.has(edge.from));
    assert.ok(ids.has(edge.to));
  }
  for (const view of scene.views)
    assert.ok(view.values.every((r) => r.length === view.values[0].length));
});

test("ablation changes later attention, preserves layout, and visibly zeros the effective head output", () => {
  const e = pkg.examples[0],
    a = compileTrace(pkg, e, e.runs.baseline),
    b = compileTrace(pkg, e, e.runs.L0H0, "L0H0");
  assert.deepEqual(
    a.views.map((v) => [v.id, v.position]),
    b.views.map((v) => [v.id, v.position]),
  );
  assert.notDeepEqual(
    e.runs.baseline.blocks[1].attention,
    e.runs.L0H0.blocks[1].attention,
  );
  assert.ok(
    b.views
      .find((v) => v.id === "b0.h0.mixed")!
      .values.flat()
      .some((v) => v !== 0),
  );
  assert.ok(
    b.views
      .find((v) => v.id === "b0.h0.gate")!
      .values.flat()
      .every((v) => v === 0),
  );
  const p = e.runs.baseline.probabilities.at(-1)!,
    ablated = e.runs.L0H0.probabilities.at(-1)!;
  assert.equal(pkg.model.vocab[p.indexOf(Math.max(...p))], "4");
  assert.equal(pkg.model.vocab[ablated.indexOf(Math.max(...ablated))], "2");
});
