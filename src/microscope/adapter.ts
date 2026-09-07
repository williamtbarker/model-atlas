import type {
  TracePackage,
  SceneIR,
  TensorView,
  Matrix,
  Operation,
  Example,
  Run,
} from "./types";
import type { ModelIR } from "../types";

export const transpose = (m: Matrix): Matrix =>
  m[0].map((_, c) => m.map((r) => r[c]));
export function compileTrace(
  pkg: TracePackage,
  example: Example,
  run: Run,
  runName = "baseline",
): SceneIR {
  const m = pkg.model,
    views: TensorView[] = [],
    connections: SceneIR["connections"] = [];
  const model: ModelIR = {
    version: "1.0",
    name: m.name,
    rootId: "model",
    source: {
      kind: "recorded-computation",
      evidence: "observed",
      notes: m.description,
    },
    entities: [
      {
        id: "model",
        parentId: null,
        kind: "Model",
        label: m.name,
        evidence: "observed",
      },
    ],
    tensors: [],
    edges: [],
  };
  const add = (
    id: string,
    label: string,
    kind: TensorView["kind"],
    values: Matrix,
    position: TensorView["position"],
    block: number,
    group: string,
    op?: Operation,
    head?: number,
    caption = "",
  ) => {
    const v: TensorView = {
      id,
      entityId: id,
      label,
      kind,
      values,
      position,
      block,
      group,
      op,
      head,
      caption,
      rowAxis:
        kind === "attention"
          ? "query position"
          : kind === "weight"
            ? "output coordinate"
            : "coordinate",
      colAxis:
        kind === "attention"
          ? "key position"
          : kind === "weight"
            ? "input coordinate"
            : "token position",
    };
    views.push(v);
    model.entities.push({
      id,
      parentId: block < 0 ? "model" : `block.${block}`,
      kind:
        kind === "attention"
          ? "AttentionGrid"
          : kind === "weight"
            ? "MatrixPlane"
            : "TokenStream",
      label,
      evidence: "observed",
      tensorId: id,
    });
    model.tensors.push({
      id,
      shape: [values.length, values[0].length],
      dtype: "float64",
      role: kind === "weight" ? "unknown" : "activation",
      evidence: "observed",
      values: values.flat(),
    });
    return id;
  };
  const edge = (
    from: string,
    to: string,
    kind: "data" | "residual" | "weight" = "data",
    via?: Connection["via"],
  ) => {
    const id = `${from}->${to}`;
    connections.push({ id, from, to, kind, via });
    model.edges.push({
      id,
      from,
      to,
      kind: kind === "residual" ? "residual" : "dataflow",
      evidence: "observed",
    });
  };
  type Connection = SceneIR["connections"][number];
  const matrix = (name: string) => pkg.weights[name] as number[][];
  const vector = (name: string) => pkg.weights[name] as number[];
  const word = example.inputIds.map((i) => matrix("embedding")[i]),
    pos = matrix("position").slice(0, example.tokens.length);
  add(
    "embedding",
    "Token + position",
    "activation",
    transpose(run.embedding),
    [-12, 0, 0],
    -1,
    "embedding",
    { kind: "embedding", word, position: pos },
    undefined,
    "Every coordinate, all tokens",
  );
  for (let l = 0; l < m.nLayers; l++) {
    model.entities.push({
      id: `block.${l}`,
      parentId: "model",
      kind: "TransformerBlock",
      label: `Block ${l + 1}`,
      evidence: "observed",
    });
    const b = run.blocks[l],
      p = `b${l}`,
      w = `blocks.${l}`,
      x = l * 164;
    const a = (
      name: string,
      label: string,
      kind: TensorView["kind"],
      vals: Matrix,
      dx: number,
      y = 0,
      op?: Operation,
      head?: number,
      caption = "",
    ) =>
      add(
        `${p}.${name}`,
        label,
        kind,
        vals,
        [x + dx, y, 0],
        l,
        name,
        op,
        head,
        caption,
      );
    const input = a(
      "input",
      "Residual in",
      "activation",
      transpose(b.input),
      0,
    );
    edge(l ? `b${l - 1}.output` : "embedding", input);
    const norm = a(
      "norm1",
      "Normalize",
      "normalization",
      transpose(b.norm1),
      11,
      0,
      {
        kind: "norm",
        input,
        gamma: vector(`${w}.ln1.gamma`),
        beta: vector(`${w}.ln1.beta`),
        epsilon: m.layerNormEpsilon,
      },
    );
    edge(input, norm);
    const headIds: string[] = [];
    for (let h = 0; h < m.nHeads; h++) {
      const y = (m.nHeads - 1 - 2 * h) * 16,
        head = `h${h}`,
        slice = (n: string) =>
          matrix(`${w}.${n}`).map((r) =>
            r.slice(h * m.dHead, (h + 1) * m.dHead),
          );
      for (const [n, dy, vals] of [
        ["q", 8, b.q[h]],
        ["k", 0, b.k[h]],
        ["v", -8, b.v[h]],
      ] as const) {
        const weight = a(
          `${head}.w${n}`,
          `W${n.toUpperCase()} · head ${h + 1}`,
          "weight",
          transpose(slice(`w${n}`)),
          18,
          y + dy,
          undefined,
          h,
          "Displayed as Wᵀ; checkpoint stores XW",
        );
        views[views.length - 1].position[2] = -11;
        const out = a(
          `${head}.${n}`,
          n.toUpperCase(),
          "activation",
          transpose(vals),
          28,
          y + dy,
          { kind: "matmul", input: norm, weight },
          h,
        );
        edge(norm, out);
        edge(weight, out, "weight");
      }
      const q = `${p}.${head}.q`,
        k = `${p}.${head}.k`,
        v = `${p}.${head}.v`;
      const scores = a(
        `${head}.scores`,
        "QK scores / √d",
        "attention",
        b.scores[h],
        38,
        y + 8,
        {
          kind: "attention",
          q,
          k,
          scale: Math.sqrt(m.dHead),
          scores: b.scores[h],
        },
        h,
        "Scaled dot products · future positions masked",
      );
      edge(q, scores);
      edge(k, scores);
      const att = a(
        `${head}.attention`,
        `Softmax · head ${h + 1}`,
        "attention",
        b.attention[h],
        44,
        y - 2,
        { kind: "rowSoftmax", input: scores },
        h,
        "Rows query · columns key · row sum = 1",
      );
      edge(scores, att);
      const rawMix = b.attention[h].map((row) =>
        b.v[h][0].map((_, d) =>
          row.reduce((sum, weight, j) => sum + weight * b.v[h][j][d], 0),
        ),
      );
      const mix = a(
        `${head}.mixed`,
        "Weighted values",
        "activation",
        transpose(rawMix),
        53,
        y - 5,
        { kind: "mix", attention: att, value: v },
        h,
      );
      edge(att, mix);
      edge(v, mix);
      const effective = a(
        `${head}.gate`,
        runName === `L${l}H${h}` ? "Head disabled" : "Head output",
        "activation",
        transpose(b.headOutputs[h]),
        60,
        y + 4,
        { kind: "gate", input: mix, factor: runName === `L${l}H${h}` ? 0 : 1 },
        h,
      );
      edge(mix, effective);
      const wo = a(
        `${head}.wo`,
        `WO · head ${h + 1}`,
        "weight",
        transpose(matrix(`${w}.wo`).slice(h * m.dHead, (h + 1) * m.dHead)),
        59,
        y - 15,
        undefined,
        h,
        "Head slice of the shared output matrix",
      );
      views[views.length - 1].position[2] = -10;
      const write = a(
        `${head}.write`,
        "Write to residual",
        "activation",
        transpose(b.projectedHeads[h]),
        70,
        y,
        { kind: "matmul", input: effective, weight: wo },
        h,
      );
      edge(effective, write);
      edge(wo, write, "weight");
      headIds.push(write);
    }
    const delta = a(
      "delta",
      "Heads add",
      "activation",
      transpose(b.attentionOutput),
      79,
      0,
      { kind: "sum", inputs: headIds },
    );
    headIds.forEach((id) => edge(id, delta));
    const res = a(
      "residual1",
      "Residual + attention",
      "residual",
      transpose(b.residual1),
      91,
      0,
      { kind: "sum", inputs: [input, delta] },
    );
    edge(delta, res);
    edge(input, res, "residual", [
      [x, 38, 0],
      [x + 91, 38, 0],
    ]);
    const n2 = a(
      "norm2",
      "Normalize",
      "normalization",
      transpose(b.norm2),
      103,
      0,
      {
        kind: "norm",
        input: res,
        gamma: vector(`${w}.ln2.gamma`),
        beta: vector(`${w}.ln2.beta`),
        epsilon: m.layerNormEpsilon,
      },
    );
    edge(res, n2);
    const w1 = a(
      "w1",
      "MLP expansion",
      "weight",
      transpose(matrix(`${w}.w1`)),
      111,
      -24,
      undefined,
      undefined,
      "16 → 32 coordinates",
    );
    views[views.length - 1].position[2] = -10;
    const up = a("up", "Expand", "activation", transpose(b.mlpUp), 116, 0, {
      kind: "matmul",
      input: n2,
      weight: w1,
      bias: vector(`${w}.b1`),
    });
    edge(n2, up);
    edge(w1, up, "weight");
    const act = a("gelu", "GELU", "activation", transpose(b.mlpAct), 129, 0, {
      kind: "gelu",
      input: up,
    });
    edge(up, act);
    const w2 = a(
      "w2",
      "MLP contraction",
      "weight",
      transpose(matrix(`${w}.w2`)),
      137,
      -24,
      undefined,
      undefined,
      "32 → 16 coordinates",
    );
    views[views.length - 1].position[2] = -10;
    const down = a(
      "down",
      "Contract",
      "activation",
      transpose(b.mlpDown),
      142,
      0,
      { kind: "matmul", input: act, weight: w2, bias: vector(`${w}.b2`) },
    );
    edge(act, down);
    edge(w2, down, "weight");
    const out = a(
      "output",
      "Residual + MLP",
      "residual",
      transpose(b.output),
      155,
      0,
      { kind: "sum", inputs: [res, down] },
    );
    edge(down, out);
    edge(res, out, "residual", [
      [x + 91, 28, 0],
      [x + 155, 28, 0],
    ]);
  }
  const last = `b${m.nLayers - 1}.output`,
    end = m.nLayers * 164;
  const norm = add(
    "finalNorm",
    "Final norm",
    "normalization",
    transpose(run.finalNorm),
    [end + 4, 0, 0],
    -1,
    "output",
    {
      kind: "norm",
      input: last,
      gamma: vector("finalNorm.gamma"),
      beta: vector("finalNorm.beta"),
      epsilon: m.layerNormEpsilon,
    },
  );
  edge(last, norm);
  const weights = add(
    "unembedding",
    "Vocabulary projection",
    "weight",
    transpose(matrix("unembedding")),
    [end + 13, -19, -10],
    -1,
    "output",
  );
  const logits = add(
    "logits",
    "Vocabulary logits",
    "activation",
    transpose(run.logits),
    [end + 20, 0, 0],
    -1,
    "output",
    {
      kind: "matmul",
      input: norm,
      weight: weights,
      bias: vector("outputBias"),
    },
  );
  edge(norm, logits);
  edge(weights, logits, "weight");
  const probs = add(
    "probabilities",
    "Next token",
    "probabilities",
    transpose(run.probabilities),
    [end + 35, 0, 0],
    -1,
    "output",
    { kind: "softmax", input: logits },
  );
  edge(logits, probs);
  // Full stored tensors are the canonical parameter inventory. Head slices are views, never additional parameters.
  model.entities.push({
    id: "parameters",
    parentId: "model",
    kind: "ParameterVolume",
    label: "Stored parameters",
    evidence: "observed",
  });
  Object.entries(pkg.weights).forEach(([name, values], i) => {
    const isMatrix = Array.isArray(values[0]),
      data = isMatrix ? (values as number[][]) : [values as number[]];
    const id = add(
      `param.${name}`,
      name,
      "weight",
      data,
      [15 + (i % 6) * 30, -68 - Math.floor(i / 6) * 31, -4],
      -1,
      "parameters",
      undefined,
      undefined,
      "Exact checkpoint storage; row-vector convention",
    );
    const entity = model.entities.find((e) => e.id === id)!;
    entity.parentId = "parameters";
    entity.kind = "ParameterTensor";
    const tensor = model.tensors.find((t) => t.id === id)!;
    tensor.role = "parameter";
    tensor.shape = isMatrix ? [data.length, data[0].length] : [data[0].length];
    const view = views.at(-1)!;
    view.rowAxis = isMatrix ? "stored row" : "vector";
    view.colAxis = isMatrix ? "stored column" : "coordinate";
  });
  model.metadata = {
    parameterCount: m.parameterCount,
    storageConvention: "row vectors; displayed weight views are transposed",
    sourceModel: m,
  };
  return {
    model,
    views,
    connections,
    tokens: example.tokens,
    extent: end + 40,
    parameterCount: m.parameterCount,
  };
}
