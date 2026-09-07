import type { Arithmetic, SceneIR, Selection, TensorView, Term } from "./types";

export function explain(scene: SceneIR, s: Selection): Arithmetic {
  const byId = new Map(scene.views.map((v) => [v.id, v])),
    v = byId.get(s.viewId)!;
  const { row: r, col: c } = s,
    result = v.values[r]?.[c] ?? 0,
    op = v.op;
  const get = (id: string) => byId.get(id)!;
  const at = (id: string, row: number, col: number) =>
    get(id).values[row][col] ?? 0;
  const base = {
    title: v.label,
    equation: "",
    terms: [] as Term[],
    result,
    note: "Recorded float64 value. Displayed decimal values are rounded.",
    mode: "value" as Arithmetic["mode"],
  };
  if (!op) return base;
  if (op.kind === "matmul") {
    const input = get(op.input),
      terms = input.values.map((_, i) => ({
        label: `${i}`,
        left: at(op.input, i, c),
        right: at(op.weight, r, i),
        value: at(op.input, i, c) * at(op.weight, r, i),
        source: { viewId: op.input, row: i, col: c },
        second: { viewId: op.weight, row: r, col: i },
      }));
    return {
      ...base,
      title: `${v.label} · coordinate ${r}`,
      equation: `y[${r}] = Σ x[k] W[k,${r}]${op.bias ? " + bias" : ""}`,
      terms,
      bias: op.bias?.[r] ?? 0,
      mode: "sum",
      note: "All products are shown. Weight planes display the transpose of row-vector checkpoint storage.",
    };
  }
  if (op.kind === "attention") {
    if (c > r)
      return {
        ...base,
        title: `Query ${r} ← key ${c}`,
        equation: "Future position: masked before softmax",
        terms: [],
        mode: "value",
        note: "The raw score is masked to −∞; the resulting attention probability is exactly zero. No contribution can travel along this route.",
      };
    const q = get(op.q),
      terms = q.values.map((_, i) => ({
        label: `${i}`,
        left: at(op.q, i, r),
        right: at(op.k, i, c),
        value: at(op.q, i, r) * at(op.k, i, c),
        source: { viewId: op.q, row: i, col: r },
        second: { viewId: op.k, row: i, col: c },
      }));
    return {
      ...base,
      title: `Query ${r} ← key ${c}`,
      equation: `score = q[${r}] · k[${c}] / √${q.values.length}`,
      terms,
      divide: op.scale,
      mode: "sum",
      note: `Products accumulate, then divide by ${op.scale.toFixed(4)}. The separate softmax plane compares the allowed keys in this query row.`,
    };
  }
  if (op.kind === "rowSoftmax") {
    if (c > r)
      return {
        ...base,
        title: `Query ${r} ← key ${c}`,
        equation: "Masked source: attention = 0",
        terms: [],
        mode: "value",
        note: "Future positions have zero probability and do not enter the denominator.",
      };
    const scores = get(op.input).values[r],
      max = Math.max(...(scores.filter((x) => x !== null) as number[])),
      exps = scores.map((x) => (x === null ? 0 : Math.exp(x - max))),
      denom = exps.reduce((s, v) => s + v, 0);
    return {
      ...base,
      title: `Softmax · query ${r}`,
      equation: "A[j] = exp(score[j] − max) / Σ exp(score − max)",
      terms: exps
        .slice(0, r + 1)
        .map((value, j) => ({
          label: scene.tokens[j] + " · " + j,
          left: value,
          right: 1 / denom,
          value: value / denom,
          source: { viewId: op.input, row: r, col: j },
        })),
      mode: "softmax",
      note: `Row maximum ${max.toFixed(5)} · denominator ${denom.toFixed(5)}. Each bar is a key's probability; the row sums to one.`,
    };
  }
  if (op.kind === "mix") {
    const a = get(op.attention),
      terms = a.values[c].map((w, j) => ({
        label: scene.tokens[j] + " · " + j,
        left: w ?? 0,
        right: at(op.value, r, j),
        value: (w ?? 0) * at(op.value, r, j),
        source: { viewId: op.value, row: r, col: j },
        second: { viewId: op.attention, row: c, col: j },
      }));
    return {
      ...base,
      title: `Value mix · coordinate ${r}`,
      equation: `z[${r}] = Σ A[${c},j] v[j,${r}]`,
      terms,
      mode: "sum",
      note: "Each value is scaled by its attention weight. Positive and negative products can cancel.",
    };
  }
  if (op.kind === "sum")
    return {
      ...base,
      title: `${v.label} · coordinate ${r}`,
      equation: "out = " + op.inputs.map((id) => get(id).label).join(" + "),
      terms: op.inputs.map((id) => ({
        label: get(id).label,
        left: at(id, r, c),
        value: at(id, r, c),
        source: { viewId: id, row: r, col: c },
      })),
      mode: "sum",
      note: "Aligned coordinates add. The residual bypass retains the unnormalized incoming state.",
    };
  if (op.kind === "norm") {
    const x = get(op.input).values.map((_, i) => at(op.input, i, c)),
      mean = x.reduce((a, b) => a + b, 0) / x.length,
      variance = x.reduce((a, b) => a + (b - mean) ** 2, 0) / x.length;
    return {
      ...base,
      title: `Normalize · coordinate ${r}`,
      equation: "y = γ (x − mean) / √(variance + ε) + β",
      terms: [
        {
          label: "input",
          left: x[r],
          value: x[r],
          source: { viewId: op.input, row: r, col: c },
        },
        { label: "centered", left: x[r] - mean, value: x[r] - mean },
        {
          label: "scaled",
          left: (x[r] - mean) / Math.sqrt(variance + op.epsilon),
          value: (x[r] - mean) / Math.sqrt(variance + op.epsilon),
        },
        { label: "learned affine", left: result, value: result },
      ],
      mode: "norm",
      note: `mean ${mean.toFixed(5)} · variance ${variance.toFixed(5)} · γ ${op.gamma[r].toFixed(5)} · β ${op.beta[r].toFixed(5)}`,
    };
  }
  if (op.kind === "gelu")
    return {
      ...base,
      title: `GELU · coordinate ${r}`,
      equation: "½x [1 + tanh(√(2/π)(x + 0.044715x³))]",
      terms: [
        {
          label: "before",
          left: at(op.input, r, c),
          value: at(op.input, r, c),
          source: { viewId: op.input, row: r, col: c },
        },
        { label: "after", left: result, value: result },
      ],
      mode: "gelu",
      note: "The exact tanh approximation used during training. Negative inputs are softened; this is not a binary gate.",
    };
  if (op.kind === "gate")
    return {
      ...base,
      title: op.factor ? "Head output" : "Head disabled",
      equation: op.factor ? "head output = A V" : "head output = 0 × (A V)",
      terms: [
        {
          label: "head gate",
          left: at(op.input, r, c),
          right: op.factor,
          value: result,
          source: { viewId: op.input, row: r, col: c },
        },
      ],
      mode: "sum",
      note: op.factor
        ? "The head is active. Its complete output feeds its slice of the shared output projection."
        : "Intervention: this head is zeroed at every token. Every subsequent operation was rerun offline.",
    };
  if (op.kind === "softmax")
    return {
      ...base,
      title: "Vocabulary softmax",
      equation: "p[i] = exp(logit[i] − max) / Σ exp(logit − max)",
      terms: get(op.input).values.map((_, i) => ({
        label: `${i}`,
        left: at(op.input, i, c),
        value: at(op.input, i, c),
        source: { viewId: op.input, row: i, col: c },
      })),
      mode: "softmax",
      note: "Only the final input position was supervised for this synthetic lookup task.",
    };
  if (op.kind === "embedding")
    return {
      ...base,
      equation: "residual = token embedding + position embedding",
      terms: [
        { label: "token", left: op.word[c][r], value: op.word[c][r] },
        {
          label: "position",
          left: op.position[c][r],
          value: op.position[c][r],
        },
      ],
      mode: "sum",
      note: "Learned token and position vectors. Coordinates are not semantic 3D directions.",
    };
  return base;
}

export function chooseCell(view: TensorView, token: number): Selection {
  if (view.kind === "attention") {
    const row = Math.min(token, view.values.length - 1),
      vals = view.values[row];
    let col = 0;
    vals.forEach((v, i) => {
      if ((v ?? 0) > (vals[col] ?? 0)) col = i;
    });
    return { viewId: view.id, row, col };
  }
  const col = Math.min(token, view.values[0].length - 1);
  let row = 0;
  view.values.forEach((values, i) => {
    if (Math.abs(values[col] ?? 0) > Math.abs(view.values[row][col] ?? 0))
      row = i;
  });
  return { viewId: view.id, row, col };
}
