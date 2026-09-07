/** Small, independently implemented operations for exact, inspectable demos.
 * These functions do not infer or execute an imported model.
 */
export const dot = (a, b) => {
  if (a.length !== b.length) throw new Error("Dot-product shape mismatch");
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
};
export function softmax(values) {
  const max = Math.max(...values);
  if (!Number.isFinite(max)) throw new Error("Softmax row has no finite entry");
  const exps = values.map((v) => Math.exp(v - max)),
    sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}
export function matvec(matrix, vector) {
  return matrix.map((row) => dot(row, vector));
}
export function attention(q, k, v, causal = true) {
  const dk = q[0].length;
  if (
    !dk ||
    q.some((x) => x.length !== dk) ||
    k.some((x) => x.length !== dk) ||
    k.length !== v.length
  )
    throw new Error("Attention shape mismatch");
  const scores = q.map((row, i) =>
    k.map((col, j) =>
      causal && j > i ? -Infinity : dot(row, col) / Math.sqrt(dk),
    ),
  );
  const weights = scores.map(softmax);
  const output = weights.map((row) =>
    v[0].map((_, d) => row.reduce((sum, w, j) => sum + w * v[j][d], 0)),
  );
  return { scores, weights, output };
}
export function route(logits, k) {
  if (!Number.isInteger(k) || k < 1 || k > logits.length)
    throw new Error("Invalid routing top-k");
  const probabilities = softmax(logits);
  const chosen = probabilities
    .map((p, id) => ({ id, p }))
    .sort((a, b) => b.p - a.p || a.id - b.id)
    .slice(0, k);
  const norm = chosen.reduce((sum, x) => sum + x.p, 0);
  return {
    probabilities,
    selected: chosen.map((x) => ({ ...x, weight: x.p / norm })),
  };
}
export function computedExample() {
  const tokens = ["The", "model", "mixes", "context"];
  const x = [
    [0.8, -0.2, 0.4],
    [0.1, 0.9, -0.3],
    [-0.5, 0.4, 0.7],
    [0.6, 0.3, -0.8],
  ];
  // Hand-chosen operands, real arithmetic. Not token embeddings from any LLM.
  const wq = [
    [0.6, -0.2, 0.4],
    [0.1, 0.7, -0.3],
  ];
  const wk = [
    [0.4, 0.3, -0.5],
    [-0.2, 0.8, 0.1],
  ];
  const wv = [
    [0.5, 0.2, 0.1],
    [-0.1, 0.3, 0.7],
    [0.4, -0.5, 0.2],
  ];
  const q = x.map((row) => matvec(wq, row)),
    k = x.map((row) => matvec(wk, row)),
    v = x.map((row) => matvec(wv, row));
  return {
    tokens,
    x,
    wq,
    wk,
    wv,
    q,
    k,
    v,
    ...attention(q, k, v),
    router: route([0.6, -0.2, 1.4, 0.1, 0.8, -0.7], 2),
  };
}
