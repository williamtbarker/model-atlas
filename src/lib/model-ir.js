/** ModelIR v1: containment, dataflow and parameter identity are separate. */
export const IR_VERSION = "1.0";
export const LIMITS = Object.freeze({
  entities: 200000,
  tensors: 100000,
  edges: 400000,
  jsonBytes: 64000000,
  headerBytes: 16000000,
  page: 128,
});
export const EVIDENCE = [
  "declared",
  "derived",
  "observed",
  "illustrative",
  "unavailable",
];

export function numel(shape) {
  if (!Array.isArray(shape)) throw new Error("Tensor shape must be an array.");
  let count = 1n;
  for (const n of shape) {
    if (!Number.isSafeInteger(n) || n < 0) return null;
    count *= BigInt(n);
  }
  return count;
}
export function countLabel(n) {
  if (n == null) return "Unknown";
  const v = typeof n === "bigint" ? n : BigInt(n);
  if (v >= 1000000000000n) return `${(Number(v) / 1e12).toFixed(2)}T`;
  if (v >= 1000000000n) return `${(Number(v) / 1e9).toFixed(2)}B`;
  if (v >= 1000000n) return `${(Number(v) / 1e6).toFixed(2)}M`;
  if (v >= 1000n) return `${(Number(v) / 1e3).toFixed(1)}K`;
  return v.toString();
}
export const shapeLabel = (shape) =>
  shape?.length ? shape.join(" × ") : shape ? "scalar" : "—";

function boundedText(value, label, max = 2000) {
  if (typeof value !== "string" || !value.length || value.length > max)
    throw new Error(`Invalid ${label}.`);
}

export function validateIR(ir) {
  if (!ir || ir.version !== IR_VERSION)
    throw new Error("Expected ModelIR version 1.0.");
  boundedText(ir.name, "model name");
  if (!ir.source || typeof ir.source !== "object")
    throw new Error("Missing source provenance.");
  boundedText(ir.source.kind, "source kind");
  if (!EVIDENCE.includes(ir.source.evidence))
    throw new Error("Invalid source evidence.");
  if (
    ir.source.urls != null &&
    (!Array.isArray(ir.source.urls) ||
      ir.source.urls.some((u) => typeof u !== "string"))
  )
    throw new Error("Invalid source URLs.");
  if (
    !Array.isArray(ir.entities) ||
    !ir.entities.length ||
    ir.entities.length > LIMITS.entities
  )
    throw new Error("Invalid entity list or entity budget exceeded.");
  if (!Array.isArray(ir.tensors) || ir.tensors.length > LIMITS.tensors)
    throw new Error("Invalid tensor list or tensor budget exceeded.");
  if (!Array.isArray(ir.edges) || ir.edges.length > LIMITS.edges)
    throw new Error("Invalid edge list or edge budget exceeded.");
  const entities = new Map(),
    tensors = new Map();
  for (const e of ir.entities) {
    boundedText(e.id, "entity ID");
    boundedText(e.kind, "entity kind");
    boundedText(e.label, "entity label");
    if (entities.has(e.id)) throw new Error(`Duplicate entity: ${e.id}`);
    if (!EVIDENCE.includes(e.evidence))
      throw new Error(`Missing or invalid evidence: ${e.id}`);
    if (
      e.tensorRefs != null &&
      (!Array.isArray(e.tensorRefs) ||
        e.tensorRefs.some((r) => typeof r !== "string"))
    )
      throw new Error(`Invalid tensor references: ${e.id}`);
    entities.set(e.id, e);
  }
  const root = entities.get(ir.rootId);
  if (!root || root.parentId != null)
    throw new Error("Missing root or invalid root parent.");
  for (const t of ir.tensors) {
    boundedText(t.id, "tensor ID");
    boundedText(t.dtype, "tensor dtype");
    if (!["parameter", "buffer", "activation", "unknown"].includes(t.role))
      throw new Error(`Invalid tensor role: ${t.id}`);
    if (tensors.has(t.id)) throw new Error(`Duplicate tensor: ${t.id}`);
    if (
      !Array.isArray(t.shape) ||
      t.shape.length > 32 ||
      t.shape.some(
        (d) =>
          !(Number.isSafeInteger(d) && d >= 0) &&
          !(typeof d === "string" && d.length > 0 && d.length < 64),
      )
    )
      throw new Error(`Invalid shape: ${t.id}`);
    if (!EVIDENCE.includes(t.evidence))
      throw new Error(`Invalid tensor evidence: ${t.id}`);
    const size = numel(t.shape);
    if (
      t.values != null &&
      (!Array.isArray(t.values) ||
        size == null ||
        BigInt(t.values.length) !== size ||
        t.values.length > 1000000 ||
        t.values.some(
          (v) => v !== null && (typeof v !== "number" || !Number.isFinite(v)),
        ))
    )
      throw new Error(`Invalid inline values: ${t.id}`);
    tensors.set(t.id, t);
  }
  const done = new Set([ir.rootId]);
  for (const e of ir.entities) {
    if (e.id !== ir.rootId && !entities.has(e.parentId))
      throw new Error(`Missing parent: ${e.id}`);
    const path = new Set();
    let cursor = e;
    while (!done.has(cursor.id)) {
      if (path.has(cursor.id)) throw new Error("Containment cycle.");
      path.add(cursor.id);
      cursor = entities.get(cursor.parentId);
      if (!cursor) throw new Error("Disconnected containment tree.");
    }
    for (const id of path) done.add(id);
    for (const ref of e.tensorRefs ?? [])
      if (!tensors.has(ref))
        throw new Error(`Missing tensor reference: ${ref}`);
    if (e.tensorId && !tensors.has(e.tensorId))
      throw new Error(`Missing tensor: ${e.tensorId}`);
  }
  const edgeIds = new Set();
  for (const edge of ir.edges) {
    boundedText(edge.id, "edge ID");
    if (edgeIds.has(edge.id)) throw new Error(`Duplicate edge: ${edge.id}`);
    edgeIds.add(edge.id);
    if (!entities.has(edge.from) || !entities.has(edge.to))
      throw new Error(`Dangling edge: ${edge.id}`);
    if (
      !["dataflow", "residual", "parameter_share", "view"].includes(edge.kind)
    )
      throw new Error(`Invalid edge kind: ${edge.id}`);
    if (!EVIDENCE.includes(edge.evidence))
      throw new Error(`Invalid edge evidence: ${edge.id}`);
  }
  return ir;
}

export class ModelIndex {
  constructor(ir) {
    validateIR(ir);
    this.ir = ir;
    this.entities = new Map(ir.entities.map((e) => [e.id, e]));
    this.tensors = new Map(ir.tensors.map((t) => [t.id, t]));
    this.children = new Map();
    this.incident = new Map();
    this.counts = new Map();
    for (const e of ir.entities) {
      if (!this.children.has(e.parentId)) this.children.set(e.parentId, []);
      this.children.get(e.parentId).push(e);
    }
    for (const edge of ir.edges)
      for (const id of [edge.from, edge.to]) {
        if (!this.incident.has(id)) this.incident.set(id, []);
        this.incident.get(id).push(edge);
      }
  }
  childList(id) {
    return this.children.get(id) ?? [];
  }
  ancestors(id) {
    const path = [];
    let e = this.entities.get(id);
    while (e) {
      path.unshift(e);
      e = this.entities.get(e.parentId);
    }
    return path;
  }
  tensorIds(id) {
    const ids = new Set(),
      pending = [id];
    while (pending.length) {
      const e = this.entities.get(pending.pop());
      if (e.tensorId) ids.add(e.tensorId);
      for (const ref of e.tensorRefs ?? []) ids.add(ref);
      for (const child of this.childList(e.id)) pending.push(child.id);
    }
    return ids;
  }
  parameterCount(id = this.ir.rootId) {
    if (this.counts.has(id)) return this.counts.get(id);
    let count = 0n;
    for (const tid of this.tensorIds(id)) {
      const t = this.tensors.get(tid);
      if (t.role === "unknown") return null;
      if (t.role === "parameter") {
        const n = numel(t.shape);
        if (n == null) return null;
        count += n;
      }
    }
    this.counts.set(id, count);
    return count;
  }
  page(id, page = 0) {
    const children = this.childList(id);
    return {
      items: children.slice(page * LIMITS.page, (page + 1) * LIMITS.page),
      total: children.length,
      pages: Math.max(1, Math.ceil(children.length / LIMITS.page)),
    };
  }
}

/** Independent layout. Edges alone establish ranks; cycles use a stable grid. */
export function layoutScope(items, edges) {
  const set = new Set(items.map((e) => e.id)),
    rank = new Map(items.map((e) => [e.id, 0]));
  const deps = edges.filter(
    (e) =>
      set.has(e.from) &&
      set.has(e.to) &&
      ["dataflow", "residual"].includes(e.kind),
  );
  const indegree = new Map(items.map((e) => [e.id, 0])),
    successors = new Map();
  for (const e of deps) {
    indegree.set(e.to, indegree.get(e.to) + 1);
    if (!successors.has(e.from)) successors.set(e.from, []);
    successors.get(e.from).push(e.to);
  }
  const queue = items.filter((e) => !indegree.get(e.id)).map((e) => e.id);
  let processed = 0;
  for (let i = 0; i < queue.length; i++) {
    const id = queue[i];
    processed++;
    for (const to of successors.get(id) ?? []) {
      rank.set(to, Math.max(rank.get(to), rank.get(id) + 1));
      indegree.set(to, indegree.get(to) - 1);
      if (!indegree.get(to)) queue.push(to);
    }
  }
  const positions = new Map();
  if (deps.length && processed === items.length) {
    const lanes = new Map();
    for (const e of items) {
      const r = rank.get(e.id);
      if (!lanes.has(r)) lanes.set(r, []);
      lanes.get(r).push(e);
    }
    const maxRank = Math.max(...rank.values());
    for (const [r, list] of lanes)
      list.forEach((e, i) =>
        positions.set(e.id, [
          (i - (list.length - 1) / 2) * 7,
          0,
          (r - maxRank / 2) * 5,
        ]),
      );
  } else {
    const cols = Math.min(8, Math.max(1, Math.ceil(Math.sqrt(items.length))));
    const rows = Math.ceil(items.length / cols);
    items.forEach((e, i) =>
      positions.set(e.id, [
        ((i % cols) - (cols - 1) / 2) * 7,
        0,
        (Math.floor(i / cols) - (rows - 1) / 2) * 5,
      ]),
    );
  }
  return positions;
}
