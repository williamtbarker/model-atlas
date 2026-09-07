import "./style.css";
import type { Entity, ModelIR, Tensor, Tile } from "./types";
import {
  ModelIndex,
  LIMITS,
  countLabel,
  shapeLabel,
  numel,
  validateIR,
} from "./lib/model-ir.js";
import { AtlasRenderer } from "./lib/renderer";
import { colorHex, primitiveFor } from "./lib/primitives";
import { TensorReader } from "./lib/tensors";
import { computedExample } from "./lib/math.js";

const $ = <T extends HTMLElement = HTMLElement>(id: string) =>
  document.getElementById(id) as T;
const text = (id: string, value: unknown) => {
  $(id).textContent = String(value ?? "—");
};
const el = <K extends keyof HTMLElementTagNameMap>(
  tag: K,
  value?: string,
  className?: string,
) => {
  const e = document.createElement(tag);
  if (value !== undefined) e.textContent = value;
  if (className) e.className = className;
  return e;
};
const button = (label: string, fn: () => void, className?: string) => {
  const e = el("button", label, className);
  e.addEventListener("click", fn);
  return e;
};
const friendly = (s: string) =>
  s.replace(/([a-z])([A-Z])/g, "$1 $2").replaceAll("_", " ");
const exact = (n: bigint | null) =>
  n == null ? "Unknown" : n.toLocaleString("en-US");
let index: ModelIndex,
  scope: string,
  selection: string,
  page = 0,
  loadGeneration = 0,
  tileGeneration = 0,
  currentTensor: Tensor | null = null,
  origin: number[] = [],
  renderer: AtlasRenderer | null = null;
const reader = new TensorReader();
let toastTimer: ReturnType<typeof setTimeout>;
function toast(message: string) {
  text("toast", message);
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ($("toast").hidden = true), 6500);
}

try {
  renderer = new AtlasRenderer($("scene"), {
    select,
    enter,
    back,
    stats: (s) => text("render-stats", s),
  });
} catch (error) {
  toast(
    "WebGL2 is unavailable. Use the component index and tensor inspector, or try a current browser with hardware acceleration.",
  );
  text("render-stats", "Graphics unavailable · index navigation available");
}

async function readJSON(blob: Blob, gzip = false): Promise<ModelIR> {
  let stream: ReadableStream<Uint8Array> = blob.stream();
  if (gzip)
    stream = (blob.stream() as ReadableStream<BufferSource>).pipeThrough(
      new DecompressionStream("gzip"),
    );
  const streamReader = stream.getReader(),
    chunks: Uint8Array[] = [];
  let size = 0;
  while (true) {
    const result = await streamReader.read();
    if (result.done) break;
    size += result.value.byteLength;
    if (size > 128000000) {
      await streamReader.cancel();
      throw new Error("Uncompressed package exceeds the 128 MB budget.");
    }
    chunks.push(result.value);
  }
  const all = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    all.set(chunk, offset);
    offset += chunk.length;
  }
  return validateIR(JSON.parse(new TextDecoder().decode(all))) as ModelIR;
}
function displayIR(ir: ModelIR) {
  index = new ModelIndex(ir);
  scope = (ir.metadata?.initialScope as string) || ir.rootId;
  if (!index.entities.has(scope)) scope = ir.rootId;
  selection = scope;
  page = 0;
  reader.reset();
  $<HTMLInputElement>("search").value = "";
  text("model-title", ir.name);
  text("model-description", ir.metadata?.architecture ?? ir.source.kind);
  text("params", countLabel(index.parameterCount()));
  text("layers", ir.metadata?.layers ?? "—");
  text("evidence", `${ir.source.evidence.toUpperCase()} STRUCTURE`);
  showScope();
}
async function loadBundled(key: string) {
  const generation = ++loadGeneration;
  $("loading").hidden = false;
  text("load-message", "Reading a compact architecture package…");
  try {
    const response = await fetch(`/models/${key}.atlas.json.gz`);
    if (!response.ok)
      throw new Error("Architecture package could not be loaded.");
    const ir = await readJSON(await response.blob(), true);
    if (generation !== loadGeneration) return;
    displayIR(ir);
  } catch (error) {
    if (generation === loadGeneration) toast((error as Error).message);
  } finally {
    if (generation === loadGeneration) $("loading").hidden = true;
  }
}
function showScope() {
  if (!index) return;
  const current = index.entities.get(scope) as Entity,
    p = index.page(scope, page),
    ids = new Set(p.items.map((e) => e.id));
  const edges = index.ir.edges.filter((e) => ids.has(e.from) && ids.has(e.to));
  renderer?.show(p.items as Entity[], edges, index, scope !== index.ir.rootId);
  text("scope-kind", friendly(current.kind).toUpperCase());
  text("scope-title", current.label);
  text(
    "scope-caption",
    p.total
      ? `${p.total.toLocaleString()} direct components · Select to inspect. Double-click to enter.`
      : "Leaf component · exact shape and coordinate inspection at right.",
  );
  $("back").toggleAttribute("disabled", scope === index.ir.rootId);
  const crumbs = $("breadcrumbs");
  crumbs.replaceChildren();
  for (const [i, e] of index.ancestors(scope).entries()) {
    if (i) crumbs.append(el("span", "/"));
    crumbs.append(button(e.label, () => navigate(e.id)));
  }
  renderList();
  select(scope);
}
function renderList() {
  if (!index) return;
  const query = $<HTMLInputElement>("search").value.trim().toLowerCase(),
    tree = $("tree");
  tree.replaceChildren();
  let items: Entity[] = [];
  if (query) {
    for (const e of index.ir.entities) {
      if (
        e.label.toLowerCase().includes(query) ||
        e.id.toLowerCase().includes(query)
      ) {
        items.push(e as Entity);
        if (items.length >= 80) break;
      }
    }
    text("list-title", "Search results");
    text("list-count", items.length === 80 ? "first 80" : items.length);
    $("pagination").hidden = true;
  } else {
    const p = index.page(scope, page);
    items = p.items as Entity[];
    text("list-title", "Components");
    text("list-count", p.total.toLocaleString());
    $("pagination").hidden = p.pages === 1;
    text("page-label", `${page + 1} / ${p.pages}`);
    $("prev-page").toggleAttribute("disabled", page === 0);
    $("next-page").toggleAttribute("disabled", page >= p.pages - 1);
  }
  if (!items.length) {
    tree.append(
      el(
        "p",
        query ? "No matching components." : "No child components.",
        "muted",
      ),
    );
    return;
  }
  for (const e of items) {
    const b = button(
      "",
      () => {
        if (query) {
          const parent = e.parentId ?? index.ir.rootId;
          navigate(parent);
          const all = index.childList(parent),
            ix = all.findIndex((x) => x.id === e.id);
          if (ix >= LIMITS.page) {
            page = Math.floor(ix / LIMITS.page);
            showScope();
          }
        }
        select(e.id);
      },
      "tree-item",
    );
    b.style.setProperty("--entity-color", colorHex(e));
    b.dataset.id = e.id;
    b.title = e.id;
    const dot = el("span", undefined, "dot");
    dot.setAttribute("aria-hidden", "true");
    b.append(dot, el("span", e.label, "name"));
    const children = index.childList(e.id).length;
    b.append(el("small", children ? `${children} ›` : e.tensorId ? "▦" : "·"));
    b.addEventListener("dblclick", () => enter(e.id));
    tree.append(b);
  }
}
function navigate(id: string) {
  if (!index.entities.has(id)) return;
  scope = id;
  page = 0;
  $<HTMLInputElement>("search").value = "";
  showScope();
}
function enter(id: string = selection) {
  const e = index?.entities.get(id);
  if (!e) return;
  if (index.childList(id).length) navigate(id);
  else if (e.tensorId) {
    select(id);
    $("coordinates").focus();
  } else
    toast(
      "This is the deepest documented operation. Its attributes show the available evidence.",
    );
}
function back() {
  const parent = index?.entities.get(scope)?.parentId;
  if (parent) navigate(parent);
}
function metric(label: string, value: string) {
  const row = el("div", undefined, "detail-metric");
  row.append(el("span", label), el("strong", value));
  return row;
}
function select(id: string) {
  const e = index?.entities.get(id) as Entity;
  if (!e) return;
  selection = id;
  renderer?.select(id);
  tileGeneration++;
  for (const item of document.querySelectorAll<HTMLElement>(".tree-item"))
    item.classList.toggle("active", item.dataset.id === id);
  text("selected-kind", friendly(e.kind));
  text("selected-title", e.label);
  text("selected-id", e.id);
  const children = index.childList(id);
  $("enter").toggleAttribute("disabled", !children.length && !e.tensorId);
  text(
    "enter",
    children.length
      ? "Enter component ↗"
      : e.tensorId
        ? "Inspect tensor ↗"
        : "Leaf operation",
  );
  $("focus").toggleAttribute("disabled", !renderer?.positions.has(id));
  const details = $("selected-details");
  details.replaceChildren();
  const count = index.parameterCount(id);
  details.append(metric("Unique parameter count", exact(count)));
  details.append(
    metric("Evidence", friendly(e.evidence)),
    metric("Direct components", children.length.toLocaleString()),
  );
  if (e.attrs?.formula)
    details.append(el("p", String(e.attrs.formula), "formula"));
  const attrs = el("dl", undefined, "attributes");
  for (const [key, value] of Object.entries(e.attrs ?? {})) {
    if (key === "formula") continue;
    const row = el("div", undefined, "attribute");
    row.append(
      el("dt", friendly(key)),
      el(
        "dd",
        typeof value === "object" ? JSON.stringify(value) : String(value),
      ),
    );
    attrs.append(row);
  }
  details.append(attrs);
  currentTensor = e.tensorId ? (index.tensors.get(e.tensorId) as Tensor) : null;
  if (currentTensor) {
    details.append(
      metric("Logical shape", shapeLabel(currentTensor.shape)),
      metric("Data type", currentTensor.dtype),
      metric("Tensor elements", exact(numel(currentTensor.shape))),
    );
    const t = currentTensor as Tensor & { storage?: unknown };
    if (t.storage) {
      const row = el("div", undefined, "attribute");
      row.append(
        el("dt", "Checkpoint storage"),
        el("dd", JSON.stringify(t.storage)),
      );
      details.append(row);
    }
  }
  if (e.tensorRefs?.length) {
    const heading = el("p", "Shared parameter references", "muted");
    details.append(heading);
    for (const ref of e.tensorRefs) details.append(el("p", ref, "path"));
  }
  const connections = index.incident.get(id) ?? [];
  if (connections.length) {
    details.append(el("div", "DECLARED CONNECTIONS", "rule-label"));
    for (const edge of connections.slice(0, 24)) {
      const other = edge.from === id ? edge.to : edge.from;
      const otherEntity = index.entities.get(other);
      const row = el("div", undefined, "attribute");
      row.append(
        el(
          "dt",
          `${edge.from === id ? "To" : "From"} ${otherEntity?.label ?? other} · ${edge.kind}`,
        ),
        el("dd", edge.label ?? "Declared relation"),
      );
      details.append(row);
    }
  }
  $("tensor-inspector").hidden = !currentTensor;
  if (currentTensor) {
    origin = currentTensor.shape.map(() => 0);
    $<HTMLInputElement>("coordinates").value = origin.join(", ");
    loadTile();
  }
  const source = $("source-details");
  source.replaceChildren(
    el("p", primitiveFor(e).meaning),
    el("p", index.ir.source.notes ?? ""),
  );
  const urls = index.ir.source.urls ?? [];
  for (const [i, url] of urls.entries()) {
    try {
      const parsed = new URL(url);
      if (!["https:", "http:"].includes(parsed.protocol)) continue;
      const link = el(
        "a",
        i === 0 ? "Primary source ↗" : `Implementation / evidence ${i + 1} ↗`,
      );
      link.href = url;
      link.target = "_blank";
      link.rel = "noreferrer";
      source.append(link, el("br"));
    } catch {}
  }
}
async function loadTile() {
  if (!currentTensor) return;
  const generation = ++tileGeneration,
    tensor = currentTensor;
  text("tensor-note", "Reading coordinate tile…");
  $("tensor-grid").replaceChildren();
  try {
    const tile = await reader.tile(tensor, origin);
    if (generation !== tileGeneration) return;
    renderTile(tile, tensor);
  } catch (error) {
    if (generation === tileGeneration) {
      text("tensor-note", (error as Error).message);
      text("scalar", "Values unavailable for this shape.");
    }
  }
}
function renderTile(tile: Tile, tensor: Tensor) {
  text("tensor-note", tile.note);
  const grid = $("tensor-grid");
  grid.style.gridTemplateColumns = `repeat(${Math.max(1, tile.cols)},1fr)`;
  const finite = tile.values.filter(
    (v) => typeof v === "number" && Number.isFinite(v),
  ) as number[];
  const scale = Math.max(1e-10, ...finite.map(Math.abs));
  tile.values.forEach((value, i) => {
    const idx = tile.indices[i];
    const cell = button(value == null ? "·" : "", () => {
      for (const b of grid.children) b.classList.remove("cell-selected");
      cell.classList.add("cell-selected");
      showScalar(idx, value);
    });
    cell.title = `[${idx.join(", ")}] = ${value ?? "unavailable"}`;
    cell.setAttribute("aria-label", cell.title);
    if (typeof value === "number" && Number.isFinite(value)) {
      const intensity = 0.15 + (0.7 * Math.abs(value)) / scale;
      cell.style.background =
        value >= 0
          ? `rgba(75,174,180,${intensity})`
          : `rgba(212,145,99,${intensity})`;
    }
    grid.append(cell);
  });
  showScalar(tile.indices[0] ?? origin, tile.values[0] ?? null);
  $("tile-left").toggleAttribute(
    "disabled",
    !origin.length || origin.at(-1) === 0,
  );
  $("tile-right").toggleAttribute(
    "disabled",
    !origin.length ||
      typeof tensor.shape.at(-1) !== "number" ||
      origin.at(-1)! + 12 >= (tensor.shape.at(-1) as number),
  );
}
function showScalar(indices: number[], value: number | string | null) {
  const box = $("scalar");
  box.replaceChildren(
    el("span", `COORDINATE [${indices.join(", ")}]`),
    el(
      "strong",
      value == null ? "Unavailable — no value source" : String(value),
    ),
  );
}

$<HTMLSelectElement>("model").addEventListener("change", (e) => {
  const key = (e.target as HTMLSelectElement).value;
  if (key !== "custom") loadBundled(key);
});
$("root").onclick = () => navigate(index.ir.rootId);
$("back").onclick = back;
$("fit").onclick = () => renderer?.fit();
$("enter").onclick = () => enter();
$("focus").onclick = () => renderer?.focus(selection);
$("planar").onclick = () => {
  const value = $("planar").getAttribute("aria-pressed") !== "true";
  $("planar").setAttribute("aria-pressed", String(value));
  renderer?.setPlanar(value);
};
$("autozoom").onclick = () => {
  if (!renderer) return;
  renderer.autoZoom = !renderer.autoZoom;
  $("autozoom").setAttribute("aria-pressed", String(renderer.autoZoom));
};
$("prev-page").onclick = () => {
  page--;
  showScope();
};
$("next-page").onclick = () => {
  page++;
  showScope();
};
let searchTimer: ReturnType<typeof setTimeout>;
$("search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(renderList, 150);
});
$("coordinate-form").addEventListener("submit", (e) => {
  e.preventDefault();
  try {
    const s = $<HTMLInputElement>("coordinates").value.trim();
    origin = s
      ? s.split(",").map((x) => {
          if (!/^\s*\d+\s*$/.test(x))
            throw new Error("Use comma-separated nonnegative integers.");
          return Number(x.trim());
        })
      : [];
    loadTile();
  } catch (error) {
    toast((error as Error).message);
  }
});
$("tile-left").onclick = () => {
  origin[origin.length - 1] = Math.max(0, origin.at(-1)! - 12);
  $<HTMLInputElement>("coordinates").value = origin.join(", ");
  loadTile();
};
$("tile-right").onclick = () => {
  origin[origin.length - 1] += 12;
  $<HTMLInputElement>("coordinates").value = origin.join(", ");
  loadTile();
};
$("help").onclick = () => $<HTMLDialogElement>("help-dialog").showModal();
$("help-close").onclick = () => $<HTMLDialogElement>("help-dialog").close();
$("import").onclick = () => $("file").click();
$("file").addEventListener("change", async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  const generation = ++loadGeneration;
  $("loading").hidden = false;
  text("load-message", "Reading your local package…");
  try {
    if (file.name.endsWith(".safetensors")) {
      const pendingReader = new TensorReader();
      const tensors = await pendingReader.openSafetensors(file);
      const entities: Entity[] = [
          {
            id: "model",
            parentId: null,
            kind: "Model",
            label: file.name,
            evidence: "observed",
          },
        ],
        parents = new Set(["model"]);
      for (const t of tensors) {
        const parts = t.id.split(".");
        let parent = "model";
        for (let i = 0; i < parts.length - 1; i++) {
          const id = "module:" + parts.slice(0, i + 1).join(".");
          if (!parents.has(id)) {
            entities.push({
              id,
              parentId: parent,
              kind: "Module",
              label: parts[i],
              evidence: "observed",
            });
            parents.add(id);
          }
          parent = id;
        }
        entities.push({
          id: "tensor:" + t.id,
          parentId: parent,
          kind:
            t.shape.length === 0
              ? "ScalarCell"
              : t.shape.length === 1
                ? "VectorColumn"
                : t.shape.length === 2
                  ? "MatrixPlane"
                  : "TensorVolume",
          label: parts.at(-1)!,
          evidence: "observed",
          tensorId: t.id,
        });
      }
      const ir: ModelIR = {
        version: "1.0",
        name: file.name,
        rootId: "model",
        entities,
        tensors,
        edges: [],
        source: {
          kind: "local tensor catalog",
          evidence: "observed",
          notes:
            "Names, shapes and storage are observed. Prefix grouping is not proof of module type or computation order. Parameter count is unknown because buffers cannot be distinguished from parameters using headers alone.",
        },
      };
      if (generation !== loadGeneration) return;
      displayIR(ir);
      reader.attach(pendingReader);
      select(selection);
    } else {
      if (file.size > LIMITS.jsonBytes)
        throw new Error(
          "Input file exceeds the 64 MB compressed/input budget.",
        );
      const ir = await readJSON(file, /\.(gz|atlas)$/.test(file.name));
      if (generation !== loadGeneration) return;
      displayIR(ir);
    }
    const selectModel = $<HTMLSelectElement>("model");
    let custom = selectModel.querySelector("option[value=custom]");
    if (!custom) {
      custom = el("option", "Imported package");
      (custom as HTMLOptionElement).value = "custom";
      selectModel.append(custom);
    }
    selectModel.value = "custom";
    toast("Package opened locally. No file was uploaded.");
  } catch (error) {
    toast((error as Error).message);
  } finally {
    if (generation === loadGeneration) $("loading").hidden = true;
    (e.target as HTMLInputElement).value = "";
  }
});
$("export").onclick = () => {
  if (!index) return;
  const blob = new Blob([JSON.stringify(index.ir)], {
      type: "application/json",
    }),
    url = URL.createObjectURL(blob),
    link = el("a");
  link.href = url;
  link.download = "model-atlas.modelir.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.target as HTMLElement).closest("button,a")) return;
  if (
    (e.target as HTMLElement).matches("input,textarea,select") ||
    document.querySelector("dialog[open]")
  )
    return;
  if (e.key === "f" || e.key === "F") {
    e.preventDefault();
    renderer?.fit();
  }
  if (e.key === "Enter") {
    e.preventDefault();
    enter();
  }
  if (e.key === "Backspace") {
    e.preventDefault();
    back();
  }
});

// Independently computed, fully inspectable arithmetic. No decorative flow pulses.
const example = computedExample();
let operation = "projection",
  step = 0,
  playing: ReturnType<typeof setInterval> | null = null;
example.tokens.forEach((word, i) => {
  $<HTMLSelectElement>("lab-token").append(
    new Option(`${i} · ${word}`, String(i)),
  );
});
$<HTMLSelectElement>("lab-token").value = "3";
const fmt = (n: number) => (Number.isFinite(n) ? n.toFixed(5) : "masked");
function stop() {
  if (playing) clearInterval(playing);
  playing = null;
  text("play", "▶ Play");
}
function labRender() {
  (document.querySelector(".lab-controls") as HTMLElement).hidden = operation === "router";
  const token = Number($<HTMLSelectElement>("lab-token").value),
    dim = Number($<HTMLSelectElement>("lab-dim").value),
    visual = $("lab-visual"),
    explanation = $("lab-explanation");
  visual.replaceChildren();
  explanation.replaceChildren();
  const max =
    operation === "projection"
      ? 3
      : operation === "attention"
        ? token + 1
        : operation === "residual"
          ? 2
          : 6;
  step = Math.min(step, max);
  $<HTMLInputElement>("scrub").max = String(max);
  $<HTMLInputElement>("scrub").value = String(step);
  text("step-count", `${step} / ${max}`);
  const table = el("table", undefined, "calc-table");
  const head = el("tr");
  let rows: { a: number; b: number; label: string }[] = [],
    title = "",
    formula = "",
    note = "";
  if (operation === "projection") {
    title = "A projection is a sum of products";
    formula = `Q[${token}, ${dim}] = Σ X[${token}, i] × W_Q[${dim}, i]`;
    rows = example.x[token].map((a, i) => ({
      a,
      b: example.wq[dim][i],
      label: `i = ${i}`,
    }));
    note =
      "Each connection contributes an input coordinate multiplied by a learned coefficient. Here the coefficients are hand-chosen. Positive and negative contributions accumulate numerically; they are not represented as color blending.";
  } else if (operation === "attention") {
    title = "Attention mixes value vectors";
    formula = `O[${token}, ${dim}] = Σ A[${token}, key] × V[key, ${dim}]`;
    rows = example.weights[token]
      .slice(0, token + 1)
      .map((a, j) => ({ a, b: example.v[j][dim], label: `key ${j}` }));
    const grid = el("div", undefined, "attention-matrix");
    grid.append(el("span", "Q ↓ / K →", "axis"));
    for (let j = 0; j < 4; j++) grid.append(el("span", String(j), "axis"));
    for (let i = 0; i < 4; i++) {
      grid.append(el("span", String(i), "axis"));
      for (let j = 0; j < 4; j++) {
        const cell = el(
          "span",
          j > i ? "×" : example.weights[i][j].toFixed(3),
          j > i ? "masked" : i === token ? "active" : "",
        );
        if (j <= i)
          cell.style.background = `rgba(87,170,187,${0.12 + 0.8 * example.weights[i][j]})`;
        grid.append(cell);
      }
    }
    visual.append(grid);
    note = `Scores are Q·K / √2, followed by the causal mask and a softmax over keys. Row ${token} sums to ${example.weights[token].reduce((a, b) => a + b, 0).toFixed(6)}. A masked cell is unavailable to this query; it is shown with a hatch, separately from zero.`;
  } else if (operation === "residual") {
    title = "Residual addition preserves the coordinate";
    formula = `Y[${token}, ${dim}] = X[${token}, ${dim}] + O[${token}, ${dim}]`;
    rows = [
      { a: example.x[token][dim], b: 1, label: "input" },
      { a: example.output[token][dim], b: 1, label: "update" },
    ];
    note =
      "The two values at the same coordinate are added. This miniature illustrates ordinary residual addition. Multi-stream hyper-connections in some architectures perform additional learned mixing and need a different trace.";
  } else {
    title = "Routing chooses and weights experts";
    formula = "softmax(logits) → top 2 → renormalize selected probabilities";
    for (let i = 0; i < 6; i++) {
      const chosen = example.router.selected.find((e) => e.id === i);
      const row = el("div", undefined, "contribution");
      row.append(el("span", `E${i}`));
      const track = el("div", undefined, "bar-track"),
        bar = el("div", undefined, "bar");
      bar.style.width = `${step > i ? example.router.probabilities[i] * 100 : 0}%`;
      if (chosen) bar.style.background = "#d7b47c";
      track.append(bar);
      row.append(track, el("span", example.router.probabilities[i].toFixed(3)));
      visual.append(row);
    }
    note =
      "This six-expert example uses softmax routing. The gold bars are the two selected experts. It does not describe hash routing or sqrt-softplus routing in the hero model.";
    explanation.append(
      el("h3", title),
      el("p", formula, "formula"),
      el("p", note),
    );
    for (const chosen of example.router.selected)
      explanation.append(
        metric(`Expert ${chosen.id} · selected weight`, fmt(chosen.weight)),
      );
    explanation.append(
      el(
        "p",
        `Selected weights sum to ${example.router.selected.reduce((s, x) => s + x.weight, 0).toFixed(6)}.`,
      ),
    );
    return;
  }
  ["Term", "Operand", "Multiplier", "Product"].forEach((s) =>
    head.append(el("th", s)),
  );
  table.append(head);
  rows.forEach((row, i) => {
    const tr = el("tr", undefined, i < step ? "done" : "pending");
    [row.label, fmt(row.a), fmt(row.b), fmt(row.a * row.b)].forEach((s) =>
      tr.append(el("td", s)),
    );
    table.append(tr);
  });
  if (operation !== "attention") visual.append(table);
  else explanation.append(table);
  const sum = rows.slice(0, step).reduce((s, r) => s + r.a * r.b, 0),
    total = rows.reduce((s, r) => s + r.a * r.b, 0);
  explanation.prepend(el("h3", title), el("p", formula, "formula"));
  explanation.append(
    el("p", `Accumulated: ${fmt(sum)}`, "calc-total"),
    el(
      "p",
      `Complete result: ${fmt(total)}. ${step} of ${rows.length} contributions included.`,
    ),
    el("p", note),
  );
}
$("lab-open").onclick = () => {
  $<HTMLDialogElement>("lab").showModal();
  labRender();
};
$("lab-close").onclick = () => {
  stop();
  $<HTMLDialogElement>("lab").close();
};
$("lab").addEventListener("close", stop);
for (const tab of document.querySelectorAll<HTMLButtonElement>("[data-op]"))
  tab.onclick = () => {
    stop();
    operation = tab.dataset.op!;
    step = 0;
    for (const other of document.querySelectorAll("[data-op]"))
      other.setAttribute("aria-selected", String(other === tab));
    labRender();
  };
$("lab-token").onchange = $("lab-dim").onchange = () => {
  stop();
  step = 0;
  labRender();
};
$("scrub").addEventListener("input", () => {
  stop();
  step = Number($<HTMLInputElement>("scrub").value);
  labRender();
});
$("step").onclick = () => {
  stop();
  step = Math.min(step + 1, Number($<HTMLInputElement>("scrub").max));
  labRender();
};
$("play").onclick = () => {
  if (playing) {
    stop();
    return;
  }
  if (step >= Number($<HTMLInputElement>("scrub").max)) step = 0;
  playing = setInterval(() => {
    step++;
    labRender();
    if (step >= Number($<HTMLInputElement>("scrub").max)) stop();
  }, 650);
  text("play", "Ⅱ Pause");
};
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stop();
});
loadBundled("deepseek4");
