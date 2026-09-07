import "./style.css";
import type { Entity, ModelIR, Tensor, Edge } from "../types";
import { countLabel, shapeLabel } from "../lib/model-ir.js";
import { compileLandscape, type Landscape, type BlockLayout } from "./layout";
import { LandscapeRenderer } from "./renderer";
const $ = <T extends HTMLElement = HTMLElement>(id: string) =>
  document.getElementById(id) as T;
let landscape: Landscape,
  renderer: LandscapeRenderer,
  selected: Entity | null = null,
  currentBlock: BlockLayout | null = null,
  currentMode = "model",
  loading = 0;
const replaceText = (id: string, text: string) => {
  $(id).textContent = text;
};
function fact(label: string, value: unknown) {
  if (value === undefined || value === null) return;
  const row = document.createElement("div");
  row.className = "fact";
  const a = document.createElement("span");
  a.textContent = label;
  const b = document.createElement("strong");
  b.textContent =
    typeof value === "number" ? value.toLocaleString() : String(value);
  row.append(a, b);
  $("selection-facts").append(row);
}
function action(label: string, cb: () => void) {
  const b = document.createElement("button");
  b.textContent = label;
  b.addEventListener("click", cb);
  $("selection-actions").append(b);
}
function children(id: string): Entity[] {
  return landscape.index.childList(id);
}
function nearestBlock(id: string) {
  const ids = new Set(landscape.index.ancestors(id).map((e: Entity) => e.id));
  return landscape.blocks.find((b) => ids.has(b.id)) ?? null;
}
function tensorFor(e: Entity): Tensor | undefined {
  if (e.tensorId) return landscape.index.tensors.get(e.tensorId);
  const child = children(e.id).find((e) => e.tensorId);
  return child ? landscape.index.tensors.get(child.tensorId!) : undefined;
}
function selectedTensor(t: Tensor) {
  const host = $("tensor-info");
  host.hidden = false;
  host.replaceChildren();
  const label = document.createElement("label");
  label.textContent = "Exact scalar coordinate";
  const input = document.createElement("input");
  input.placeholder = t.shape.map(() => 0).join(", ");
  input.value = t.shape.map(() => 0).join(", ");
  input.setAttribute("aria-label", "Tensor coordinate, comma separated");
  const result = document.createElement("code");
  const update = () => {
    const raw = input.value.trim();
    const indices = raw ? raw.split(",").map((s) => Number(s.trim())) : [];
    if (
      indices.length !== t.shape.length ||
      indices.some(
        (v, i) =>
          !Number.isSafeInteger(v) ||
          v < 0 ||
          typeof t.shape[i] !== "number" ||
          v >= Number(t.shape[i]),
      )
    ) {
      result.textContent = "Enter one valid index per dimension.";
      return;
    }
    let offset = 0n;
    for (let i = 0; i < indices.length; i++)
      offset = offset * BigInt(t.shape[i] as number) + BigInt(indices[i]);
    const value =
      offset <= BigInt(Number.MAX_SAFE_INTEGER)
        ? t.values?.[Number(offset)]
        : undefined;
    result.textContent = `${t.id}\n[${indices.join(", ")}]\nflat index ${offset.toLocaleString()}\nvalue: ${value === undefined || value === null ? "not included in architecture package" : value}`;
  };
  input.addEventListener("input", update);
  host.append(label, input, result);
  update();
}
function compressionLabel(value: unknown): string {
  return Number(value) === 0 ? "Uncompressed" : `${value}:1`;
}
function formulaFor(e: Entity, t?: Tensor): string | null {
  const a = e.attrs ?? {};
  if (
    e.kind === "ProjectionBlock" &&
    a.storageOrientation === "[input, output]" &&
    t?.role === "parameter"
  ) {
    const bias = children(e.id).some(
      (child) =>
        child.tensorId &&
        landscape.index.tensors.get(child.tensorId)?.shape.length === 1,
    );
    return `y = x W${bias ? " + b" : ""}   (row vectors; W stored as input × output)`;
  }
  if (typeof a.formula === "string") return a.formula;
  if (
    e.kind === "ProjectionBlock" &&
    t?.role === "parameter" &&
    t.shape.length === 2
  )
    return "y = W x";
  return null;
}
function describe(e: Entity): string {
  const a = e.attrs ?? {};
  switch (e.kind) {
    case "TransformerBlock":
      return "A complete decoder block. Residual streams pass through attention and feed-forward branches while retaining their place in the full model.";
    case "CompressedAttention":
      return Number(a.compressionRatio) === 0
        ? "Query heads read uncompressed key/value memory. This attention module has no compressed-history branch."
        : `Query heads read a local window and eligible compressed history. The ${a.compressionRatio}:1 ratio acts along sequence positions, not hidden coordinates.`;
    case "HyperConnectionMix": {
      const streams =
        typeof a.streams === "number" ? a.streams : nearestBlock(e.id)?.streams;
      return `${streams === undefined ? "Residual streams" : `${streams} residual streams`} are combined or updated by the declared learned mixing map. Crossings show permitted mixing, not measured coefficient strength.`;
    }
    case "ExpertCluster":
      return "Every mark is an available expert. Selection depends on the input; no active set is fabricated in this architecture view.";
    case "ScoreRouter":
      return "The router computes scores and chooses a sparse subset of experts. Inspect its declared top-k rule and learned gate matrix.";
    case "HashRouter":
      return "This router uses a token-ID routing table. It is distinct from the learned score routers in later blocks.";
    case "SparseIndexer":
      return "A separate indexer selects eligible compressed-memory positions. These selection dependencies are distinct from the KV payload.";
    case "Compressor":
      return "Sequence positions are pooled into compressed memory. The projection, positional terms and pooling form a compound operation.";
    case "SharedExpert":
      return "A shared feed-forward path accompanies routed experts. Its parameters are separate from the selectable expert bank.";
    case "AttentionHead":
      return "A view of attention mixing. Conceptual heads share KV or projection storage where the architecture declares it.";
    case "NormalizationPlane":
      return "Normalization rescales coordinates according to the declared operator. It does not change the coordinate count.";
    case "EmbeddingTable":
      return "An index selects one learned vector from this table. The full table shape is available; its trained values are not bundled.";
    case "VocabularyPlane":
      return "A learned map produces vocabulary scores. Actual logits require an input and a recorded execution.";
    case "ProjectionBlock": {
      const t = tensorFor(e);
      if (typeof a.rotaryDimensions === "number")
        return `A deterministic rotary position transform acts on ${a.rotaryDimensions} declared coordinates. It is not a learned projection matrix.`;
      if (!t || t.role !== "parameter")
        return "A declared coordinate transformation. Its mathematical rule and actual dependencies identify the operation; no learned matrix is inferred from its appearance.";
      if (a.storageOrientation === "[input, output]")
        return "The stored matrix has input-coordinate rows and output-coordinate columns. The formula below uses row vectors to match those exact stored axes.";
      return "A parameterized transformation between coordinate spaces. The declared tensor shape and operation specify how its coordinates are mapped.";
    }
    default:
      if (e.attrs?.expertIndex !== undefined)
        return "This expert has its own learned projections. Open a tensor to inspect exact dimensions and scalar addresses.";
      if (e.tensorId)
        return "Logical tensor dimensions are preserved independently of packed checkpoint storage. Grid lines are guides, not fabricated weight values.";
      return "A declared component of the model. Its identity, parameters and dependencies remain accessible at this level of the landscape.";
  }
}
function connectionLens(selectedEntity: Entity) {
  let host = document.getElementById("connection-list");
  if (!host) {
    host = document.createElement("section");
    host.id = "connection-list";
    $("selection-actions").after(host);
  }
  host.replaceChildren();
  host.hidden = false;
  const computationEdges = (id: string): Edge[] =>
    ((landscape.index.incident.get(id) ?? []) as Edge[]).filter(
      (edge) => edge.kind === "dataflow" || edge.kind === "residual",
    );
  let owner = selectedEntity,
    edges = computationEdges(owner.id);
  if (!edges.length) {
    const containing = [
      ...(landscape.index.ancestors(selectedEntity.id) as Entity[]),
    ]
      .reverse()
      .find(
        (candidate) =>
          candidate.id !== selectedEntity.id &&
          computationEdges(candidate.id).length,
      );
    if (containing) {
      owner = containing;
      edges = computationEdges(owner.id);
    }
  }
  const title = document.createElement("h3");
  title.textContent = "Connections";
  host.append(title);
  if (owner.id !== selectedEntity.id) {
    const note = document.createElement("p");
    note.className = "connection-owner";
    note.textContent = `Connections of containing operation: ${owner.label} (${owner.kind}).${selectedEntity.tensorId ? " These edges belong to the containing operation, not the selected tensor." : ""}`;
    host.append(note);
  }
  if (!edges.length) {
    const note = document.createElement("p");
    note.textContent =
      "No computation edges are declared for this component or its containing operations.";
    host.append(note);
    return;
  }
  for (const direction of ["incoming", "outgoing"] as const) {
    const matching = edges.filter((edge) =>
      direction === "incoming" ? edge.to === owner.id : edge.from === owner.id,
    );
    if (!matching.length) continue;
    const section = document.createElement("div");
    section.className = "connection-group";
    const heading = document.createElement("h4");
    heading.textContent = `${direction === "incoming" ? "Inputs" : "Outputs"} (${matching.length})`;
    section.append(heading);
    for (const edge of matching.slice(0, 6)) {
      const endpointId = direction === "incoming" ? edge.from : edge.to;
      const endpoint = landscape.index.entities.get(endpointId) as
        | Entity
        | undefined;
      if (!endpoint) continue;
      const button = document.createElement("button");
      button.className = "connection-link";
      button.type = "button";
      button.dataset.edgeId = edge.id;
      const label = document.createElement("span");
      label.className = "connection-label";
      label.textContent =
        edge.label ?? (edge.kind === "residual" ? "Residual path" : "Dataflow");
      const target = document.createElement("strong");
      target.className = "connection-endpoint";
      target.textContent = `${direction === "incoming" ? "From" : "To"} ${endpoint.label}`;
      const meta = document.createElement("small");
      meta.className = "connection-evidence";
      meta.textContent = `${endpoint.kind} · ${edge.kind} · ${edge.evidence}`;
      button.title = `${edge.id}: ${edge.from} → ${edge.to}`;
      button.append(label, target, meta);
      button.addEventListener("click", () => {
        inspect(endpoint.id);
        renderer.approach(endpoint.id);
      });
      section.append(button);
    }
    if (matching.length > 6) {
      const rest = document.createElement("p");
      rest.className = "connection-more";
      rest.textContent = `${matching.length - 6} additional ${direction} edges are declared.`;
      section.append(rest);
    }
    host.append(section);
  }
}
function inspect(id: string) {
  const e = landscape.index.entities.get(id) as Entity | undefined;
  if (!e) return;
  selected = e;
  currentBlock = nearestBlock(id) ?? currentBlock;
  renderer.select(id);
  $("panel-body").hidden = false;
  $("collapse-panel").textContent = "−";
  replaceText("selection-kind", e.kind.replace(/([a-z])([A-Z])/g, "$1 $2"));
  replaceText("selection-title", e.label);
  replaceText("selection-description", describe(e));
  $("selection-facts").replaceChildren();
  $("selection-actions").replaceChildren();
  $("tensor-info").hidden = true;
  const a = e.attrs ?? {};
  const count = landscape.index.parameterCount(e.id);
  if (count !== null) fact("Referenced parameters", count.toLocaleString());
  if (a.index !== undefined) fact("Block index", a.index);
  if (a.heads !== undefined || a.queryHeads !== undefined)
    fact("Query heads", a.heads ?? a.queryHeads);
  if (a.headDim !== undefined) fact("Head width", a.headDim);
  if (a.localWindow !== undefined) fact("Local window", a.localWindow);
  if (a.compressionRatio !== undefined)
    fact("Sequence compression", compressionLabel(a.compressionRatio));
  if (a.streams !== undefined) fact("Residual streams", a.streams);
  if (a.storageOrientation !== undefined)
    fact("Stored matrix axes", a.storageOrientation);
  if (a.routedExperts !== undefined) fact("Routed experts", a.routedExperts);
  if (a.activeRoutedExperts !== undefined)
    fact("Selected per token", a.activeRoutedExperts);
  if (a.sharedExperts !== undefined) fact("Shared experts", a.sharedExperts);
  if (a.count !== undefined) fact("Members", a.count);
  if (a.topK !== undefined) fact("Top-k", a.topK);
  if (a.expertIndex !== undefined) fact("Expert index", a.expertIndex);
  if (a.ratio !== undefined)
    fact("Sequence compression", compressionLabel(a.ratio));
  const t = tensorFor(e);
  if (t) {
    fact("Logical shape", shapeLabel(t.shape));
    fact("Logical dtype", t.dtype);
    const storage = (
      t as Tensor & {
        storage?: {
          dtype?: string;
          shape?: (number | string)[];
          encoding?: string;
        };
      }
    ).storage;
    if (storage?.dtype) fact("Stored dtype", storage.dtype);
    if (storage?.shape && storage.shape.join(",") !== t.shape.join(","))
      fact("Packed storage shape", shapeLabel(storage.shape));
    fact("Tensor role", t.role);
    if (t.shape.every((s) => typeof s === "number")) selectedTensor(t);
  }
  const formula = formulaFor(e, t);
  $("formula").hidden = !formula;
  if (formula) replaceText("formula", formula);
  if (e.kind === "TransformerBlock") {
    action("Open attention", () => renderer.navigate("attention", e.id));
    action("Open experts", () => renderer.navigate("experts", e.id));
  } else action("Approach component", () => renderer.approach(e.id));
  const kids = children(e.id).filter(
    (e) =>
      e.tensorId ||
      e.kind === "ProjectionBlock" ||
      e.kind === "SharedExpert" ||
      e.kind === "ScoreRouter" ||
      e.kind === "HashRouter",
  );
  for (const child of kids.slice(0, 5))
    action(child.label, () => {
      inspect(child.id);
      renderer.approach(child.id);
    });
  if (e.kind === "ExpertCluster") {
    const bank = children(e.id).some((e) => e.attrs?.expertIndex !== undefined)
      ? e
      : children(e.id).find((e) => e.kind === "ExpertCluster");
    if (bank) {
      const experts = children(bank.id).filter(
        (e) => e.attrs?.expertIndex !== undefined,
      );
      if (experts.length) {
        const label = document.createElement("label");
        label.textContent = "Choose any expert";
        label.className = "expert-picker";
        const select = document.createElement("select");
        select.setAttribute("aria-label", "Expert index");
        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = "Choose an expert…";
        placeholder.disabled = true;
        placeholder.selected = true;
        select.append(placeholder);
        for (const ex of experts) {
          const o = document.createElement("option");
          o.value = ex.id;
          o.textContent = `Expert ${ex.attrs!.expertIndex}`;
          select.append(o);
        }
        select.addEventListener("change", () => {
          inspect(select.value);
          renderer.approach(select.value);
        });
        $("selection-actions").append(label, select);
      }
    }
  }
  connectionLens(e);
  if (currentBlock) {
    updatePosition(currentBlock.id, currentMode);
  }
}
function overviewFacts() {
  selected = null;
  const connections = document.getElementById("connection-list");
  if (connections) {
    connections.replaceChildren();
    connections.hidden = true;
  }
  replaceText("selection-kind", "COMPLETE ARCHITECTURE");
  replaceText("selection-title", "Follow the computation.");
  replaceText(
    "selection-description",
    "The whole decoder is unfolded here. Scroll into any block; its summaries resolve into projections, memory paths, routers and individual experts.",
  );
  $("selection-facts").replaceChildren();
  $("selection-actions").replaceChildren();
  $("formula").hidden = true;
  $("tensor-info").hidden = true;
  fact("Core blocks", landscape.blocks.filter((b) => !b.auxiliary).length);
  fact("Auxiliary blocks", landscape.blocks.filter((b) => b.auxiliary).length);
  fact(
    "Unique parameters",
    landscape.index.parameterCount(landscape.model.rootId)?.toLocaleString(),
  );
  fact("Addressable tensors", landscape.model.tensors.length);
  fact("Metadata package", "No full model weights");
  action("Enter first block", () =>
    renderer.navigate("block", landscape.blocks[0].id),
  );
  const aux = landscape.blocks.find((b) => b.auxiliary);
  if (aux) action("Auxiliary module", () => renderer.navigate("block", aux.id));
}
function updatePosition(id: string | null, mode: string) {
  currentMode = mode;
  if (id)
    currentBlock = landscape.blocks.find((b) => b.id === id) ?? currentBlock;
  const b = currentBlock;
  replaceText(
    "position-label",
    mode === "model" ? "All core blocks" : (b?.label ?? "Model"),
  );
  for (const button of document.querySelectorAll<HTMLButtonElement>(
    "[data-view]",
  ))
    button.classList.toggle("active", button.dataset.view === mode);
  for (const button of $("block-map").children)
    button.classList.toggle(
      "active",
      mode !== "model" && (button as HTMLButtonElement).dataset.id === id,
    );
  const title = $("map-title");
  title.style.fontSize = mode === "model" ? "" : "30px";
  $("map-description").hidden = mode !== "model";
  $("title-facts").hidden = mode !== "model";
}
function navigation() {
  const map = $("block-map");
  map.replaceChildren();
  for (const block of landscape.blocks.filter((b) => !b.auxiliary)) {
    const b = document.createElement("button");
    b.dataset.id = block.id;
    b.title = block.label;
    b.setAttribute("aria-label", block.label);
    const entity = landscape.index.entities.get(block.id);
    if (Number(entity?.attrs?.compressionRatio) === 4) b.classList.add("csa");
    b.addEventListener("click", () =>
      renderer.navigate(
        currentMode === "model" ? "block" : currentMode,
        block.id,
      ),
    );
    map.append(b);
  }
  for (const el of document.querySelectorAll<HTMLButtonElement>("[data-view]"))
    el.onclick = () => {
      renderer.navigate(el.dataset.view!, currentBlock?.id);
      if (el.dataset.view === "model") overviewFacts();
    };
  $("reset-view").onclick = () => {
    renderer.fitModel();
    overviewFacts();
  };
  for (const [id, direction] of [
    ["previous-block", -1],
    ["next-block", 1],
  ] as const)
    $(id).onclick = () => {
      const core = landscape.blocks.filter((b) => !b.auxiliary),
        i = Math.max(
          0,
          core.findIndex((b) => b.id === currentBlock?.id),
        );
      renderer.navigate(
        currentMode === "model" ? "block" : currentMode,
        core[(i + direction + core.length) % core.length].id,
      );
    };
}
let searchTimer: ReturnType<typeof setTimeout>;
function search() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    const query = $<HTMLInputElement>("search").value.toLowerCase().trim();
    const host = $("search-results");
    host.replaceChildren();
    if (query.length < 2) return;
    const terms = query.split(/\s+/);
    let found = 0;
    for (const e of landscape.model.entities) {
      if (!terms.every((t) => (e.id + " " + e.label).toLowerCase().includes(t)))
        continue;
      const b = document.createElement("button");
      b.textContent = e.label;
      const s = document.createElement("small");
      s.textContent = e.id;
      b.append(s);
      b.addEventListener("click", () => {
        inspect(e.id);
        renderer.approach(e.id);
      });
      host.append(b);
      if (++found === 35) break;
    }
    if (!found) host.textContent = "No matching components.";
  }, 160);
}
async function load(name: string) {
  const request = ++loading;
  $("loading").hidden = false;
  try {
    const res = await fetch(`/models/${name}.atlas.json.gz`);
    if (!res.ok || !res.body)
      throw new Error(`Package unavailable (${res.status}).`);
    const ir: ModelIR = await new Response(
      res.body.pipeThrough(new DecompressionStream("gzip")),
    ).json();
    if (request !== loading) return;
    landscape = compileLandscape(ir);
    currentBlock = landscape.blocks[0];
    const nameParts = ir.name.split(" · ")[0].split("-");
    $("map-title").replaceChildren();
    const first = document.createElement("span");
    first.textContent = nameParts.shift() ?? ir.name;
    const br = document.createElement("br"),
      em = document.createElement("em");
    em.textContent = nameParts.join("-") + ".";
    $("map-title").append(first, br, em);
    replaceText(
      "map-eyebrow",
      "MODEL ARCHITECTURE / " +
        String(ir.metadata?.architecture ?? "TRANSFORMER").split(" · ")[0],
    );
    const facts = $("title-facts");
    facts.replaceChildren();
    for (const [label, value] of [
      ["parameters", countLabel(landscape.index.parameterCount(ir.rootId))],
      ["core blocks", landscape.blocks.filter((b) => !b.auxiliary).length],
    ]) {
      const d = document.createElement("div"),
        strong = document.createElement("strong"),
        small = document.createElement("span");
      strong.textContent = String(value);
      small.textContent = String(label);
      d.append(strong, small);
      facts.append(d);
    }
    navigation();
    renderer.setScene(landscape);
    overviewFacts();
    $("loading").hidden = true;
  } catch (e) {
    $("loading").replaceChildren();
    const strong = document.createElement("strong");
    strong.textContent = "The architecture could not load.";
    const p = document.createElement("span");
    p.textContent = e instanceof Error ? e.message : String(e);
    $("loading").append(strong, p);
    console.error(e);
  }
}
try {
  renderer = new LandscapeRenderer($("canvas-host"), {
    select: inspect,
    position: updatePosition,
    stats: (s) => replaceText("stats", s),
  });
  $<HTMLSelectElement>("model-choice").addEventListener("change", (e) =>
    load((e.target as HTMLSelectElement).value),
  );
  $("search").addEventListener("input", search);
  $("about").onclick = () => $<HTMLDialogElement>("about-dialog").showModal();
  $("close-about").onclick = () => $<HTMLDialogElement>("about-dialog").close();
  $("collapse-panel").onclick = () => {
    const hidden = !$("panel-body").hidden;
    $("panel-body").hidden = hidden;
    $("collapse-panel").textContent = hidden ? "+" : "−";
  };
  $("face-on").onclick = () => {
    const b = $("face-on"),
      active = b.getAttribute("aria-pressed") !== "true";
    b.setAttribute("aria-pressed", String(active));
    renderer.setFlat(active);
  };
  window.addEventListener("keydown", (e) => {
    if ((e.target as HTMLElement).matches("input,select,textarea")) return;
    const modes = ["model", "block", "attention", "experts"];
    if (/^[1-4]$/.test(e.key)) {
      renderer.navigate(modes[Number(e.key) - 1], currentBlock?.id);
      if (e.key === "1") overviewFacts();
    }
    if (e.key === "Escape") $<HTMLDialogElement>("about-dialog").close();
  });
  load("deepseek4");
} catch (e) {
  $("loading").textContent =
    "WebGL2 is required to view the landscape. " + String(e);
}
