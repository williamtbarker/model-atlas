import "./style.css";
import { compileTrace } from "./adapter";
import { explain, chooseCell } from "./arithmetic";
import { Microscope } from "./renderer";
import type {
  Arithmetic,
  Chapter,
  Example,
  SceneIR,
  Selection,
  TracePackage,
} from "./types";

const $ = <T extends HTMLElement = HTMLElement>(id: string) =>
  document.getElementById(id) as T;
const fmt = (n: number, d = 5) =>
  Number.isFinite(n) ? (n >= 0 ? "+" : "") + n.toFixed(d) : "unavailable";
let pkg: TracePackage,
  example: Example,
  scene: SceneIR,
  baseline: SceneIR,
  microscope: Microscope;
let layer = 0,
  head = 0,
  token = 7,
  chapter: Chapter = "attention",
  ablation = "baseline",
  selection: Selection,
  arithmetic: Arithmetic;
const chapters: Record<
  Chapter,
  { title: string; description: string; group: string }
> = {
  overview: {
    title: "One continuous computation.",
    description: "Two blocks. Every recorded coordinate has a place.",
    group: "input",
  },
  embedding: {
    title: "A token gets coordinates.",
    description: "Token and position vectors add to form the residual stream.",
    group: "embedding",
  },
  projection: {
    title: "Same input. Different questions.",
    description:
      "Fixed matrices make queries, keys, and values from the same state.",
    group: "q",
  },
  attention: {
    title: "Where to read.",
    description:
      "Queries compare with keys. Each row competes for one unit of attention.",
    group: "attention",
  },
  write: {
    title: "What gets written.",
    description:
      "Attention scales values. The output projection writes them into the residual.",
    group: "mixed",
  },
  residual: {
    title: "Preserve. Add. Cancel.",
    description: "The incoming state survives while signed updates add to it.",
    group: "residual1",
  },
  mlp: {
    title: "Expand the possibilities.",
    description:
      "A wider coordinate space, a nonlinearity, then a return to the residual.",
    group: "up",
  },
  output: {
    title: "From coordinates to a prediction.",
    description:
      "The final position becomes a distribution over the vocabulary.",
    group: "probabilities",
  },
  parameters: {
    title: "Every learned parameter.",
    description:
      "Complete stored tensors. The attention-head slices refer to this same storage.",
    group: "parameters",
  },
};

function option(select: HTMLSelectElement, value: string, label: string) {
  const o = document.createElement("option");
  o.value = value;
  o.textContent = label;
  select.append(o);
}
function selectionForChapter() {
  const id =
    chapter === "parameters"
      ? "param.embedding"
      : chapter === "embedding"
        ? "embedding"
        : chapter === "output"
          ? "probabilities"
          : chapter === "projection"
            ? `b${layer}.h${head}.q`
            : chapter === "attention"
              ? `b${layer}.h${head}.attention`
              : chapter === "write"
                ? `b${layer}.h${head}.mixed`
                : `b${layer}.${chapters[chapter].group}`;
  return chooseCell(scene.views.find((v) => v.id === id)!, token);
}
function navigate(next: Chapter) {
  chapter = next;
  document.querySelectorAll("[data-chapter]").forEach((b) => {
    if ((b as HTMLElement).dataset.chapter === chapter)
      b.setAttribute("aria-current", "step");
    else b.removeAttribute("aria-current");
  });
  $("stage-title").textContent = chapters[chapter].title;
  $("stage-description").textContent = chapters[chapter].description;
  updateKicker();
  microscope.navigate(chapter, layer, head, token);
  select(selectionForChapter());
}
function updateKicker() {
  $("stage-kicker").textContent =
    chapter === "overview"
      ? "RECORDED FORWARD PASS"
      : chapter === "embedding"
        ? "LEARNED EMBEDDINGS"
        : chapter === "output"
          ? "FINAL POSITION"
          : `BLOCK ${layer + 1} / HEAD ${head + 1}${ablation !== "baseline" ? " / INTERVENTION" : ""}`;
}
function createSelectors() {
  for (const [id, count, label] of [
    ["layers", pkg.model.nLayers, "Block"],
    ["heads", pkg.model.nHeads, "Head"],
  ] as const) {
    const host = $(id);
    host.replaceChildren();
    for (let i = 0; i < count; i++) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = `${label} ${i + 1}`;
      b.setAttribute("aria-pressed", String(i === 0));
      b.addEventListener("click", () => {
        if (id === "layers") layer = i;
        else head = i;
        refreshButtons();
        navigate(chapter);
      });
      host.append(b);
    }
  }
  const examples = $<HTMLSelectElement>("example");
  pkg.examples.forEach((e, i) =>
    option(examples, e.id, `Example ${i + 1} · ${e.target}`),
  );
  examples.addEventListener("change", () => {
    example = pkg.examples.find((e) => e.id === examples.value)!;
    ablation = "baseline";
    token = example.tokens.length - 1;
    updateScene();
    navigate(chapter);
  });
  const operations = $<HTMLSelectElement>("operation");
  scene.views.forEach((v) =>
    option(
      operations,
      v.id,
      `${v.block >= 0 ? "B" + (v.block + 1) + " · " : ""}${v.head !== undefined ? "H" + (v.head + 1) + " · " : ""}${v.label}`,
    ),
  );
  operations.addEventListener("change", () => {
    const v = scene.views.find((v) => v.id === operations.value)!;
    select(chooseCell(v, token));
    microscope.zoomTo(v.id);
  });
}
function refreshButtons() {
  Array.from($("layers").children).forEach((b, i) =>
    b.setAttribute("aria-pressed", String(i === layer)),
  );
  Array.from($("heads").children).forEach((b, i) =>
    b.setAttribute("aria-pressed", String(i === head)),
  );
  const disabled = ablation === `L${layer}H${head}`;
  $("disable-button").setAttribute("aria-pressed", String(disabled));
  $("disable-button").textContent =
    `${disabled ? "Restore" : "Disable"} B${layer + 1}/H${head + 1}`;
  updateKicker();
}
function updateScene() {
  baseline = compileTrace(pkg, example, example.runs.baseline);
  scene =
    ablation === "baseline"
      ? baseline
      : compileTrace(pkg, example, example.runs[ablation], ablation);
  microscope.setScene(scene, baseline);
  drawTokens();
  drawPrediction();
  refreshButtons();
  if (selection && scene.views.some((v) => v.id === selection.viewId))
    select(selection);
}
function drawTokens() {
  const host = $("tokens");
  host.replaceChildren();
  example.tokens.forEach((text, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "token" + (i === token ? " query" : "");
    b.setAttribute("aria-pressed", String(i === token));
    b.setAttribute(
      "aria-label",
      `Position ${i}: ${text}${i === token ? ", selected query" : ""}`,
    );
    const label = document.createElement("span");
    label.textContent = text;
    const ix = document.createElement("small");
    ix.textContent = String(i);
    b.append(label, ix);
    b.addEventListener("click", () => {
      token = i;
      drawTokens();
      select(selectionForChapter());
    });
    host.append(b);
  });
  $("task-description").textContent =
    `Find the value for ${example.tokens.at(-1)}`;
}
function drawPrediction() {
  const p = example.runs[ablation].probabilities.at(-1)!,
    base = example.runs.baseline.probabilities.at(-1)!,
    pred = p.indexOf(Math.max(...p));
  $("prediction").textContent = pkg.model.vocab[pred];
  $("prediction-confidence").textContent = `${(p[pred] * 100).toFixed(2)}%`;
  $("expected").textContent = `answer: ${example.target}`;
  $("prediction-change").textContent =
    ablation === "baseline"
      ? ""
      : `P(${example.target}): ${(base[example.targetId] * 100).toFixed(1)}% → ${(p[example.targetId] * 100).toFixed(1)}% · ${ablation} disabled`;
}
function select(s: Selection) {
  const view = scene.views.find((v) => v.id === s.viewId)!;
  s = {
    viewId: s.viewId,
    row: Math.max(0, Math.min(s.row, view.values.length - 1)),
    col: Math.max(0, Math.min(s.col, view.values[0].length - 1)),
  };
  selection = s;
  arithmetic = explain(scene, s);
  microscope.select(s, arithmetic);
  if (view.block >= 0) layer = view.block;
  if (view.head !== undefined) head = view.head;
  if (view.kind === "attention") token = s.row;
  else if (view.kind !== "weight") token = s.col;
  microscope.setContext(layer, head, token);
  refreshButtons();
  drawTokens();
  $<HTMLSelectElement>("operation").value = s.viewId;
  const rs = $<HTMLSelectElement>("row"),
    cs = $<HTMLSelectElement>("col");
  rs.replaceChildren();
  cs.replaceChildren();
  view.values.forEach((_, i) =>
    option(
      rs,
      String(i),
      view.kind === "attention"
        ? `${i} · ${example.tokens[i]}`
        : view.id === "probabilities" || view.id === "logits"
          ? `${i} · ${pkg.model.vocab[i]}`
          : String(i),
    ),
  );
  view.values[0].forEach((_, i) =>
    option(
      cs,
      String(i),
      view.kind === "weight" ? String(i) : `${i} · ${example.tokens[i]}`,
    ),
  );
  rs.value = String(s.row);
  cs.value = String(s.col);
  $("row-label").textContent = view.rowAxis;
  $("col-label").textContent = view.colAxis;
  const masked = view.kind === "attention" && s.col > s.row;
  $("value").textContent = masked ? "MASKED" : fmt(arithmetic.result, 6);
  $("value-label").textContent = arithmetic.title;
  const bv =
    baseline.views.find((v) => v.id === s.viewId)!.values[s.row][s.col] ?? 0;
  $("baseline-value").textContent =
    ablation === "baseline"
      ? ""
      : `baseline ${fmt(bv, 6)} · Δ ${fmt(arithmetic.result - bv, 6)}`;
  $("equation").textContent =
    arithmetic.equation || `${view.rowAxis} ${s.row}, ${view.colAxis} ${s.col}`;
  $("computation-note").textContent = arithmetic.note;
  $("tensor-shape").textContent =
    `${view.values.length} × ${view.values[0].length} · float64`;
  drawTerms();
  drawChart();
  drawProgress(0, false);
  const hasTerms = arithmetic.terms.length > 0;
  $<HTMLButtonElement>("play-button").disabled = !hasTerms;
  $<HTMLInputElement>("scrubber").disabled = !hasTerms;
  Array.from($("tokens").children).forEach((b, i) =>
    b.classList.toggle("source", view.kind === "attention" && i === s.col),
  );
}
function drawTerms() {
  const host = $("terms");
  host.replaceChildren();
  arithmetic.terms.forEach((t, i) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "term-row " + (t.value >= 0 ? "pos" : "neg");
    row.dataset.index = String(i);
    row.setAttribute(
      "aria-label",
      `${t.label}: ${t.left}${t.right !== undefined ? " times " + t.right : ""} equals ${t.value}`,
    );
    for (const text of [
      t.label,
      fmt(t.left, 3) + (t.right !== undefined ? " × " + fmt(t.right, 3) : ""),
      fmt(t.value, 5),
    ]) {
      const span = document.createElement("span");
      span.textContent = text;
      row.append(span);
    }
    row.addEventListener("click", () =>
      microscope.setProgress((i + 1) / Math.max(1, arithmetic.terms.length)),
    );
    host.append(row);
  });
}
function drawChart() {
  const values = arithmetic.terms.map((t) => t.value),
    max = Math.max(0.001, ...values.map(Math.abs)),
    w = 288,
    h = 92,
    n = Math.max(1, values.length),
    step = 260 / n;
  const bars = values
    .map((v, i) => {
      const height = (Math.abs(v) / max) * 31,
        y = v >= 0 ? 43 - height : 43;
      return `<rect data-bar="${i}" x="${14 + i * step + 1}" y="${y}" width="${Math.max(2, step - 3)}" height="${Math.max(0.8, height)}" fill="${v >= 0 ? "#58dfcd" : "#f38b91"}" opacity=".2"/>`;
    })
    .join("");
  $("signed-chart").innerHTML =
    `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Signed contributions above and below zero. Scale ${max.toFixed(4)}."><line x1="9" x2="279" y1="43" y2="43" stroke="#506974" stroke-width="1"/>${bars}<text x="9" y="88" fill="#6d909f" font-size="10" font-family="monospace">− ${max.toFixed(3)}</text><text x="279" y="88" text-anchor="end" fill="#6d909f" font-size="10" font-family="monospace">+ ${max.toFixed(3)}</text></svg>`;
}
function drawProgress(p: number, active: boolean) {
  $<HTMLInputElement>("scrubber").value = String(Math.round(p * 1000));
  $("play-button").textContent = active
    ? "Ⅱ Pause contributions"
    : "▶ Follow contributions";
  if (!arithmetic) return;
  const n = arithmetic.terms.length,
    k = Math.min(n, Math.floor(p * n + 0.00001));
  let sum = arithmetic.terms.slice(0, k).reduce((a, t) => a + t.value, 0);
  if (p >= 1) {
    sum += arithmetic.bias ?? 0;
    if (arithmetic.divide) sum /= arithmetic.divide;
  }
  $("sum-label").textContent = arithmetic.divide
    ? p >= 1
      ? "Scaled dot product"
      : "Dot product (before scale)"
    : arithmetic.mode === "sum"
      ? "Accumulated sum"
      : "Recorded output";
  $("running-sum").textContent = fmt(
    arithmetic.mode === "sum" ? sum : arithmetic.result,
    6,
  );
  document
    .querySelectorAll(".term-row")
    .forEach((r, i) => r.classList.toggle("reached", i < k));
  document
    .querySelectorAll("[data-bar]")
    .forEach((r, i) => r.setAttribute("opacity", i < k ? "1" : ".2"));
}
function about() {
  const m = pkg.model,
    metrics = m.metrics.heldOut;
  $("about-content").replaceChildren();
  for (const text of [
    `${m.name} is a ${m.parameterCount.toLocaleString()}-parameter transformer trained from scratch to read three key/value pairs and retrieve the value for the final query. It is a synthetic teaching model, not a pretrained language model.`,
    `${m.nLayers} pre-normalized blocks · ${m.nHeads} heads per block · ${m.dModel} residual coordinates · ${m.dFF} MLP coordinates. All matrices, intermediate values, and coordinates shown here are recorded from its actual computation.`,
    `The held-out evaluation contains ${metrics.examples} queries, separated by entire dictionaries: ${(metrics.accuracy * 100).toFixed(1)}% accuracy. These five examples were deliberately selected for visible ablation effects.`,
    "Disabling a head loads a complete recorded rerun, including all downstream layers. It is an intervention, not a unique explanation of the answer. “Changes only” colors the difference from baseline; exact values remain in the inspector.",
    "The scene shows coordinate arrays, not a projection of meaning into physical space. Color encodes sign and magnitude relative to each baseline tensor’s maximum. Animation time explains order; it is not measured execution time.",
    "No inference runs in the browser. The supplied trace is about 1.7 MB compressed. Source, training code, mathematical checks, and visual encoding notes are included in the bundle.",
  ]) {
    const p = document.createElement("p");
    p.textContent = text;
    $("about-content").append(p);
  }
  $<HTMLDialogElement>("about-dialog").showModal();
}

async function start() {
  try {
    const res = await fetch("/traces/recall.json.gz");
    if (!res.ok) throw new Error(`Trace download failed (${res.status})`);
    if (!res.body) throw new Error("No trace data");
    pkg = await new Response(
      res.body.pipeThrough(new DecompressionStream("gzip")),
    ).json();
    if (!pkg.model?.nLayers || !pkg.examples?.length)
      throw new Error("The bundled trace is not valid.");
    example = pkg.examples[0];
    scene = baseline = compileTrace(pkg, example, example.runs.baseline);
    $("model-name").textContent = pkg.model.name;
    $("model-size").textContent =
      `${pkg.model.parameterCount.toLocaleString()} parameters · ${pkg.model.nLayers} blocks`;
    $("loading").hidden = true;
    $("workspace").hidden = false;
    microscope = new Microscope($("canvas-host"), {
      select,
      stats: (s) => ($("stats").textContent = s),
      progress: drawProgress,
      hover: (s) => {
        const host = $("hover-value");
        host.hidden = !s;
        if (s) {
          const v = scene.views.find((v) => v.id === s.viewId)!;
          host.textContent = `${v.label} [${s.row}, ${s.col}]\n${v.kind === "attention" && s.col > s.row ? "causally masked" : fmt(v.values[s.row][s.col] ?? 0, 6)}`;
        }
      },
    });
    microscope.setScene(scene, baseline);
    createSelectors();
    drawTokens();
    drawPrediction();
    navigate("attention");
    document
      .querySelectorAll<HTMLButtonElement>("[data-chapter]")
      .forEach((b) =>
        b.addEventListener("click", () =>
          navigate(b.dataset.chapter as Chapter),
        ),
      );
    for (const id of ["row", "col"])
      $(id).addEventListener("change", () =>
        select({
          viewId: selection.viewId,
          row: Number($<HTMLSelectElement>("row").value),
          col: Number($<HTMLSelectElement>("col").value),
        }),
      );
    $("play-button").addEventListener("click", () => microscope.play());
    $("scrubber").addEventListener("input", () =>
      microscope.setProgress(
        Number($<HTMLInputElement>("scrubber").value) / 1000,
      ),
    );
    $("focus-button").addEventListener("click", () =>
      microscope.focusCalculation(),
    );
    $("disable-button").addEventListener("click", () => {
      ablation =
        ablation === `L${layer}H${head}` ? "baseline" : `L${layer}H${head}`;
      updateScene();
    });
    $("difference").addEventListener("change", () =>
      microscope.setDifference($<HTMLInputElement>("difference").checked),
    );
    $("flat-button").addEventListener("click", () => {
      const b = $("flat-button"),
        value = b.getAttribute("aria-pressed") !== "true";
      b.setAttribute("aria-pressed", String(value));
      microscope.setFlat(value);
    });
    $("copy-value").addEventListener("click", async () => {
      const raw = scene.views.find((v) => v.id === selection.viewId)!.values[
          selection.row
        ][selection.col],
        text =
          raw === null
            ? "null (causal score mask; −Infinity before softmax)"
            : String(raw);
      try {
        await navigator.clipboard.writeText(text);
        $("copy-value").textContent = "Copied";
        setTimeout(
          () => ($("copy-value").textContent = "Copy exact value"),
          1200,
        );
      } catch {
        $("copy-value").textContent = text;
      }
    });
    $("about-button").addEventListener("click", about);
    $("close-about").addEventListener("click", () =>
      $<HTMLDialogElement>("about-dialog").close(),
    );
    window.addEventListener("keydown", (e) => {
      if ((e.target as HTMLElement).matches("input,select,textarea,button"))
        return;
      if (e.code === "Space") {
        e.preventDefault();
        microscope.play();
      }
      if (e.key === "Escape") $<HTMLDialogElement>("about-dialog").close();
    });
  } catch (err) {
    $("loading").hidden = false;
    $("loading").textContent =
      `Could not open the microscope: ${err instanceof Error ? err.message : String(err)}. Use a current Chrome, Safari, or Firefox with WebGL2 enabled. The README includes local startup steps.`;
    console.error(err);
  }
}
start();
