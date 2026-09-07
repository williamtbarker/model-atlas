import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type {
  SceneIR,
  TensorView,
  Selection,
  Arithmetic,
  Chapter,
} from "./types";

const PITCH = 0.78,
  CELL = 0.68;
const POS = new THREE.Color("#58dfcd"),
  NEG = new THREE.Color("#f38b91"),
  INK = new THREE.Color("#102329");
const AMBER = new THREE.Color("#f4ca7b");
export const RENDER_BUDGET = {
  cells: 18000,
  numericPlanes: 10,
  labels: 36,
  paths: 64,
  pixelRatio: 1.7,
};
type Visual = {
  view: TensorView;
  group: THREE.Group;
  mesh: THREE.InstancedMesh | null;
  plate: THREE.Mesh;
  numbers: THREE.Mesh | null;
  label: HTMLButtonElement;
  level: number;
  scale: number;
  width: number;
  height: number;
};
type Flight = {
  start: number;
  duration: number;
  from: THREE.Vector3;
  to: THREE.Vector3;
  lookFrom: THREE.Vector3;
  lookTo: THREE.Vector3;
};
type Flow = {
  curve: THREE.CatmullRomCurve3;
  dot: THREE.Mesh;
  value: number;
  start: THREE.Vector3;
  end: THREE.Vector3;
};

/** Renderer consumes numerical scene entities. Model architecture interpretation lives in adapter.ts. */
export class Microscope {
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(38, 1, 0.1, 1800);
  readonly renderer: THREE.WebGLRenderer;
  readonly controls: OrbitControls;
  private ir: SceneIR | null = null;
  private visuals = new Map<string, Visual>();
  private root = new THREE.Group();
  private edges = new THREE.Group();
  private focus = new THREE.Group();
  private labels: HTMLDivElement;
  private selection: Selection | null = null;
  private arithmetic: Arithmetic | null = null;
  private ray = new THREE.Raycaster();
  private pointer = new THREE.Vector2();
  private pointerDown = new THREE.Vector2();
  private box = new THREE.BoxGeometry(1, 1, 1);
  private plane = new THREE.PlaneGeometry(1, 1);
  private sphere = new THREE.SphereGeometry(0.095, 10, 8);
  private edgesGeometry = new THREE.EdgesGeometry(
    new THREE.BoxGeometry(1, 1, 1),
  );
  private matrix = new THREE.Matrix4();
  private color = new THREE.Color();
  private lineMaterial = new THREE.LineBasicMaterial({
    color: "#88b9b5",
    transparent: true,
    opacity: 0.24,
  });
  private flight: Flight | null = null;
  private frame = 0;
  private inFrame = false;
  private dirty = true;
  private lastTime = 0;
  private tick = 0;
  private resizeObserver: ResizeObserver;
  private flows: Flow[] = [];
  private bars: THREE.Mesh[] = [];
  private accumulator: THREE.Mesh | null = null;
  private accumulatorOrigin = new THREE.Vector3();
  private accumulatorScale = 1;
  private readout: THREE.CanvasTexture | null = null;
  private readoutCanvas: HTMLCanvasElement | null = null;
  private lastReadout = "";
  private progress = 0;
  private playing = false;
  private playStart = 0;
  private token = 7;
  private layer = 0;
  private head = 0;
  private chapter: Chapter = "attention";
  private detailStyle: "values" | "delta" = "values";
  private baseline: SceneIR | null = null;
  private reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  private flat = false;
  private ruler: THREE.Group;
  private activeSignature = "";

  constructor(
    private host: HTMLElement,
    private cb: {
      select: (s: Selection) => void;
      stats: (s: string) => void;
      progress: (p: number, playing: boolean) => void;
      hover: (s: Selection | null) => void;
    },
  ) {
    this.scene.background = new THREE.Color("#070e13");
    this.scene.fog = new THREE.FogExp2("#070e13", 0.0007);
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "low-power",
    });
    this.renderer.setPixelRatio(
      Math.min(devicePixelRatio, RENDER_BUDGET.pixelRatio),
    );
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.setAttribute(
      "aria-label",
      "3D computation. Drag to orbit, scroll to zoom, click a numerical cell to inspect. Equivalent controls and values appear beside the canvas.",
    );
    this.host.append(this.renderer.domElement);
    this.labels = document.createElement("div");
    this.labels.className = "world-labels";
    this.host.append(this.labels);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.13;
    this.controls.minDistance = 4;
    this.controls.maxDistance = 1050;
    this.controls.addEventListener("change", () => this.invalidate());
    this.controls.addEventListener("start", () => {
      this.flight = null;
      this.invalidate();
    });
    this.scene.add(this.root, this.edges, this.focus);
    this.scene.add(new THREE.AmbientLight("#bdffef", 1.8));
    const light = new THREE.DirectionalLight("#c8e6ff", 2);
    light.position.set(50, 90, 150);
    this.scene.add(light);
    this.ruler = new THREE.Group();
    this.scene.add(this.ruler);
    const grid = new THREE.GridHelper(500, 100, "#244039", "#10252a");
    grid.rotation.x = Math.PI / 2;
    grid.position.set(155, 0, -18);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.3;
    this.scene.add(grid);
    this.renderer.domElement.addEventListener("pointerdown", (e) =>
      this.pointerDown.set(e.clientX, e.clientY),
    );
    this.renderer.domElement.addEventListener("pointerup", (e) => {
      if (
        this.pointerDown.distanceTo(new THREE.Vector2(e.clientX, e.clientY)) < 4
      ) {
        const hit = this.hit(e);
        if (hit) this.cb.select(hit);
      }
    });
    this.renderer.domElement.addEventListener("pointermove", (e) => {
      if (e.buttons) return;
      const hit = this.hit(e);
      this.renderer.domElement.style.cursor = hit ? "crosshair" : "grab";
      this.cb.hover(hit);
    });
    this.renderer.domElement.addEventListener("dblclick", (e) => {
      const hit = this.hit(e);
      if (hit) this.zoomTo(hit.viewId);
    });
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);
    this.resize();
    this.camera.position.set(66, 35, 88);
    this.controls.target.set(42, 13, 0);
    this.controls.update();
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) this.invalidate();
    });
  }
  private resize() {
    const w = this.host.clientWidth,
      h = this.host.clientHeight;
    if (w && h) {
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
      this.invalidate();
    }
  }
  private invalidate() {
    this.dirty = true;
    if (!this.frame && !this.inFrame)
      this.frame = requestAnimationFrame((t) => this.render(t));
  }
  private disposeGroup(group: THREE.Group) {
    for (const object of [...group.children]) {
      group.remove(object);
      object.traverse((o) => {
        const mesh = o as THREE.Mesh;
        const mat = mesh.material;
        if (mat) {
          for (const m of Array.isArray(mat) ? mat : [mat]) {
            const map = (m as THREE.MeshBasicMaterial).map;
            if (map) map.dispose();
            m.dispose();
          }
        }
        if (
          mesh.geometry &&
          ![this.box, this.plane, this.sphere, this.edgesGeometry].includes(
            mesh.geometry as any,
          )
        )
          mesh.geometry.dispose();
        if (o instanceof THREE.InstancedMesh) o.dispose();
      });
    }
  }
  setScene(ir: SceneIR, baseline: SceneIR) {
    this.ir = ir;
    this.baseline = baseline;
    // Stable IDs retain their world transforms, geometry and camera across recorded interventions.
    if (
      this.visuals.size &&
      this.activeSignature === ir.views.map((v) => v.id).join("|")
    ) {
      for (const view of ir.views) {
        const v = this.visuals.get(view.id)!;
        v.view = view;
        v.label.textContent = view.label;
        const base = baseline.views.find((t) => t.id === view.id)!;
        v.scale = Math.max(
          0.0001,
          ...base.values.flat().map((n) => Math.abs(n ?? 0)),
        );
        this.paint(v);
        if (v.numbers) {
          this.dropNumbers(v);
        }
      }
    } else {
      this.disposeGroup(this.root);
      this.disposeGroup(this.edges);
      this.labels.replaceChildren();
      this.visuals.clear();
      for (const view of ir.views) {
        const width = view.values[0].length * PITCH,
          height = view.values.length * PITCH,
          group = new THREE.Group();
        group.position.fromArray(view.position);
        const plate = new THREE.Mesh(
          this.box,
          new THREE.MeshBasicMaterial({
            color:
              view.kind === "weight"
                ? "#23333e"
                : view.kind === "residual"
                  ? "#574834"
                  : "#19333b",
            transparent: true,
            opacity: 0.45,
          }),
        );
        plate.scale.set(width + 0.22, height + 0.22, 0.1);
        plate.position.z = -0.13;
        group.add(plate);
        const outline = new THREE.LineSegments(
          this.edgesGeometry,
          new THREE.LineBasicMaterial({
            color:
              view.kind === "weight"
                ? "#647087"
                : view.kind === "residual"
                  ? "#e6bd75"
                  : "#568b91",
            transparent: true,
            opacity: 0.36,
          }),
        );
        outline.scale.set(width + 0.22, height + 0.22, 0.1);
        outline.position.z = -0.13;
        group.add(outline);
        const label = document.createElement("button");
        label.className = "world-label";
        label.type = "button";
        label.textContent = view.label;
        label.addEventListener("click", () => this.zoomTo(view.id));
        this.labels.append(label);
        const base = baseline.views.find((v) => v.id === view.id) ?? view;
        const max = Math.max(
          0.0001,
          ...base.values.flat().map((n) => Math.abs(n ?? 0)),
        );
        const visual: Visual = {
          view,
          group,
          plate,
          mesh: null,
          numbers: null,
          label,
          level: 0,
          scale: max,
          width,
          height,
        };
        this.visuals.set(view.id, visual);
        this.root.add(group);
      }
      for (const edge of ir.connections) {
        const a = this.visuals.get(edge.from)!,
          b = this.visuals.get(edge.to)!;
        const start = a.group.position
          .clone()
          .add(new THREE.Vector3(a.width / 2 + 0.3, 0, 0));
        const end = b.group.position
          .clone()
          .add(new THREE.Vector3(-b.width / 2 - 0.3, 0, 0));
        const points = [
          start,
          ...(edge.via?.map((p) => new THREE.Vector3(...p)) ?? [
            start
              .clone()
              .lerp(end, 0.35)
              .add(new THREE.Vector3(0, 0, 2)),
            start
              .clone()
              .lerp(end, 0.7)
              .add(new THREE.Vector3(0, 0, 2)),
          ]),
          end,
        ];
        const curve = new THREE.CatmullRomCurve3(points, false, "centripetal");
        const line = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(curve.getPoints(45)),
          new THREE.LineBasicMaterial({
            color:
              edge.kind === "residual"
                ? "#e5b86d"
                : edge.kind === "weight"
                  ? "#718099"
                  : "#689a9b",
            transparent: true,
            opacity:
              edge.kind === "residual"
                ? 0.5
                : edge.kind === "weight"
                  ? 0.17
                  : 0.22,
          }),
        );
        line.userData.connection = edge;
        this.edges.add(line);
        const tip = curve.getPoint(0.99),
          dir = tip.clone().sub(curve.getPoint(0.94)).normalize();
        const arrow = new THREE.Mesh(
          new THREE.ConeGeometry(0.12, 0.42, 5),
          new THREE.MeshBasicMaterial({
            color: edge.kind === "residual" ? "#e5b86d" : "#689a9b",
            transparent: true,
            opacity: 0.4,
          }),
        );
        arrow.position.copy(tip);
        arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
        this.edges.add(arrow);
      }
      this.activeSignature = ir.views.map((v) => v.id).join("|");
    }
    this.updateEmphasis();
    this.invalidate();
  }
  private makeCells(v: Visual) {
    const count = v.view.values.length * v.view.values[0].length;
    v.mesh = new THREE.InstancedMesh(
      this.box,
      new THREE.MeshBasicMaterial({
        vertexColors: false,
        transparent: true,
        opacity: 1,
      }),
      count,
    );
    v.mesh.userData.viewId = v.view.id;
    v.group.add(v.mesh);
    this.paint(v);
  }
  private paint(v: Visual) {
    if (!v.mesh) return;
    const baseline = this.baseline?.views.find((t) => t.id === v.view.id),
      cols = v.view.values[0].length;
    let index = 0;
    for (let r = 0; r < v.view.values.length; r++)
      for (let c = 0; c < cols; c++) {
        let value = v.view.values[r][c];
        const masked = v.view.kind === "attention" && c > r;
        if (this.detailStyle === "delta")
          value = (value ?? 0) - (baseline?.values[r][c] ?? 0);
        const magnitude = Math.min(1, Math.abs(value ?? 0) / v.scale);
        this.matrix.makeScale(
          CELL,
          CELL,
          masked ? 0.025 : 0.035 + 0.14 * magnitude,
        );
        this.matrix.setPosition(
          (c - (cols - 1) / 2) * PITCH,
          ((v.view.values.length - 1) / 2 - r) * PITCH,
          masked ? -0.015 : 0.025 + 0.07 * magnitude,
        );
        v.mesh.setMatrixAt(index, this.matrix);
        if (masked) this.color.set("#121b24");
        else
          this.color
            .copy(INK)
            .lerp(
              (value ?? 0) >= 0 ? POS : NEG,
              0.1 + 0.85 * Math.sqrt(magnitude),
            );
        v.mesh.setColorAt(index++, this.color);
      }
    v.mesh.instanceMatrix.needsUpdate = true;
    if (v.mesh.instanceColor) v.mesh.instanceColor.needsUpdate = true;
    v.mesh.computeBoundingSphere();
  }
  setDifference(value: boolean) {
    this.detailStyle = value ? "delta" : "values";
    for (const v of this.visuals.values()) {
      this.paint(v);
      this.dropNumbers(v);
    }
    this.invalidate();
  }
  private dropNumbers(v: Visual) {
    if (!v.numbers) return;
    v.group.remove(v.numbers);
    const m = v.numbers.material as THREE.MeshBasicMaterial;
    m.map?.dispose();
    m.dispose();
    v.numbers = null;
  }
  private makeNumbers(v: Visual) {
    const rows = v.view.values.length,
      cols = v.view.values[0].length,
      canvas = document.createElement("canvas");
    canvas.width = cols * 100;
    canvas.height = rows * 64;
    const ctx = canvas.getContext("2d")!;
    ctx.font = "500 24px ui-monospace, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const baseline = this.baseline?.views.find((t) => t.id === v.view.id);
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        if (v.view.kind === "attention" && c > r) {
          ctx.strokeStyle = "#425263";
          ctx.lineWidth = 1.5;
          for (let k = -60; k < 100; k += 17) {
            ctx.beginPath();
            ctx.moveTo(c * 100 + Math.max(0, k), r * 64 + Math.max(0, -k));
            ctx.lineTo(
              c * 100 + Math.min(100, k + 64),
              r * 64 + Math.min(64, 100 - k),
            );
            ctx.stroke();
          }
          continue;
        }
        let val = v.view.values[r][c] ?? 0;
        if (this.detailStyle === "delta") val -= baseline?.values[r][c] ?? 0;
        ctx.fillStyle = "#eaf8f6";
        ctx.fillText(
          (val >= 0 ? "+" : "") + val.toFixed(2),
          c * 100 + 50,
          r * 64 + 32,
        );
      }
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = Math.min(
      4,
      this.renderer.capabilities.getMaxAnisotropy(),
    );
    const mesh = new THREE.Mesh(
      this.plane,
      new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        depthWrite: false,
      }),
    );
    mesh.scale.set(v.width, v.height, 1);
    mesh.position.z = 0.26;
    v.group.add(mesh);
    v.numbers = mesh;
  }
  private screenSize(v: Visual) {
    return (
      (v.height * this.host.clientHeight) /
      (2 *
        Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) *
        this.camera.position.distanceTo(v.group.position))
    );
  }
  private updateLOD() {
    this.camera.updateMatrixWorld();
    const frustum = new THREE.Frustum().setFromProjectionMatrix(
      new THREE.Matrix4().multiplyMatrices(
        this.camera.projectionMatrix,
        this.camera.matrixWorldInverse,
      ),
    );
    const ranked = [...this.visuals.values()]
      .map((v) => ({
        v,
        size: this.screenSize(v),
        visible: frustum.intersectsSphere(
          new THREE.Sphere(v.group.position, Math.hypot(v.width, v.height) / 2),
        ),
      }))
      .sort(
        (a, b) =>
          b.size / b.v.view.values.length - a.size / a.v.view.values.length,
      );
    let cells = 0,
      numbers = 0,
      labels = 0;
    const occupied: { x: number; y: number; w: number; h: number }[] = [];
    for (const item of ranked) {
      const { v, size } = item,
        cellSize = size / v.view.values.length,
        selected = v.view.id === this.selection?.viewId;
      v.group.visible = item.visible;
      const needCells =
        item.visible &&
        cellSize > (v.mesh ? 2.3 : 3.0) &&
        cells + v.view.values.length * v.view.values[0].length <=
          RENDER_BUDGET.cells;
      if (needCells) {
        if (!v.mesh) this.makeCells(v);
        cells += v.mesh!.count;
      } else if (v.mesh) {
        v.group.remove(v.mesh);
        v.mesh.dispose();
        (v.mesh.material as THREE.Material).dispose();
        v.mesh = null;
      }
      const needNumbers =
        needCells &&
        cellSize > (v.numbers ? 17 : 22) &&
        numbers < RENDER_BUDGET.numericPlanes;
      if (needNumbers) {
        if (!v.numbers) this.makeNumbers(v);
        numbers++;
      } else this.dropNumbers(v);
      v.level = needNumbers ? 2 : needCells ? 1 : 0;
      const pos = v.group.position
        .clone()
        .add(new THREE.Vector3(0, v.height / 2 + 1.1, 0.3))
        .project(this.camera);
      const x = (pos.x * 0.5 + 0.5) * this.host.clientWidth,
        y = (-pos.y * 0.5 + 0.5) * this.host.clientHeight;
      const shown =
        item.visible &&
        pos.z < 1 &&
        size > 28 &&
        x > 25 &&
        x < this.host.clientWidth - 25 &&
        y > 10 &&
        y < this.host.clientHeight - 20 &&
        labels < RENDER_BUDGET.labels;
      const box = { x: x - 75, y: y - 12, w: 150, h: 23 };
      const overlap = occupied.some(
        (b) =>
          box.x < b.x + b.w &&
          box.x + box.w > b.x &&
          box.y < b.y + b.h &&
          box.y + box.h > b.y,
      );
      v.label.hidden = !shown || (!selected && overlap);
      if (!v.label.hidden) {
        v.label.style.transform = `translate(${x}px,${y}px) translate(-50%,-50%)`;
        v.label.classList.toggle("selected", selected);
        occupied.push(box);
        labels++;
      }
    }
    this.cb.stats(
      `${cells.toLocaleString()} cells · ${labels} labels · ${this.renderer.info.render.calls} draws`,
    );
  }
  private cellWorld(s: Selection) {
    const v = this.visuals.get(s.viewId)!;
    return v.group.position
      .clone()
      .add(
        new THREE.Vector3(
          (s.col - (v.view.values[0].length - 1) / 2) * PITCH,
          ((v.view.values.length - 1) / 2 - s.row) * PITCH,
          0.35,
        ),
      );
  }
  private hit(e: PointerEvent | MouseEvent): Selection | null {
    const b = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(
      ((e.clientX - b.left) / b.width) * 2 - 1,
      (-(e.clientY - b.top) / b.height) * 2 + 1,
    );
    this.ray.setFromCamera(this.pointer, this.camera);
    const hits = this.ray.intersectObjects(
      [...this.visuals.values()]
        .filter((v) => v.group.visible && v.mesh)
        .map((v) => v.mesh!),
      false,
    );
    if (hits.length && hits[0].instanceId !== undefined) {
      const id = hits[0].object.userData.viewId,
        v = this.visuals.get(id)!,
        index = hits[0].instanceId;
      return {
        viewId: id,
        row: Math.floor(index / v.view.values[0].length),
        col: index % v.view.values[0].length,
      };
    }
    return null;
  }
  setContext(layer: number, head: number, token: number) {
    this.layer = layer;
    this.head = head;
    this.token = token;
    this.updateEmphasis();
    this.invalidate();
  }
  select(s: Selection, a: Arithmetic) {
    this.selection = s;
    this.arithmetic = a;
    this.playing = false;
    this.progress = 0;
    this.buildFocus();
    this.updateEmphasis();
    this.cb.progress(0, false);
    this.invalidate();
  }
  private highlight(s: Selection, color: THREE.Color, opacity = 1) {
    const edges = new THREE.LineSegments(
      this.edgesGeometry,
      new THREE.LineBasicMaterial({ color, transparent: true, opacity }),
    );
    edges.scale.set(CELL + 0.08, CELL + 0.08, 0.35);
    edges.position.copy(this.cellWorld(s));
    this.focus.add(edges);
  }
  private buildFocus() {
    this.disposeGroup(this.focus);
    this.flows = [];
    this.bars = [];
    this.accumulator = null;
    this.readout = null;
    this.readoutCanvas = null;
    this.lastReadout = "";
    if (!this.selection || !this.arithmetic) return;
    const a = this.arithmetic,
      dest = this.cellWorld(this.selection),
      target = this.visuals.get(this.selection.viewId)!;
    this.highlight(this.selection, AMBER);
    const values = a.terms.map((t) => t.value),
      partial = values.reduce<number[]>(
        (arr, value) => [...arr, (arr.at(-1) ?? 0) + value],
        [],
      ),
      max = Math.max(
        0.001,
        ...values.map(Math.abs),
        ...(a.mode === "sum" || (a.mode === "softmax" && a.divide)
          ? partial.map(Math.abs)
          : []),
        Math.abs(a.result),
      );
    const center = target.group.position
      .clone()
      .add(new THREE.Vector3(0, -target.height / 2 - 5, 3));
    const boardWidth = Math.max(5, a.terms.length * 0.46);
    const baseline = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        center.clone().add(new THREE.Vector3(-boardWidth / 2 - 0.5, 0, 0)),
        center.clone().add(new THREE.Vector3(boardWidth / 2 + 0.5, 0, 0)),
      ]),
      new THREE.LineBasicMaterial({
        color: "#65757c",
        transparent: true,
        opacity: 0.7,
      }),
    );
    this.focus.add(baseline);
    a.terms.forEach((term, i) => {
      const color = term.value >= 0 ? POS : NEG;
      if (term.source) {
        this.highlight(term.source, color, 0.85);
        const start = this.cellWorld(term.source),
          mid = start.clone().lerp(dest, 0.5);
        mid.z += 4 + i * 0.09;
        const curve = new THREE.CatmullRomCurve3([start, mid, dest]);
        const line = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(curve.getPoints(35)),
          new THREE.LineBasicMaterial({
            color,
            transparent: true,
            opacity: 0.22,
          }),
        );
        this.focus.add(line);
        const dot = new THREE.Mesh(
          this.sphere,
          new THREE.MeshBasicMaterial({ color }),
        );
        dot.position.copy(start);
        dot.scale.setScalar(
          0.55 + 1.25 * Math.sqrt(Math.abs(term.value) / max),
        );
        this.focus.add(dot);
        this.flows.push({ curve, dot, value: term.value, start, end: dest });
      }
      if (term.second) this.highlight(term.second, AMBER, 0.65);
      const bar = new THREE.Mesh(
        this.box,
        new THREE.MeshBasicMaterial({
          color,
          transparent: true,
          opacity: 0.85,
        }),
      );
      const height = Math.max(0.015, (Math.abs(term.value) / max) * 2.8);
      bar.scale.set(0.32, height, 0.16);
      bar.position
        .copy(center)
        .add(
          new THREE.Vector3(
            (i - (a.terms.length - 1) / 2) * 0.46,
            (Math.sign(term.value) * height) / 2,
            0,
          ),
        );
      bar.userData.termIndex = i;
      this.focus.add(bar);
      this.bars.push(bar);
    });
    if (a.terms.length) {
      this.accumulatorOrigin
        .copy(center)
        .add(new THREE.Vector3(boardWidth / 2 + 1.25, 0, 0));
      this.accumulatorScale = 2.8 / max;
      const ghost = new THREE.Mesh(
        this.box,
        new THREE.MeshBasicMaterial({
          color: AMBER,
          transparent: true,
          opacity: 0.13,
        }),
      );
      const finalValue =
        a.mode === "sum"
          ? a.result
          : a.mode === "softmax" && a.divide
            ? values.reduce((s, v) => s + v, 0)
            : a.result;
      const height = Math.max(
        0.01,
        Math.abs(finalValue) * this.accumulatorScale,
      );
      ghost.scale.set(0.6, height, 0.1);
      ghost.position
        .copy(this.accumulatorOrigin)
        .add(new THREE.Vector3(0, (Math.sign(finalValue) * height) / 2, -0.2));
      this.focus.add(ghost);
      this.accumulator = new THREE.Mesh(
        this.box,
        new THREE.MeshBasicMaterial({ color: AMBER }),
      );
      this.focus.add(this.accumulator);
      const canvas = document.createElement("canvas");
      canvas.width = 512;
      canvas.height = 96;
      this.readoutCanvas = canvas;
      this.readout = new THREE.CanvasTexture(canvas);
      this.readout.colorSpace = THREE.SRGBColorSpace;
      const label = new THREE.Mesh(
        this.plane,
        new THREE.MeshBasicMaterial({
          map: this.readout,
          transparent: true,
          depthWrite: false,
        }),
      );
      label.scale.set(7.2, 1.35, 1);
      label.position
        .copy(this.accumulatorOrigin)
        .add(new THREE.Vector3(0, -4.2, 0.3));
      this.focus.add(label);
    }
    this.renderProgress();
  }
  setProgress(p: number) {
    this.progress = Math.max(0, Math.min(1, p));
    this.playing = false;
    this.renderProgress();
    this.cb.progress(this.progress, false);
    this.invalidate();
  }
  play() {
    if (this.playing) {
      this.playing = false;
      this.cb.progress(this.progress, false);
    } else {
      if (this.progress > 0.995) this.progress = 0;
      this.playStart = performance.now() - this.progress * 9000;
      this.playing = true;
      this.invalidate();
    }
  }
  private renderProgress() {
    const a = this.arithmetic,
      n = Math.max(1, a?.terms.length ?? 1);
    this.flows.forEach((flow, i) => {
      const p = Math.max(0, Math.min(1, this.progress * n - i));
      flow.dot.position.copy(flow.curve.getPoint(p));
      flow.dot.visible = p < 1 && p > 0 && flow.value !== 0;
    });
    this.bars.forEach((bar, i) => {
      (bar.material as THREE.MeshBasicMaterial).opacity =
        this.progress * n >= i + 1 - 0.000001 ? 0.95 : 0.16;
    });
    if (a && this.accumulator) {
      const k = Math.min(n, Math.floor(this.progress * n + 0.000001)),
        additive = a.mode === "sum" || (a.mode === "softmax" && a.divide);
      let value = additive
        ? a.terms.slice(0, k).reduce((s, t) => s + t.value, 0)
        : k
          ? a.terms[Math.min(k - 1, a.terms.length - 1)].value
          : 0;
      if (this.progress >= 1) {
        if (a.mode === "sum") {
          value += a.bias ?? 0;
          if (a.divide) value /= a.divide;
        } else value = a.result;
      }
      const height = Math.max(0.008, Math.abs(value) * this.accumulatorScale);
      this.accumulator.scale.set(0.6, height, 0.2);
      this.accumulator.position
        .copy(this.accumulatorOrigin)
        .add(new THREE.Vector3(0, (Math.sign(value) * height) / 2, 0));
      (this.accumulator.material as THREE.MeshBasicMaterial).color.copy(
        value >= 0 ? POS : NEG,
      );
      const label = `${additive ? "Σ" : "value"} ${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
      if (label !== this.lastReadout && this.readoutCanvas) {
        const ctx = this.readoutCanvas.getContext("2d")!;
        ctx.clearRect(0, 0, 512, 96);
        ctx.fillStyle = "#f4dfb3";
        ctx.font = "500 40px ui-monospace, Menlo, monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, 256, 48);
        this.readout!.needsUpdate = true;
        this.lastReadout = label;
      }
    }
  }
  private updateEmphasis() {
    for (const v of this.visuals.values()) {
      const relevant =
        v.view.block === this.layer &&
        (v.view.head === undefined || v.view.head === this.head);
      const opacity = this.chapter === "overview" ? 0.7 : relevant ? 1 : 0.38;
      if (v.mesh)
        (v.mesh.material as THREE.MeshBasicMaterial).opacity = opacity;
      (v.plate.material as THREE.MeshBasicMaterial).opacity = opacity * 0.42;
    }
  }
  navigate(chapter: Chapter, layer: number, head: number, token: number) {
    this.chapter = chapter;
    this.layer = layer;
    this.head = head;
    this.token = token;
    this.updateEmphasis();
    const x = layer * 164,
      y =
        (this.visuals.get(`b${layer}.h${head}.attention`)?.view.position[1] ??
          -2) + 2;
    const poses: Record<Chapter, [number, number, number]> = {
      overview: [this.ir!.extent / 2, 0, this.ir!.extent * 0.89],
      embedding: [-5, 0, 48],
      projection: [x + 23, y, 53],
      attention: [x + 40, y + 1, 37],
      write: [x + 61, y - 1, 52],
      residual: [x + 85, 0, 48],
      mlp: [x + 126, 0, 76],
      output: [this.ir!.extent - 17, 0, 58],
      parameters: [90, -125, 250],
    };
    const [cx, cy, distance] = poses[chapter];
    this.fly(new THREE.Vector3(cx, cy, 0), distance);
  }
  private fly(target: THREE.Vector3, distance: number) {
    const offset = this.flat
      ? new THREE.Vector3(0, 0, distance)
      : new THREE.Vector3(distance * 0.18, distance * 0.18, distance);
    this.flight = {
      start: performance.now(),
      duration: this.reduced ? 1 : 1050,
      from: this.camera.position.clone(),
      to: target.clone().add(offset),
      lookFrom: this.controls.target.clone(),
      lookTo: target,
    };
    this.invalidate();
  }
  zoomTo(id: string) {
    const v = this.visuals.get(id);
    if (!v) return;
    this.fly(
      v.group.position.clone().add(new THREE.Vector3(0, -2, 0)),
      Math.max(16, v.height * 2.25, v.width * 2.1),
    );
    this.cb.select({
      viewId: id,
      row: Math.min(this.selection?.row ?? 0, v.view.values.length - 1),
      col: Math.min(
        this.selection?.viewId === id ? this.selection.col : this.token,
        v.view.values[0].length - 1,
      ),
    });
  }
  setFlat(value: boolean) {
    this.flat = value;
    if (this.selection) this.zoomTo(this.selection.viewId);
    else this.navigate(this.chapter, this.layer, this.head, this.token);
  }
  focusCalculation() {
    if (!this.selection) return;
    const v = this.visuals.get(this.selection.viewId)!,
      bounds = new THREE.Box3();
    bounds.expandByPoint(
      v.group.position
        .clone()
        .add(new THREE.Vector3(-v.width / 2, -v.height / 2 - 8, -1)),
    );
    bounds.expandByPoint(
      v.group.position
        .clone()
        .add(new THREE.Vector3(v.width / 2, v.height / 2 + 3, 2)),
    );
    for (const term of this.arithmetic?.terms ?? [])
      for (const s of [term.source, term.second])
        if (s) bounds.expandByPoint(this.cellWorld(s));
    const center = bounds.getCenter(new THREE.Vector3()),
      size = bounds.getSize(new THREE.Vector3()),
      fov = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)),
      distance =
        Math.max(
          size.y / 2 / fov,
          size.x / 2 / (fov * this.camera.aspect),
          10,
        ) *
          1.5 +
        size.z;
    this.fly(center, distance);
  }
  private render(time: number) {
    this.frame = 0;
    if (document.hidden) return;
    if (time - this.lastTime < 25) {
      this.frame = requestAnimationFrame((t) => this.render(t));
      return;
    }
    this.inFrame = true;
    this.lastTime = time;
    let moving = false;
    if (this.flight) {
      const p = Math.min(1, (time - this.flight.start) / this.flight.duration),
        t = p * p * (3 - 2 * p);
      this.camera.position.lerpVectors(this.flight.from, this.flight.to, t);
      this.controls.target.lerpVectors(
        this.flight.lookFrom,
        this.flight.lookTo,
        t,
      );
      if (p >= 1) this.flight = null;
      moving = true;
    }
    if (this.playing) {
      this.progress = Math.min(1, (time - this.playStart) / 9000);
      this.renderProgress();
      if (this.progress >= 1) this.playing = false;
      this.cb.progress(this.progress, this.playing);
      moving = true;
    }
    moving = this.controls.update() || moving;
    if (this.dirty || moving) {
      this.updateLOD();
      this.updateEmphasis();
      this.renderer.render(this.scene, this.camera);
      this.dirty = false;
      this.tick++;
    }
    this.inFrame = false;
    if (moving || this.flight || this.playing)
      this.frame = requestAnimationFrame((t) => this.render(t));
  }
  dispose() {
    if (this.frame) cancelAnimationFrame(this.frame);
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.disposeGroup(this.root);
    this.disposeGroup(this.edges);
    this.disposeGroup(this.focus);
    this.box.dispose();
    this.plane.dispose();
    this.sphere.dispose();
    this.edgesGeometry.dispose();
    this.renderer.dispose();
    this.labels.remove();
  }
}
