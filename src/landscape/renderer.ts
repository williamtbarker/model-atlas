import * as THREE from "three";
import { headGrouping } from "./visual-encoding";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  projectLandscapeLinks,
  type Landscape,
  type Glyph,
  type BlockLayout,
  type LandscapeLink,
} from "./layout";

const COLORS = {
  attention: 0x64cbc9,
  expert: 0xdcb179,
  residual: 0xc392b5,
  parameter: 0x9fbcac,
  neutral: 0x71989c,
};
const BG = 0x080e13;
const BUDGET = {
  objects: 650,
  labels: 42,
  pixels: 1.7,
  expertMarks: 48000,
  drawCalls: 1800,
};
type Visual = {
  glyph: Glyph;
  group: THREE.Group | null;
  label: HTMLButtonElement | null;
  cost?: number;
  instances?: number;
};
type Callbacks = {
  select: (id: string) => void;
  position: (id: string | null, mode: string) => void;
  stats: (s: string) => void;
};
export class LandscapeRenderer {
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(43, 1, 0.1, 18000);
  readonly renderer: THREE.WebGLRenderer;
  readonly controls: OrbitControls;
  private data: Landscape | null = null;
  private visuals = new Map<string, Visual>();
  private root = new THREE.Group();
  private edgeGroup = new THREE.Group();
  private focus = new THREE.Group();
  private labelLayer = document.createElement("div");
  private plane = new THREE.PlaneGeometry(1, 1);
  private diamond = new THREE.CircleGeometry(1, 4);
  private sphere = new THREE.SphereGeometry(1, 8, 6);
  private ray = new THREE.Raycaster();
  private pointer = new THREE.Vector2();
  private down = new THREE.Vector2();
  private resizeObserver: ResizeObserver;
  private frame = 0;
  private dirty = true;
  private inFrame = false;
  private lastTime = 0;
  private selected: string | null = null;
  private block: BlockLayout | null = null;
  private mode = "model";
  private flat = false;
  private visibleKey = "";
  private visibleGlyphs = new Set<string>();
  private paths: {
    link: LandscapeLink;
    start: number;
    count: number;
    color: THREE.Color;
    residual: boolean;
  }[] = [];
  private edgeGeometry: THREE.BufferGeometry | null = null;
  private blockLevels = new Map<string, number>();
  private focusedExpert: string | null = null;
  private detailLabels: {
    id: string;
    text: string;
    position: THREE.Vector3;
    node?: HTMLButtonElement;
  }[] = [];
  private clearDetailLabels() {
    for (const item of this.detailLabels) item.node?.remove();
    this.detailLabels = [];
  }
  private flight: {
    start: number;
    duration: number;
    from: THREE.Vector3;
    to: THREE.Vector3;
    lookFrom: THREE.Vector3;
    lookTo: THREE.Vector3;
  } | null = null;
  private reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  constructor(
    private host: HTMLElement,
    private cb: Callbacks,
  ) {
    this.scene.background = new THREE.Color(BG);
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "low-power",
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, BUDGET.pixels));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setClearColor(BG);
    this.renderer.domElement.setAttribute(
      "aria-label",
      "Continuous model architecture. Scroll to reveal operations, click to inspect, drag to orbit.",
    );
    this.renderer.domElement.tabIndex = 0;
    host.append(this.renderer.domElement);
    this.labelLayer.className = "landscape-labels";
    host.append(this.labelLayer);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.13;
    this.controls.minDistance = 8;
    this.controls.maxDistance = 6000;
    this.controls.maxPolarAngle = Math.PI;
    this.controls.addEventListener("change", () => this.invalidate());
    this.controls.addEventListener("start", () => {
      this.flight = null;
    });
    this.scene.add(this.root, this.edgeGroup, this.focus);
    this.camera.position.set(500, -480, 1900);
    this.controls.target.set(500, -480, 0);
    this.controls.update();
    this.renderer.domElement.addEventListener("pointerdown", (e) =>
      this.down.set(e.clientX, e.clientY),
    );
    this.renderer.domElement.addEventListener("pointerup", (e) => {
      if (this.down.distanceTo(new THREE.Vector2(e.clientX, e.clientY)) > 5)
        return;
      const hit = this.hit(e);
      if (hit) this.cb.select(hit);
    });
    this.renderer.domElement.addEventListener("dblclick", (e) => {
      const hit = this.hit(e);
      if (hit) this.approach(hit);
    });
    this.renderer.domElement.addEventListener("pointermove", (e) => {
      if (!e.buttons)
        this.renderer.domElement.style.cursor = this.hit(e)
          ? "pointer"
          : "grab";
    });
    this.renderer.domElement.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      this.cb.stats("Graphics context lost. Reload to recover.");
    });
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);
    this.resize();
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) this.invalidate();
    });
  }
  private resize() {
    const w = this.host.clientWidth,
      h = this.host.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.invalidate();
  }
  private invalidate() {
    this.dirty = true;
    if (!this.frame && !this.inFrame)
      this.frame = requestAnimationFrame((t) => this.render(t));
  }
  private disposeGroup(group: THREE.Object3D) {
    group.traverse((o) => {
      const m = o as THREE.Mesh;
      if (
        m.geometry &&
        ![this.plane, this.diamond, this.sphere].includes(m.geometry as any)
      )
        m.geometry.dispose();
      if (m.material)
        for (const mat of Array.isArray(m.material)
          ? m.material
          : [m.material]) {
          (mat as THREE.MeshBasicMaterial).map?.dispose();
          mat.dispose();
        }
      if (o instanceof THREE.InstancedMesh) o.dispose();
    });
    group.clear();
  }
  setScene(data: Landscape) {
    this.disposeGroup(this.root);
    this.disposeGroup(this.edgeGroup);
    this.disposeGroup(this.focus);
    this.labelLayer.replaceChildren();
    this.visuals.clear();
    this.clearDetailLabels();
    this.data = data;
    this.visibleKey = "";
    this.paths = [];
    this.blockLevels.clear();
    for (const glyph of data.glyphs)
      this.visuals.set(glyph.id, { glyph, group: null, label: null });
    this.block = data.blocks.find((b) => !b.auxiliary) ?? null;
    this.mode = "model";
    this.selected = null;
    this.focusedExpert = null;
    this.fitModel();
    this.invalidate();
  }
  private line(points: THREE.Vector3[], color: number, opacity = 1) {
    return new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthWrite: false,
      }),
    );
  }
  private segments(points: number[], color: number, opacity = 1) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(points, 3),
    );
    return new THREE.LineSegments(
      geometry,
      new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthWrite: false,
      }),
    );
  }
  private mesh(geometry: THREE.BufferGeometry, color: number, opacity = 1) {
    return new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({
        color,
        transparent: opacity < 1,
        opacity,
        side: THREE.DoubleSide,
        depthWrite: opacity === 1,
      }),
    );
  }
  private mark(
    group: THREE.Group,
    x: number,
    y: number,
    w: number,
    h: number,
    color: number,
    z = 0,
    opacity = 1,
  ) {
    const m = this.mesh(this.plane, color, opacity);
    m.position.set(x, y, z);
    m.scale.set(w, h, 1);
    group.add(m);
    return m;
  }
  private marks(
    group: THREE.Group,
    rects: number[][],
    color: number,
    opacity = 1,
  ) {
    if (!rects.length) return;
    const m = new THREE.InstancedMesh(
      this.plane,
      new THREE.MeshBasicMaterial({
        color,
        transparent: opacity < 1,
        opacity,
        side: THREE.DoubleSide,
        depthWrite: opacity === 1,
      }),
      rects.length,
    );
    const mat = new THREE.Matrix4();
    rects.forEach(([x, y, w, h, z = 0], i) => {
      mat.makeScale(w, h, 1);
      mat.setPosition(x, y, z);
      m.setMatrixAt(i, mat);
    });
    group.add(m);
  }
  private declared(g: Glyph, key: string): unknown {
    if (g.attrs?.[key] !== undefined) return g.attrs[key];
    return this.data!.index.ancestors(g.entityId)
      .slice()
      .reverse()
      .find((e: any) => e.attrs?.[key] !== undefined)?.attrs?.[key];
  }
  private headCount(g: Glyph) {
    return Number(
      this.declared(g, "heads") ??
        this.declared(g, "queryHeads") ??
        g.attrs?.count ??
        1,
    );
  }
  private grid(
    group: THREE.Group,
    w: number,
    h: number,
    nx: number,
    ny: number,
    color: number,
    opacity = 0.35,
    z = 0,
  ) {
    const a: number[] = [];
    for (let i = 0; i <= nx; i++) {
      const x = -w / 2 + (w * i) / nx;
      a.push(x, -h / 2, z, x, h / 2, z);
    }
    for (let j = 0; j <= ny; j++) {
      const y = -h / 2 + (h * j) / ny;
      a.push(-w / 2, y, z, w / 2, y, z);
    }
    group.add(this.segments(a, color, opacity));
  }
  private matrixSheet(
    group: THREE.Group,
    w: number,
    h: number,
    color: number,
    shape?: any[],
    x = 0,
    y = 0,
    z = 0,
  ) {
    const sheet = new THREE.Group();
    sheet.position.set(x, y, z);
    group.add(sheet);
    this.mark(sheet, 0, 0, w, h, color, 0, 0.06);
    const ratio = shape?.length === 2 ? Number(shape[1]) / Number(shape[0]) : 1;
    const nx = Math.max(3, Math.min(16, Math.round(7 * Math.sqrt(ratio))));
    const ny = Math.max(3, Math.min(16, Math.round(7 / Math.sqrt(ratio))));
    this.grid(sheet, w, h, nx, ny, color, 0.42, 0.02);
    const outline = [
      new THREE.Vector3(-w / 2, -h / 2, 0.02),
      new THREE.Vector3(-w / 2, h / 2, 0.02),
      new THREE.Vector3(w / 2, h / 2, 0.02),
      new THREE.Vector3(w / 2, -h / 2, 0.02),
    ];
    sheet.add(
      new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints(outline),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8 }),
      ),
    );
    return sheet;
  }
  private createVisual(v: Visual) {
    const g = v.glyph,
      group = new THREE.Group();
    group.position.fromArray(g.position);
    group.userData.entityId = g.entityId;
    v.group = group;
    this.root.add(group);
    const c = COLORS[g.role],
      w = g.width,
      h = g.height;
    const hit = this.mesh(this.plane, 0xffffff, 0);
    hit.scale.set(Math.max(5, w), Math.max(5, h), 1);
    hit.position.z = -0.3;
    hit.material.depthWrite = false;
    hit.userData.entityId = g.entityId;
    group.add(hit);
    if (g.kind === "stream") {
      const n = Number(g.attrs?.streams ?? g.attrs?.residualStreams ?? 1);
      const lines: number[][] = [];
      for (let i = 0; i < n; i++)
        lines.push([0, (i - (n - 1) / 2) * 1.6, w, 0.2, 0]);
      this.marks(group, lines, COLORS.residual, 0.8);
      if (g.attrs?.semanticBackbone) {
        const block = this.data!.blocks.find((b) => b.id === g.blockId);
        const direction = block?.direction ?? 1;
        const a = new THREE.Shape();
        a.moveTo(0, 0);
        a.lineTo(-direction * 1.4, 0.8);
        a.lineTo(-direction * 1.4, -0.8);
        a.closePath();
        const arrow = this.mesh(
          new THREE.ShapeGeometry(a),
          COLORS.residual,
          0.9,
        );
        arrow.position.set(direction * (w / 2 - 2), 0, 0.3);
        group.add(arrow);
      }
    } else if (g.kind === "heads") {
      const n = this.headCount(g);
      const shared = String(this.declared(g, "KV") ?? "").includes("shared");
      const kv = Number(this.declared(g, "kvHeads") ?? (shared ? 1 : n));
      const grouping = headGrouping(n, kv);
      const points: number[] = [];
      for (const head of grouping.heads) {
        const x = head.queryX * w * 0.95,
          base = head.kvX * w * 0.72;
        let px = base,
          py = -h * 0.35,
          pz = 0;
        for (let j = 1; j <= 7; j++) {
          const t = j / 7,
            nx = base + (x - base) * t * t,
            ny = -h * 0.35 + h * 0.77 * t,
            nz = Math.sin(t * Math.PI) * Math.min(2.5, h * 0.12);
          points.push(px, py, pz, nx, ny, nz);
          px = nx;
          py = ny;
          pz = nz;
        }
      }
      group.add(this.segments(points, c, 0.45));
      this.marks(
        group,
        grouping.keys.map((key) => [
          key.x * w * 0.72,
          -h * 0.35,
          kv === 1 ? w * 0.16 : Math.max(0.1, (w / kv) * 0.45),
          h * 0.045,
          0.3,
        ]),
        c,
        0.9,
      );
      this.mark(group, 0, h * 0.42, w * 0.96, 0.22, c, 0.3, 0.9);
      this.marks(
        group,
        grouping.heads.map((head) => [
          head.queryX * w * 0.95,
          h * 0.46,
          Math.max(0.1, (w / n) * 0.56),
          h * 0.06,
          0.3,
        ]),
        c,
        0.95,
      );
      if (g.attrs?.semanticSummary) {
        const ratio = Number(g.attrs?.compressionRatio ?? 0);
        if (ratio) {
          const bank = new THREE.Group();
          bank.position.set(w * 0.36, -h * 0.4, 0);
          group.add(bank);
          this.marks(
            bank,
            Array.from({ length: 8 }, (_, i) => [
              (i - 3.5) * w * 0.023,
              0,
              w * 0.012,
              h * 0.14,
            ]),
            c,
            0.5,
          );
        }
      }
    } else if (g.kind === "state") {
      // A recurrent state is a fixed-size memory, not a token-by-token attention grid.
      const stateShape = Array.isArray(g.attrs?.stateShape)
        ? g.attrs.stateShape
        : g.shape;
      this.matrixSheet(
        group,
        w * 0.38,
        h * 0.59,
        c,
        stateShape?.slice(-2),
        0,
        0,
        0.3,
      );
      if (
        g.attrs?.sourceKind === "LinearAttention" ||
        g.attrs?.temporalRecurrence === true
      ) {
        const loop = new THREE.CubicBezierCurve3(
          new THREE.Vector3(w * 0.2, 0, 0.2),
          new THREE.Vector3(w * 0.52, h * 0.72, 1),
          new THREE.Vector3(-w * 0.52, h * 0.72, 1),
          new THREE.Vector3(-w * 0.2, 0, 0.2),
        );
        group.add(this.line(loop.getPoints(28), c, 0.65));
        group.add(
          this.segments(
            [
              -w * 0.2,
              0,
              0.3,
              -w * 0.24,
              h * 0.12,
              0.3,
              -w * 0.2,
              0,
              0.3,
              -w * 0.32,
              h * 0.025,
              0.3,
            ],
            c,
            0.9,
          ),
        );
        group.add(
          this.line(
            [
              new THREE.Vector3(-w * 0.48, -h * 0.28, 0),
              new THREE.Vector3(-w * 0.2, -h * 0.13, 0),
            ],
            c,
            0.75,
          ),
        );
        group.add(
          this.line(
            [
              new THREE.Vector3(w * 0.2, -h * 0.13, 0),
              new THREE.Vector3(w * 0.48, -h * 0.28, 0),
            ],
            c,
            0.75,
          ),
        );
      }
    } else if (g.kind === "convolution") {
      const n = Math.max(
        1,
        Math.min(12, Number(g.attrs?.kernelSize ?? g.attrs?.convKernel ?? 1)),
      );
      this.grid(group, w, h * 0.4, n, 1, c, 0.55, 0);
      const a: number[] = [];
      for (let i = 0; i < n; i++) {
        const x = ((i + 0.5) / n - 0.5) * w;
        a.push(x, -h * 0.2, 0, 0, -h * 0.42, 0.1);
      }
      group.add(this.segments(a, c, 0.5));
    } else if (g.kind === "gate") {
      group.add(
        this.mesh(
          new THREE.RingGeometry(
            Math.min(w, h) * 0.28,
            Math.min(w, h) * 0.32,
            24,
          ),
          c,
          0.8,
        ),
      );
      group.add(
        this.segments(
          [
            -w * 0.12,
            -h * 0.12,
            0.1,
            w * 0.12,
            h * 0.12,
            0.1,
            -w * 0.12,
            h * 0.12,
            0.1,
            w * 0.12,
            -h * 0.12,
            0.1,
          ],
          c,
          0.9,
        ),
      );
    } else if (g.kind === "rotary") {
      const radius = Math.min(w, h) * 0.38;
      const arc = Array.from(
        { length: 25 },
        (_, i) =>
          new THREE.Vector3(
            Math.cos((i / 24) * Math.PI * 1.65) * radius,
            Math.sin((i / 24) * Math.PI * 1.65) * radius,
            0,
          ),
      );
      group.add(this.line(arc, c, 0.6));
      group.add(
        this.segments(
          [
            0,
            0,
            0.1,
            radius,
            0,
            0.1,
            0,
            0,
            0.1,
            radius * 0.45,
            radius * 0.89,
            0.1,
          ],
          c,
          0.9,
        ),
      );
    } else if (g.kind === "vision") {
      this.matrixSheet(group, w * 0.92, h * 0.65, c, undefined, 0, 0, 0);
      this.grid(group, w * 0.75, h * 0.49, 4, 4, c, 0.3, 0.3);
      // Patch lattice denotes image encoding; no image pixels or activations are invented.
    } else if (g.kind === "experts") {
      const ids = this.expertIds(g);
      const count = Number(
        g.attrs?.count ?? g.attrs?.routedExperts ?? ids.length,
      );
      const n = Math.max(1, Math.min(1024, count));
      const cols = Math.ceil(Math.sqrt((n * w) / h)),
        rows = Math.ceil(n / cols),
        sx = w / cols,
        sy = h / rows;
      const marks = new THREE.InstancedMesh(
        this.plane,
        new THREE.MeshBasicMaterial({ color: c, side: THREE.DoubleSide }),
        n,
      );
      const matrix = new THREE.Matrix4();
      for (let i = 0; i < n; i++) {
        matrix.makeScale(sx * 0.64, sy * 0.55, 1);
        matrix.setPosition(
          ((i % cols) - (cols - 1) / 2) * sx,
          (Math.floor(i / cols) - (rows - 1) / 2) * sy,
          0,
        );
        marks.setMatrixAt(i, matrix);
      }
      marks.userData.expertIds = ids;
      group.add(marks);
      const brackets: number[] = [];
      const q = Math.min(w, h) * 0.12;
      for (const sign of [-1, 1])
        brackets.push(
          (sign * w) / 2,
          -h / 2 - q,
          0,
          (sign * w) / 2,
          h / 2 + q,
          0,
          (sign * w) / 2,
          h / 2 + q,
          0,
          sign * (w / 2 - q),
          h / 2 + q,
          0,
        );
      group.add(this.segments(brackets, c, 0.3));
    } else if (
      g.kind === "matrix" ||
      g.kind === "embedding" ||
      g.kind === "output"
    ) {
      if (g.kind === "embedding" || g.kind === "output") {
        for (let i = 2; i >= 0; i--)
          this.matrixSheet(
            group,
            w,
            h,
            c,
            g.shape,
            i * 1.9,
            -i * 1.3,
            -i * 1.2,
          );
      } else this.matrixSheet(group, w, h, c, g.shape);
    } else if (g.kind === "mix") {
      if (
        g.attrs?.sourceKind === "HyperConnectionMix" &&
        typeof g.attrs?.streams === "number"
      ) {
        const n = Math.max(1, Math.min(8, Number(g.attrs.streams))),
          points: number[] = [],
          ports: number[][] = [];
        for (let a = 0; a < n; a++)
          for (let b = 0; b < n; b++)
            points.push(
              -w / 2,
              ((a - (n - 1) / 2) * h) / (n + 1),
              0,
              w / 2,
              ((b - (n - 1) / 2) * h) / (n + 1),
              0,
            );
        group.add(this.segments(points, c, 0.3));
        for (let i = 0; i < n; i++)
          ports.push(
            [-w / 2, ((i - (n - 1) / 2) * h) / (n + 1), 0.5, 0.5],
            [w / 2, ((i - (n - 1) / 2) * h) / (n + 1), 0.5, 0.5],
          );
        this.marks(group, ports, c);
      } else {
        // Addition is not a learned all-to-all residual mixing map.
        group.add(
          this.mesh(
            new THREE.RingGeometry(
              Math.min(w, h) * 0.32,
              Math.min(w, h) * 0.35,
              24,
            ),
            c,
            0.7,
          ),
        );
        group.add(
          this.segments(
            [
              -w * 0.2,
              0,
              0.2,
              w * 0.2,
              0,
              0.2,
              0,
              -h * 0.2,
              0.2,
              0,
              h * 0.2,
              0.2,
            ],
            c,
            0.9,
          ),
        );
      }
    } else if (g.kind === "norm") {
      this.mark(group, 0, 0, w * 0.5, h, c, 0, 0.28);
      this.marks(
        group,
        Array.from({ length: 6 }, (_, i) => [
          0,
          ((i - 2.5) * h) / 6,
          w,
          0.1,
          0.2,
        ]),
        c,
        0.85,
      );
    } else if (g.kind === "router") {
      const m = this.mesh(this.diamond, c, 0.7);
      m.scale.set(w * 0.32, h * 0.5, 1);
      m.rotation.z = Math.PI / 2;
      group.add(m);
      const ports = Math.min(
        12,
        Number(g.attrs?.topK ?? g.attrs?.activeExperts ?? 0),
      );
      this.marks(
        group,
        Array.from({ length: ports }, (_, i) => [
          w * 0.5,
          ((i - (ports - 1) / 2) * h) / (ports + 1),
          w * 0.18,
          0.13,
        ]),
        c,
        0.8,
      );
    } else if (g.kind === "memory") {
      if (g.attrs?.sourceKind === "SparseIndexer") {
        // Empty candidate slots and a selector. No positions are claimed active.
        this.grid(group, w, h * 0.45, 8, 1, c, 0.45, h * 0.02);
        const selector = this.mesh(this.diamond, c, 0.55);
        selector.scale.set(w * 0.13, h * 0.2, 1);
        selector.position.set(0, -h * 0.35, 0.1);
        group.add(selector);
        group.add(
          this.line(
            [
              new THREE.Vector3(-w * 0.4, -h * 0.25, 0),
              new THREE.Vector3(w * 0.4, -h * 0.25, 0),
            ],
            c,
            0.6,
          ),
        );
      } else {
        const ratio = Number(g.attrs?.ratio ?? g.attrs?.compressionRatio ?? 1);
        const n = Math.min(32, Math.max(2, ratio));
        const pts: number[] = [];
        for (let i = 0; i < n; i++) {
          const x = (i / (n - 1) - 0.5) * w;
          pts.push(
            x,
            h * 0.42,
            0,
            x,
            h * 0.17,
            0,
            x,
            h * 0.17,
            0,
            x * 0.18,
            -h * 0.32,
            0,
          );
        }
        group.add(this.segments(pts, c, 0.5));
        this.mark(group, 0, -h * 0.38, w * 0.3, h * 0.14, c, 0, 0.9);
      }
    } else if (g.kind === "activation") {
      const e = this.data!.index.entities.get(g.entityId);
      const tensors = this.parameterChildren(g.entityId);
      if (e?.kind === "SharedExpert" || e?.attrs?.expertIndex !== undefined) {
        tensors
          .slice(0, 3)
          .forEach((item, i) =>
            this.matrixSheet(
              group,
              w * 0.29,
              h * 0.65,
              c,
              item.tensor.shape,
              (i - 1) * w * 0.31,
              (i % 2) * h * 0.15,
              0,
            ),
          );
      } else {
        // The curve is a symbolic nonlinear-operator icon, not sampled activations.
        const curve = Array.from({ length: 33 }, (_, i) => {
          const x = (i / 32) * 5 - 2.5;
          return new THREE.Vector3(
            (x / 5) * w,
            ((x / (1 + Math.exp(-x)) - 0.85) / 3) * h,
            0,
          );
        });
        group.add(this.line(curve, c, 0.9));
        group.add(
          this.segments(
            [
              -w / 2,
              -h * 0.28,
              -0.1,
              w / 2,
              -h * 0.28,
              -0.1,
              -w * 0.3,
              -h / 2,
              -0.1,
              -w * 0.3,
              h / 2,
              -0.1,
            ],
            c,
            0.2,
          ),
        );
      }
    } else {
      const ring = new THREE.RingGeometry(
        Math.min(w, h) * 0.23,
        Math.min(w, h) * 0.27,
        24,
      );
      group.add(this.mesh(ring, c, 0.75));
    }
    v.cost = 0;
    v.instances = 0;
    group.traverse((o) => {
      if (!o.userData.entityId) o.userData.entityId = g.entityId;
      if ((o as THREE.Mesh).isMesh || (o as THREE.Line).isLine) v.cost!++;
      if (o instanceof THREE.InstancedMesh) v.instances! += o.count;
    });
  }
  private expertIds(g: Glyph) {
    const bankId = String(g.attrs?.bankEntityId ?? g.id);
    return this.data!.index.childList(bankId)
      .filter((e: any) => typeof e.attrs?.expertIndex === "number")
      .sort((a: any, b: any) => a.attrs.expertIndex - b.attrs.expertIndex)
      .map((e: any) => e.id);
  }
  private projectedSize(g: Glyph) {
    return (
      (g.width * this.host.clientHeight) /
      (2 *
        Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) *
        this.camera.position.distanceTo(new THREE.Vector3(...g.position)))
    );
  }
  private updateLOD() {
    if (!this.data) return;
    this.camera.updateMatrixWorld();
    const frustum = new THREE.Frustum().setFromProjectionMatrix(
      new THREE.Matrix4().multiplyMatrices(
        this.camera.projectionMatrix,
        this.camera.matrixWorldInverse,
      ),
    );
    const blockLevels = new Map<string, number>();
    for (const b of this.data.blocks) {
      const size =
        (100 * this.host.clientHeight) /
        (2 *
          Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) *
          this.camera.position.distanceTo(new THREE.Vector3(...b.position)));
      const previous = this.blockLevels.get(b.id) ?? 1;
      blockLevels.set(
        b.id,
        previous > 1
          ? size < 305
            ? 1
            : size > 700
              ? 3
              : 2
          : size > 360
            ? 2
            : 1,
      );
    }
    this.blockLevels = blockLevels;
    const ranked = [...this.visuals.values()]
      .map((v) => ({
        v,
        size: this.projectedSize(v.glyph),
        visible: frustum.intersectsSphere(
          new THREE.Sphere(
            new THREE.Vector3(...v.glyph.position),
            Math.hypot(v.glyph.width, v.glyph.height) / 2,
          ),
        ),
      }))
      .sort((a, b) => b.size - a.size);
    const active = new Set<string>();
    let count = 0,
      labels = 0;
    const occupied: { x: number; y: number; w: number; h: number }[] = [];
    let geometry = 0,
      instances = 0;
    for (const { v, size, visible } of ranked) {
      const g = v.glyph,
        level = g.blockId ? blockLevels.get(g.blockId)! : 3;
      let wanted =
        visible &&
        g.level <= level &&
        !(g.attrs?.semanticSummary && level >= 2) &&
        count < BUDGET.objects &&
        geometry + (v.cost ?? 8) <= BUDGET.drawCalls &&
        instances + (v.instances ?? 0) <= BUDGET.expertMarks;
      if (wanted) {
        if (!v.group) this.createVisual(v);
        wanted =
          geometry + v.cost! <= BUDGET.drawCalls &&
          instances + v.instances! <= BUDGET.expertMarks;
      }
      if (wanted) {
        v.group!.visible = true;
        count++;
        active.add(g.id);
        geometry += v.cost!;
        instances += v.instances!;
      } else if (v.group) {
        this.root.remove(v.group);
        this.disposeGroup(v.group);
        v.group = null;
      }
      const selected = g.id === this.selected;
      const isBlock = g.attrs?.semanticBackbone;
      const wantsLabel =
        wanted &&
        (isBlock ? size > 60 : size > 45) &&
        (level > 1 || isBlock || !g.blockId);
      if (wantsLabel && labels < BUDGET.labels - this.detailLabels.length) {
        const p = new THREE.Vector3(...g.position);
        p.y += isBlock ? -68 : g.height / 2 + 3;
        p.project(this.camera);
        const x = (p.x * 0.5 + 0.5) * this.host.clientWidth,
          y = (-p.y * 0.5 + 0.5) * this.host.clientHeight;
        const box = { x: x - 64, y: y - 10, w: 128, h: 22 };
        const uiCollision =
          (x < 255 && y < 310) ||
          (x > this.host.clientWidth - 310 &&
            y > 70 &&
            y < this.host.clientHeight - 120) ||
          y < 65 ||
          y > this.host.clientHeight - 120;
        const overlap = occupied.some(
          (b) =>
            box.x < b.x + b.w &&
            box.x + box.w > b.x &&
            box.y < b.y + b.h &&
            box.y + box.h > b.y,
        );
        const show =
          !uiCollision &&
          (selected || !overlap) &&
          x > 30 &&
          x < this.host.clientWidth - 30;
        if (show) {
          if (!v.label) {
            v.label = document.createElement("button");
            v.label.className = "landscape-label";
            v.label.addEventListener("click", () => {
              this.cb.select(g.entityId);
              this.approach(g.entityId);
            });
            this.labelLayer.append(v.label);
          }
          v.label.replaceChildren();
          const name = document.createElement("span");
          name.textContent = isBlock ? g.label : g.label;
          v.label.append(name);
          if (g.shape && size > 100) {
            const sub = document.createElement("small");
            sub.textContent = g.shape.join(" × ");
            v.label.append(sub);
          }
          v.label.hidden = false;
          v.label.classList.toggle("selected", selected);
          v.label.style.left = x + "px";
          v.label.style.top = y + "px";
          occupied.push(box);
          labels++;
        } else if (v.label) {
          v.label.remove();
          v.label = null;
        }
      } else if (v.label) {
        v.label.remove();
        v.label = null;
      }
    }
    if (this.focusedExpert) {
      const e = this.data.index.entities.get(this.focusedExpert),
        g = e ? this.visuals.get(e.parentId)?.glyph : null;
      this.focus.visible = !!g && this.projectedSize(g) > 260;
    } else this.focus.visible = true;
    for (const item of this.detailLabels) {
      if (!this.focus.visible) {
        item.node?.remove();
        item.node = undefined;
        continue;
      }
      const p = item.position.clone().project(this.camera);
      if (p.z < -1 || p.z > 1) {
        item.node?.remove();
        item.node = undefined;
        continue;
      }
      if (!item.node) {
        item.node = document.createElement("button");
        item.node.className = "landscape-label detail-label";
        item.node.textContent = item.text;
        item.node.addEventListener("click", () => this.cb.select(item.id));
        this.labelLayer.append(item.node);
      }
      item.node.style.left = (p.x * 0.5 + 0.5) * this.host.clientWidth + "px";
      item.node.style.top = (-p.y * 0.5 + 0.5) * this.host.clientHeight + "px";
      labels++;
    }
    const key = [...active].sort().join("|");
    if (key !== this.visibleKey) {
      this.visibleKey = key;
      this.visibleGlyphs = active;
      this.rebuildLinks();
    }
    this.cb.stats(
      `${count} visible components · ${labels} labels · ${this.data.blocks.filter((b) => !b.auxiliary).length} core blocks`,
    );
  }
  private rebuildLinks() {
    this.disposeGroup(this.edgeGroup);
    this.paths = [];
    if (!this.data) return;
    const vertices: number[] = [],
      colors: number[] = [];
    const links = projectLandscapeLinks(this.data, this.visibleGlyphs);
    for (const link of links) {
      const a = this.visuals.get(link.from)?.glyph,
        b = this.visuals.get(link.to)?.glyph;
      if (!a || !b) continue;
      const ab = this.data.blocks.find((k) => k.id === a.blockId),
        bb = this.data.blocks.find((k) => k.id === b.blockId);
      const crossBlock = ab && bb && ab.id !== bb.id;
      const residual =
        link.kind === "residual" ||
        !!(a.attrs?.semanticBackbone && b.attrs?.semanticBackbone);
      const color = residual ? COLORS.residual : COLORS[b.role];
      const multiplicity = crossBlock ? Math.min(ab.streams, bb.streams) : 1;
      for (let lane = 0; lane < multiplicity; lane++) {
        const from = new THREE.Vector3(...a.position),
          to = new THREE.Vector3(...b.position);
        if (crossBlock) {
          from.x += (ab.direction * a.width) / 2;
          to.x -= (bb.direction * b.width) / 2;
          from.y += (lane - (multiplicity - 1) / 2) * 1.6;
          to.y += (lane - (multiplicity - 1) / 2) * 1.6;
        }
        let curve: THREE.CubicBezierCurve3;
        const dy = Math.abs(to.y - from.y),
          dx = to.x - from.x;
        if (crossBlock && dy > 50) {
          const bend = ab.direction * (22 + lane * 2);
          curve = new THREE.CubicBezierCurve3(
            from,
            from.clone().add(new THREE.Vector3(bend, 0, 1)),
            to.clone().add(new THREE.Vector3(bend, 0, 1)),
            to,
          );
        } else {
          const bend = Math.min(18, Math.max(4, Math.abs(dx) * 0.32));
          curve = new THREE.CubicBezierCurve3(
            from,
            from
              .clone()
              .add(new THREE.Vector3(Math.sign(dx || 1) * bend, 0, 1.1)),
            to
              .clone()
              .add(new THREE.Vector3(-Math.sign(dx || 1) * bend, 0, 1.1)),
            to,
          );
        }
        const points = curve.getPoints(crossBlock ? 24 : 12),
          start = vertices.length / 3;
        for (let i = 1; i < points.length; i++)
          vertices.push(...points[i - 1].toArray(), ...points[i].toArray());
        if (!crossBlock) {
          const p = curve.getPoint(0.86),
            t = curve.getTangent(0.86),
            perp = new THREE.Vector3(-t.y, t.x, 0);
          const back = p.clone().addScaledVector(t, -0.8);
          vertices.push(
            ...p.toArray(),
            ...back.clone().addScaledVector(perp, 0.35).toArray(),
            ...p.toArray(),
            ...back.clone().addScaledVector(perp, -0.35).toArray(),
          );
        }
        const count = vertices.length / 3 - start;
        for (let i = 0; i < count; i++) colors.push(0, 0, 0);
        this.paths.push({
          link,
          start,
          count,
          color: new THREE.Color(color),
          residual,
        });
      }
    }
    this.edgeGeometry = new THREE.BufferGeometry();
    this.edgeGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(vertices, 3),
    );
    this.edgeGeometry.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(colors, 3),
    );
    this.edgeGroup.add(
      new THREE.LineSegments(
        this.edgeGeometry,
        new THREE.LineBasicMaterial({ vertexColors: true, depthWrite: false }),
      ),
    );
    this.emphasize();
  }
  private emphasize() {
    const colors = this.edgeGeometry?.getAttribute("color") as
      | THREE.BufferAttribute
      | undefined;
    if (colors) {
      for (const path of this.paths) {
        const selected =
          path.link.from === this.selected || path.link.to === this.selected;
        const c = path.color
          .clone()
          .multiplyScalar(selected ? 1 : path.residual ? 0.6 : 0.32);
        for (let i = path.start; i < path.start + path.count; i++)
          colors.setXYZ(i, c.r, c.g, c.b);
      }
      colors.needsUpdate = true;
    }
    for (const v of this.visuals.values())
      if (v.label)
        v.label.classList.toggle("selected", v.glyph.id === this.selected);
  }
  select(id: string) {
    this.clearDetailLabels();
    this.selected = id;
    this.focusedExpert = null;
    this.disposeGroup(this.focus);
    const v = this.visuals.get(id);
    let anchor = v?.glyph;
    if (!anchor && this.data) {
      const ancestors = this.data.index.ancestors(id).slice().reverse();
      anchor = ancestors
        .map((e: any) => this.visuals.get(e.id)?.glyph)
        .find(Boolean);
    }
    if (anchor) {
      this.block =
        this.data!.blocks.find((b) => b.id === anchor!.blockId) ?? this.block;
      const outline = new THREE.Group();
      outline.position.fromArray(anchor.position);
      const w = anchor.width + 2,
        h = anchor.height + 2;
      const a: number[] = [];
      const q = Math.min(2.5, w * 0.15, h * 0.15);
      for (const sx of [-1, 1])
        for (const sy of [-1, 1])
          a.push(
            (sx * w) / 2,
            (sy * h) / 2,
            1.5,
            sx * (w / 2 - q),
            (sy * h) / 2,
            1.5,
            (sx * w) / 2,
            (sy * h) / 2,
            1.5,
            (sx * w) / 2,
            sy * (h / 2 - q),
            1.5,
          );
      outline.add(this.segments(a, 0xd9f1df, 0.9));
      this.focus.add(outline);
    }
    const expert = this.data?.index
      .ancestors(id)
      .slice()
      .reverse()
      .find((e: any) => e.attrs?.expertIndex !== undefined);
    if (expert) {
      const bank = this.visuals.get(expert.parentId)?.glyph;
      if (bank) this.expertDetail(expert.id, bank, false);
    }
    this.emphasize();
    this.cb.position(this.block?.id ?? null, this.mode);
    this.invalidate();
  }
  private boundsForModel() {
    const box = new THREE.Box3();
    for (const g of this.data?.glyphs ?? []) {
      if (g.level !== 1) continue;
      box.expandByPoint(
        new THREE.Vector3(
          g.position[0] - g.width / 2,
          g.position[1] - g.height / 2,
          g.position[2],
        ),
      );
      box.expandByPoint(
        new THREE.Vector3(
          g.position[0] + g.width / 2,
          g.position[1] + g.height / 2,
          g.position[2],
        ),
      );
    }
    return box;
  }
  fitModel() {
    if (!this.data) return;
    this.mode = "model";
    this.clearDetailLabels();
    this.selected = null;
    this.focusedExpert = null;
    this.disposeGroup(this.focus);
    this.emphasize();
    const box = this.boundsForModel(),
      center = box.getCenter(new THREE.Vector3()),
      size = box.getSize(new THREE.Vector3());
    const usableWidth = Math.max(
      0.4,
      (this.host.clientWidth - 450) / this.host.clientWidth,
    );
    const distance =
      Math.max(
        (size.y /
          (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)))) *
          1.22,
        size.x /
          (2 *
            Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) *
            this.camera.aspect *
            usableWidth),
      ) * 1.02;
    center.x += 50;
    this.fly(center, distance);
    this.cb.position(null, "model");
  }
  navigate(mode: string, blockId?: string) {
    if (!this.data) return;
    if (mode === "model") {
      this.fitModel();
      return;
    }
    this.mode = mode;
    this.block =
      this.data.blocks.find((b) => b.id === blockId) ??
      this.block ??
      this.data.blocks[0];
    const b = this.block;
    let target = new THREE.Vector3(...b.position),
      distance = 205;
    let select = b.id;
    if (mode === "attention") {
      target.y += 34;
      distance = 120;
      select = b.attentionId ?? b.id;
    }
    if (mode === "experts") {
      target.y -= 40;
      distance = 118;
      select = b.expertsId ?? b.id;
    }
    this.fly(target, distance);
    this.cb.select(select);
    this.cb.position(b.id, mode);
  }
  approach(id: string) {
    if (!this.data) return;
    const ancestors = this.data.index.ancestors(id).slice().reverse();
    const expert = ancestors.find(
      (e: any) => e.attrs?.expertIndex !== undefined,
    );
    if (expert) {
      const bank = this.visuals.get(expert.parentId)?.glyph;
      if (bank) {
        this.mode = "detail";
        this.cb.select(id);
        this.expertDetail(expert.id, bank, true);
        return;
      }
    }
    let g = this.visuals.get(id)?.glyph;
    if (!g)
      g = ancestors
        .map((e: any) => this.visuals.get(e.id)?.glyph)
        .find(Boolean);
    if (!g) return;
    if (g.attrs?.semanticBackbone) {
      this.navigate("block", g.blockId);
      return;
    }
    if (g.attrs?.semanticSummary) {
      this.navigate(
        g.role === "attention" ? "attention" : "experts",
        g.blockId,
      );
      return;
    }
    this.mode = "detail";
    this.fly(
      new THREE.Vector3(...g.position),
      Math.max(15, g.width * 1.45, g.height * 2.4),
    );
    this.cb.select(id);
  }
  private parameterChildren(id: string) {
    return this.data!.index.childList(id).flatMap((e: any) => {
      const t = e.tensorId ? this.data!.index.tensors.get(e.tensorId) : null;
      if (t?.role === "parameter" && t.shape.length === 2)
        return [{ entity: e, tensor: t }];
      return this.data!.index.childList(e.id)
        .filter((c: any) => c.tensorId)
        .flatMap((c: any) => {
          const t = this.data!.index.tensors.get(c.tensorId);
          return t?.role === "parameter" && t.shape.length === 2
            ? [{ entity: c, tensor: t }]
            : [];
        });
    });
  }
  private expertDetail(id: string, bank: Glyph, fly: boolean) {
    this.clearDetailLabels();
    this.disposeGroup(this.focus);
    this.focusedExpert = id;
    const ids = this.expertIds(bank),
      n = ids.indexOf(id);
    if (n < 0) return;
    const cols = Math.ceil(Math.sqrt((ids.length * bank.width) / bank.height)),
      rows = Math.ceil(ids.length / cols);
    const x =
        bank.position[0] + (((n % cols) - (cols - 1) / 2) * bank.width) / cols,
      y =
        bank.position[1] +
        ((Math.floor(n / cols) - (rows - 1) / 2) * bank.height) / rows;
    const root = new THREE.Group();
    root.position.set(x, y, 7);
    this.focus.add(root);
    const tensors = this.parameterChildren(id);
    tensors.slice(0, 3).forEach((item, i) => {
      const sheet = this.matrixSheet(
        root,
        4.3,
        7,
        COLORS.expert,
        item.tensor.shape,
        (i - 1) * 5.5,
        0,
        0,
      );
      sheet.traverse((o) => {
        o.userData.entityId = item.entity.id;
      });
      this.detailLabels.push({
        id: item.entity.id,
        text: item.entity.label + " · " + item.tensor.shape.join(" × "),
        position: new THREE.Vector3(x + (i - 1) * 5.5, y + 4.4, 7),
      });
      if (item.entity.id === this.selected) {
        const m = this.mesh(new THREE.RingGeometry(0.22, 0.3, 16), 0xeaffee);
        m.position.set((i - 1) * 5.5, 4.4, 0.3);
        root.add(m);
      }
    });
    // Sheets are expanded storage, not an invented sequential expert computation.
    root.add(
      this.line(
        [
          new THREE.Vector3(0, -4.3, 0),
          new THREE.Vector3(0, -5.2, 0),
          new THREE.Vector3(0, -5.2, -7),
        ],
        COLORS.expert,
        0.5,
      ),
    );
    if (fly) this.fly(new THREE.Vector3(x, y, 7), 37);
    this.invalidate();
  }
  setFlat(value: boolean) {
    this.flat = value;
    this.fly(
      this.controls.target.clone(),
      this.camera.position.distanceTo(this.controls.target),
    );
  }
  private fly(target: THREE.Vector3, distance: number) {
    const offset = this.flat
      ? new THREE.Vector3(0, 0, distance)
      : new THREE.Vector3(distance * 0.04, distance * 0.13, distance);
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
  private hit(e: PointerEvent | MouseEvent) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(
      ((e.clientX - r.left) / r.width) * 2 - 1,
      (-(e.clientY - r.top) / r.height) * 2 + 1,
    );
    this.ray.setFromCamera(this.pointer, this.camera);
    const roots = [...this.visuals.values()]
      .filter((v) => v.group?.visible)
      .map((v) => v.group!);
    if (this.focus.visible) roots.unshift(this.focus);
    for (const hit of this.ray.intersectObjects(roots, true)) {
      if (hit.object.userData.expertIds && hit.instanceId !== undefined) {
        const id = hit.object.userData.expertIds[hit.instanceId];
        if (id) return id;
      }
      const id = hit.object.userData.entityId;
      if (id) return id;
    }
    return null;
  }
  private render(t: number) {
    this.frame = 0;
    if (document.hidden) return;
    if (t - this.lastTime < 24) {
      this.frame = requestAnimationFrame((v) => this.render(v));
      return;
    }
    this.inFrame = true;
    this.lastTime = t;
    let moving = false;
    if (this.flight) {
      const p = Math.min(1, (t - this.flight.start) / this.flight.duration),
        k = p * p * (3 - 2 * p);
      this.camera.position.lerpVectors(this.flight.from, this.flight.to, k);
      this.controls.target.lerpVectors(
        this.flight.lookFrom,
        this.flight.lookTo,
        k,
      );
      if (p === 1) this.flight = null;
      moving = true;
    }
    moving = this.controls.update() || moving;
    if (moving || this.dirty) {
      this.updateLOD();
      this.renderer.render(this.scene, this.camera);
      this.dirty = false;
    }
    this.inFrame = false;
    if (moving || this.flight)
      this.frame = requestAnimationFrame((v) => this.render(v));
  }
  dispose() {
    if (this.frame) cancelAnimationFrame(this.frame);
    this.controls.dispose();
    this.resizeObserver.disconnect();
    for (const g of [this.root, this.edgeGroup, this.focus])
      this.disposeGroup(g);
    this.plane.dispose();
    this.diamond.dispose();
    this.sphere.dispose();
    this.renderer.dispose();
    this.labelLayer.remove();
  }
}
