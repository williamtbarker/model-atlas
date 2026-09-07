import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { Entity, Edge, Tile } from "../types";
import { layoutScope } from "./model-ir.js";
import { primitiveFor } from "./primitives";

type Index = {
  childList: (id: string) => Entity[];
  parameterCount: (id: string) => bigint | null;
};
type Callbacks = {
  select: (id: string) => void;
  enter: (id: string) => void;
  back: () => void;
  stats: (text: string) => void;
};
type Label = {
  id: string;
  element: HTMLButtonElement;
  position: THREE.Vector3;
  size: number;
};
export const VISUAL_BUDGET = {
  nodes: 128,
  edges: 256,
  labels: 40,
  detail: 192,
  tileCells: 144,
  pixelRatio: 1.6,
};

/** Bounded scene materialization. No architecture adapters or model names. */
export class AtlasRenderer {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(44, 1, 0.05, 10000);
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  root = new THREE.Group();
  labels: Label[] = [];
  positions = new Map<string, THREE.Vector3>();
  entities = new Map<string, Entity>();
  picking: THREE.Object3D[] = [];
  selected: string | null = null;
  private outline: THREE.Box3Helper;
  private labelLayer: HTMLDivElement;
  private raycaster = new THREE.Raycaster();
  private index: Index | null = null;
  private detail = new THREE.Group();
  private detailKey = "";
  private box = new THREE.BoxGeometry(1, 1, 1);
  private plane = new THREE.PlaneGeometry(1, 1);
  private frame = 0;
  private lastTime = 0;
  private idle = true;
  private resizeObserver: ResizeObserver;
  private flight: {
    from: THREE.Vector3;
    to: THREE.Vector3;
    start: number;
    duration: number;
    targetFrom: THREE.Vector3;
    targetTo: THREE.Vector3;
  } | null = null;
  private canGoBack = false;
  private baseDistance = 50;
  private semanticCooldown = 0;
  private isPointerDown = false;
  private down = new THREE.Vector2();
  private reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  autoZoom = true;
  planar = false;

  constructor(
    private host: HTMLElement,
    private callbacks: Callbacks,
  ) {
    this.scene.background = new THREE.Color(0x09131e);
    this.scene.fog = new THREE.FogExp2(0x09131e, 0.0014);
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "low-power",
    });
    this.renderer.setPixelRatio(
      Math.min(devicePixelRatio, VISUAL_BUDGET.pixelRatio),
    );
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.setAttribute(
      "aria-label",
      "Interactive model architecture. Use the component list for keyboard navigation.",
    );
    this.renderer.domElement.setAttribute("tabindex", "0");
    host.append(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.maxPolarAngle = Math.PI * 0.48;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 2500;
    this.controls.addEventListener("change", () => {
      this.idle = false;
    });
    this.controls.addEventListener("start", () => {
      this.flight = null;
    });
    this.labelLayer = document.createElement("div");
    this.labelLayer.className = "scene-labels";
    host.append(this.labelLayer);
    this.scene.add(new THREE.HemisphereLight(0xe0f1ff, 0x142333, 2.5));
    const sun = new THREE.DirectionalLight(0xfff0d6, 3);
    sun.position.set(-10, 30, 10);
    this.scene.add(sun);
    const fill = new THREE.DirectionalLight(0x4ba8ff, 2);
    fill.position.set(20, 10, -20);
    this.scene.add(fill);
    const grid = new THREE.GridHelper(800, 160, 0x203545, 0x152534);
    grid.position.y = -1.05;
    this.scene.add(grid);
    this.scene.add(this.root);
    this.outline = new THREE.Box3Helper(new THREE.Box3(), 0xf6d798);
    this.outline.visible = false;
    this.scene.add(this.outline);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);
    this.renderer.domElement.addEventListener("pointerdown", (e) => {
      this.isPointerDown = true;
      this.down.set(e.clientX, e.clientY);
    });
    this.renderer.domElement.addEventListener("pointerup", (e) => {
      this.isPointerDown = false;
      if (this.down.distanceTo(new THREE.Vector2(e.clientX, e.clientY)) > 5)
        return;
      const hit = this.hit(e);
      if (hit) this.callbacks.select(hit);
    });
    this.renderer.domElement.addEventListener("dblclick", (e) => {
      const hit = this.hit(e);
      if (hit) this.callbacks.enter(hit);
    });
    this.renderer.domElement.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      this.callbacks.stats(
        "Graphics context lost. Reload to recover; component navigation remains available.",
      );
    });
    this.resize();
    this.animate(0);
  }
  private resize() {
    const w = this.host.clientWidth,
      h = this.host.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.idle = false;
  }
  private disposeGroup(group: THREE.Object3D) {
    group.traverse((obj) => {
      const o = obj as THREE.Mesh;
      if ((obj as THREE.InstancedMesh).isInstancedMesh)
        (obj as THREE.InstancedMesh).dispose();
      if (o.geometry && o.geometry !== this.box && o.geometry !== this.plane)
        o.geometry.dispose();
      const m = o.material;
      if (Array.isArray(m)) m.forEach((x) => x.dispose());
      else m?.dispose();
    });
    group.clear();
  }
  show(items: Entity[], edges: Edge[], index: Index, canGoBack: boolean) {
    this.disposeGroup(this.root);
    this.detail = new THREE.Group();
    this.root.add(this.detail);
    this.detailKey = "";
    this.labelLayer.replaceChildren();
    this.labels = [];
    this.picking = [];
    this.positions.clear();
    this.entities.clear();
    this.index = index;
    this.canGoBack = canGoBack;
    this.selected = null;
    this.outline.visible = false;
    this.flight = null;
    this.semanticCooldown = performance.now() + 1600;
    const list = items.slice(0, VISUAL_BUDGET.nodes),
      positions = layoutScope(list, edges);
    const batches = new Map<string, Entity[]>();
    for (const e of list) {
      this.entities.set(e.id, e);
      const p = positions.get(e.id);
      this.positions.set(e.id, new THREE.Vector3(...p));
      if (!batches.has(e.kind)) batches.set(e.kind, []);
      batches.get(e.kind)!.push(e);
    }
    const transform = new THREE.Object3D();
    for (const batch of batches.values()) {
      const s = primitiveFor(batch[0]);
      const material = new THREE.MeshStandardMaterial({
        color: s.color,
        roughness: 0.38,
        metalness: 0.22,
        transparent: true,
        opacity: 0.85,
      });
      const mesh = new THREE.InstancedMesh(this.box, material, batch.length);
      mesh.userData.ids = batch.map((e) => e.id);
      batch.forEach((e, i) => {
        transform.position.copy(this.positions.get(e.id)!);
        transform.scale.set(s.width, s.height, s.depth);
        transform.updateMatrix();
        mesh.setMatrixAt(i, transform.matrix);
      });
      mesh.instanceMatrix.needsUpdate = true;
      mesh.computeBoundingSphere();
      this.root.add(mesh);
      this.picking.push(mesh);
    }
    // Edges are strictly package-declared relations. No containment connectors.
    const visibleEdges = edges
      .filter(
        (e) =>
          e.kind !== "view" &&
          this.positions.has(e.from) &&
          this.positions.has(e.to),
      )
      .slice(0, VISUAL_BUDGET.edges);
    for (const edge of visibleEdges) {
      const a = this.positions.get(edge.from)!,
        b = this.positions.get(edge.to)!;
      const skip = edge.kind === "residual",
        share = edge.kind === "parameter_share";
      const points = [
        a.clone().add(new THREE.Vector3(0, 0.8, 0)),
        a
          .clone()
          .lerp(b, 0.35)
          .add(new THREE.Vector3(skip ? 3 : 0, skip ? 3 : 0.9, 0)),
        a
          .clone()
          .lerp(b, 0.68)
          .add(new THREE.Vector3(skip ? 3 : 0, skip ? 3 : 0.9, 0)),
        b.clone().add(new THREE.Vector3(0, 0.8, 0)),
      ];
      const curve = new THREE.CatmullRomCurve3(points);
      const geometry = new THREE.BufferGeometry().setFromPoints(
        curve.getPoints(24),
      );
      const color = share ? 0x93a5b6 : skip ? 0xd9a16c : 0x4d8499;
      const line = share
        ? new THREE.Line(
            geometry,
            new THREE.LineDashedMaterial({
              color,
              dashSize: 0.3,
              gapSize: 0.18,
              transparent: true,
              opacity: 0.65,
            }),
          )
        : new THREE.Line(
            geometry,
            new THREE.LineBasicMaterial({
              color,
              transparent: true,
              opacity: 0.6,
            }),
          );
      if (share) (line as THREE.Line).computeLineDistances();
      this.root.add(line);
      if (!share) {
        const at = curve.getPoint(0.78),
          dir = curve.getTangent(0.78).normalize();
        const cone = new THREE.Mesh(
          new THREE.ConeGeometry(0.1, 0.32, 5),
          new THREE.MeshBasicMaterial({ color }),
        );
        cone.position.copy(at);
        cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
        this.root.add(cone);
      }
    }
    for (const e of list) {
      const button = document.createElement("button");
      button.className = "scene-label";
      button.textContent = e.label;
      button.tabIndex = -1;
      button.addEventListener("click", () => this.callbacks.select(e.id));
      button.addEventListener("dblclick", () => this.callbacks.enter(e.id));
      this.labelLayer.append(button);
      this.labels.push({
        id: e.id,
        element: button,
        position: this.positions
          .get(e.id)!
          .clone()
          .add(new THREE.Vector3(0, 1.3, 1.85)),
        size: 0,
      });
    }
    this.fit();
    this.callbacks.stats(
      `${list.length} objects · ${visibleEdges.length} connections · ${VISUAL_BUDGET.nodes}-object budget`,
    );
    this.idle = false;
  }
  fit() {
    const points = [...this.positions.values()];
    const bounds = new THREE.Box3().setFromPoints(
      points.length ? points : [new THREE.Vector3()],
    );
    const center = bounds.getCenter(new THREE.Vector3()),
      size = bounds.getSize(new THREE.Vector3());
    this.baseDistance = Math.max(
      15,
      Math.max(size.x / Math.min(this.camera.aspect, 1.5), size.z) * 1.25 + 12,
    );
    this.controls.target.copy(center);
    const d = this.baseDistance;
    this.camera.position
      .copy(center)
      .add(
        this.planar
          ? new THREE.Vector3(0.01, d, 0.01)
          : new THREE.Vector3(d * 0.38, d * 0.78, d * 0.7),
      );
    this.controls.update();
    this.idle = false;
  }
  setPlanar(value: boolean) {
    this.planar = value;
    this.fit();
  }
  select(id: string) {
    this.selected = id;
    const p = this.positions.get(id),
      e = this.entities.get(id);
    if (p && e) {
      const s = primitiveFor(e);
      this.outline.box.setFromCenterAndSize(
        p,
        new THREE.Vector3(s.width + 0.16, s.height + 0.16, s.depth + 0.16),
      );
      this.outline.visible = true;
    } else this.outline.visible = false;
    for (const label of this.labels)
      label.element.classList.toggle("selected", label.id === id);
    this.idle = false;
  }
  focus(id: string) {
    const target = this.positions.get(id);
    if (!target) return;
    const delta = this.camera.position
      .clone()
      .sub(this.controls.target)
      .normalize()
      .multiplyScalar(12);
    this.flight = {
      from: this.camera.position.clone(),
      to: target.clone().add(delta),
      start: performance.now(),
      duration: this.reduced ? 1 : 600,
      targetFrom: this.controls.target.clone(),
      targetTo: target.clone(),
    };
    this.idle = false;
  }
  private hit(event: PointerEvent | MouseEvent): string | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.raycaster.setFromCamera(
      new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        (-(event.clientY - rect.top) / rect.height) * 2 + 1,
      ),
      this.camera,
    );
    const hit = this.raycaster.intersectObjects(this.picking, false)[0];
    return hit
      ? (hit.object.userData.ids?.[hit.instanceId ?? 0] ??
          hit.object.userData.id ??
          null)
      : null;
  }
  private screenSize(id: string): number {
    const p = this.positions.get(id);
    if (!p) return 0;
    return (
      (4.4 * this.host.clientHeight) /
      (2 *
        Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) *
        this.camera.position.distanceTo(p))
    );
  }
  private updateLabels() {
    const w = this.host.clientWidth,
      h = this.host.clientHeight,
      frustum = new THREE.Frustum().setFromProjectionMatrix(
        new THREE.Matrix4().multiplyMatrices(
          this.camera.projectionMatrix,
          this.camera.matrixWorldInverse,
        ),
      );
    const sorted = [...this.labels].sort(
      (a, b) =>
        Number(b.id === this.selected) - Number(a.id === this.selected) ||
        this.camera.position.distanceToSquared(a.position) -
          this.camera.position.distanceToSquared(b.position),
    );
    let count = 0;
    const occupied: { x: number; y: number }[] = [];
    for (const label of sorted) {
      const p = label.position.clone().project(this.camera),
        x = ((p.x + 1) * w) / 2,
        y = ((1 - p.y) * h) / 2;
      const size = this.screenSize(label.id);
      label.size = size;
      const visible =
        frustum.containsPoint(label.position) &&
        size > 27 &&
        count < VISUAL_BUDGET.labels &&
        (!occupied.some(
          (o) => Math.abs(o.x - x) < 110 && Math.abs(o.y - y) < 26,
        ) ||
          label.id === this.selected);
      label.element.hidden = !visible;
      if (visible) {
        count++;
        occupied.push({ x, y });
        label.element.style.transform = `translate(${x}px,${y}px) translate(-50%,0)`;
      }
    }
  }
  private updateDetail() {
    // Hysteresis: reveal actual immediate children at 160 px; retract below 125.
    const e = this.selected ? this.entities.get(this.selected) : null;
    const size = e ? this.screenSize(e.id) : 0;
    const show = !!e && (size > 160 || (this.detailKey === e.id && size > 125));
    const key = show ? e!.id : "";
    if (key === this.detailKey) return;
    this.disposeGroup(this.detail);
    this.detailKey = key;
    if (!show || !this.index) return;
    const children = this.index.childList(e!.id).slice(0, VISUAL_BUDGET.detail);
    if (!children.length) return;
    const transform = new THREE.Object3D(),
      s = primitiveFor(e!),
      position = this.positions.get(e!.id)!;
    const cols = Math.min(16, Math.ceil(Math.sqrt(children.length))),
      rows = Math.ceil(children.length / cols);
    const mesh = new THREE.InstancedMesh(
      this.box,
      new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.5,
        metalness: 0.1,
      }),
      children.length,
    );
    children.forEach((child, i) => {
      transform.position.set(
        position.x + ((i % cols) - (cols - 1) / 2) * (3.8 / cols),
        s.height / 2 + 0.2,
        position.z + (Math.floor(i / cols) - (rows - 1) / 2) * (2.3 / rows),
      );
      transform.scale.set(3.2 / cols, 0.28, 1.8 / rows);
      transform.updateMatrix();
      mesh.setMatrixAt(i, transform.matrix);
      mesh.setColorAt(i, new THREE.Color(primitiveFor(child).color));
    });
    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingSphere();
    this.detail.add(mesh);
  }
  private animate = (time: number) => {
    this.frame = requestAnimationFrame(this.animate);
    if (document.hidden) return;
    if (time - this.lastTime < 32) return;
    this.lastTime = time;
    if (this.flight) {
      const t = Math.min(1, (time - this.flight.start) / this.flight.duration),
        ease = 1 - (1 - t) ** 3;
      this.camera.position.lerpVectors(this.flight.from, this.flight.to, ease);
      this.controls.target.lerpVectors(
        this.flight.targetFrom,
        this.flight.targetTo,
        ease,
      );
      if (t === 1) this.flight = null;
      this.idle = false;
    }
    this.controls.update();
    if (this.idle) return;
    this.camera.updateMatrixWorld();
    this.updateLabels();
    this.updateDetail();
    this.renderer.render(this.scene, this.camera);
    this.idle = true;
    if (
      this.autoZoom &&
      !this.isPointerDown &&
      !this.flight &&
      time > this.semanticCooldown
    ) {
      if (
        this.selected &&
        this.screenSize(this.selected) >
          Math.min(this.host.clientWidth * 0.7, 480) &&
        this.index?.childList(this.selected).length
      ) {
        this.semanticCooldown = time + 1500;
        this.callbacks.enter(this.selected);
      } else if (
        this.canGoBack &&
        this.camera.position.distanceTo(this.controls.target) >
          this.baseDistance * 2.4
      ) {
        this.semanticCooldown = time + 1500;
        this.callbacks.back();
      }
    }
  };
  dispose() {
    cancelAnimationFrame(this.frame);
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.disposeGroup(this.scene);
    this.box.dispose();
    this.plane.dispose();
    this.renderer.dispose();
    this.labelLayer.remove();
    this.renderer.domElement.remove();
  }
}
