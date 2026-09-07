import { ModelIndex } from "../lib/model-ir.js";
import type { Entity, ModelIR, Evidence, Shape } from "../types";

export type Position = [number, number, number];
export type GlyphKind =
  | "matrix"
  | "norm"
  | "mix"
  | "heads"
  | "memory"
  | "router"
  | "experts"
  | "activation"
  | "stream"
  | "embedding"
  | "output"
  | "generic";
export interface Glyph {
  id: string;
  entityId: string;
  kind: GlyphKind;
  label: string;
  position: Position;
  width: number;
  height: number;
  blockId?: string;
  level: 1 | 2 | 3;
  role: "attention" | "expert" | "residual" | "parameter" | "neutral";
  shape?: Shape;
  attrs?: Record<string, unknown>;
}
export interface BlockLayout {
  id: string;
  label: string;
  index: number;
  position: Position;
  auxiliary: boolean;
  attentionId?: string;
  expertsId?: string;
  routerId?: string;
  streams: number;
  glyphIds: string[];
  /** Local operator x increases in this direction, matching the serpentine chain. */
  direction: 1 | -1;
}
export interface LandscapeLink {
  id: string;
  from: string;
  to: string;
  kind: "dataflow" | "residual";
  evidence: Evidence;
  /** Actual ModelIR edges that were projected to these endpoints. */
  sourceEdgeIds: string[];
  label?: string;
}
export interface Landscape {
  index: ModelIndex;
  model: ModelIR;
  blocks: BlockLayout[];
  glyphs: Glyph[];
  links: LandscapeLink[];
  width: number;
  height: number;
}

const natural = new Intl.Collator("en", { numeric: true });
const numberAttr = (e: Entity, key: string, fallback = 0) => {
  const value = e.attrs?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
};
const numericIndex = (e: Entity) =>
  numberAttr(e, "index", Number.MAX_SAFE_INTEGER);
const compareEntities = (a: Entity, b: Entity) =>
  numericIndex(a) - numericIndex(b) || natural.compare(a.id, b.id);

/** No model names or checkpoint-name conventions are used by this adapter. */
export function compileLandscape(ir: ModelIR): Landscape {
  const index = new ModelIndex(ir);
  const byId = index.entities as Map<string, Entity>;
  const children = (id: string): Entity[] => index.childList(id);
  const ancestors = (id: string): Entity[] => index.ancestors(id);
  const allBlocks = ir.entities
    .filter((e) => e.kind === "TransformerBlock")
    .sort(compareEntities);
  const declaredScope =
    typeof ir.metadata?.initialScope === "string"
      ? ir.metadata.initialScope
      : ir.rootId;
  let core = allBlocks.filter((e) =>
    ancestors(e.id).some((p) => p.id === declaredScope),
  );
  if (!core.length) core = allBlocks;
  const coreIds = new Set(core.map((e) => e.id));
  const auxiliary = allBlocks.filter((e) => !coreIds.has(e.id));
  const glyphs: Glyph[] = [];
  const glyphById = new Map<string, Glyph>();
  const blocks: BlockLayout[] = [];
  const blockById = new Map<string, BlockLayout>();
  const columns = Math.min(10, Math.max(1, core.length));
  const rows = Math.max(1, Math.ceil(core.length / columns));

  const ownShape = (e: Entity): Shape | undefined => {
    const tensorEntity = e.tensorId
      ? e
      : children(e.id).find(
          (c) =>
            c.tensorId && index.tensors.get(c.tensorId)?.role === "parameter",
        );
    const tensor = tensorEntity?.tensorId
      ? index.tensors.get(tensorEntity.tensorId)
      : undefined;
    if (tensor) return [...tensor.shape];
    return Array.isArray(e.attrs?.shape)
      ? ([...e.attrs.shape] as Shape)
      : undefined;
  };
  const descendantHas = (
    e: Entity,
    test: (candidate: Entity) => boolean,
  ): boolean => {
    const pending = [...children(e.id)];
    while (pending.length) {
      const candidate = pending.pop()!;
      if (test(candidate)) return true;
      pending.push(...children(candidate.id));
    }
    return false;
  };
  const kindOf = (e: Entity): GlyphKind => {
    if (e.kind === "NormalizationPlane") return "norm";
    if (e.kind === "HyperConnectionMix" || e.kind === "ResidualBus")
      return "mix";
    if (
      e.kind === "CompressedAttention" ||
      e.kind === "AttentionHead" ||
      e.kind === "AttentionGrid"
    )
      return "heads";
    if (e.kind === "Compressor" || e.kind === "SparseIndexer") return "memory";
    if (
      e.kind === "HashRouter" ||
      e.kind === "ScoreRouter" ||
      e.kind === "RouterJunction"
    )
      return "router";
    if (e.kind === "ExpertCluster") return "experts";
    if (
      e.kind === "ProjectionBlock" ||
      e.kind === "MatrixPlane" ||
      e.kind === "TensorVolume"
    )
      return "matrix";
    if (e.kind === "ActivationLayer" || e.kind === "SharedExpert")
      return "activation";
    if (e.kind === "EmbeddingTable") return "embedding";
    if (e.kind === "VocabularyPlane") return "output";
    if (e.kind === "TokenStream" || e.kind === "TransformerBlock")
      return "stream";
    if (
      e.kind === "RepeatedModuleArray" &&
      descendantHas(e, (c) => typeof c.attrs?.headIndex === "number")
    )
      return "heads";
    return "generic";
  };

  const emit = (
    e: Entity,
    block: BlockLayout | undefined,
    x: number,
    y: number,
    width: number,
    height: number,
    level: 1 | 2 | 3,
    role: Glyph["role"],
    overrides: Partial<Glyph> = {},
  ): Glyph => {
    const previous = glyphById.get(e.id);
    if (previous) return previous;
    const shape = ownShape(e);
    const glyph: Glyph = {
      id: e.id,
      entityId: e.id,
      kind: kindOf(e),
      label: e.label,
      position: block
        ? [block.position[0] + x * block.direction, block.position[1] + y, 0]
        : [x, y, 0],
      width,
      height,
      blockId: block?.id,
      level,
      role,
      shape,
      attrs: { ...e.attrs, sourceKind: e.kind, valueStatus: "unavailable" },
      ...overrides,
    };
    glyphs.push(glyph);
    glyphById.set(glyph.id, glyph);
    block?.glyphIds.push(glyph.id);
    return glyph;
  };

  /** Project only the supplied graph to immediate children to get a flow rank. */
  const flowRanks = (parent: Entity, items: Entity[]) => {
    const ids = new Set(items.map((e) => e.id));
    const owner = (id: string): string | undefined => {
      let entity = byId.get(id);
      while (entity && entity.id !== parent.id) {
        if (ids.has(entity.id)) return entity.id;
        entity = entity.parentId ? byId.get(entity.parentId) : undefined;
      }
      return undefined;
    };
    const incoming = new Map(items.map((e) => [e.id, new Set<string>()]));
    const outgoing = new Map(items.map((e) => [e.id, new Set<string>()]));
    for (const edge of ir.edges) {
      if (edge.kind !== "dataflow" && edge.kind !== "residual") continue;
      const from = owner(edge.from),
        to = owner(edge.to);
      if (!from || !to || from === to) continue;
      outgoing.get(from)!.add(to);
      incoming.get(to)!.add(from);
    }
    const ranks = new Map(items.map((e) => [e.id, 0]));
    const degree = new Map([...incoming].map(([id, refs]) => [id, refs.size]));
    const queue = items.filter((e) => degree.get(e.id) === 0).map((e) => e.id);
    for (let i = 0; i < queue.length; i++) {
      const id = queue[i];
      for (const next of outgoing.get(id)!) {
        ranks.set(next, Math.max(ranks.get(next)!, ranks.get(id)! + 1));
        degree.set(next, degree.get(next)! - 1);
        if (!degree.get(next)) queue.push(next);
      }
    }
    // Cyclic/unranked components receive deterministic positions, not invented edges.
    if (queue.length !== items.length)
      items.forEach((e, i) => ranks.set(e.id, i));
    return { ranks, incoming, outgoing, order: queue };
  };

  const attentionDetail = (attention: Entity, block: BlockLayout) => {
    const items = children(attention.id)
      .filter((e) => !e.tensorId && e.attrs?.headIndex == null)
      .slice(0, 18);
    const flow = flowRanks(attention, items);
    const maxRank = Math.max(1, ...flow.ranks.values());
    const tails = new Map(items.map((e) => [e.id, 0]));
    [...flow.order].reverse().forEach((id) => {
      tails.set(
        id,
        Math.max(
          0,
          ...[...flow.outgoing.get(id)!].map((to) => 1 + tails.get(to)!),
        ),
      );
    });
    const branches = new Map<string, number>();
    const firstFork = flow.order.find((id) => flow.outgoing.get(id)!.size > 1);
    if (firstFork) {
      const starts = [...flow.outgoing.get(firstFork)!].sort(
        (a, b) => tails.get(b)! - tails.get(a)! || natural.compare(a, b),
      );
      starts.forEach((start, lane) => {
        const pending = [start],
          seen = new Set<string>();
        while (pending.length) {
          const id = pending.pop()!;
          if (seen.has(id)) continue;
          seen.add(id);
          // A convergence returns to the primary lane.
          branches.set(
            id,
            branches.has(id) ? Math.min(branches.get(id)!, lane) : lane,
          );
          pending.push(...flow.outgoing.get(id)!);
        }
      });
    }
    for (const entity of items) {
      const kind = kindOf(entity);
      let y =
        branches.get(entity.id) === 1
          ? 29
          : branches.get(entity.id)! >= 2
            ? 18
            : 47;
      if (kind === "memory") y = 18;
      if (kind === "stream") y = 29;
      const x = -46 + (flow.ranks.get(entity.id)! / maxRank) * 92;
      const w =
        kind === "norm"
          ? 2.2
          : kind === "heads"
            ? 11
            : kind === "stream"
              ? 3
              : 8.3;
      const h = kind === "heads" ? 15 : kind === "matrix" ? 10 : 7;
      emit(entity, block, x, y, w, h, 2, "attention");
    }
    return {
      entryGlyphId: flow.order.find((id) => flow.incoming.get(id)!.size === 0),
      exitGlyphId: [...flow.order]
        .reverse()
        .find((id) => flow.outgoing.get(id)!.size === 0),
    };
  };

  for (const [order, entity] of [...core, ...auxiliary].entries()) {
    const isAuxiliary = !coreIds.has(entity.id);
    const row = isAuxiliary ? order - core.length : Math.floor(order / columns);
    const direction: 1 | -1 = isAuxiliary || row % 2 === 0 ? 1 : -1;
    const column = isAuxiliary
      ? columns + 1
      : direction === 1
        ? order % columns
        : columns - 1 - (order % columns);
    const block: BlockLayout = {
      id: entity.id,
      label: entity.label,
      index: numberAttr(entity, "index", order),
      position: [column * 120, -row * 160, 0],
      auxiliary: isAuxiliary,
      streams: Math.max(1, numberAttr(entity, "residualStreams", 1)),
      glyphIds: [],
      direction,
    };
    blocks.push(block);
    blockById.set(block.id, block);
    emit(
      entity,
      block,
      0,
      0,
      100,
      Math.max(3, block.streams * 1.5),
      1,
      "residual",
      {
        attrs: {
          ...entity.attrs,
          sourceKind: entity.kind,
          streams: block.streams,
          semanticBackbone: true,
          valueStatus: "unavailable",
        },
      },
    );
    const direct = children(entity.id).filter((e) => !e.tensorId);
    const attention = direct.find(
      (e) => e.kind === "CompressedAttention" || e.kind === "AttentionHead",
    );
    const experts = direct.find((e) => e.kind === "ExpertCluster");
    const dense = direct.find((e) => e.kind === "ActivationLayer");
    const flow = flowRanks(entity, direct);
    const residuals = direct
      .filter(
        (e) =>
          (e.kind === "HyperConnectionMix" || e.kind === "ResidualBus") &&
          (flow.incoming.get(e.id)!.size || flow.outgoing.get(e.id)!.size),
      )
      .sort((a, b) => flow.ranks.get(a.id)! - flow.ranks.get(b.id)!);
    residuals
      .slice(0, 4)
      .forEach((e, i) =>
        emit(e, block, [-43, -8, 9, 47][i], 0, 7, 8, 1, "residual"),
      );
    const norms = direct.filter((e) => e.kind === "NormalizationPlane");
    norms
      .slice(0, 2)
      .forEach((e, i) =>
        emit(
          e,
          block,
          -41,
          i ? -17 : 15,
          2.3,
          8,
          2,
          i ? "expert" : "attention",
        ),
      );

    if (attention) {
      block.attentionId = attention.id;
      emit(attention, block, 0, 37, 94, 32, 1, "attention", {
        attrs: {
          ...attention.attrs,
          sourceKind: attention.kind,
          semanticSummary: true,
          valueStatus: "unavailable",
        },
      });
      Object.assign(
        glyphById.get(attention.id)!.attrs!,
        attentionDetail(attention, block),
      );
    }
    if (experts) {
      const inner = children(experts.id).filter((e) => !e.tensorId);
      const bank = inner.find(
        (e) =>
          e.kind === "ExpertCluster" &&
          (typeof e.attrs?.count === "number" ||
            children(e.id).some(
              (c) => typeof c.attrs?.expertIndex === "number",
            )),
      );
      const router = inner.find((e) =>
        ["HashRouter", "ScoreRouter", "RouterJunction"].includes(e.kind),
      );
      block.expertsId = bank?.id ?? experts.id;
      block.routerId = router?.id;
      emit(experts, block, 3, -39, 94, 32, 1, "expert", {
        attrs: {
          ...experts.attrs,
          sourceKind: experts.kind,
          bankEntityId: bank?.id,
          semanticSummary: true,
          valueStatus: "unavailable",
        },
      });
      if (bank) emit(bank, block, 0, -44, 69, 28, 2, "expert");
      if (router) emit(router, block, -40, -24, 10, 9, 2, "expert");
      const input = inner.find((e) => e.kind === "TokenStream");
      if (input) emit(input, block, -48, -15, 3, 6, 2, "expert");
      const shared = inner.find((e) => e.kind === "SharedExpert");
      if (shared) emit(shared, block, 45, -43, 11, 20, 2, "expert");
      const merge = inner.find((e) => e.kind === "ResidualBus");
      if (merge) emit(merge, block, 43, -17, 7, 7, 2, "expert");
      Object.assign(glyphById.get(experts.id)!.attrs!, {
        entryGlyphId: input?.id ?? router?.id ?? bank?.id,
        exitGlyphId: merge?.id ?? bank?.id,
      });
    } else if (dense) {
      emit(dense, block, 1, -38, 86, 26, 1, "expert", {
        attrs: {
          ...dense.attrs,
          sourceKind: dense.kind,
          semanticSummary: true,
          valueStatus: "unavailable",
        },
      });
      const inner = children(dense.id)
        .filter((e) => !e.tensorId)
        .slice(0, 5);
      inner.forEach((e, i) =>
        emit(
          e,
          block,
          -35 + (i * 70) / Math.max(1, inner.length - 1),
          -39,
          17,
          23,
          2,
          "expert",
        ),
      );
      Object.assign(glyphById.get(dense.id)!.attrs!, {
        entryGlyphId: inner[0]?.id,
        exitGlyphId: inner.at(-1)?.id,
      });
    }
    // Unrecognized direct operations remain individually inspectable within the cap.
    const remaining = direct.filter(
      (e) => !glyphById.has(e.id) && e.kind !== "TokenStream",
    );
    remaining
      .slice(0, Math.max(0, 35 - block.glyphIds.length))
      .forEach((e, i) => emit(e, block, -35 + i * 12, -57, 8, 5, 3, "neutral"));
  }

  const blockDescendants = new Map<string, BlockLayout[]>();
  for (const block of blocks) {
    for (const ancestor of ancestors(block.id)) {
      if (ancestor.id === block.id) continue;
      if (!blockDescendants.has(ancestor.id))
        blockDescendants.set(ancestor.id, []);
      blockDescendants.get(ancestor.id)!.push(block);
    }
  }
  for (const descendants of blockDescendants.values())
    descendants.sort((a, b) => a.index - b.index);

  // Top-level non-block stages are modest landmark glyphs. Repeated arrays are
  // represented by their persistent descendants, not by additional container boxes.
  const first = blocks.find((b) => !b.auxiliary);
  const last = [...blocks].reverse().find((b) => !b.auxiliary);
  let endSlot = 0,
    embeddingSlot = 0;
  for (const entity of children(ir.rootId)) {
    if (
      entity.kind === "RepeatedModuleArray" ||
      blockById.has(entity.id) ||
      entity.tensorId
    )
      continue;
    const upstream = entity.kind === "EmbeddingTable";
    const base = upstream ? first : last;
    const direction = base?.direction ?? 1;
    const x =
      (base?.position[0] ?? 0) +
      (upstream ? -84 : 80 + endSlot++ * 38) * direction;
    emit(
      entity,
      undefined,
      x,
      (base?.position[1] ?? 0) + (upstream ? embeddingSlot++ * 68 : 0),
      upstream ? 22 : 15,
      upstream || entity.kind === "VocabularyPlane" ? 58 : 22,
      1,
      entity.kind === "HyperConnectionMix" ? "residual" : "neutral",
    );
  }

  const endpoint = (id: string, source: boolean): string | undefined => {
    // A collection-level edge identifies the exit/entry of its ordered collection.
    // This preserves the supplied group edge; it does not invent adjacent-block edges.
    const collection = blockDescendants.get(id);
    if (collection?.length && !glyphById.has(id))
      return collection[source ? collection.length - 1 : 0].id;
    let entity = byId.get(id);
    while (entity) {
      if (glyphById.has(entity.id)) return entity.id;
      entity = entity.parentId ? byId.get(entity.parentId) : undefined;
    }
    return undefined;
  };
  const links: LandscapeLink[] = [];
  const linkByPair = new Map<string, LandscapeLink>();
  for (const edge of ir.edges) {
    if (edge.kind !== "dataflow" && edge.kind !== "residual") continue;
    const from = endpoint(edge.from, true),
      to = endpoint(edge.to, false);
    if (!from || !to || from === to) continue;
    const key = `${edge.kind}\u0000${from}\u0000${to}\u0000${edge.evidence}`;
    const existing = linkByPair.get(key);
    if (existing) {
      existing.sourceEdgeIds.push(edge.id);
      continue;
    }
    const link: LandscapeLink = {
      id: edge.id,
      from,
      to,
      kind: edge.kind,
      evidence: edge.evidence,
      label: edge.label,
      sourceEdgeIds: [edge.id],
    };
    links.push(link);
    linkByPair.set(key, link);
  }
  const minX = Math.min(0, ...glyphs.map((g) => g.position[0] - g.width / 2));
  const maxX = Math.max(0, ...glyphs.map((g) => g.position[0] + g.width / 2));
  const minY = Math.min(0, ...glyphs.map((g) => g.position[1] - g.height / 2));
  const maxY = Math.max(0, ...glyphs.map((g) => g.position[1] + g.height / 2));
  return {
    index,
    model: ir,
    blocks,
    glyphs,
    links,
    width: maxX - minX,
    height: Math.max(rows * 160 - 40, maxY - minY),
  };
}

/** Re-project connectivity when a block crosses a semantic level-of-detail boundary.
 * Call only after the visible set changes. Source ModelIR edge IDs remain attached.
 */
export function projectLandscapeLinks(
  landscape: Landscape,
  visibleGlyphIds: ReadonlySet<string>,
): LandscapeLink[] {
  const glyphs = new Map(landscape.glyphs.map((g) => [g.id, g]));
  const resolve = (id: string, source: boolean): string | undefined => {
    if (visibleGlyphIds.has(id)) return id;
    const glyph = glyphs.get(id);
    const port = glyph?.attrs?.[source ? "exitGlyphId" : "entryGlyphId"];
    if (typeof port === "string" && visibleGlyphIds.has(port)) return port;
    const lineage = landscape.index.ancestors(id) as Entity[];
    for (let i = lineage.length - 1; i >= 0; i--) {
      if (visibleGlyphIds.has(lineage[i].id)) return lineage[i].id;
    }
    return undefined;
  };
  const pairs = new Map<string, LandscapeLink>();
  for (const edge of landscape.links) {
    const from = resolve(edge.from, true),
      to = resolve(edge.to, false);
    if (!from || !to || from === to) continue;
    const key = `${edge.kind}\u0000${from}\u0000${to}\u0000${edge.evidence}`;
    const existing = pairs.get(key);
    if (existing) {
      existing.sourceEdgeIds.push(...edge.sourceEdgeIds);
    } else
      pairs.set(key, {
        ...edge,
        from,
        to,
        sourceEdgeIds: [...edge.sourceEdgeIds],
      });
  }
  return [...pairs.values()];
}
