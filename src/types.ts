export type Evidence =
  | "declared"
  | "derived"
  | "observed"
  | "illustrative"
  | "unavailable";
export type Shape = (number | string)[];
export interface Entity {
  id: string;
  parentId: string | null;
  kind: string;
  label: string;
  evidence: Evidence;
  attrs?: Record<string, unknown>;
  tensorId?: string;
  tensorRefs?: string[];
}
export interface Tensor {
  id: string;
  shape: Shape;
  dtype: string;
  role: "parameter" | "buffer" | "activation" | "unknown";
  evidence: Evidence;
  values?: (number | null)[];
  source?: { offsets?: number[]; file?: string };
}
export interface Edge {
  id: string;
  from: string;
  to: string;
  kind: "dataflow" | "residual" | "parameter_share" | "view";
  label?: string;
  evidence: Evidence;
}
export interface ModelIR {
  version: "1.0";
  name: string;
  rootId: string;
  source: { kind: string; evidence: Evidence; notes?: string; urls?: string[] };
  metadata?: Record<string, unknown>;
  entities: Entity[];
  tensors: Tensor[];
  edges: Edge[];
}
export interface Tile {
  shape: number[];
  origin: number[];
  rows: number;
  cols: number;
  values: (number | string | null)[];
  indices: number[][];
  note: string;
}
