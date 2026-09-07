import type { ModelIR } from "../types";

export type Matrix = (number | null)[][];
export interface BlockTrace {
  input: number[][];
  norm1: number[][];
  q: number[][][];
  k: number[][][];
  v: number[][][];
  scores: Matrix[];
  attention: number[][][];
  headOutputs: number[][][];
  sourceWrites: number[][][];
  sourceContributions: number[][][][];
  projectedHeads: number[][][];
  attentionOutput: number[][];
  residual1: number[][];
  norm2: number[][];
  mlpUp: number[][];
  mlpAct: number[][];
  mlpDown: number[][];
  output: number[][];
}
export interface Run {
  embedding: number[][];
  blocks: BlockTrace[];
  finalNorm: number[][];
  logits: number[][];
  probabilities: number[][];
}
export interface Example {
  id: string;
  label: string;
  tokens: string[];
  inputIds: number[];
  targetId: number;
  target: string;
  strongestAblation: string;
  runs: Record<string, Run>;
}
export interface TracePackage {
  format: string;
  version: number;
  model: {
    name: string;
    description: string;
    dModel: number;
    dHead: number;
    dFF: number;
    nHeads: number;
    nLayers: number;
    parameterCount: number;
    vocab: string[];
    layerNormEpsilon: number;
    activation: string;
    metrics: any;
    provenance: any;
  };
  weights: Record<string, number[] | number[][]>;
  examples: Example[];
}
export type Operation =
  | { kind: "matmul"; input: string; weight: string; bias?: number[] }
  | { kind: "attention"; q: string; k: string; scale: number; scores: Matrix }
  | { kind: "rowSoftmax"; input: string }
  | { kind: "mix"; attention: string; value: string }
  | { kind: "sum"; inputs: string[] }
  | {
      kind: "norm";
      input: string;
      gamma: number[];
      beta: number[];
      epsilon: number;
    }
  | { kind: "gelu"; input: string }
  | { kind: "gate"; input: string; factor: number }
  | { kind: "softmax"; input: string }
  | { kind: "embedding"; word: number[][]; position: number[][] };
export interface TensorView {
  id: string;
  entityId: string;
  label: string;
  caption: string;
  kind:
    | "activation"
    | "weight"
    | "attention"
    | "residual"
    | "normalization"
    | "probabilities";
  values: Matrix;
  position: [number, number, number];
  rowAxis: string;
  colAxis: string;
  block: number;
  head?: number;
  op?: Operation;
  group: string;
}
export interface Connection {
  id: string;
  from: string;
  to: string;
  kind: "data" | "residual" | "weight";
  via?: [number, number, number][];
}
export interface SceneIR {
  model: ModelIR;
  views: TensorView[];
  connections: Connection[];
  tokens: string[];
  extent: number;
  parameterCount: number;
}
export type Chapter =
  | "overview"
  | "embedding"
  | "projection"
  | "attention"
  | "write"
  | "residual"
  | "mlp"
  | "output"
  | "parameters";
export interface Selection {
  viewId: string;
  row: number;
  col: number;
}
export interface Term {
  label: string;
  left: number;
  right?: number;
  value: number;
  source?: Selection;
  second?: Selection;
}
export interface Arithmetic {
  title: string;
  equation: string;
  terms: Term[];
  result: number;
  note: string;
  bias?: number;
  divide?: number;
  mode: "sum" | "softmax" | "norm" | "gelu" | "value";
}
