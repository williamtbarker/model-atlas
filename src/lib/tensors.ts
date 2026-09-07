import type { Tensor, Tile } from "../types";
import { LIMITS, numel } from "./model-ir.js";

const WIDTHS: Record<string, number> = {
  F64: 8,
  F32: 4,
  F16: 2,
  BF16: 2,
  I64: 8,
  U64: 8,
  I32: 4,
  U32: 4,
  I16: 2,
  U16: 2,
  I8: 1,
  U8: 1,
  BOOL: 1,
};
export function flatIndex(shape: number[], indices: number[]): bigint {
  if (shape.length !== indices.length)
    throw new Error(`Expected ${shape.length} indices.`);
  return shape.reduce((offset, size, i) => {
    if (
      !Number.isSafeInteger(indices[i]) ||
      indices[i] < 0 ||
      indices[i] >= size
    )
      throw new Error(`Axis ${i}: index must be in 0…${size - 1}.`);
    return offset * BigInt(size) + BigInt(indices[i]);
  }, 0n);
}
export function decodeHalf(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1,
    exp = (bits >> 10) & 31,
    frac = bits & 1023;
  return exp === 31
    ? frac
      ? NaN
      : sign * Infinity
    : exp === 0
      ? (sign * 2 ** -14 * frac) / 1024
      : sign * 2 ** (exp - 15) * (1 + frac / 1024);
}
function decode(view: DataView, dtype: string): number | string {
  switch (dtype) {
    case "F64":
      return view.getFloat64(0, true);
    case "F32":
      return view.getFloat32(0, true);
    case "F16":
      return decodeHalf(view.getUint16(0, true));
    case "BF16": {
      const b = new DataView(new ArrayBuffer(4));
      b.setUint32(0, view.getUint16(0, true) << 16, true);
      return b.getFloat32(0, true);
    }
    case "I64":
      return view.getBigInt64(0, true).toString();
    case "U64":
      return view.getBigUint64(0, true).toString();
    case "I32":
      return view.getInt32(0, true);
    case "U32":
      return view.getUint32(0, true);
    case "I16":
      return view.getInt16(0, true);
    case "U16":
      return view.getUint16(0, true);
    case "I8":
      return view.getInt8(0);
    case "U8":
      return view.getUint8(0);
    case "BOOL":
      return view.getUint8(0) ? 1 : 0;
    default:
      throw new Error(`Scalar decoding is not implemented for ${dtype}.`);
  }
}

/** Optional local file access. Normal architecture packages never need this. */
export class TensorReader {
  file: File | null = null;
  dataStart = 0;
  private cache = new Map<string, number | string | null>();
  private epoch = 0;
  async openSafetensors(file: File): Promise<Tensor[]> {
    if (file.size < 10) throw new Error("Truncated safetensors file.");
    const first = new DataView(await file.slice(0, 8).arrayBuffer());
    const size = first.getBigUint64(0, true);
    if (
      size > BigInt(LIMITS.headerBytes) ||
      size < 2n ||
      size + 8n > BigInt(file.size)
    )
      throw new Error("Invalid header length or 16 MB header budget exceeded.");
    const header = JSON.parse(await file.slice(8, 8 + Number(size)).text());
    const tensors: Tensor[] = [],
      spans: number[][] = [];
    for (const [id, value] of Object.entries(header)) {
      if (id === "__metadata__") continue;
      const t = value as {
        shape: number[];
        dtype: string;
        data_offsets: number[];
      };
      const n = numel(t.shape),
        span = t.data_offsets;
      if (
        n == null ||
        !Array.isArray(span) ||
        span.length !== 2 ||
        span.some((x) => !Number.isSafeInteger(x) || x < 0) ||
        span[1] < span[0] ||
        span[1] + 8 + Number(size) > file.size
      )
        throw new Error(`Invalid tensor bounds: ${id}`);
      if (
        WIDTHS[t.dtype] &&
        n * BigInt(WIDTHS[t.dtype]) !== BigInt(span[1] - span[0])
      )
        throw new Error(`Shape/byte mismatch: ${id}`);
      spans.push(span);
      tensors.push({
        id,
        shape: t.shape,
        dtype: t.dtype,
        role: "unknown",
        evidence: "observed",
        source: { offsets: span, file: file.name },
      });
    }
    if (tensors.length > LIMITS.tensors)
      throw new Error("Tensor budget exceeded.");
    spans.sort((a, b) => a[0] - b[0]);
    let end = 0;
    for (const [start, stop] of spans) {
      if (start !== end)
        throw new Error("Overlapping or incomplete safetensors data ranges.");
      end = stop;
    }
    if (end + 8 + Number(size) !== file.size)
      throw new Error("Unindexed safetensors bytes.");
    this.epoch++;
    this.file = file;
    this.dataStart = 8 + Number(size);
    this.cache.clear();
    return tensors;
  }
  reset(): void {
    this.epoch++;
    this.file = null;
    this.dataStart = 0;
    this.cache.clear();
  }
  attach(other: TensorReader): void {
    this.reset();
    this.file = other.file;
    this.dataStart = other.dataStart;
  }
  async scalar(
    tensor: Tensor,
    indices: number[],
  ): Promise<number | string | null> {
    if (tensor.shape.some((d) => typeof d !== "number"))
      throw new Error("Symbolic axes need a runtime trace.");
    const flat = flatIndex(tensor.shape as number[], indices);
    if (tensor.values) return tensor.values[Number(flat)] ?? null;
    if (
      !this.file ||
      !tensor.source?.offsets ||
      tensor.source.file !== this.file.name
    )
      return null;
    const width = WIDTHS[tensor.dtype];
    if (!width) return null;
    const key = `${tensor.id}:${flat}`;
    if (this.cache.has(key)) return this.cache.get(key)!;
    const offset =
      BigInt(this.dataStart + tensor.source.offsets[0]) + flat * BigInt(width);
    if (offset > BigInt(Number.MAX_SAFE_INTEGER))
      throw new Error("File offset exceeds browser exact-integer range.");
    const epoch = this.epoch,
      file = this.file;
    const v = decode(
      new DataView(
        await file.slice(Number(offset), Number(offset) + width).arrayBuffer(),
      ),
      tensor.dtype,
    );
    if (epoch === this.epoch && file === this.file) {
      if (this.cache.size >= 2048)
        this.cache.delete(this.cache.keys().next().value!);
      this.cache.set(key, v);
    }
    return v;
  }
  async tile(tensor: Tensor, origin: number[], size = 12): Promise<Tile> {
    if (tensor.shape.some((d) => typeof d !== "number"))
      throw new Error("Symbolic axes need a runtime trace.");
    const shape = tensor.shape as number[];
    if (shape.some((d) => d === 0))
      return {
        shape,
        origin,
        rows: 0,
        cols: 0,
        indices: [],
        values: [],
        note: "Empty tensor.",
      };
    flatIndex(shape, origin);
    const rows =
      shape.length < 2 ? 1 : Math.min(size, shape.at(-2)! - origin.at(-2)!);
    const cols =
      shape.length === 0 ? 1 : Math.min(size, shape.at(-1)! - origin.at(-1)!);
    const indices: number[][] = [];
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        const idx = [...origin];
        if (shape.length > 1) idx[idx.length - 2] += r;
        if (shape.length) idx[idx.length - 1] += c;
        indices.push(idx);
      }
    const values = await Promise.all(
      indices.map((idx) => this.scalar(tensor, idx)),
    );
    return {
      shape,
      origin,
      rows,
      cols,
      indices,
      values,
      note: values.every((v) => v == null)
        ? "Coordinate tile. Values unavailable; no weights or trace loaded."
        : "Individual values, no averaging. Color range uses this visible tile only.",
    };
  }
}
