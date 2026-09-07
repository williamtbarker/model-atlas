/** Package names are a catalog concern. Scene geometry only consumes ModelIR. */
export const modelCatalog = [
  { key: "deepseek4", label: "DeepSeek V4-Pro" },
  { key: "llama4-maverick", label: "Llama 4 Maverick" },
  { key: "qwen35-397b", label: "Qwen3.5 397B-A17B" },
  { key: "gpt2", label: "GPT-2" },
] as const;
export type ModelKey = (typeof modelCatalog)[number]["key"];
export type ViewMode = "model" | "block" | "attention" | "experts" | "detail";
export interface ViewState {
  model: ModelKey;
  entity: string | null;
  mode: ViewMode;
}
const modes = new Set<string>([
  "model",
  "block",
  "attention",
  "experts",
  "detail",
]);
export function isModelKey(value: string | null): value is ModelKey {
  return modelCatalog.some((model) => model.key === value);
}
export function parseViewURL(href: string): ViewState {
  const params = new URL(href).searchParams;
  const model = params.get("model"),
    mode = params.get("view");
  return {
    model: isModelKey(model) ? model : "deepseek4",
    entity: params.get("entity") || null,
    mode:
      mode && modes.has(mode)
        ? (mode as ViewMode)
        : params.has("entity")
          ? "detail"
          : "model",
  };
}
export function viewURL(href: string, view: ViewState): string {
  const url = new URL(href);
  url.searchParams.set("model", view.model);
  url.searchParams.set("view", view.mode);
  if (view.entity) url.searchParams.set("entity", view.entity);
  else url.searchParams.delete("entity");
  return url.href;
}
export function sameView(a: ViewState, b: ViewState): boolean {
  return a.model === b.model && a.entity === b.entity && a.mode === b.mode;
}
/** Both successes and failures must check ownership before changing current UI. */
export class LatestRequest {
  private controller: AbortController | null = null;
  begin(): AbortController {
    this.controller?.abort();
    this.controller = new AbortController();
    return this.controller;
  }
  isCurrent(request: AbortController): boolean {
    return this.controller === request && !request.signal.aborted;
  }
}
