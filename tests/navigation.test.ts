import test from "node:test";
import assert from "node:assert/strict";
import {
  LatestRequest,
  modelCatalog,
  parseViewURL,
  viewURL,
  type ViewState,
} from "../src/landscape/navigation";

test("shared view URLs round-trip exact tensor identities across every catalog model", () => {
  for (const { key } of modelCatalog) {
    const state: ViewState = {
      model: key,
      entity: "tensor:layers.2.experts.0/w1 + bias#weight",
      mode: "detail",
    };
    const href = viewURL("http://127.0.0.1:4173/?unrelated=keep", state);
    assert.deepEqual(parseViewURL(href), state);
    assert.equal(new URL(href).searchParams.get("unrelated"), "keep");
  }
  const clean = viewURL("https://example.test/?entity=stale", {
    model: "gpt2",
    entity: null,
    mode: "model",
  });
  assert.equal(new URL(clean).searchParams.has("entity"), false);
});

test("URL input cannot select an arbitrary package path or unknown navigation mode", () => {
  assert.deepEqual(
    parseViewURL("https://example.test/?model=../../secret&view=execute"),
    { model: "deepseek4", entity: null, mode: "model" },
  );
  assert.equal(
    parseViewURL("https://example.test/?model=gpt2&entity=layers.0").mode,
    "detail",
  );
});

test("obsolete loads lose ownership of both the success and error UI", async () => {
  const requests = new LatestRequest();
  const first = requests.begin();
  let rejectOld!: (error: Error) => void;
  const failing = new Promise<never>((_, reject) => {
    rejectOld = reject;
  });
  const displayed: string[] = [];
  const oldResult = failing.catch(() => {
    if (requests.isCurrent(first)) displayed.push("obsolete error");
  });
  const second = requests.begin();
  assert.equal(first.signal.aborted, true);
  if (requests.isCurrent(second)) displayed.push("new model");
  rejectOld(new Error("late failure"));
  await oldResult;
  if (requests.isCurrent(first)) displayed.push("obsolete success");
  assert.deepEqual(displayed, ["new model"]);
});
