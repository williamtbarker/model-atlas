/** Bounded display samples preserve the declared GQA sharing relation. */
export function headGrouping(
  queryCount: number,
  kvCount: number,
  budget = 256,
) {
  if (
    ![queryCount, kvCount, budget].every(
      (n) => Number.isSafeInteger(n) && n > 0,
    ) ||
    queryCount % kvCount !== 0
  )
    throw new Error(
      "Head counts must be positive integers with equal-sized KV groups.",
    );
  const count = Math.min(queryCount, budget);
  const indices = Array.from({ length: count }, (_, i) =>
    count === 1 ? 0 : Math.floor((i * (queryCount - 1)) / (count - 1)),
  );
  const normalized = (index: number, total: number) =>
    total === 1 ? 0 : index / (total - 1) - 0.5;
  const heads = indices.map((index) => {
    const kvIndex = Math.floor(index / (queryCount / kvCount));
    return {
      index,
      kvIndex,
      queryX: normalized(index, queryCount),
      kvX: normalized(kvIndex, kvCount),
    };
  });
  const keys = [...new Set(heads.map((h) => h.kvIndex))].map((index) => ({
    index,
    x: normalized(index, kvCount),
  }));
  return { heads, keys, sampled: count < queryCount };
}
