import type { GenerationRecord, GenerationStatus } from "./types";

const generations = new Map<string, GenerationRecord>();
const byDedupeKey = new Map<string, string>(); // dedupeKey -> generationId

export const memoryStore = {
  create(gen: GenerationRecord) {
    generations.set(gen.id, gen);
    byDedupeKey.set(gen.dedupeKey, gen.id);
    return gen;
  },

  get(id: string) {
    return generations.get(id) ?? null;
  },

  findByDedupeKey(dedupeKey: string) {
    const id = byDedupeKey.get(dedupeKey);
    if (!id) return null;
    return generations.get(id) ?? null;
  },

  update(id: string, patch: Partial<GenerationRecord>) {
    const current = generations.get(id);
    if (!current) return null;

    // Prevent accidental dedupeKey mutation (identity)
    if ("dedupeKey" in patch && (patch as any).dedupeKey !== current.dedupeKey) {
      throw new Error("dedupeKey is immutable");
    }

    const next = { ...current, ...patch };
    generations.set(id, next);
    return next;
  },

  setStatus(id: string, status: GenerationStatus) {
    return this.update(id, { status });
  },
};