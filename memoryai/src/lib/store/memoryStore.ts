import type { GenerationRecord, GenerationStatus } from "./types";

const generations = new Map<string, GenerationRecord>();

export const memoryStore = {
  create(gen: GenerationRecord) {
    generations.set(gen.id, gen);
    return gen;
  },
  get(id: string) {
    return generations.get(id) ?? null;
  },
  update(id: string, patch: Partial<GenerationRecord>) {
    const current = generations.get(id);
    if (!current) return null;
    const next = { ...current, ...patch };
    generations.set(id, next);
    return next;
  },
  setStatus(id: string, status: GenerationStatus) {
    return this.update(id, { status });
  },
};
