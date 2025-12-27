import type { GenerationRepository } from "@/lib/ports/generationRepo";
import { store } from "@/lib/store";

export const memoryGenerationRepo: GenerationRepository = {
  create: (gen) => store.create(gen),
  get: (id) => store.get(id),
  update: (id, patch) => store.update(id, patch),
  setStatus: (id, status) => store.setStatus(id, status),
  findByDedupeKey: (key) => store.findByDedupeKey(key),
};
