import type { GenerationRepository } from "@/lib/ports/generationRepo";
import { store } from "@/lib/store";

/**
 * Adapter: wraps your existing in-memory store behind the GenerationRepository port.
 * Later you can swap to a DB implementation without touching services/routes/workers.
 */
export const memoryGenerationRepo: GenerationRepository = {
  create: (gen) => store.create(gen),
  get: (id) => store.get(id),
  update: (id, patch) => store.update(id, patch),
  setStatus: (id, status) => store.setStatus(id, status),
};
