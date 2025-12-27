import type { GenerationRecord, GenerationStatus } from "@/lib/store/types";

export type GenerationPatch = Partial<
  Omit<GenerationRecord, "id" | "createdAt" | "dedupeKey">
>;

export interface GenerationRepository {
  create(gen: GenerationRecord): GenerationRecord;
  get(id: string): GenerationRecord | null;
  update(id: string, patch: GenerationPatch): GenerationRecord | null;
  setStatus(id: string, status: GenerationStatus): GenerationRecord | null;

  findByDedupeKey(dedupeKey: string): GenerationRecord | null;
}

