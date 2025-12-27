import type { PresetId } from "@/lib/presets";

export type GenerationStatus =
  | "queued"
  | "creating"
  | "running"
  | "downloading"
  | "succeeded"
  | "failed"
  | "needs_review";

export type GenerationRecord = {
  id: string;
  createdAt: number;
  status: GenerationStatus;

  assetId?: string;
  imagePath?: string;
  presetId: PresetId;
  durationSec: number;
  dedupeKey: string;


  provider?: string;
  outputVideoPath?: string;

  error?: string;
};
