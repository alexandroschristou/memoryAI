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

  imagePath?: string;
  presetId: PresetId;
  durationSec: number;

  provider?: string;
  outputVideoPath?: string;

  error?: string;
};
