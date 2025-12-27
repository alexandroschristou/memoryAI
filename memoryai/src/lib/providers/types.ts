import type { PresetId } from "@/lib/presets";

export type GenerationInput = {
  generationId: string;
  imagePath: string;
  presetId: PresetId;
  durationSec: number;
};

export type GenerationOutput = {
  videoPath: string;
  provider: string;
};

export interface VideoProvider {
  readonly name: string;
  generate(input: GenerationInput): Promise<GenerationOutput>;
}
