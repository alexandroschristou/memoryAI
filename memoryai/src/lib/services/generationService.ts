import crypto from "node:crypto";
import type { PresetId } from "@/lib/presets";
import { getPreset } from "@/lib/presets";
import type { GenerationRepository } from "@/lib/ports/generationRepo";
import type { JobQueue } from "@/lib/ports/jobQueue";
import type { VideoProviderFactory } from "@/lib/ports/videoProvider";

const MAX_DURATION = 5;

export type CreateGenerationInput = {
  imagePath: string;
  presetId: PresetId;
  // userId?: string; // add later without changing architecture
};

export type GenerationServiceDeps = {
  repo: GenerationRepository;
  queue: JobQueue;
  providerFactory: VideoProviderFactory;
  now: () => number;
};
export function createGeneration(deps: GenerationServiceDeps, input: CreateGenerationInput) {
  const preset = getPreset(input.presetId);
  const durationSec = Math.min(preset.defaultDurationSec, MAX_DURATION);

  const providerName = deps.providerFactory.get().name; // 결정 now, store for later

  const id = crypto.randomUUID();
  deps.repo.create({
    id,
    createdAt: deps.now(),
    status: "queued",
    imagePath: input.imagePath,
    presetId: input.presetId,
    durationSec,
    provider: providerName, // ✅ IMPORTANT
  });

  deps.queue.enqueueGeneration(id);
  return { generationId: id };
}


export function getGeneration(deps: GenerationServiceDeps, id: string) {
  return deps.repo.get(id);
}

/**
 * Worker entrypoint: runs a single generation job.
 * Keeps orchestration in one place.
 */
export async function runGeneration(deps: GenerationServiceDeps, generationId: string): Promise<void> {
  const gen = deps.repo.get(generationId);
  if (!gen) return;

  // Don't rerun completed jobs
  if (gen.status === "succeeded" || gen.status === "failed") return;

  if (!gen.imagePath) {
    deps.repo.update(generationId, { status: "failed", error: "Missing imagePath" });
    return;
  }

  // ✅ if provider isn't stored (older records), fall back to current env provider
  const providerName = gen.provider;

  try {
    deps.repo.setStatus(generationId, "creating");

    const provider = deps.providerFactory.get(providerName);

    deps.repo.setStatus(generationId, "running");

    const result = await provider.generate({
      generationId,
      imagePath: gen.imagePath,
      presetId: gen.presetId,
      durationSec: gen.durationSec,
    });

    deps.repo.setStatus(generationId, "downloading");

    // For FakeProvider/your RunwayProvider, `generate()` already outputs a local mp4 path.
    // Later when you introduce storage abstraction, this is where you'd upload to S3, etc.

    deps.repo.update(generationId, {
      status: "succeeded",
      provider: result.provider,
      outputVideoPath: result.videoPath,
      error: undefined,
    });
  } catch (e: any) {
    deps.repo.update(generationId, {
      status: "failed",
      error: e?.message ?? "Unknown error",
    });
  }
}

