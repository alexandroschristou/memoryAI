import crypto from "node:crypto";
import type { Logger } from "@/lib/ports/logger";
import type { GenerationErrorCode } from "./errors";
import type { PresetId } from "@/lib/presets";
import { getPreset } from "@/lib/presets";
import type { GenerationRepository } from "@/lib/ports/generationRepo";
import type { JobQueue } from "@/lib/ports/jobQueue";
import type { VideoProviderFactory } from "@/lib/ports/videoProvider";
import type { UploadRepository } from "@/lib/ports/uploadRepo";


const MAX_DURATION = 5;

export type CreateGenerationInput = {
  assetId: string;
  presetId: PresetId;
};


export type GenerationServiceDeps = {
  repo: GenerationRepository;
  uploadRepo: UploadRepository;
  queue: JobQueue;
  providerFactory: VideoProviderFactory;
  logger: Logger;
  now: () => number;
};


function makeDedupeKey(assetId: string, presetId: PresetId) {
  return `${assetId}:${presetId}`;
}

export function createGeneration(deps: GenerationServiceDeps, input: CreateGenerationInput) {
  deps.logger.info("GENERATION_CREATE_REQUESTED", {
    presetId: input.presetId,
    assetId: input.assetId,
  });

  const asset = deps.uploadRepo.get(input.assetId);
  if (!asset) {
    deps.logger.warn("UPLOAD_ASSET_NOT_FOUND", { assetId: input.assetId });
    throw new Error("Unknown assetId");
  }

  const dedupeKey = makeDedupeKey(input.assetId, input.presetId);

  const existing = deps.repo.findByDedupeKey(dedupeKey);
  if (existing) {
    deps.logger.info("GENERATION_DEDUPED", {
      assetId: input.assetId,
      presetId: input.presetId,
      generationId: existing.id,
      status: existing.status,
    });

    return { generationId: existing.id, deduped: true as const };
  }

  const preset = getPreset(input.presetId);
  const durationSec = Math.min(preset.defaultDurationSec, MAX_DURATION);
  const providerName = deps.providerFactory.get().name;

  const id = crypto.randomUUID();

  deps.repo.create({
    id,
    createdAt: deps.now(),
    status: "queued",
    assetId: input.assetId,
    imagePath: asset.imagePath,
    presetId: input.presetId,
    durationSec,
    dedupeKey,               // ✅ required now
    provider: providerName,
  });

  deps.logger.info("GENERATION_QUEUED", { generationId: id, provider: providerName });
  deps.queue.enqueueGeneration(id);

  return { generationId: id, deduped: false as const };
}




export function getGeneration(deps: GenerationServiceDeps, id: string) {
  return deps.repo.get(id);
}

/**
 * Worker entrypoint: runs a single generation job.
 * Keeps orchestration in one place.
 */
export async function runGeneration(
  deps: GenerationServiceDeps,
  generationId: string
): Promise<void> {
  const gen = deps.repo.get(generationId);
  if (!gen) return;

  if (gen.status === "succeeded" || gen.status === "failed") return;

  deps.logger.info("GENERATION_STARTED", {
    generationId,
    provider: gen.provider,
  });

  try {
    deps.repo.setStatus(generationId, "creating");

    const provider = deps.providerFactory.get(gen.provider);

    deps.repo.setStatus(generationId, "running");

    const result = await provider.generate({
      generationId,
      imagePath: gen.imagePath!,
      presetId: gen.presetId,
      durationSec: gen.durationSec,
    });

    deps.repo.setStatus(generationId, "downloading");

    deps.repo.update(generationId, {
      status: "succeeded",
      outputVideoPath: result.videoPath,
      error: undefined,
    });

    deps.logger.info("GENERATION_SUCCEEDED", {
      generationId,
      provider: result.provider,
    });
  } catch (err: any) {
    const errorCode: GenerationErrorCode = "PROVIDER_FAILED";

    deps.repo.update(generationId, {
      status: "failed",
      error: errorCode,
    });

    deps.logger.error("GENERATION_FAILED", {
      generationId,
      errorCode,
      message: err?.message,
    });
  }
}


