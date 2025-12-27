import { memoryGenerationRepo } from "@/lib/adapters/memoryGenerationRepo";
import { mvpJobQueue } from "@/lib/adapters/mvpJobQueue";
import { envVideoProviderFactory } from "@/lib/adapters/envVideoProviderFactory";
import type { GenerationServiceDeps } from "./generationService";
import { consoleLogger } from "@/lib/adapters/consoleLogger";
import { memoryUploadRepo } from "@/lib/adapters/memoryUploadRepo";

export const generationServiceDeps: GenerationServiceDeps = {
  repo: memoryGenerationRepo,
  uploadRepo: memoryUploadRepo,
  queue: mvpJobQueue,
  providerFactory: envVideoProviderFactory,
  logger: consoleLogger,
  now: () => Date.now(),
};
