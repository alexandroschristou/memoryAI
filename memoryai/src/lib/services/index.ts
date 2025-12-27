import { memoryGenerationRepo } from "@/lib/adapters/memoryGenerationRepo";
import { mvpJobQueue } from "@/lib/adapters/mvpJobQueue";
import { envVideoProviderFactory } from "@/lib/adapters/envVideoProviderFactory";
import type { GenerationServiceDeps } from "./generationService";

export const generationServiceDeps: GenerationServiceDeps = {
  repo: memoryGenerationRepo,
  queue: mvpJobQueue,
  providerFactory: envVideoProviderFactory,
  now: () => Date.now(),
};
