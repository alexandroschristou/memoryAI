import type { GenerationServiceDeps } from "./generationService";
import { memoryGenerationRepo } from "@/lib/adapters/memoryGenerationRepo";
import { mvpJobQueue } from "@/lib/adapters/mvpJobQueue";
import type { VideoProviderFactory } from "@/lib/ports/videoProvider";
import { FakeProvider } from "@/lib/providers/fake";
import { consoleLogger } from "@/lib/adapters/consoleLogger";
import { memoryUploadRepo } from "@/lib/adapters/memoryUploadRepo";


const fakeFactory: VideoProviderFactory = {
  get: () => new FakeProvider(), // ignores name + env entirely
};

export const generationServiceDepsFake: GenerationServiceDeps = {
  repo: memoryGenerationRepo,
  uploadRepo: memoryUploadRepo,
  queue: mvpJobQueue,
  providerFactory: fakeFactory,
  logger: consoleLogger,
  now: () => Date.now(),
};


