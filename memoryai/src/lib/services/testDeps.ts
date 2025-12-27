import type { GenerationServiceDeps } from "./generationService";
import { memoryGenerationRepo } from "@/lib/adapters/memoryGenerationRepo";
import { mvpJobQueue } from "@/lib/adapters/mvpJobQueue";
import type { VideoProviderFactory } from "@/lib/ports/videoProvider";
import { FakeProvider } from "@/lib/providers/fake";

const fakeFactory: VideoProviderFactory = {
  get: () => new FakeProvider(), // ignores name + env entirely
};

export const generationServiceDepsFake: GenerationServiceDeps = {
  repo: memoryGenerationRepo,
  queue: mvpJobQueue,
  providerFactory: fakeFactory,
  now: () => Date.now(),
};
