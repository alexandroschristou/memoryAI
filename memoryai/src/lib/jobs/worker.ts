import { generationServiceDeps } from "@/lib/services";
import { runGeneration } from "@/lib/services/generationService";

export async function runGenerationJob(generationId: string): Promise<void> {
  await runGeneration(generationServiceDeps, generationId);
}
