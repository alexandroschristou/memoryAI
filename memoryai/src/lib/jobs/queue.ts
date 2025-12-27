import { runGenerationJob } from "./worker";

/**
 * MVP "queue": fire-and-forget async.
 * Swap later with Inngest/QStash without changing your API contract.
 */
export function enqueueGeneration(generationId: string) {
  // Don’t await: emulate background processing
  void runGenerationJob(generationId);
}
