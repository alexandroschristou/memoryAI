export interface JobQueue {
  enqueueGeneration(generationId: string): void;
}
