import type { JobQueue } from "@/lib/ports/jobQueue";
import { enqueueGeneration } from "@/lib/jobs/queue";

/**
 * Adapter: wraps your current fire-and-forget queue behind a port.
 * Later swap to BullMQ/SQS/QStash/Inngest.
 */
export const mvpJobQueue: JobQueue = {
  enqueueGeneration,
};
