import type { Logger } from "@/lib/ports/logger";

function log(
  level: "INFO" | "WARN" | "ERROR",
  event: string,
  data?: Record<string, unknown>
) {
  const payload = {
    ts: new Date().toISOString(),
    level,
    event,
    ...data,
  };

  // Single JSON log line (easy to parse later)
  console.log(JSON.stringify(payload));
}

export const consoleLogger: Logger = {
  info: (event, data) => log("INFO", event, data),
  warn: (event, data) => log("WARN", event, data),
  error: (event, data) => log("ERROR", event, data),
};
