import type { VideoProvider } from "./types";
import { FakeProvider } from "./fake";
import { RunwayProvider } from "./runway";
import { config } from "@/lib/config";

export type ProviderName = "runway" | "fake";

export function getProviderByName(name: ProviderName): VideoProvider {
  switch (name) {
    case "runway":
      return new RunwayProvider();
    case "fake":
    default:
      return new FakeProvider();
  }
}

export function getProvider(): VideoProvider {
  // env-driven default
  return getProviderByName((config.videoProvider as ProviderName) ?? "fake");
}
