import type { VideoProvider } from "@/lib/providers/types";

export interface VideoProviderFactory {
  get(name?: string): VideoProvider;
}
