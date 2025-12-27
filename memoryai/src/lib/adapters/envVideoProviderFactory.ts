import type { VideoProviderFactory } from "@/lib/ports/videoProvider";
import { getProvider, getProviderByName, type ProviderName } from "@/lib/providers";

/**
 * Adapter: chooses provider based on env/config by default, or forces an explicit provider when provided.
 */
export const envVideoProviderFactory: VideoProviderFactory = {
  get: (name?: string) => {
    if (!name) return getProvider();
    return getProviderByName(name as ProviderName);
  },
};
