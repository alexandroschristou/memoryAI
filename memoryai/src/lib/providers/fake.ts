import fs from "node:fs/promises";
import path from "node:path";
import type { VideoProvider, GenerationInput, GenerationOutput } from "./types";

/**
 * Fake provider: copies a bundled placeholder MP4 into a per-generation output file.
 * This lets you build the whole app flow before integrating a real model API.
 */
export class FakeProvider implements VideoProvider {
  public readonly name = "fake";

  async generate(input: GenerationInput): Promise<GenerationOutput> {
    const outDir = path.join(process.cwd(), ".local", "outputs");
    await fs.mkdir(outDir, { recursive: true });

    const placeholder = path.join(process.cwd(), "src", "lib", "providers", "placeholder.mp4");
    const videoPath = path.join(outDir, `${input.generationId}.mp4`);

    await fs.copyFile(placeholder, videoPath);
    return { videoPath, provider: this.name };
  }
}
