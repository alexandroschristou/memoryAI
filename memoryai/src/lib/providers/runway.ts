import fs from "node:fs/promises";
import path from "node:path";
import type { VideoProvider, GenerationInput, GenerationOutput } from "./types";
import { config } from "@/lib/config";
import {buildPromptFromPreset} from "@/lib/presets";

const RUNWAY_VERSION = "2024-11-06";
const MODEL = "gen4_turbo";
const RATIO = "960:960";

export class RunwayProvider implements VideoProvider {
  public readonly name = "runway";

  async generate(input: GenerationInput): Promise<GenerationOutput> {
    const outDir = path.join(process.cwd(), ".local", "outputs");
    await fs.mkdir(outDir, { recursive: true });
    const videoPath = path.join(outDir, `${input.generationId}.mp4`);

    // 1️⃣ Upload image to Runway
    const uploadUri = await this.uploadImage(input.imagePath);

    // 2️⃣ Start generation
    const promptText = buildPromptFromPreset(input.presetId);

    const createRes = await fetch(
      `${config.runway.apiBase}/v1/image_to_video`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${config.runway.apiKey}`,
          "X-Runway-Version": RUNWAY_VERSION,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: MODEL,
          promptImage: uploadUri,
          promptText,
          duration: input.durationSec,
          ratio: RATIO,
        }),
      }
    );

    if (!createRes.ok) {
      const text = await createRes.text();
      throw new Error(`Runway task creation failed: ${text}`);
    }

    const { id: taskId } = await createRes.json();

    // 3️⃣ Poll task status
    let videoUrl: string | null = null;
    const start = Date.now();
    const timeoutMs = 3 * 60 * 1000;

    while (!videoUrl) {
      if (Date.now() - start > timeoutMs) {
        throw new Error("Runway generation timed out");
      }

      await sleep(3000);

      const statusRes = await fetch(
        `${config.runway.apiBase}/v1/tasks/${taskId}`,
        {
          headers: {
            Authorization: `Bearer ${config.runway.apiKey}`,
            "X-Runway-Version": RUNWAY_VERSION,
          },
        }
      );

      const data = await statusRes.json();

      if (data.status === "FAILED") {
        throw new Error("Runway generation failed");
      }

      if (data.status === "SUCCEEDED") {
        videoUrl = data.output?.[0] ?? null;
      }
    }

    // 4️⃣ Download video
    const videoRes = await fetch(videoUrl!);
    const buffer = Buffer.from(await videoRes.arrayBuffer());
    await fs.writeFile(videoPath, buffer);

    return { videoPath, provider: this.name };
  }

  private async uploadImage(imagePath: string): Promise<string> {
    // 1) Create upload "slot"
    const createRes = await fetch(`${config.runway.apiBase}/v1/uploads`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${config.runway.apiKey}`,
        "X-Runway-Version": RUNWAY_VERSION,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        filename: path.basename(imagePath),
        type: "ephemeral",
      }),
    });

    if (!createRes.ok) {
      const text = await createRes.text();
      throw new Error(`Runway upload slot creation failed: ${text}`);
    }

    const { uploadUrl, fields, runwayUri } = (await createRes.json()) as {
      uploadUrl: string;
      fields: Record<string, string>;
      runwayUri: string;
    };

    // 2) Upload file bytes via multipart/form-data to the provided uploadUrl
    const bytes = await fs.readFile(imagePath);

    // Node 18+ has global FormData/Blob
    const form = new FormData();
    for (const [k, v] of Object.entries(fields)) form.append(k, v);

    // If you know the real mime type, set it; otherwise omit type.
    const filename = path.basename(imagePath);
    const blob = new Blob([bytes]); // or: new Blob([bytes], { type: "image/png" })
    form.append("file", blob, filename);

    const uploadRes = await fetch(uploadUrl, { method: "POST", body: form });
    if (!uploadRes.ok) {
      const text = await uploadRes.text();
      // Per Runway docs: if upload fails, create a NEW /v1/uploads and retry from scratch.
      throw new Error(`Runway file upload failed: ${text}`);
    }

    return runwayUri; // <-- this is what you pass to promptImage
  }

}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

