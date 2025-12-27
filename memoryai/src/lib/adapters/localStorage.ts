import fs from "node:fs/promises";
import path from "node:path";
import type { Storage, SaveUploadInput } from "@/lib/ports/storage";

export const localStorage: Storage = {
  async saveUpload(input: SaveUploadInput): Promise<{ imagePath: string }> {
    const uploadDir = path.join(process.cwd(), ".local", "uploads");
    await fs.mkdir(uploadDir, { recursive: true });

    const imagePath = path.join(uploadDir, `${input.assetId}.${input.ext}`);
    await fs.writeFile(imagePath, input.bytes);

    return { imagePath };
  },
};
