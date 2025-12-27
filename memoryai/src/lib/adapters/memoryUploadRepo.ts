import type { UploadAsset, UploadRepository } from "@/lib/ports/uploadRepo";

class MemoryUploadRepo implements UploadRepository {
  private readonly map = new Map<string, UploadAsset>();

  create(asset: UploadAsset): void {
    this.map.set(asset.id, asset);
  }

  get(id: string): UploadAsset | null {
    return this.map.get(id) ?? null;
  }
}

export const memoryUploadRepo = new MemoryUploadRepo();
