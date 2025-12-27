export type UploadAsset = {
  id: string;                 // assetId
  imagePath: string;          // absolute path on server (MVP)
  contentType: string;        // e.g. image/png
  createdAt: number;
};

export interface UploadRepository {
  create(asset: UploadAsset): void;
  get(id: string): UploadAsset | null;
}
