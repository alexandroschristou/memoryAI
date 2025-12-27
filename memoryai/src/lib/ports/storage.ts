export type SaveUploadInput = {
  assetId: string;
  bytes: Buffer;
  ext: string;          // "png" | "jpg" | ...
};

export interface Storage {
  saveUpload(input: SaveUploadInput): Promise<{ imagePath: string }>;
  // later: saveOutputVideo(), getStream(), signedUrl(), etc.
}
