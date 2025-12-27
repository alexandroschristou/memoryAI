import { NextResponse } from "next/server";
import crypto from "node:crypto";
import { memoryUploadRepo } from "@/lib/adapters/memoryUploadRepo";
import { localStorage } from "@/lib/adapters/localStorage";

export const runtime = "nodejs";

export async function POST(req: Request) {
  const form = await req.formData();
  const file = form.get("file");

  if (!(file instanceof File)) {
    return NextResponse.json({ error: "Missing file" }, { status: 400 });
  }

  if (!file.type.startsWith("image/")) {
    return NextResponse.json({ error: "Only image uploads are supported" }, { status: 400 });
  }

  const buf = Buffer.from(await file.arrayBuffer());
  const id = crypto.randomUUID();

  const ext = file.type === "image/png" ? "png" : "jpg";

  const { imagePath } = await localStorage.saveUpload({
    assetId: id,
    bytes: buf,
    ext,
  });

  memoryUploadRepo.create({
    id,
    imagePath,
    contentType: file.type,
    createdAt: Date.now(),
  });

  // ✅ IMPORTANT: return only assetId to the client (no server paths)
  return NextResponse.json({ assetId: id });
}
