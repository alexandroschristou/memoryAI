import { NextResponse } from "next/server";
import path from "node:path";
import fs from "node:fs/promises";
import crypto from "node:crypto";

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

  const uploadDir = path.join(process.cwd(), ".local", "uploads");
  await fs.mkdir(uploadDir, { recursive: true });

  const ext = file.type === "image/png" ? "png" : "jpg";
  const imagePath = path.join(uploadDir, `${id}.${ext}`);

  await fs.writeFile(imagePath, buf);

  return NextResponse.json({ assetId: id, imagePath });
}
