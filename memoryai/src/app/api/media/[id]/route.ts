import { NextResponse } from "next/server";
import fs from "node:fs/promises";
import { store } from "@/lib/store";

export const runtime = "nodejs";

type Ctx = { params: Promise<{ id: string }> };

export async function GET(_: Request, ctx: Ctx) {
  const { id } = await ctx.params;

  const gen = store.get(id);
  if (!gen?.outputVideoPath) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  const bytes = await fs.readFile(gen.outputVideoPath);
  return new NextResponse(bytes, {
    headers: {
      "Content-Type": "video/mp4",
      "Cache-Control": "no-store",
    },
  });
}
