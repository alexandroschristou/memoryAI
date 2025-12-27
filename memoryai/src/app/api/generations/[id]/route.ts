import { NextResponse } from "next/server";
import { generationServiceDeps } from "@/lib/services";
import { getGeneration } from "@/lib/services/generationService";

export const runtime = "nodejs";

type Ctx = { params: Promise<{ id: string }> };

export async function GET(_: Request, ctx: Ctx) {
  const { id } = await ctx.params;

  const gen = getGeneration(generationServiceDeps, id);
  if (!gen) return NextResponse.json({ error: "Not found" }, { status: 404 });

  return NextResponse.json({
    id: gen.id,
    status: gen.status,
    presetId: gen.presetId,
    durationSec: gen.durationSec,
    provider: gen.provider,
    error: gen.error,
    hasVideo: Boolean(gen.outputVideoPath),
  });
}
