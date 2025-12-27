import { NextResponse } from "next/server";
import type { PresetId } from "@/lib/presets";
import { generationServiceDeps } from "@/lib/services";
import { createGeneration } from "@/lib/services/generationService";

export const runtime = "nodejs";

type Body = {
  imagePath: string;
  presetId: PresetId;
};

export async function POST(req: Request) {
  const body = (await req.json()) as Body;

  try {
    const result = createGeneration(generationServiceDeps, body);
    return NextResponse.json(result);
  } catch (e: any) {
    return NextResponse.json(
      { error: e?.message ?? "Invalid request" },
      { status: 400 }
    );
  }
}
