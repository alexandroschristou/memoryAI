export type PresetId = "gentle_motion" | "cinematic_pan" | "old_film";

export type Preset = {
  id: PresetId;
  label: string;
  description: string;
  defaultDurationSec: number;
  promptBase: string;
};

/**
 * Global correctness constraints.
 * These MUST apply to every preset to prevent visual breakage.
 */
const GLOBAL_CONSTRAINTS = [
  "Maintain the exact composition and layout of the input image.",
  "Do not add, remove, invent, or change any objects, people, text, logos, or background elements.",
  "Do not change identity, facial features, clothing, brand marks, or object geometry.",
  "No morphing, warping, melting, or shape drift.",
  "No scene cuts, transitions, or new camera angles.",
  "Keep lighting and colors consistent; avoid flicker.",
  "Keep the image sharp and stable; avoid jitter and temporal artifacts.",
] as const;

export const PRESETS: Preset[] = [
  {
    id: "gentle_motion",
    label: "Gentle motion",
    description: "Subtle, stable movement. Best for portraits and calm scenes.",
    defaultDurationSec: 5,
    promptBase:
      "Animate the image with very subtle, natural motion only. Use micro-movements and minimal parallax. Keep everything realistic and stable.",
  },
  {
    id: "cinematic_pan",
    label: "Cinematic pan",
    description: "Slow, smooth camera movement with a cinematic feel.",
    defaultDurationSec: 5,
    promptBase:
      "Create a slow, smooth cinematic camera pan with extremely mild parallax. The subject must remain stable and unchanged.",
  },
  {
    id: "old_film",
    label: "Old film",
    description: "Vintage vibe: soft grain, warm tone, gentle jitter.",
    defaultDurationSec: 5,
    promptBase:
      "Animate the image with subtle motion inspired by vintage film. Add gentle grain and slight organic jitter without altering structure or details.",
  },
];

export function getPreset(id: PresetId): Preset {
  const found = PRESETS.find((p) => p.id === id);
  if (!found) throw new Error(`Unknown preset: ${id}`);
  return found;
}

/**
 * Builds the final prompt text for a preset.
 * This is the SINGLE source of truth for prompt generation.
 */
export function buildPromptFromPreset(id: PresetId): string {
  const preset = getPreset(id);

  return [
    preset.promptBase,
    "",
    "Constraints:",
    ...GLOBAL_CONSTRAINTS.map((c) => `- ${c}`),
  ].join("\n");
}
