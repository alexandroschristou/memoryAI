"use client";

import { useMemo, useState } from "react";
import { PRESETS, type PresetId } from "@/lib/presets";
import { useRouter } from "next/navigation";

export default function HomePage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [presetId, setPresetId] = useState<PresetId>("gentle_motion");
  const [busy, setBusy] = useState(false);
  const preset = useMemo(() => PRESETS.find(p => p.id === presetId)!, [presetId]);

  async function onSubmit() {
    if (!file) return;
    setBusy(true);
    try {
      // 1) upload
      const fd = new FormData();
      fd.append("file", file);
      const upRes = await fetch("/api/upload", { method: "POST", body: fd });
      if (!upRes.ok) throw new Error(await upRes.text());
      const { assetId  } = await upRes.json();

      // 2) create generation
      const genRes = await fetch("/api/generations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ assetId , presetId }),
      });
      if (!genRes.ok) throw new Error(await genRes.text());
      const { generationId } = await genRes.json();

      router.push(`/generate/${generationId}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main style={{ maxWidth: 720, margin: "40px auto", padding: 16, fontFamily: "system-ui" }}>
      <h1 style={{ fontSize: 28, fontWeight: 700 }}>Memory Video MVP</h1>
      <p style={{ opacity: 0.8 }}>
        Upload an image, pick a style, get a short animated clip.
      </p>

      <div style={{ marginTop: 24, display: "grid", gap: 12 }}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          disabled={busy}
        />

        <label style={{ display: "grid", gap: 6 }}>
          <span style={{ fontWeight: 600 }}>Style</span>
          <select value={presetId} onChange={(e) => setPresetId(e.target.value as PresetId)} disabled={busy}>
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>
          <span style={{ fontSize: 13, opacity: 0.75 }}>{preset.description}</span>
        </label>

        <button
          onClick={onSubmit}
          disabled={!file || busy}
          style={{ padding: "10px 14px", fontWeight: 700 }}
        >
          {busy ? "Working..." : "Generate video"}
        </button>
      </div>
    </main>
  );
}
