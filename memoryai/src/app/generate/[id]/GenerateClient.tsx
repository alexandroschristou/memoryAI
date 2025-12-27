"use client";

import { useEffect, useState } from "react";

type Gen = {
  id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  provider?: string;
  error?: string;
  hasVideo: boolean;
};

export default function GenerateClient({ id }: { id: string }) {
  const [gen, setGen] = useState<Gen | null>(null);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function poll() {
      const res = await fetch(`/api/generations/${encodeURIComponent(id)}`, {
        cache: "no-store",
      });

      if (!res.ok) {
        if (alive) timer = setTimeout(poll, 1500);
        return;
      }

      const data = (await res.json()) as Gen;
      if (!alive) return;

      setGen(data);

      if (data.status === "queued" || data.status === "running") {
        timer = setTimeout(poll, 1500);
      }
    }

    void poll();

    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [id]);

  return (
    <main
      style={{
        maxWidth: 720,
        margin: "40px auto",
        padding: 16,
        fontFamily: "system-ui",
      }}
    >
      <h1 style={{ fontSize: 22, fontWeight: 700 }}>Generation</h1>

      {!gen && <p>Loading…</p>}

      {gen && (
        <>
          <p>
            Status: <b>{gen.status}</b>
            {gen.provider ? ` (provider: ${gen.provider})` : ""}
          </p>

          {gen.status === "failed" && (
            <p style={{ color: "crimson" }}>
              Failed: {gen.error ?? "Unknown error"}
            </p>
          )}

          {gen.status === "succeeded" && gen.hasVideo && (
            <div style={{ marginTop: 16, display: "grid", gap: 12 }}>
              <video
                controls
                src={`/api/media/${encodeURIComponent(id)}`}
                style={{ width: "100%", borderRadius: 8 }}
              />
              <a href={`/api/media/${encodeURIComponent(id)}`} download>
                Download MP4
              </a>
            </div>
          )}

          {(gen.status === "queued" || gen.status === "running") && (
            <p style={{ opacity: 0.8 }}>Generating…</p>
          )}
        </>
      )}
    </main>
  );
}
