# MemoryAI

MemoryAI is a service-oriented, preset-driven image-to-video generation SaaS.

Users upload an image, select a motion preset, and receive a generated video.
The system is designed to evolve from a safe MVP into a production-grade SaaS
without architectural rewrites.

This repository currently contains a **fully working MVP** with a **clean,
future-proof architecture**.

---

## Core principles

- **No free-form prompts**  
  Users select presets only. This guarantees visual correctness and prevents
  model hallucinations or broken outputs.

- **Service-oriented architecture**  
  Business logic is isolated from frameworks, APIs, and infrastructure.

- **Provider-agnostic**  
  Runway is used today. A local or self-hosted model can replace it later
  without touching the core logic.

- **Safe by default**  
  Fake providers, mock adapters, and deterministic execution prevent accidental
  credit usage.

---

## High-level architecture

MemoryAI follows a **Ports & Adapters (Hexagonal) architecture**.

```
UI / API Routes
      ↓
 Service Layer (business logic)
      ↓
 Ports (interfaces)
      ↓
 Adapters (Runway, Fake, memory store, local FS, etc.)
```

---

## Project structure

```
.local/                     Runtime artifacts (gitignored)
src/
├─ app/
│  ├─ globals.css
│  ├─ layout.tsx
│  ├─ page.tsx
│  ├─ generate/[id]/
│  │  ├─ GenerateClient.tsx
│  │  └─ page.tsx
│  └─ api/
│     ├─ upload/route.ts
│     ├─ generations/
│     │  ├─ route.ts
│     │  └─ [id]/route.ts
│     └─ media/[id]/route.ts
│
├─ lib/
│  ├─ config.ts
│  ├─ presets.ts
│  │
│  ├─ services/             Core business logic
│  │  ├─ generationService.ts
│  │  ├─ index.ts            Composition root
│  │  └─ testDeps.ts         Fake-provider deps (safe testing)
│  │
│  ├─ ports/                Interfaces (contracts)
│  │  ├─ generationRepo.ts
│  │  ├─ jobQueue.ts
│  │  ├─ videoProvider.ts
│  │  ├─ storage.ts
│  │  ├─ uploadRepo.ts
│  │  ├─ clock.ts
│  │  └─ logger.ts
│  │
│  ├─ adapters/             Port implementations
│  │  ├─ envVideoProviderFactory.ts
│  │  ├─ memoryGenerationRepo.ts
│  │  └─ mvpJobQueue.ts
│  │
│  ├─ providers/            Video generation backends
│  │  ├─ fake.ts
│  │  ├─ runway.ts
│  │  ├─ types.ts
│  │  ├─ index.ts
│  │  └─ placeholder.mp4
│  │
│  └─ store/                In-memory persistence (MVP)
│     ├─ types.ts
│     ├─ memoryStore.ts
│     └─ index.ts
```

---

## Presets

Users never define prompts directly.

Each preset defines:
- base prompt
- global correctness constraints
- default duration

Presets live in:
```
src/lib/presets.ts
```

---

## Service layer

The service layer owns all orchestration:
- validation
- provider selection
- lifecycle management
- failure handling

Core file:
```
src/lib/services/generationService.ts
```

---

## Deterministic provider selection

The provider is chosen at generation creation time and stored on the record.
Queued jobs are not affected by environment changes.

---

## Generation lifecycle

```
queued → creating → running → downloading → succeeded
                       ↘
                        failed / needs_review
```

---

## Testing without spending credits

Use the fake provider:

.env.local:
```
VIDEO_PROVIDER=fake
```

Or force fake deps:
```
import { generationServiceDepsFake } from "@/lib/services/testDeps";
```

---

## Roadmap

- Database adapter
- Authentication
- Billing & quotas
- Object storage (S3/R2)
- Admin dashboard

---

## License

TBD
