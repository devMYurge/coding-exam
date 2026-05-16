# Setup

Quick setup for both questions. Each is independent.

---

## Question 1 — On-Call Assistant (FastAPI · Python 3.11+)

A FastAPI app with three routes: `/v1` (TF-IDF keyword search), `/v2` (semantic search via fastembed), `/v3` (LangChain agent with a `readFile` tool). SOPs in `data/` are loaded on startup.

```bash
cd question-1

# one-time: create + activate a virtualenv, install deps
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Phase 3 (the LLM agent) needs an OpenAI key:
export OPENAI_API_KEY='sk-...'        # only required for /v3

python main.py                         # serves on http://127.0.0.1:8001
```

Then open:

- http://127.0.0.1:8001/v1 — keyword search page
- http://127.0.0.1:8001/v2 — semantic search page
- http://127.0.0.1:8001/v3 — agent chat page

Stop with `Ctrl+C`. Re-running just needs `source .venv/bin/activate && python main.py` (skip the install steps).

---

## Question 2 — Antigravity Particle Background (Vite · React 19 · R3F)

A tiny Vite project hosting the actual Medusae particle engine source (`src/Medusae.jsx`) — open it to read or edit the GLSL shaders, per-frame physics, and cursor logic.

```bash
cd question-2

npm install           # one-time
npm run dev           # Vite dev server on http://localhost:5173
```

Other scripts:

```bash
npm run build         # production build into dist/
npm run preview       # serve the built dist/ locally
```

### Where to edit

- `src/Medusae.jsx` — the particle engine: vertex + fragment shaders, the `100 × 55` instanced grid, the cursor-follow lerp, the per-frame `useFrame` loop.
- `src/defaults.js` — every tunable knob (cursor radius/strength/drag, halo amplitude/frequency/rim, particle base/active size, blob scale, rotation, the three particle colors).
- `src/App.jsx` — pass overrides via `config={{ ... }}`; anything not set falls back to `defaults.js`.

Vite hot-reloads on save, so a shader tweak shows up immediately.
