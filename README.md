# Témoin — neutral visual witness for remote second-hand trade

Témoin recognizes a used item by its visible flaws — like a fingerprint — and
produces a sealed constat proving the delivered (or returned) item is the one
that was sold. It is not an image describer: a verifiable investigator agent
runs a multi-step enquiry (autonomous zooms, SAM segmentation, challenge-code
checks, photo requests), decides under deterministic guardrails, and seals
every conclusion with its audit trail.

Live demo: publishing on Streamlit Community Cloud — the public link will be
added here (deploy guide below).

## Features

- **Wear fingerprint** — every flaw mapped (kind, zone, severity, bbox).
- **Challenge code** — short unique code (e.g. `K7`), valid 15 minutes,
  handwritten next to the item; proves the photo was taken now.
- **Guided checkpoints** — the 3 most distinctive flaws to re-photograph.
- **Investigator agent** — tool loop (`zoom`, `segment`, `check_challenge`,
  `request_photo`, `match_defects`) with a sealed, auditable trail.
- **Deterministic verdicts** — CONFORME / ENDOMMAGÉ / À VÉRIFIER /
  NON CONFORME; the model proposes, the rules dispose (freshness cap,
  2-contradiction rule for accusations).
- **Devil's advocate** — every verdict is self-attacked (up to 3
  objections) before sealing; strong objections downgrade to À VÉRIFIER.
- **Evidence board** — side-by-side flaw crops (sale ↔ delivery) with
  match statuses, rendered from the sealed certificate.
- **Sealed certificate** — SHA-256 chain + QR + WhatsApp share link.
- **Precise segmentation** — SAM 3.1 masks overlaid on the photo.
- **Voice declaration** — transcribed with Muse Voice Transcribe.
- **Video constat** — walkaround video fused into multi-view confirmed flaws.
- **Materials + condition grade** — observed vs declared, A/B/C/D note.
- **Forensics** — retouch signals plus photo quality coaching.
- **Depth Matrix** — CPU-only relief from two views or a video (no GPU),
  parallax GIF and interactive 3D cloud; unmeasured zones stay empty, never
  invented.
- **Monocular AI depth** — dense indicative relief from a single photo
  (MiDaS-small ONNX on CPU, no GPU); always labeled as estimated, never
  proof by itself.
- **FR/EN interface**, phone-friendly flows (Sale / Delivery / Verify).

## Evaluation

20 annotated pairs (`eval/pairs/`, labels in `eval/labels.csv`): same object
under transforms, missing challenge codes, object-not-visible fraud attempts.
Full run: `python -m eval.run_eval`. Report: `eval/RESULTS.md`.

- Accuracy: **18/20 (90%)** (see `eval/RESULTS.md`)
- False accusations (NON CONFORME on a same-object pair): **0 (0 %)** (goal: 0 %)
- Different-object pairs cleared as same: **0** (goal: 0)

Known gaps: damaged-between-photos and same-model-different-unit need new
photos (see `eval/pairs/README.md`).

## Meta models used

- **Muse Spark Open** (`meta-models/muse-spark-open-1026` via Baseten) —
  the reasoning core: constats, investigator agent, verdicts.
- **SAM 3.1** (`sam-3.1` via Meta Model API) — pixel-exact object masks.
- **Muse Voice Transcribe** (`muse-voice-transcribe-1.0` via Meta Model API)
  — one-shot transcription of voice declarations.
- Built with **Muse Code** (Meta, agentic CLI).

## Built during the hackathon (Oct 27 – Nov 4)

Built for the Meta Global AI Developer Hackathon (Indie Developer track,
*Build for Your World*), October 27 – November 4, 2026. One commit per step,
public repository. The demo shows only what was built during the event.

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
# put your keys in .env: BASETEN_API_KEY=... and MODEL_API_KEY=...
.venv\Scripts\python -m streamlit run app.py
```

CLI:

```powershell
.venv\Scripts\python -m temoin constat photo.jpg --code K7
.venv\Scripts\python -m temoin compare sale.jpg delivery.jpg --sale-code K7 --delivery-code A8
.venv\Scripts\python -m temoin segment photo.jpg --concept headphones
.venv\Scripts\python -m temoin transcribe note.wav --language-bias french
.venv\Scripts\python -m temoin video-constat tour.mp4 --code K7
.venv\Scripts\python -m temoin forensics photo.jpg
.venv\Scripts\python -m temoin stereo left.jpg right.jpg --bbox 400,400,600,600
.venv\Scripts\python -m temoin video-depth tour.mp4 --cloud cloud.html
.venv\Scripts\python -m temoin mono-depth photo.jpg --cloud cloud.html
.venv\Scripts\python -m temoin board cert-1.json --sale sale.jpg --delivery delivery.jpg
.venv\Scripts\python -m temoin verify cert-1.json
.venv\Scripts\python -m eval.run_eval
```

Tests and lint from the project root: `python -m pytest`, `ruff check .`,
`ruff format .` (244 tests, offline with a fake client).

## Deploy your own (Streamlit Community Cloud)

1. Push this repository to a public GitHub repo.
2. On share.streamlit.io, create an app from the repo (entrypoint `app.py`,
   Python 3.11).
3. In the app Secrets, paste:
   ```toml
   BASETEN_API_KEY = "your-baseten-key"
   MODEL_API_KEY = "your-meta-model-api-key"
   ```
4. Deploy. Note: cloud storage is ephemeral — sealed certificates should be
   downloaded (JSON + QR) to keep them.

## Model notes

- Core endpoint: `https://inference.baseten.co/v1`, model
  `meta-models/muse-spark-open-1026`.
- Tool calling (`tools` parameter): **supported**. A minimal
  `chat.completions` call with a `get_time` function (tested 2026-10-28)
  returned `tool_calls=[get_time]` with no text answer.
- SAM concepts must be short **English** noun phrases (`headphones` works,
  `casque` finds nothing — tested 2026-10-30).
- Meta Model API: `https://api.meta.ai/v1` (key `MODEL_API_KEY`). When it is
  unreachable, Témoin degrades gracefully (constat boxes, typed declaration)
  and never invents masks or transcripts.
- Mono AI relief: MiDaS v2.1 small ONNX (official isl-org release, ~64 MB),
  run on CPU via OpenCV DNN in ~1 s. Downloaded once on first use into
  `data/models/` (git-ignored) with SHA-256 verification; offline use fails
  with a clear message instead of a fake relief.

## Privacy

Photos stay local in `data/` (git-ignored) with deletion possible. Faces and
people are signaled, never identified. No keys or personal data in code, logs
or git.
