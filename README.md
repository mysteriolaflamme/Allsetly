# Temoin

Neutral visual witness for remote second-hand trade: it recognizes an item by its visible flaws — like a fingerprint — and produces a sealed constat proving the delivered (or returned) item is the one that was sold.

Built for the Meta Global AI Developer Hackathon with the open-weights model `meta-models/muse-spark-open-1026` via Baseten inference.

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install openai python-dotenv pillow streamlit pydantic pytest ruff
# put your key in .env: BASETEN_API_KEY=...
.venv\Scripts\python -m temoin constat photo.jpg --code K7
```

Tests and lint from the project root: `python -m pytest`, `ruff check .`, `ruff format .`.

## Model notes

- Endpoint: `https://inference.baseten.co/v1`, model `meta-models/muse-spark-open-1026`.
- Tool calling (`tools` parameter): **supported**. A minimal `chat.completions` call with a `get_time` function (tested 2026-10-28) returned `tool_calls=[get_time]` with no text answer.
