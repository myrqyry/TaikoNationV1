<!-- .github/copilot-instructions.md for TaikoNationV1 -->
# TaikoNationV1 — Copilot instructions for code edits

These instructions are written for an AI coding agent editing this repository. Keep guidance concrete and tied to files and patterns actually present in the codebase.

1) Big picture (what changes look like)
- Purpose: PyTorch transformer that maps audio features (.npy / spectrograms) -> taiko chart token sequences. Training and data code lives at the repo root (see `train_transformer.py`, `transformer_model.py`, `transformer_dataset.py`).
- Web UI: `web/` contains a Flask (legacy) and a FastAPI-compatible server (`web/server.py`, `web/server_fastapi.py`) used for dev, generation, and human evaluation.

2) Key files to read before editing
- Model: `transformer_model.py` — encoder/decoder transformer, positional encodings, audio projection.
- Data pipeline: `transformer_dataset.py` — expects `input_charts_nr/` and `input_songs/` and `output/genre_labels.json`; tokenizer in `tokenization.py`; audio feature helpers in `audio_processing.py`.
- Training entrypoint: `train_transformer.py` — uses `config/default.yaml`, wandb (offline by default), and saves models under `training.save_path`.
- Config: `config/default.yaml` — single source of hyperparameters and data/workflow toggles.
- Web UI docs: `web/web-setup-readme.md` — contains practical run instructions and FastAPI notes.

3) Developer workflows (exact commands the repo expects)
- Install deps: `pip install -r requirements.txt` (optionally `pip install -r web/requirements.txt` for the web UI).
- Train model (default config):
  - `python train_transformer.py --config config/default.yaml`
  - The script respects `dry_run` (quick single-fold) and `training.k_folds`.
- Run web UI (legacy):
  - `cd web && python server.py` (Flask) — serves at port 5000 by default.
- Run web UI (recommended FastAPI):
  - `cd web && uvicorn server_fastapi:socket_app --host 127.0.0.1 --port 5001 --reload`
- Generate chart (CLI example in README):
  - `python generate_chart.py model.pth input_songs/song.npy output_chart.osu --difficulty oni` (confirm `generate_chart.py` exists before using).

4) Conventions and patterns to follow
- Config-driven: Put hyperparameters and dataset settings into `config/*.yaml`. Editors should prefer reading/writing YAML used by `train_transformer.py`.
- Tokenizer alignment: `tokenization.py` and `transformer_dataset.py` assume `[CLS]` and `[PAD]` tokens exist; when changing token IDs, update both tokenizer and dataset padding logic.
- Sequence lengths: The code pads/truncates to `data.max_sequence_length` (default 512). Keep new model inputs/outputs compatible or add conversion code in `transformer_dataset.py`.
- Genre/difficulty conditioning: Model expects `genre_id` and `difficulty` in batches. Any change to genre handling must preserve `output/genre_labels.json` format and `genre_vocab` building in `transformer_dataset.py`.
- Device handling: training uses `torch.device("cuda" if available else "cpu")`. Avoid forcing CUDA-only code in changes unless you add safe fallbacks.

5) Testing & quick checks
- After edits to model/dataset: run a very short dry run: set `dry_run: true` in `config/default.yaml` or pass a small dataset to validate shapes without full training.
- Run `python -m pip install -r requirements.txt` in a clean venv and run `python train_transformer.py --config config/default.yaml` to ensure no immediate import/runtime errors.

6) Integration & external dependencies
- wandb: training uses Weights & Biases; scripts set `WANDB_MODE=offline` by default. Avoid assuming an online account.
- Web: FastAPI + uvicorn recommended for production; Flask server kept for compatibility. Background tasks should run in separate worker processes (Celery/RQ) — the repo documents this in `web/web-setup-readme.md`.

7) Avoid these common pitfalls
- Don’t change token names/IDs without updating `tokenization.py`, `transformer_dataset.py`, and any saved vocab files in `output/`.
- Don’t assume variable-length batches are already supported — padding is currently used and padding masks are commented out in `transformer_model.py`.
- When modifying the transformer API, update `train_transformer.py` call site (model constructor args and forward signature) and any web endpoints that call generation logic.

8) Small, high-value edits you can make proactively
- Add shape assertions in the forward pass of `transformer_model.py` (non-breaking, helps debugging).
- Add a short `scripts/check_data.py` that validates `input_charts_nr/` and `input_songs/` alignment (useful for CI).

If a change involves behavior not discoverable from files (external deployments, GPU setup), ask a human and include suggested commands to reproduce locally.

---
If anything in these notes is unclear or you want more detail in a specific area (data format, generation CLI, web API endpoints), tell me which area to expand and I'll iterate.
