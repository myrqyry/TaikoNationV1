"""Project CLI entrypoints used by pyproject console scripts."""

from __future__ import annotations

from taikonation.generation.generator import main as generate_main
from taikonation.training.trainer import load_config, main as train_main


def train() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train TaikoNation model.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    train_main(load_config(args.config))


def generate() -> None:
    generate_main()


def serve() -> None:
    import uvicorn

    uvicorn.run("web.server_fastapi:socket_app", host="127.0.0.1", port=5000, reload=False)
