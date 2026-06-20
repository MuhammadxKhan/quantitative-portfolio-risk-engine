
from __future__ import annotations

from pathlib import Path

from engine.io_utils import load_config
from engine.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent


def main():
    cfg = load_config(ROOT / "config" / "default.yml")

    try:
        written = run_pipeline(cfg)
    except Exception as exc:
        # if a live data pull fails, fall back to sample data so it still runs
        if cfg.source != "yfinance":
            raise
        print(f"\nlive data failed ({exc}) - falling back to sample data")
        cfg.source = "sample"
        written = run_pipeline(cfg)

    print("\nSaved outputs:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    print("\nDone - charts are in the outputs/ folder.")

    input("\nPress Enter to close...")


if __name__ == "__main__":
    main()


