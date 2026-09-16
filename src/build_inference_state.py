"""Create the compact state used by fast prediction CLI calls."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from src.inference_state import save_inference_state
except ModuleNotFoundError:  # pragma: no cover
    from inference_state import save_inference_state  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact point-in-time inference state.")
    parser.add_argument("--clean-dir", type=Path, default=Path("data/interim/hltv_final_clean"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/inference_state.json"))
    parser.add_argument("--elo-k", type=float, default=48.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_inference_state(args.clean_dir, args.output, elo_k=args.elo_k)
    print(f"Saved inference state: {args.output}")


if __name__ == "__main__":
    main()
