"""Command-line entry point for the leakage-safe feature dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # Supports both `python -m src.build_features` and direct script execution.
    from src.feature_engineering import build_feature_dataset_from_dir
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI smoke tests
    from feature_engineering import build_feature_dataset_from_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build point-in-time CS2 features without using current-match outcomes."
    )
    parser.add_argument("--clean-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--min-player-history-maps", type=int, default=5)
    parser.add_argument("--elo-k", type=float, default=48.0)
    parser.add_argument("--elo-base", type=float, default=1500.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = build_feature_dataset_from_dir(
        args.clean_dir,
        elo_k=args.elo_k,
        elo_base=args.elo_base,
        min_player_history_maps=args.min_player_history_maps,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_csv, index=False)
    repaired = int(features.get("datetime_repaired", 0).sum())
    print(f"Saved feature dataset: {args.output_csv}")
    print(f"Rows: {len(features):,}")
    print(f"Columns: {len(features.columns):,}")
    print(f"Repaired inconsistent timestamps: {repaired:,}")


if __name__ == "__main__":
    main()
