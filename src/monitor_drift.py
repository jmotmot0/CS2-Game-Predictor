"""Monitor feature-distribution and missingness drift between time windows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.modeling import (
        MODEL_FEATURES,
        TRAIN_END,
        VALIDATION_END,
        load_feature_dataset,
        update_artifact_manifest,
        write_json,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from modeling import (  # type: ignore[no-redef]
        MODEL_FEATURES,
        TRAIN_END,
        VALIDATION_END,
        load_feature_dataset,
        update_artifact_manifest,
        write_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare reference and recent feature windows.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features_dataset.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--reference-start",
        default="",
        help="Reference-window start; defaults to 180 days before --reference-end.",
    )
    parser.add_argument("--reference-end", default=TRAIN_END)
    parser.add_argument("--current-start", default=VALIDATION_END)
    parser.add_argument("--current-end", default="")
    parser.add_argument("--psi-warning", type=float, default=0.10)
    parser.add_argument("--psi-critical", type=float, default=0.25)
    parser.add_argument("--missing-shift-warning", type=float, default=0.10)
    return parser.parse_args()


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(float)
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy(float)
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if len(reference_values) < bins or len(current_values) == 0:
        return float("nan")
    quantiles = np.quantile(reference_values, np.linspace(0.0, 1.0, bins + 1))
    inner_edges = np.unique(quantiles[1:-1])
    edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))
    if len(edges) < 3:
        same_constant = np.allclose(reference_values, reference_values[0]) and np.allclose(
            current_values, reference_values[0]
        )
        return 0.0 if same_constant else float("nan")
    reference_share = np.histogram(reference_values, bins=edges)[0].astype(float)
    current_share = np.histogram(current_values, bins=edges)[0].astype(float)
    reference_share = np.clip(reference_share / reference_share.sum(), epsilon, None)
    current_share = np.clip(current_share / current_share.sum(), epsilon, None)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def main() -> None:
    args = parse_args()
    if not 0 < args.psi_warning < args.psi_critical:
        raise ValueError("PSI thresholds must satisfy 0 < warning < critical")
    if not 0 < args.missing_shift_warning <= 1:
        raise ValueError("--missing-shift-warning must be in (0, 1]")

    frame = load_feature_dataset(args.features)
    reference_end = pd.Timestamp(args.reference_end, tz="UTC")
    reference_start = (
        pd.Timestamp(args.reference_start, tz="UTC")
        if args.reference_start
        else reference_end - pd.Timedelta(days=180)
    )
    current_start = pd.Timestamp(args.current_start, tz="UTC")
    current_end = (
        pd.Timestamp(args.current_end, tz="UTC") if args.current_end else None
    )
    timestamps = frame["match_datetime_utc"]
    if reference_start >= reference_end:
        raise ValueError("Reference start must be earlier than reference end")
    reference = frame[(timestamps >= reference_start) & (timestamps < reference_end)]
    current_mask = timestamps >= current_start
    if current_end is not None:
        current_mask &= timestamps < current_end
    current = frame[current_mask]
    if reference.empty or current.empty:
        raise ValueError(
            f"Drift windows must be non-empty: reference={len(reference)}, current={len(current)}"
        )

    rows: list[dict[str, object]] = []
    critical_features: list[str] = []
    warning_features: list[str] = []
    for feature in MODEL_FEATURES:
        psi = population_stability_index(reference[feature], current[feature])
        reference_missing = float(reference[feature].isna().mean())
        current_missing = float(current[feature].isna().mean())
        missing_shift = current_missing - reference_missing
        level = "stable"
        if np.isfinite(psi) and psi >= args.psi_critical:
            level = "critical"
            critical_features.append(feature)
        elif (
            np.isfinite(psi) and psi >= args.psi_warning
        ) or abs(missing_shift) >= args.missing_shift_warning:
            level = "warning"
            warning_features.append(feature)
        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "reference_missing_rate": reference_missing,
                "current_missing_rate": current_missing,
                "missing_rate_change": missing_shift,
                "level": level,
            }
        )

    rows.sort(
        key=lambda row: (
            0 if row["level"] == "critical" else 1 if row["level"] == "warning" else 2,
            -(float(row["psi"]) if np.isfinite(row["psi"]) else -1.0),
        )
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "critical" if critical_features else "warning" if warning_features else "stable",
        "interpretation": (
            "PSI is a monitoring signal, not evidence of target leakage or model failure. "
            "Critical features should trigger review on newly observed matches."
        ),
        "windows": {
            "reference": {
                "start": reference_start.isoformat(),
                "end_exclusive": reference_end.isoformat(),
                "rows": len(reference),
                "date_min": reference["match_datetime_utc"].min(),
                "date_max": reference["match_datetime_utc"].max(),
            },
            "current": {
                "start": current_start.isoformat(),
                "end_exclusive": current_end.isoformat() if current_end is not None else None,
                "rows": len(current),
                "date_min": current["match_datetime_utc"].min(),
                "date_max": current["match_datetime_utc"].max(),
            },
        },
        "thresholds": {
            "psi_warning": args.psi_warning,
            "psi_critical": args.psi_critical,
            "absolute_missing_rate_change_warning": args.missing_shift_warning,
        },
        "summary": {
            "critical_feature_count": len(critical_features),
            "warning_feature_count": len(warning_features),
            "stable_feature_count": len(MODEL_FEATURES) - len(critical_features) - len(warning_features),
            "critical_features": critical_features,
            "warning_features": warning_features,
        },
        "features": rows,
    }
    output_path = args.output_dir / "drift_report.json"
    write_json(output_path, report)
    update_artifact_manifest(args.output_dir, [output_path.name])
    print(
        f"Drift status: {report['status']}; critical={len(critical_features)}, "
        f"warning={len(warning_features)}"
    )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
