"""Retrospective, nested chronological feature selection; never replaces production.

Run: python -m src.feature_selection_experiment
The protocol is written before fitting. The already used 2026 holdout is a
diagnostic, not a new independent confirmation. All models are kept in memory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.modeling import (
    MODEL_FEATURES, augment_team_swap, catboost_classifier, elo_probabilities,
    load_feature_dataset, metric_row, sha256_file, symmetrized_probability, tune_elo,
)

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (10, 20, 30, 40)
SEEDS = (42, 43, 44)
FOLDS = ("2025-04-01", "2025-07-01", "2025-10-01")
METHODS = ("PVC", "PFI_LogLoss")


def feature_bundles() -> list[list[str]]:
    """Never split mutually dependent one-hot encodings."""
    grouped = [["bo1", "bo3", "bo5"], ["is_lan", "is_online"]]
    grouped.extend([[name] for name in MODEL_FEATURES if name not in sum(grouped, [])])
    return grouped


def split_masks(frame: pd.DataFrame, start: str) -> dict[str, np.ndarray]:
    boundary = pd.Timestamp(start, tz="UTC")
    times = frame["match_datetime_utc"]
    stop_end = boundary - pd.DateOffset(months=3)
    fit_end = boundary - pd.DateOffset(months=6)
    masks = {
        "fit": (times < fit_end).to_numpy(),
        "early_stop": ((times >= fit_end) & (times < stop_end)).to_numpy(),
        "ranking": ((times >= stop_end) & (times < boundary)).to_numpy(),
        "evaluation": ((times >= boundary) & (times < boundary + pd.DateOffset(months=3))).to_numpy(),
    }
    if min(int(mask.sum()) for mask in masks.values()) == 0:
        raise ValueError(f"Empty temporal partition for {start}")
    return masks


def elo_features(frame: pd.DataFrame, k: float) -> pd.DataFrame:
    """Recompute BOTH Elo-dependent features with the fold-local K.

    Opponents' pre-match ratings are appended only after the entire equal-time
    batch has been read, matching compute_team_history_features semantics.
    """
    _, first, second = elo_probabilities(frame, k=k)
    opponents: dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
    team1 = frame["team1_id"].astype(int).to_numpy()
    team2 = frame["team2_id"].astype(int).to_numpy()
    timestamps = frame["match_datetime_utc"].astype("int64").to_numpy()
    opponent_diff = np.empty(len(frame))
    start = 0
    while start < len(frame):
        end = start + 1
        while end < len(frame) and timestamps[end] == timestamps[start]:
            end += 1
        for i in range(start, end):
            left, right = opponents[team1[i]], opponents[team2[i]]
            opponent_diff[i] = (np.mean(left) if left else np.nan) - (np.mean(right) if right else np.nan)
        for i in range(start, end):
            opponents[team1[i]].append(second[i])
            opponents[team2[i]].append(first[i])
        start = end
    return pd.DataFrame({"diff_elo_pre": first - second, "diff_avg_opp_elo_last_10": opponent_diff}, index=frame.index)


def point_losses(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def monthly_permutation(times: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Permute rows only inside the same calendar month, never across windows."""
    labels = times.dt.strftime("%Y-%m").to_numpy()
    permutation = np.arange(len(times))
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        permutation[indices] = rng.permutation(indices)
    return permutation


def rank_features(model, x: pd.DataFrame, y: np.ndarray, times: pd.Series,
                  seed: int, repeats: int) -> list[dict]:
    bundles = feature_bundles()
    pvc = dict(zip(MODEL_FEATURES, model.get_feature_importance()))
    baseline = float(point_losses(y, symmetrized_probability(model, x)).mean())
    rng = np.random.default_rng(seed)
    # Common permutation draws reduce Monte Carlo noise in comparisons.
    permutations = [monthly_permutation(times, rng) for _ in range(repeats)]
    rows = []
    for group in bundles:
        changes = []
        for permutation in permutations:
            shuffled = x.copy()
            shuffled.loc[:, group] = x.iloc[permutation][group].to_numpy()
            loss = point_losses(y, symmetrized_probability(model, shuffled)).mean()
            changes.append(float(loss - baseline))
        for method, score, std in [
            ("PVC", sum(float(pvc[name]) for name in group), 0.0),
            ("PFI_LogLoss", float(np.mean(changes)), float(np.std(changes, ddof=1))),
        ]:
            rows.append({"method": method, "bundle": "|".join(group), "features": group,
                         "score": score, "score_per_feature": score / len(group),
                         "permutation_sd": std, "ranking_base_logloss": baseline})
    return rows


def select_features(ranking: list[dict], method: str, budget: int) -> list[str]:
    if not 1 <= budget <= len(MODEL_FEATURES):
        raise ValueError("Invalid feature budget")
    candidates = sorted((r for r in ranking if r["method"] == method),
                        key=lambda r: (-r["score_per_feature"], r["bundle"]))
    selected: set[str] = set()
    for row in candidates:
        if len(selected) + len(row["features"]) <= budget:
            selected.update(row["features"])
    if len(selected) != budget:
        raise ValueError(f"Cannot fill budget {budget} without splitting an encoding")
    return [feature for feature in MODEL_FEATURES if feature in selected]


def fit_model(x: pd.DataFrame, y: np.ndarray, masks: dict, features: list[str], seed: int):
    training, target = augment_team_swap(x.loc[masks["fit"], features], y[masks["fit"]])
    stopping, stop_target = augment_team_swap(x.loc[masks["early_stop"], features], y[masks["early_stop"]])
    model = catboost_classifier(iterations=1000, seed=seed)
    model.set_params(thread_count=4)
    model.fit(training, target, eval_set=(stopping, stop_target), early_stopping_rounds=100)
    return model


def paired_weekly_interval(differences: np.ndarray, times: pd.Series,
                           seed: int = 924, draws: int = 4000) -> tuple[float, float]:
    """Paired weekly-block bootstrap; not a guarantee against all dependence."""
    day = times.dt.tz_convert(None).dt.normalize()
    weeks = day - pd.to_timedelta(day.dt.dayofweek, unit="D")
    grouped = pd.DataFrame({"week": weeks.to_numpy(), "delta": differences}).groupby("week")["delta"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(grouped), size=(draws, len(grouped)))
    means = grouped["sum"].to_numpy()[sample].sum(axis=1) / grouped["count"].to_numpy()[sample].sum(axis=1)
    return tuple(float(value) for value in np.quantile(means, [.025, .975]))


def summarize(predictions: pd.DataFrame) -> list[dict]:
    # Average per-match losses over seeds, NOT probabilities (no seed ensemble).
    matches = predictions.groupby(["method", "budget", "match_id", "match_datetime_utc"], as_index=False).agg(loss=("loss", "mean"), correct=("correct", "mean"))
    baseline = matches[(matches.method == "PVC") & (matches.budget == 40)][["match_id", "loss"]].rename(columns={"loss": "full_loss"})
    rows = []
    for (method, budget), part in matches.groupby(["method", "budget"]):
        joined = part.merge(baseline, on="match_id", validate="one_to_one")
        delta = (joined.loss - joined.full_loss).to_numpy()
        lower, upper = paired_weekly_interval(delta, pd.to_datetime(joined.match_datetime_utc, utc=True))
        rows.append({"method": method, "budget": int(budget), "n_matches": len(part),
                     "log_loss": float(part.loss.mean()), "accuracy": float(part.correct.mean()),
                     "delta_vs_40": float(delta.mean()), "delta_ci_low": lower, "delta_ci_high": upper})
    return sorted(rows, key=lambda row: (row["log_loss"], row["budget"], row["method"]))


def protected_hashes() -> dict[str, str]:
    paths = [ROOT / "data/processed/features_dataset.csv"]
    paths += list((ROOT / "data/interim/hltv_final_clean").glob("*.csv"))
    paths += [p for p in (ROOT / "artifacts").iterdir() if p.is_file()]
    refresh = ROOT / "tmp/full_refresh_2026-09-28"
    paths += [p for p in refresh.rglob("*") if p.is_file()]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run(output: Path, repeats: int = 20) -> None:
    if repeats < 2:
        raise ValueError("At least two permutation repeats required")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    before = protected_hashes()
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_evaluation_starts": FOLDS, "seeds": SEEDS,
        "budgets": BUDGETS, "methods": METHODS, "permutation_repeats": repeats,
        "split": "fit < S-6 months; early stop [S-6,S-3); ranking [S-3,S); evaluation [S,S+3)",
        "elo": "K selected from 8,16,24,32,40,48,64 on early-stop window; both Elo inputs recomputed causally",
        "feature_bundles": feature_bundles(),
        "ranking": "descending bundle importance / number of columns; alphabetical tie-break; greedy exact column budget",
        "permutation": "joint within bundle and calendar month, same draws for all bundles, symmetrized probability",
        "fit": "CatBoost depth=6 lr=.04 l2=5 max1000 early100 threads4, mirrored rows; no calibration",
        "selection": "minimum development mean per-match loss, averaged over three seeds; ties fewer columns then method name",
        "uncertainty": "4000 paired weekly-block bootstrap draws over per-match seed-mean losses; exploratory, no multiplicity correction",
        "historical_diagnostic": "2026 Q1, already used in earlier project evaluation, not an untouched holdout; no procedure selection on it",
        "limitations": ["Feature definitions and windows designed retrospectively, not preregistered on new data",
                        "Correlated inputs can substitute for each other; PFI is model-dependent, not causal",
                        "Time of publication and match end are unavailable; same historical-data assumptions as baseline",
                        "40 inputs are an engineered candidate set, not a proven globally optimal feature set"],
        "source_code_sha256": sha256_file(Path(__file__)), "protected_sha256_before": before,
    }
    dump(output / "protocol.json", protocol)
    frame = load_feature_dataset(ROOT / "data/processed/features_dataset.csv")
    y = frame.team1_win.astype(int).to_numpy()
    metrics, rankings, selections, predictions, fold_info = [], [], [], [], []

    def run_fold(start: str, phase: str, choices: list[tuple[str, int]]) -> None:
        masks = split_masks(frame, start)
        # No evaluation target enters tuning, ranking or early stopping.
        k, _, _, _, k_scores = tune_elo(frame, masks["early_stop"])
        x = frame[MODEL_FEATURES].copy()
        recalculated = elo_features(frame, k)
        x[recalculated.columns] = recalculated
        fold_info.append({"phase": phase, "start": start, "elo_k": k, "elo_candidates": k_scores,
                          "counts": {key: int(mask.sum()) for key, mask in masks.items()}})
        for seed in SEEDS:
            full = fit_model(x, y, masks, MODEL_FEATURES, seed)
            rank = rank_features(full, x.loc[masks["ranking"]].reset_index(drop=True), y[masks["ranking"]],
                                 frame.loc[masks["ranking"], "match_datetime_utc"].reset_index(drop=True), seed, repeats)
            rankings.extend([{**r, "phase": phase, "fold": start, "seed": seed} for r in rank])
            model_cache = {tuple(MODEL_FEATURES): full}
            for method, budget in choices:
                features = select_features(rank, method, budget)
                key = tuple(features)
                if key not in model_cache:
                    model_cache[key] = fit_model(x, y, masks, features, seed)
                model = model_cache[key]
                p = symmetrized_probability(model, x.loc[masks["evaluation"], features])
                tags = {"phase": phase, "fold": start, "seed": seed, "method": method, "budget": budget}
                metrics.append({**tags, "trees": model.tree_count_, "n": len(p), **metric_row(y[masks["evaluation"]], p)})
                selections.append({**tags, "features": features})
                part = frame.loc[masks["evaluation"], ["match_id", "match_datetime_utc", "team1_win"]].copy()
                for name, value in tags.items():
                    part[name] = value
                part["probability"] = p
                part["loss"] = point_losses(part.team1_win.to_numpy(), p)
                part["correct"] = ((p >= .5) == part.team1_win.to_numpy()).astype(int)
                predictions.append(part)
            pd.DataFrame(metrics).to_csv(output / "fold_metrics.csv", index=False)
            print(f"{phase} {start} seed={seed} K={k:g} done; {time.monotonic()-started:.1f}s", flush=True)

    choices = [(method, budget) for method in METHODS for budget in BUDGETS]
    for start in FOLDS:
        run_fold(start, "development", choices)
    development = summarize(pd.concat(predictions, ignore_index=True))
    winner = development[0]
    # Frozen before even evaluating the historical 2026 diagnostic.
    dump(output / "selection_decision.json", {"frozen_utc": datetime.now(timezone.utc).isoformat(), "winner": winner, "development": development})
    selected = (winner["method"], winner["budget"])
    diagnostic_choices = list(dict.fromkeys([("PVC", 40), ("PFI_LogLoss", 40), selected]))
    run_fold("2026-01-01", "historical_diagnostic", diagnostic_choices)
    all_predictions = pd.concat(predictions, ignore_index=True)
    diagnostic = summarize(all_predictions[all_predictions.phase == "historical_diagnostic"])
    pd.DataFrame(rankings).drop(columns="features").to_csv(output / "importance.csv", index=False)
    dump(output / "selected_features.json", selections)
    all_predictions.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(development).to_csv(output / "development_summary.csv", index=False)
    pd.DataFrame(diagnostic).to_csv(output / "historical_diagnostic.csv", index=False)
    after = protected_hashes()
    changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
    summary = {"winner": winner, "development": development, "historical_diagnostic": diagnostic,
               "folds": fold_info, "elapsed_seconds": time.monotonic()-started,
               "protected_files_checked": len(before), "protected_files_changed": changed,
               "baseline_model_replaced": False}
    dump(output / "summary.json", summary)
    if changed:
        raise RuntimeError(f"Protected files changed during experiment: {changed}")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/feature_selection_2026-09-29")
    parser.add_argument("--permutation-repeats", type=int, default=20)
    args = parser.parse_args()
    run(args.output, args.permutation_repeats)
