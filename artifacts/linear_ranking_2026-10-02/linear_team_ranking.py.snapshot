"""Train a genuine linear own-team PairLogit ranker on the frozen 46 inputs.

The shared scoring function is s(x) = w.T @ z(x).  Median imputation and
standardisation are fitted to training *team objects*, before forming paired
differences.  LogisticRegression without an intercept minimises the pairwise
logistic loss on z(A)-z(B); its class predictions are never used as team scores.
The six shared match-context inputs cancel in the linear score difference.
This module does not modify the frozen CatBoost models or their inference CLI.
"""
from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.modeling import (
    chronological_masks, library_versions, metric_row, sha256_file, write_json,
)
from src.team_ranking import (
    EXTENDED_TEAM_FEATURES, SHARED_FEATURES, team_ranking_decision,
    team_ranking_probability, team_ranking_scores, team_rows,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN_DIR = ROOT / "artifacts/supervisor_revision_2026-09-30"
DEFAULT_OUTPUT = ROOT / "artifacts/linear_ranking_2026-10-02"
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)


def binary_target(target, size: int) -> np.ndarray:
    values = np.asarray(target)
    if values.ndim != 1 or len(values) != size or not size or not np.isin(values, [0, 1]).all():
        raise ValueError("One binary outcome per nonempty match frame is required")
    return values.astype(int)


def pair_log_loss(target, margin) -> np.ndarray:
    """Unclipped per-match PairLogit = BCE(sigmoid(margin)); overflow safe."""
    values = np.asarray(margin, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("One finite score difference per match is required")
    y = binary_target(target, len(values))
    return np.logaddexp(0.0, -(2 * y - 1) * values)


class LinearTeamRanker:
    """A single linear score for either own-team feature vector, not a classifier label."""

    def __init__(self, feature_names=None, *, C=1.0, max_iter=2000, tol=1e-8):
        self.feature_names = list(EXTENDED_TEAM_FEATURES if feature_names is None else feature_names)
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        if not np.isfinite(self.C) or self.C <= 0 or self.max_iter < 1 or self.tol <= 0:
            raise ValueError("C, max_iter and tol must be positive")

    def fit(self, frame: pd.DataFrame, target) -> "LinearTeamRanker":
        y = binary_target(target, len(frame))
        if np.unique(y).size != 2:
            raise ValueError("Training outcomes must include both classes")
        own = team_rows(frame, self.feature_names)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        filled = self.imputer_.fit_transform(own)
        self.scaler_ = StandardScaler()
        transformed = self.scaler_.fit_transform(filled)
        differences = transformed[0::2] - transformed[1::2]
        self.estimator_ = LogisticRegression(
            C=self.C, fit_intercept=False, solver="lbfgs", max_iter=self.max_iter,
            tol=self.tol, random_state=42,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.estimator_.fit(differences, y)
        self.convergence_warnings_ = [
            str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
        self.n_iter_ = int(np.max(self.estimator_.n_iter_))
        if self.convergence_warnings_ or self.n_iter_ >= self.max_iter:
            raise RuntimeError("Linear ranker did not converge; do not publish this fit")
        self.training_match_count_ = len(frame)
        self.training_team_row_count_ = len(own)
        self.all_missing_training_features_ = [
            name for name in self.feature_names if own[name].isna().all()
        ]
        self.zero_pair_variance_features_ = [
            name for name, column in zip(self.feature_names, differences.T) if np.all(column == 0)
        ]
        return self

    def transform_team_rows(self, rows: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise ValueError("Fit the linear ranker before prediction")
        ordered = rows.loc[:, self.feature_names].copy()
        ordered = ordered.replace([np.inf, -np.inf], np.nan)
        return self.scaler_.transform(self.imputer_.transform(ordered))

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """Return raw scores for own-team objects, never 0/1 class labels."""
        transformed = self.transform_team_rows(rows)
        return self.estimator_.decision_function(transformed)

    def pair_differences(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.transform_team_rows(team_rows(frame, self.feature_names))
        return transformed[0::2] - transformed[1::2]

    def scores(self, frame: pd.DataFrame) -> np.ndarray:
        return team_ranking_scores(self, frame, self.feature_names)

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        return team_ranking_probability(self, frame, self.feature_names)

    def decision(self, frame: pd.DataFrame):
        return team_ranking_decision(self, frame, self.feature_names)


def evaluate(model: LinearTeamRanker, frame: pd.DataFrame, target):
    y = binary_target(target, len(frame))
    scores = model.scores(frame)
    margin = scores[:, 0] - scores[:, 1]
    probability = expit(margin)
    choice, ties = model.decision(frame)
    metrics = metric_row(y, probability)
    metrics.update(
        accuracy=float(np.mean(choice == y)),
        pair_log_loss=float(np.mean(pair_log_loss(y, margin))),
        exact_score_ties=int(ties.sum()),
        n=len(frame),
    )
    return probability, scores, choice, ties, metrics


def verify_frozen(directory: Path) -> tuple[str, dict]:
    manifest_path = directory / "results_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Frozen research file changed: {name}")
    return sha256_file(manifest_path), manifest


def select_regularization(frame, target, masks, *, feature_names=None, grid=DEFAULT_C_GRID):
    """Fit on train only; choose C by validation PairLogit, never test quality."""
    candidates = tuple(float(value) for value in grid)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("C grid must be nonempty and unique")
    y = binary_target(target, len(frame))
    fits = []
    records = []
    for C in candidates:
        model = LinearTeamRanker(feature_names, C=C)
        model.fit(frame.loc[masks["train"]], y[masks["train"]])
        *_, metrics = evaluate(model, frame.loc[masks["validation"]], y[masks["validation"]])
        records.append({"C": C, "n_iter": model.n_iter_, "converged": True, **metrics})
        fits.append((metrics["pair_log_loss"], C, model))
    _, selected_C, selected_model = min(fits, key=lambda item: (item[0], item[1]))
    return selected_model, records, selected_C


def run_experiment(frozen_dir=DEFAULT_FROZEN_DIR, output=DEFAULT_OUTPUT, *, grid=DEFAULT_C_GRID):
    """Write an independent reproducible experiment; never overwrite completed outputs."""
    import joblib

    frozen_dir = Path(frozen_dir).resolve()
    output = Path(output).resolve()
    if output == frozen_dir or frozen_dir in output.parents or output in frozen_dir.parents:
        raise ValueError("Output must be separate from the frozen research directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Completed or partial output exists; use a new directory")
    frozen_hash, frozen_manifest = verify_frozen(frozen_dir)
    protocol = json.loads((frozen_dir / "protocol.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(frozen_dir / "extended_features.csv", low_memory=False)
    frame["match_datetime_utc"] = pd.to_datetime(frame.match_datetime_utc, utc=True, errors="raise")
    if frame.match_id.duplicated().any() or frame.match_datetime_utc.isna().any():
        raise ValueError("Frozen data IDs/timestamps must be valid and unique")
    if frame.sort_values(["match_datetime_utc", "match_id"]).match_id.tolist() != frame.match_id.tolist():
        raise ValueError("Frozen data order is not chronological")
    masks = dict(zip(["train", "validation", "test"], chronological_masks(
        frame, train_end=protocol["training_end_exclusive"],
        validation_end=protocol["validation_end_exclusive"],
    )))
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    if counts != protocol["split_counts"]:
        raise ValueError("The split must exactly match the frozen protocol")
    y = binary_target(frame.team1_win.to_numpy(), len(frame))
    model, candidates, selected_C = select_regularization(frame, y, masks, grid=grid)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates).to_csv(output / "validation_c_grid.csv", index=False)
    metrics = []
    for split in ("validation", "test"):
        part = frame.loc[masks[split]]
        p, scores, decision, ties, measured = evaluate(model, part, y[masks[split]])
        table = part.loc[:, ["match_id", "match_datetime_utc", "team1_id", "team2_id", "team1_win"]].copy()
        table["linear_rank46"] = p
        table["score_team1"] = scores[:, 0]
        table["score_team2"] = scores[:, 1]
        table["prediction"] = decision
        table["exact_score_tie"] = ties
        table.to_csv(output / f"{split}_predictions.csv", index=False)
        metrics.append({"split": split, "model": "linear_rank46", **measured})
        old_predictions = pd.read_csv(frozen_dir / f"{split}_predictions.csv")
        np.testing.assert_array_equal(table.match_id, old_predictions.match_id)
        np.testing.assert_array_equal(table.team1_win, old_predictions.team1_win)
    pd.DataFrame(metrics).to_csv(output / "model_metrics.csv", index=False)
    comparison = pd.read_csv(frozen_dir / "comparison_metrics.csv")
    comparison = comparison[comparison.model.isin(["elo", "ranking_40_s42", "ranking_46_s42"])].copy()
    comparison["source"] = "frozen_supervisor_revision"
    current = pd.DataFrame(metrics)
    current["source"] = "new_linear_own_team_pairlogit"
    pd.concat([comparison, current], ignore_index=True).to_csv(output / "comparison_metrics.csv", index=False)
    parameter_table = pd.DataFrame({
        "feature": model.feature_names,
        "median_imputation": model.imputer_.statistics_,
        "scaler_mean": model.scaler_.mean_,
        "scaler_scale": model.scaler_.scale_,
        "standardized_weight": model.estimator_.coef_[0],
        "raw_unit_weight": model.estimator_.coef_[0] / model.scaler_.scale_,
        "shared_context": [name in SHARED_FEATURES for name in model.feature_names],
        "zero_training_pair_variance": [name in model.zero_pair_variance_features_ for name in model.feature_names],
    })
    parameter_table.to_csv(output / "parameters.csv", index=False)
    joblib.dump(model, output / "linear_ranker.joblib")
    write_json(output / "feature_schema.json", {
        "feature_order": model.feature_names, "feature_count": len(model.feature_names),
        "shared_cancelled_features": list(SHARED_FEATURES),
        "potentially_effective_own_features": len(model.feature_names) - len(SHARED_FEATURES),
        "all_missing_training_features": model.all_missing_training_features_,
        "zero_training_pair_variance_features": model.zero_pair_variance_features_,
    })
    write_json(output / "selection.json", {
        "selected_C": selected_C, "C_grid": list(grid),
        "criterion": "Minimum validation mean unclipped PairLogit; ties choose smaller C",
        "test_used_for_selection": False,
        "refit_train_plus_validation": False,
        "selected_fit_iterations": model.n_iter_, "all_grid_fits_converged": True,
    })
    write_json(output / "protocol.json", {
        "kind": "linear_own_team_pairwise_ranking",
        "score": "s(x)=w^T z(x), one common coefficient vector for both teams",
        "probability": "sigmoid(s(A)-s(B)); no Platt calibration",
        "loss": "log(1+exp(-(2*y-1)*(s(A)-s(B)))) with sklearn L2 regularization",
        "estimator": "LogisticRegression(solver=lbfgs, fit_intercept=False, max_iter=2000, tol=1e-8)",
        "preprocessing": "Median imputation then StandardScaler, both fit only on training own-team rows; no missing indicators. All-empty training inputs retained with zero imputation.",
        "preprocessing_fit_rows": model.training_team_row_count_,
        "training_match_count": model.training_match_count_,
        "pair_construction": "Transform own A/B rows first, then z(A)-z(B); no mirrored augmentation",
        "context_limitation": "Six shared context fields cancel in the linear score difference. The 46-input schema has 40 potentially active own-team inputs. No context-by-team interactions are introduced.",
        "winner_rule": "Higher raw team score; exact ties choose lower stable team ID",
        "training_end_exclusive": protocol["training_end_exclusive"],
        "validation_end_exclusive": protocol["validation_end_exclusive"],
        "split_counts": counts,
        "test_status": protocol["test_status"],
        "availability": protocol["availability"],
        "previous_logistic_difference": "Old LR40 used difference-column median imputation with missingness indicators, scaling of mirrored difference rows, a fitted intercept and probability symmetrization. This is a newly fitted own46 linear PairLogit pipeline, not a relabelled old model. Both use logistic loss; a performance difference does not isolate the loss alone.",
        "source_manifest_sha256": frozen_hash,
        "input_sha256": sha256_file(frozen_dir / "extended_features.csv"),
        "library_versions": library_versions(),
        "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
            Path(__file__), ROOT / "src/team_ranking.py", ROOT / "src/modeling.py",
            ROOT / "tests/test_linear_team_ranking.py",
        )},
    })
    shutil.copyfile(Path(__file__), output / "linear_team_ranking.py.snapshot")
    final_hash, final_manifest = verify_frozen(frozen_dir)
    if final_hash != frozen_hash or final_manifest != frozen_manifest:
        raise RuntimeError("Frozen research artifacts changed during the experiment")
    write_json(output / "integrity.json", {
        "frozen_files_verified_before_and_after": len(frozen_manifest["files"]),
        "source_manifest_sha256_before": frozen_hash,
        "source_manifest_sha256_after": final_hash,
        "frozen_files_unchanged": True,
        "validation_and_test_ids_equal_frozen_predictions": True,
    })
    write_json(output / "results_manifest.json", {
        "files": {path.name: sha256_file(path) for path in sorted(output.iterdir())
                  if path.is_file() and path.name != "results_manifest.json"},
    })
    print(json.dumps({"output": str(output), "selected_C": selected_C, "metrics": metrics}, ensure_ascii=False, indent=2))
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
    args = parser.parse_args()
    run_experiment(args.frozen_dir, args.output, grid=args.c_grid)


if __name__ == "__main__":
    # Import the canonical module so joblib records an importable class path,
    # not __main__.LinearTeamRanker when invoked via python -m.
    from src.linear_team_ranking import main as canonical_main

    canonical_main()
