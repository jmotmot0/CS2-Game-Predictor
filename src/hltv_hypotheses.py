"""Research HLTV signals, not feature-importance rankings.

Fixed hypotheses, observed win curves, strength/coverage adjustment and
chronological forecasting. Includes an analytic Newton solver for penalized
logistic regression. Does not modify production data, models or refresh files.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from src.modeling import elo_probabilities, load_feature_dataset, sha256_file

ROOT = Path(__file__).resolve().parents[1]
STARTS = ("2025-04-01", "2025-07-01", "2025-10-01", "2026-01-01")
LAMBDA = .001
QUALITY = ["player_coverage", "player_history", "team_history", "map_history"]


@dataclass(frozen=True)
class Hypothesis:
    key: str
    title: str
    question: str
    feature: str
    scale: float
    unit: str
    eligibility: str


HYPOTHESES = [
    Hypothesis("form", "Недавняя форма", "Помогает ли доля побед за последние 10 матчей сверх силы команды?",
               "diff_winrate_last_10", .1, "+10 п.п. доли побед", "history"),
    Hypothesis("opponents", "Сила прошлых соперников", "Информативен ли уровень соперников последних 10 матчей?",
               "diff_avg_opp_elo_last_10", 100, "+100 Elo соперников", "history"),
    Hypothesis("rating", "Рейтинг игроков", "Добавляет ли средний прошлый Rating игроков информацию о победе?",
               "diff_lineup_player_rating_mean", .1, "+0,1 Rating", "players"),
    Hypothesis("adr", "Урон за раунд", "Помогает ли прошлый ADR игроков, если учесть силу команды?",
               "diff_lineup_player_adr_mean", 5, "+5 ADR", "players"),
    Hypothesis("kast", "KAST игроков", "Добавляет ли прошлый KAST информацию сверх силы и полноты истории?",
               "diff_lineup_player_kast_mean", 5, "+5 п.п. KAST", "players"),
    Hypothesis("opening", "Первые убийства и смерти", "Полезен ли средний прошлый баланс первых убийств и смертей?",
               "diff_lineup_player_opening_diff_mean", .5, "+0,5 баланса за карту", "players"),
    Hypothesis("roster", "Стабильность состава", "Связано ли сохранение игроков с победой сверх силы команды?",
               "diff_roster_overlap_prev_ratio", .2, "+20 п.п. сохранённых игроков", "roster"),
    Hypothesis("maps", "Победы на картах серии", "Даёт ли история побед на известных из veto картах дополнительный сигнал?",
               "diff_avg_map_wr_before", .1, "+10 п.п. побед на картах", "maps"),
    Hypothesis("ct", "Раунды за CT", "Полезна ли историческая доля выигранных CT-раундов на картах серии?",
               "diff_avg_map_ct_wr_before", .1, "+10 п.п. CT-раундов", "maps"),
    Hypothesis("t", "Раунды за T", "Полезна ли историческая доля выигранных T-раундов на картах серии?",
               "diff_avg_map_t_wr_before", .1, "+10 п.п. T-раундов", "maps"),
    Hypothesis("history", "Объём истории игроков", "Предсказывает ли объём доступной истории сверх силы и покрытия состава?",
               "player_history", 1, "удвоение отношения (1 + карт)", "players"),
    Hypothesis("coverage", "Полнота истории состава", "Предсказывает ли само наличие статистики игроков?",
               "player_coverage", .2, "+20 п.п. покрытия состава", "roster"),
]


def objective_gradient_hessian(beta: np.ndarray, x: np.ndarray, y: np.ndarray,
                               ridge: float = LAMBDA) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean Bernoulli NLL + lambda/2 ||beta||^2; no intercept."""
    z = x @ beta
    p = expit(z)
    loss = np.mean(np.logaddexp(0, z) - y * z) + ridge * (beta @ beta) / 2
    gradient = x.T @ (p - y) / len(y) + ridge * beta
    hessian = (x.T * (p * (1 - p))) @ x / len(y) + ridge * np.eye(x.shape[1])
    return float(loss), gradient, hessian


@dataclass
class LogisticFit:
    beta: np.ndarray
    scales: np.ndarray
    iterations: int
    gradient_max: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return expit((x / self.scales) @ self.beta)

    def coefficient(self, index: int) -> float:
        return float(self.beta[index] / self.scales[index])


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = LAMBDA) -> LogisticFit:
    """Damped Newton, training-only RMS scaling; odd model gives exact symmetry."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or len(x) != len(y) or len(y) == 0 or not np.isfinite(x).all():
        raise ValueError("Invalid logistic training arrays")
    if not np.isin(y, [0, 1]).all() or ridge <= 0:
        raise ValueError("Binary targets and positive ridge required")
    scales = np.maximum(np.sqrt(np.mean(x*x, axis=0)), 1e-8)
    scaled = x / scales
    beta = np.zeros(x.shape[1])
    for iteration in range(1, 101):
        loss, grad, hessian = objective_gradient_hessian(beta, scaled, y, ridge)
        norm = float(np.max(np.abs(grad)))
        if norm < 1e-8:
            return LogisticFit(beta, scales, iteration, norm)
        direction = np.linalg.solve(hessian, grad)
        step = 1.0
        while step >= 2**-24:
            candidate = beta - step * direction
            z = scaled @ candidate
            value = np.mean(np.logaddexp(0, z) - y*z) + ridge*(candidate @ candidate)/2
            if value <= loss - 1e-4*step*(grad @ direction):
                beta = candidate
                break
            step /= 2
        else:
            raise RuntimeError("Newton line search failed")
    raise RuntimeError("Logistic fit did not converge")


def losses(y, p) -> np.ndarray:
    probability = np.clip(np.asarray(p), 1e-6, 1-1e-6)
    return -(np.asarray(y)*np.log(probability) + (1-np.asarray(y))*np.log1p(-probability))


def elo_history(frame: pd.DataFrame, k: float = 32) -> tuple[np.ndarray, np.ndarray]:
    """Fixed K, chronological equal-time updates, including opponent snapshots."""
    _, r1, r2 = elo_probabilities(frame, k=k)
    histories = defaultdict(lambda: deque(maxlen=10))
    a, b = frame.team1_id.to_numpy(), frame.team2_id.to_numpy()
    times = frame.match_datetime_utc.astype("int64").to_numpy()
    opponent = np.empty(len(frame))
    start = 0
    while start < len(frame):
        end = start + 1
        while end < len(frame) and times[end] == times[start]:
            end += 1
        for i in range(start, end):
            left, right = histories[a[i]], histories[b[i]]
            opponent[i] = (np.mean(left) if left else np.nan) - (np.mean(right) if right else np.nan)
        for i in range(start, end):
            histories[a[i]].append(r2[i])
            histories[b[i]].append(r1[i])
        start = end
    return r1-r2, opponent


def prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    frame = frame.copy()
    elo, opponent = elo_history(frame)
    frame["diff_elo_pre"] = elo
    frame["diff_avg_opp_elo_last_10"] = opponent
    # Relabel focal team independently of winner / covariates. One row per match.
    hashed = pd.util.hash_array(frame.match_id.astype("int64").to_numpy())
    signs = np.where(hashed % 2 == 0, 1., -1.)
    y = np.where(signs == 1, frame.team1_win, 1-frame.team1_win).astype(int)
    design = pd.DataFrame(index=frame.index)
    design["elo"] = elo / 400
    a, b = frame.team1_team_rank, frame.team2_team_rank
    valid_a, valid_b = a.gt(0) & a.notna(), b.gt(0) & b.notna()
    known = valid_a & valid_b
    design["rank"] = np.where(known, np.log(b.where(valid_b, 1) / a.where(valid_a, 1)), 0)
    design["rank_missing_difference"] = valid_a.astype(float)-valid_b.astype(float)
    design["elo_rank_unknown"] = design.elo * (~known)
    for context in ["bo3", "bo5", "is_lan"]:
        for strength in ["elo", "rank"]:
            design[f"{strength}_{context}"] = design[strength]*frame[context]
    design["player_coverage"] = frame.diff_lineup_history_coverage
    for name, source in [("player_history", "lineup_player_maps_played_mean"),
                         ("team_history", "matches_before"), ("map_history", "avg_map_count_before")]:
        design[name] = np.log2((1+frame[f"team1_{source}"].fillna(0)) /
                               (1+frame[f"team2_{source}"].fillna(0)))
    for hypothesis in HYPOTHESES:
        if hypothesis.feature not in design:
            design[hypothesis.feature] = frame[hypothesis.feature]
    design = design.mul(signs, axis=0)
    frame["focal_sign"] = signs
    frame["target"] = y
    return frame, design, y


def eligible(frame: pd.DataFrame, design: pd.DataFrame, h: Hypothesis) -> np.ndarray:
    mask = frame.team1_matches_before.ge(10) & frame.team2_matches_before.ge(10)
    mask &= np.isfinite(design[h.feature])
    mask &= np.isfinite(design[QUALITY]).all(axis=1)
    if h.eligibility in ("players", "roster"):
        mask &= frame.team1_roster_size.eq(5) & frame.team2_roster_size.eq(5)
    if h.eligibility == "players":
        mask &= frame.team1_lineup_history_coverage.ge(.8) & frame.team2_lineup_history_coverage.ge(.8)
    if h.eligibility == "maps":
        mask &= frame.bo3.eq(1) | frame.bo5.eq(1)
        mask &= frame.team1_avg_map_count_before.ge(5) & frame.team2_avg_map_count_before.ge(5)
        mask &= frame.team1_series_maps_known.ge(2) & frame.team2_series_maps_known.ge(2)
    return mask.to_numpy()


def time_masks(times: pd.Series, start: str) -> tuple[np.ndarray, np.ndarray]:
    boundary = pd.Timestamp(start, tz="UTC")
    return (times < boundary).to_numpy(copy=True), ((times >= boundary) & (times < boundary+pd.DateOffset(months=3))).to_numpy(copy=True)


def weekly_interval(values: np.ndarray, times: pd.Series, *, alpha: float = .05,
                    draws: int = 10000, seed: int = 2951) -> tuple[float, float]:
    """Descriptive paired calendar-week bootstrap, keeping all matches in a week."""
    dates = pd.to_datetime(times, utc=True).dt.tz_localize(None).dt.normalize()
    week = dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")
    grouped = pd.DataFrame({"week": week.to_numpy(), "value": np.asarray(values)}).groupby("week").value.agg(["sum", "count"])
    if len(grouped) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), size=(draws, len(grouped)))
    estimates = grouped["sum"].to_numpy()[indices].sum(axis=1) / grouped["count"].to_numpy()[indices].sum(axis=1)
    return tuple(float(v) for v in np.quantile(estimates, [alpha/2, 1-alpha/2]))


def guarded_hashes() -> dict[str, str]:
    paths = [ROOT / "data/processed/features_dataset.csv"]
    paths += list((ROOT / "data/interim/hltv_final_clean").glob("*.csv"))
    paths += [p for p in (ROOT / "artifacts").iterdir() if p.is_file()]
    paths += [p for p in (ROOT / "tmp/full_refresh_2026-09-28").rglob("*") if p.is_file()]
    paths += [p for p in (ROOT / "artifacts/feature_selection_2026-09-29").rglob("*") if p.is_file()]
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(paths)}


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    beginning = time.monotonic()
    before = guarded_hashes()
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Which historical HLTV data add predictive information beyond team strength and data coverage?",
        "hypotheses": [asdict(h) for h in HYPOTHESES],
        "quarter_starts": STARTS, "elo_k_fixed": 32, "ridge_on_rms_scaled_features": LAMBDA,
        "training": "All eligible matches before the tested quarter. No tuning on its outcomes.",
        "eligibility": ">=10 prior matches per team and finite coverage controls. Players: five-player rosters, >=80% known players. Maps: BO3/BO5, >=5 mean prior maps and >=2 series maps with history per team.",
        "orientation": "One row per match; focal team chosen by fixed pandas match-id hash parity, independent of outcome.",
        "models": "Odd zero-intercept logistic, strength; strength+candidate; strength+coverage controls; strength+coverage+candidate; candidate alone for descriptive coefficient.",
        "strength": "Elo/400, log(rank2/rank1) when both known, rank-known signed difference, Elo x rank-unknown, Elo/rank x BO3/BO5/LAN",
        "coverage_controls": QUALITY,
        "candidate_exclusion": "Remove the candidate itself from the coverage controls before fitting its nested models.",
        "primary_endpoint": "Mean held-later-quarter loss(base adjusted) - loss(base adjusted + candidate); positive means useful.",
        "uncertainty": "10000 paired weekly bootstrap draws, 95% descriptive intervals; also 1-.05/12 intervals across 12 primary comparisons, no guarantee against longer dependence.",
        "curves": "Bin thresholds learned from eligible rows before April 2025 only; later-quarter empirical win rates and out-of-time baseline expectation.",
        "sensitivity_subgroups": ["abs(Elo difference)<=100", "all ten players have >=5 maps in this database"],
        "limits": ["Retrospective hypotheses informed by prior project experience; not a fresh independent confirmation",
                   "Rank publication snapshots and exact match end times unavailable",
                   "Linear-logit conditional associations are not causal effects and can miss nonlinearity",
                   "Coverage controls are proxies, not perfect removal of source-selection bias",
                   "No automatic deployment or selection of a new production feature set"],
        "protected_sha256": before, "source_sha256": sha256_file(Path(__file__)),
        "versions": {"python": platform.python_version(), **{lib: version(lib) for lib in ["numpy", "pandas", "scipy", "scikit-learn"]}},
    }
    dump(output / "protocol.json", protocol)
    original = load_feature_dataset(ROOT / "data/processed/features_dataset.csv")
    frame, design, y = prepare(original)
    strengths = ["elo", "rank", "rank_missing_difference", "elo_rank_unknown",
                 "elo_bo3", "rank_bo3", "elo_bo5", "rank_bo5", "elo_is_lan", "rank_is_lan"]
    predictions, coefficients, fold_metrics, bin_records = [], [], [], []
    convergence, quality_checks = [], []
    for h in HYPOTHESES:
        mask = eligible(frame, design, h)
        controls = [c for c in QUALITY if c != h.feature]
        configs = {"alone": [], "strength": strengths, "adjusted": strengths+controls}
        all_part = []
        for start in STARTS:
            train, test = time_masks(frame.match_datetime_utc, start)
            train &= mask
            test &= mask
            if train.sum() < 100 or test.sum() < 100:
                raise ValueError(f"Insufficient eligible sample {h.key} {start}: {train.sum()}/{test.sum()}")
            part = frame.loc[test, ["match_id", "match_datetime_utc", "target", "diff_elo_pre",
                                    "team1_lineup_history_coverage", "team2_lineup_history_coverage"]].copy()
            part["hypothesis"] = h.key
            part["fold"] = start
            part["value"] = design.loc[test, h.feature].to_numpy()
            for adjustment, baseline_features in configs.items():
                features = baseline_features + [h.feature]
                x = design[features].to_numpy().copy()
                x[:, -1] /= h.scale
                if not np.isfinite(x[mask]).all():
                    raise ValueError(f"Non-finite model input for {h.key}")
                model = fit_logistic(x[train], y[train])
                convergence.append(model.gradient_max)
                if adjustment != "alone":
                    base = fit_logistic(x[train, :-1], y[train])
                    convergence.append(base.gradient_max)
                    part[f"p_{adjustment}_base"] = base.predict(x[test, :-1])
                    part[f"p_{adjustment}_plus"] = model.predict(x[test])
                    delta = losses(y[test], part[f"p_{adjustment}_base"]) - losses(y[test], part[f"p_{adjustment}_plus"])
                    fold_metrics.append({"hypothesis": h.key, "fold": start, "adjustment": adjustment,
                                         "train_n": int(train.sum()), "test_n": int(test.sum()),
                                         "base_loss": float(losses(y[test], part[f"p_{adjustment}_base"]).mean()),
                                         "plus_loss": float(losses(y[test], part[f"p_{adjustment}_plus"]).mean()),
                                         "gain": float(delta.mean())})
                coefficient = model.coefficient(-1)
                coefficients.append({"hypothesis": h.key, "fold": start, "adjustment": adjustment,
                                     "coefficient": coefficient, "odds_ratio": float(np.exp(coefficient)),
                                     "unit": h.unit, "train_n": int(train.sum()), "gradient_max": model.gradient_max})
                # Independent implementation check on one representative fit per question.
                if adjustment == "adjusted" and start == STARTS[0]:
                    from sklearn.linear_model import LogisticRegression
                    reference = LogisticRegression(C=1/(int(train.sum())*LAMBDA), fit_intercept=False,
                                                   solver="lbfgs", max_iter=2000, tol=1e-10)
                    reference.fit(x[train]/model.scales, y[train])
                    error = float(np.max(np.abs(reference.predict_proba(x[test]/model.scales)[:, 1]-model.predict(x[test]))))
                    if error > 1e-5:
                        raise AssertionError(f"Independent solver mismatch {error}")
                    quality_checks.append({"hypothesis": h.key, "max_probability_error_vs_sklearn": error})
            all_part.append(part)
        part = pd.concat(all_part, ignore_index=True)
        # These dates define bins only; no outcomes or later quantiles are used.
        early = mask & (frame.match_datetime_utc < pd.Timestamp(STARTS[0], tz="UTC")).to_numpy()
        if h.key in ("roster", "coverage"):
            edges = np.array([-np.inf, -1e-9, 1e-9, np.inf])
        else:
            quantiles = np.unique(np.quantile(design.loc[early, h.feature], [.2, .4, .6, .8]))
            edges = np.r_[-np.inf, quantiles, np.inf]
        bins = pd.cut(part.value, edges, labels=False, include_lowest=True)
        for bin_id in sorted(bins.dropna().unique()):
            subset = part[bins == bin_id]
            lower, upper = weekly_interval(subset.target.to_numpy(), subset.match_datetime_utc, draws=4000)
            residual = subset.target-subset.p_adjusted_base
            residual_lo, residual_hi = weekly_interval(residual.to_numpy(), subset.match_datetime_utc, draws=4000)
            bin_records.append({"hypothesis": h.key, "bin": int(bin_id), "n": len(subset),
                                "value_mean": float(subset.value.mean()), "win_rate": float(subset.target.mean()),
                                "win_ci_low": lower, "win_ci_high": upper,
                                "strength_expected": float(subset.p_strength_base.mean()),
                                "adjusted_expected": float(subset.p_adjusted_base.mean()),
                                "plus_expected": float(subset.p_adjusted_plus.mean()),
                                "excess_wins": float(residual.mean()), "excess_ci_low": residual_lo, "excess_ci_high": residual_hi,
                                "lower_edge": None if not np.isfinite(edges[bin_id]) else float(edges[bin_id]),
                                "upper_edge": None if not np.isfinite(edges[bin_id+1]) else float(edges[bin_id+1])})
        predictions.append(part)
        print(f"{h.key}: {len(part)} later-quarter matches; {time.monotonic()-beginning:.1f}s", flush=True)
        pd.DataFrame(fold_metrics).to_csv(output / "fold_metrics.csv", index=False)
    forecasts = pd.concat(predictions, ignore_index=True)
    results, subgroup_results = [], []
    for h in HYPOTHESES:
        part = forecasts[forecasts.hypothesis == h.key]
        row = {"hypothesis": h.key, "title": h.title, "n": len(part)}
        for adjustment in ("strength", "adjusted"):
            delta = losses(part.target, part[f"p_{adjustment}_base"])-losses(part.target, part[f"p_{adjustment}_plus"])
            lower, upper = weekly_interval(delta, part.match_datetime_utc)
            row.update({f"{adjustment}_gain": float(delta.mean()), f"{adjustment}_ci_low": lower, f"{adjustment}_ci_high": upper})
            if adjustment == "adjusted":
                wide_low, wide_high = weekly_interval(delta, part.match_datetime_utc, alpha=.05/len(HYPOTHESES))
                row["family_ci_low"], row["family_ci_high"] = wide_low, wide_high
                for subgroup, selector in {
                    "close_elo": part.diff_elo_pre.abs() <= 100,
                    "full_player_coverage": part.team1_lineup_history_coverage.eq(1) & part.team2_lineup_history_coverage.eq(1),
                }.items():
                    sub = part[selector]
                    if len(sub) < 100:
                        continue
                    delta_sub = delta[selector.to_numpy()]
                    lo, hi = weekly_interval(delta_sub, sub.match_datetime_utc)
                    subgroup_results.append({"hypothesis": h.key, "subgroup": subgroup, "n": len(sub),
                                             "gain": float(delta_sub.mean()), "ci_low": lo, "ci_high": hi})
        coef = [r["coefficient"] for r in coefficients if r["hypothesis"] == h.key and r["adjustment"] == "adjusted"]
        row["adjusted_or_median"] = float(np.exp(np.median(coef)))
        row["adjusted_or_min"] = float(np.exp(min(coef)))
        row["adjusted_or_max"] = float(np.exp(max(coef)))
        row["positive_coefficient_folds"] = sum(value > 0 for value in coef)
        gains = [r["gain"] for r in fold_metrics if r["hypothesis"] == h.key and r["adjustment"] == "adjusted"]
        row["positive_gain_folds"] = sum(value > 0 for value in gains)
        row["unit"] = h.unit
        results.append(row)
    pd.DataFrame(results).to_csv(output / "hypothesis_results.csv", index=False)
    pd.DataFrame(coefficients).to_csv(output / "coefficients.csv", index=False)
    pd.DataFrame(bin_records).to_csv(output / "win_curves.csv", index=False)
    pd.DataFrame(subgroup_results).to_csv(output / "sensitivity.csv", index=False)
    forecasts.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    after = guarded_hashes()
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    audit = {"protected_files": len(before), "changed": changed, "fits": len(convergence),
             "gradient_max": max(convergence), "independent_solver_checks": quality_checks,
             "prediction_rows": len(forecasts), "distinct_evaluation_matches": int(forecasts.match_id.nunique()),
             "elapsed_seconds": time.monotonic()-beginning, "working_model_replaced": False}
    dump(output / "audit.json", audit)
    if changed:
        raise RuntimeError(f"Protected files changed: {changed}")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/hltv_research_2026-09-29")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)
