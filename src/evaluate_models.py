"""Verify saved test predictions and create compact evaluation figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.modeling import metric_row, update_artifact_manifest, write_json
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from modeling import metric_row, update_artifact_manifest, write_json  # type: ignore[no-redef]


PREDICTION_COLUMNS = {
    "Historical prior": "prob_historical_prior",
    "Elo": "prob_elo",
    "Logistic regression": "prob_logistic_regression",
    "Histogram gradient boosting": "prob_histogram_gradient_boosting",
    "CatBoost": "prob_catboost",
    "CatBoost + Platt": "prob_catboost_platt",
}

MODEL_LABELS = {
    "Historical prior": "Априорная доля",
    "Elo": "Elo",
    "Logistic regression": "Логистическая регрессия",
    "Histogram gradient boosting": "Гистограммный бустинг",
    "CatBoost": "CatBoost",
    "CatBoost + Platt": "CatBoost + Platt",
}

FEATURE_LABELS = {
    "bo1": "Формат BO1",
    "bo3": "Формат BO3",
    "bo5": "Формат BO5",
    "is_lan": "Режим LAN",
    "is_online": "Режим онлайн",
    "rank_available": "Доступны оба рейтинга",
    "diff_elo_pre": "Δ Elo до матча",
    "diff_matches_before": "Δ матчей в истории",
    "diff_days_since_last_match": "Δ дней после матча",
    "diff_activity_7d": "Δ активности за 7 дней",
    "diff_activity_30d": "Δ активности за 30 дней",
    "diff_activity_90d": "Δ активности за 90 дней",
    "diff_overall_winrate": "Δ общей доли побед",
    "diff_winrate_last_5": "Δ доли побед за 5 матчей",
    "diff_winrate_last_10": "Δ доли побед за 10 матчей",
    "diff_winrate_last_20": "Δ доли побед за 20 матчей",
    "diff_win_streak": "Δ серий побед",
    "diff_loss_streak": "Δ серий поражений",
    "diff_avg_opp_elo_last_10": "Δ Elo прошлых соперников",
    "diff_h2h_wins_all": "Δ очных побед за всю историю",
    "diff_h2h_wins_last5": "Δ очных побед за 5 встреч",
    "diff_roster_size": "Δ размера состава",
    "diff_roster_overlap_prev": "Δ совпавших игроков",
    "diff_roster_overlap_prev_ratio": "Δ стабильности состава",
    "diff_lineup_history_coverage": "Δ доли игроков с историей",
    "diff_lineup_players_with_history": "Δ игроков с историей",
    "diff_lineup_player_rating_mean": "Δ среднего рейтинга игроков",
    "diff_lineup_player_adr_mean": "Δ среднего ADR игроков",
    "diff_lineup_player_kast_mean": "Δ среднего KAST игроков",
    "diff_lineup_player_opening_diff_mean": "Δ разницы первых убийств",
    "diff_lineup_player_maps_played_mean": "Δ карт в истории игроков",
    "diff_avg_map_count_before": "Δ карт в пуле карт",
    "diff_avg_map_wr_before": "Δ доли побед на картах",
    "diff_avg_map_ct_wr_before": "Δ доли CT-раундов",
    "diff_avg_map_t_wr_before": "Δ доли T-раундов",
    "diff_series_maps_known": "Δ известных карт серии",
    "diff_veto_pick_rate_before": "Δ частоты выбора карт",
    "diff_veto_remove_rate_before": "Δ частоты исключения карт",
    "diff_veto_leftover_rate_before": "Δ частоты decider-карт",
    "diff_rank": "Δ рейтинга HLTV",
}

ABLATION_LABELS = {
    "Rank + context": "Рейтинг и контекст",
    "+ Elo and form": "+ Elo и форма",
    "+ Roster and players": "+ Состав и игроки",
    "+ Map pool and veto": "+ Пул карт и veto",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recalculate metrics from held-out predictions and draw evaluation figures."
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure directory; defaults to ARTIFACTS/figures.",
    )
    parser.add_argument("--metric-tolerance", type=float, default=1e-9)
    return parser.parse_args()


def calibration_points(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int = 10,
) -> tuple[list[float], list[float]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    predicted: list[float] = []
    observed: list[float] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            predicted.append(float(probability[mask].mean()))
            observed.append(float(target[mask].mean()))
    return predicted, observed


def main() -> None:
    args = parse_args()
    prediction_path = args.artifacts / "test_predictions.csv"
    metrics_path = args.artifacts / "model_metrics.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Predictions not found: {prediction_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Training metrics not found: {metrics_path}")

    figures = args.output_dir or args.artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.read_csv(prediction_path)
    target = prediction_frame["team1_win"].astype(int).to_numpy()
    missing = [column for column in PREDICTION_COLUMNS.values() if column not in prediction_frame]
    if missing:
        raise ValueError(f"test_predictions.csv is missing probability columns: {missing}")

    calculated = {
        model: metric_row(target, prediction_frame[column].to_numpy())
        for model, column in PREDICTION_COLUMNS.items()
    }
    stored = pd.read_csv(metrics_path).set_index("model")
    for model, values in calculated.items():
        for metric, value in values.items():
            difference = abs(float(stored.loc[model, metric]) - value)
            if difference > args.metric_tolerance:
                raise ValueError(
                    f"Stored metric mismatch for {model}/{metric}: difference={difference}"
                )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    metrics = pd.DataFrame(
        [{"model": model, **values} for model, values in calculated.items()]
    ).set_index("model")
    order = list(PREDICTION_COLUMNS)
    short = ["Априорная", "Elo", "Логрег", "ГистБуст", "CatBoost", "CatBoost\n+ Platt"]
    positions = np.arange(len(order))

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    axes[0].bar(positions - 0.18, metrics.loc[order, "accuracy"], 0.36, label="Точность")
    axes[0].bar(positions + 0.18, metrics.loc[order, "roc_auc"], 0.36, label="ROC-AUC")
    axes[0].set_xticks(positions, short)
    axes[0].set_ylim(0.45, 0.80)
    axes[0].set_title("Качество классификации и ранжирования")
    axes[0].legend(frameon=False)
    axes[1].bar(positions - 0.18, metrics.loc[order, "log_loss"], 0.36, label="LogLoss")
    axes[1].bar(positions + 0.18, metrics.loc[order, "brier"], 0.36, label="Brier")
    axes[1].set_xticks(positions, short)
    axes[1].set_title("Качество вероятностей (меньше — лучше)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(figures / "model_metrics.png", dpi=180, facecolor="white")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 5.6))
    for model in ["Elo", "Logistic regression", "Histogram gradient boosting", "CatBoost"]:
        probability = prediction_frame[PREDICTION_COLUMNS[model]].to_numpy()
        false_positive, true_positive, _ = roc_curve(target, probability)
        axis.plot(
            false_positive,
            true_positive,
            linewidth=2,
            label=f"{MODEL_LABELS[model]} (AUC={calculated[model]['roc_auc']:.3f})",
        )
    axis.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
    axis.set_xlabel("Доля ложноположительных прогнозов")
    axis.set_ylabel("Доля истинноположительных прогнозов")
    axis.set_title("ROC-кривые на будущем тестовом периоде")
    axis.legend(frameon=False, fontsize=9, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(figures / "roc_curves.png", dpi=180, facecolor="white")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 5.6))
    for model in ["Elo", "Logistic regression", "CatBoost", "CatBoost + Platt"]:
        probability = prediction_frame[PREDICTION_COLUMNS[model]].to_numpy()
        predicted, observed = calibration_points(target, probability)
        axis.plot(
            predicted,
            observed,
            marker="o",
            linewidth=1.8,
            label=f"{MODEL_LABELS[model]} (ECE={calculated[model]['ece_10']:.3f})",
        )
    axis.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1, label="Идеальная калибровка")
    axis.set_xlabel("Средняя предсказанная вероятность")
    axis.set_ylabel("Наблюдаемая доля побед")
    axis.set_title("Калибровка вероятностей")
    axis.legend(frameon=False, fontsize=8.5)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(figures / "calibration.png", dpi=180, facecolor="white")
    plt.close(figure)

    importance_path = args.artifacts / "feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(14).sort_values("importance")
        labels = importance["feature"].map(FEATURE_LABELS).fillna(importance["feature"])
        figure, axis = plt.subplots(figsize=(8.5, 6.0))
        axis.barh(labels, importance["importance"])
        axis.set_xlabel("Важность признака в CatBoost")
        axis.set_title("Наиболее значимые предматчевые признаки")
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure.savefig(figures / "feature_importance.png", dpi=180, facecolor="white")
        plt.close(figure)

    ablation_path = args.artifacts / "ablation_metrics.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        if not ablation.empty:
            figure, axis = plt.subplots(figsize=(8.7, 4.8))
            positions = np.arange(len(ablation))
            axis.plot(positions, ablation["roc_auc"], marker="o", linewidth=2, label="ROC-AUC")
            axis.plot(positions, ablation["accuracy"], marker="s", linewidth=2, label="Точность")
            labels = ablation["feature_set"].map(ABLATION_LABELS).fillna(ablation["feature_set"])
            axis.set_xticks(positions, labels, rotation=10)
            axis.set_title("Абляция групп предматчевых признаков")
            axis.legend(frameon=False)
            axis.spines[["top", "right"]].set_visible(False)
            figure.tight_layout()
            figure.savefig(figures / "ablation.png", dpi=180, facecolor="white")
            plt.close(figure)

    write_json(
        args.artifacts / "evaluation_summary.json",
        {
            "verified_rows": len(prediction_frame),
            "metric_tolerance": args.metric_tolerance,
            "metrics": calculated,
            "figures": sorted(path.name for path in figures.glob("*.png")),
        },
    )
    try:
        figure_names = [str(path.relative_to(args.artifacts)) for path in figures.glob("*.png")]
    except ValueError:
        figure_names = []
    update_artifact_manifest(
        args.artifacts,
        ["evaluation_summary.json", *figure_names],
    )
    print(f"Verified {len(prediction_frame):,} held-out predictions")
    print(f"Figures saved to {figures}")


if __name__ == "__main__":
    main()
