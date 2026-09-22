"""Парное ранжирование команд по их собственным предматчевым признакам.

В отличие от ``src.pairwise``, здесь используются не зеркальные разности,
а отдельный вектор исторических показателей каждой команды.
Оценка учитывает контекст: личные встречи зависят от соперника, а показатели
по картам — от общего набора карт, объявленного до начала серии.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling import MODEL_FEATURES
from src.research_design import FEATURE_GROUPS


LEGACY_TO_TEAM_FEATURE = {
    name: "rank_score" if name == "diff_rank" else name.removeprefix("diff_")
    for name in MODEL_FEATURES
}
TEAM_MODEL_FEATURES = [LEGACY_TO_TEAM_FEATURE[name] for name in MODEL_FEATURES]
SHARED_FEATURES = tuple(name for name in MODEL_FEATURES if not name.startswith("diff_"))
TEAM_FEATURE_GROUPS = {
    group: {
        "label": definition["label"],
        "features": {
            LEGACY_TO_TEAM_FEATURE[name]: (
                "Отрицательное место команды в рейтинге HLTV: −rank(team)"
                if name == "diff_rank" else description
            )
            for name, description in definition["features"].items()
        },
    }
    for group, definition in FEATURE_GROUPS.items()
}
EXTRA_INDIVIDUAL_FEATURES = {
    "lineup_player_rating_max": "Максимум исторических средних Rating пяти игроков текущего состава",
    "lineup_player_rating_min": "Минимум исторических средних Rating пяти игроков текущего состава",
    "lineup_player_rating_std": "Стандартное отклонение исторических средних Rating пяти игроков (ddof=0)",
}
EXTRA_JOINT_EXPERIENCE_FEATURES = {
    "roster_same_matches_90d": "Число предыдущих серий текущей команды за 90 дней с точно текущей пятёркой",
    "roster_consecutive_before": "Число непосредственно предшествующих подряд серий текущей команды с этой пятёркой",
    "roster_pair_experience_90d": "Среднее число совместных предыдущих серий за 90 дней для 10 пар текущей пятёрки, независимо от команды",
}
EXTENDED_TEAM_FEATURES = (
    TEAM_MODEL_FEATURES + list(EXTRA_INDIVIDUAL_FEATURES) + list(EXTRA_JOINT_EXPERIENCE_FEATURES)
)
EXTENDED_TEAM_FEATURE_GROUPS = {
    group: {"label": value["label"], "features": dict(value["features"])}
    for group, value in TEAM_FEATURE_GROUPS.items()
}
EXTENDED_TEAM_FEATURE_GROUPS["individual"]["features"].update(EXTRA_INDIVIDUAL_FEATURES)
EXTENDED_TEAM_FEATURE_GROUPS["cohesion"]["label"] = "Сохранность состава и опыт совместной игры (S)"
EXTENDED_TEAM_FEATURE_GROUPS["cohesion"]["features"].update(EXTRA_JOINT_EXPERIENCE_FEATURES)


def team_columns_for(groups, *, extended: bool = False) -> list[str]:
    """Вернуть признаки выбранных групп в порядке, принятом для модели."""
    definitions = EXTENDED_TEAM_FEATURE_GROUPS if extended else TEAM_FEATURE_GROUPS
    order = EXTENDED_TEAM_FEATURES if extended else TEAM_MODEL_FEATURES
    chosen = {name for group in groups for name in definitions[group]["features"]}
    return [name for name in order if name in chosen]


def team_rows(frame: pd.DataFrame, feature_names=None) -> pd.DataFrame:
    """Преобразовать строку матча в две строки команд, не используя ``diff_*``.

    Общий контекст копируется в обе строки, собственные показатели берутся
    из ``team1_*`` и ``team2_*``. Меньшее место HLTV означает более сильную
    команду, поэтому используется ``rank_score = -own_rank``.
    Пропуски сохраняются. Таргет, итоговый счёт и ID команд не входят
    в матрицу признаков.
    """
    names = list(TEAM_MODEL_FEATURES if feature_names is None else feature_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("A nonempty, unique feature list is required")
    unknown = set(names) - set(EXTENDED_TEAM_FEATURES)
    if unknown:
        raise ValueError(f"Unknown own-team features: {sorted(unknown)}")
    values = np.empty((2 * len(frame), len(names)), dtype=float)
    missing = []
    for position, name in enumerate(names):
        if name in SHARED_FEATURES:
            if name not in frame.columns:
                missing.append(name)
                continue
            common = frame[name].to_numpy(dtype=float, na_value=np.nan)
            values[0::2, position] = common
            values[1::2, position] = common
            continue
        suffix = "team_rank" if name == "rank_score" else name
        for team_number, offset in ((1, 0), (2, 1)):
            source = f"team{team_number}_{suffix}"
            if source not in frame.columns:
                missing.append(source)
                continue
            own = frame[source].to_numpy(dtype=float, na_value=np.nan)
            values[offset::2, position] = -own if name == "rank_score" else own
    if missing:
        raise ValueError(f"Own-team dataset is incompatible; missing columns: {missing}")
    values[~np.isfinite(values)] = np.nan
    return pd.DataFrame(values, columns=names)


def team_ranking_pool(frame: pd.DataFrame, target, feature_names=None):
    """Создать группу из двух команд и пару «победитель — проигравший» на серию."""
    from catboost import Pool

    y = np.asarray(target)
    if y.ndim != 1 or len(y) != len(frame) or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("One binary outcome per nonempty match frame is required")
    y = y.astype(int)
    start = np.arange(len(y), dtype=int) * 2
    labels = np.column_stack((y, 1 - y)).ravel()
    pairs = np.column_stack((start + (1 - y), start + y))
    return Pool(
        team_rows(frame, feature_names), label=labels,
        group_id=np.repeat(np.arange(len(y)), 2), pairs=pairs,
    )


def team_ranking_scores(model, frame: pd.DataFrame, feature_names=None) -> np.ndarray:
    """Вернуть ``[score_A, score_B]`` для каждого матча без усреднения."""
    scores = np.asarray(model.predict(team_rows(frame, feature_names)), dtype=float)
    if scores.size != 2 * len(frame) or not np.isfinite(scores).all():
        raise ValueError("The ranker must return one finite score per team object")
    return scores.reshape(len(frame), 2)


def team_ranking_probability(model, frame: pd.DataFrame, feature_names=None) -> np.ndarray:
    """Вероятность победы A: sigmoid(score_A − score_B), без калибровки."""
    from scipy.special import expit
    scores = team_ranking_scores(model, frame, feature_names)
    return expit(scores[:, 0] - scores[:, 1])


def team_ranking_decision(model, frame: pd.DataFrame, feature_names=None):
    """Выбрать большую оценку и отдельно отметить случаи точного равенства.

    При равенстве выбирается меньший постоянный ID команды, а не первая
    колонка. Это отдельное правило выбора, не часть обученной оценки.
    ID нужны только при равенстве; такие случаи учитываются при оценке качества.
    """
    scores = team_ranking_scores(model, frame, feature_names)
    ties = scores[:, 0] == scores[:, 1]
    choice = (scores[:, 0] > scores[:, 1]).astype(int)
    if ties.any():
        if not {"team1_id", "team2_id"}.issubset(frame.columns):
            raise ValueError("Stable team IDs are required to resolve exact score ties")
        ids = frame.loc[:, ["team1_id", "team2_id"]].to_numpy(dtype=float)
        if not np.isfinite(ids[ties]).all() or np.any(ids[ties, 0] == ids[ties, 1]):
            raise ValueError("Exact score ties require distinct, finite team IDs")
        choice[ties] = (ids[ties, 0] < ids[ties, 1]).astype(int)
    return choice, ties


def swap_team_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Поменять местами парные столбцы ``team1_*`` и ``team2_*``.

    Это проверка независимости от порядка команд, не дополнение обучающей
    выборки. Таргет меняется на противоположный, если он есть в таблице.
    Разности ``diff_*`` меняют знак для согласованности, хотя team_rows
    не использует их. Остальные столбцы сохраняются.
    """
    swapped = frame.copy()
    for column in frame.columns:
        if column.startswith("team1_"):
            partner = "team2_" + column[len("team1_"):]
            if partner in frame.columns:
                swapped[column] = frame[partner].to_numpy()
                swapped[partner] = frame[column].to_numpy()
        elif column.startswith("diff_"):
            swapped[column] = -frame[column]
    if "team1_win" in frame.columns:
        swapped["team1_win"] = 1 - frame["team1_win"]
    return swapped
