"""Парное обучение CatBoost: одна обучающая пара на серию.

Объект — команда в текущем матче, описанная разностями с соперником
и общим предматчевым контекстом. Сохраняется та же информация, что у
классификатора; независимый от соперника глобальный рейтинг не строится.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import expit
from src.modeling import swap_team_perspective


def paired_rows(features: pd.DataFrame) -> pd.DataFrame:
    a = features.reset_index(drop=True)
    b = swap_team_perspective(a, a.columns)
    values = np.empty((2 * len(a), len(a.columns)), dtype=float)
    values[0::2] = a.to_numpy(dtype=float)
    values[1::2] = b.to_numpy(dtype=float)
    return pd.DataFrame(values, columns=a.columns)


def ranking_pool(features: pd.DataFrame, target: np.ndarray):
    from catboost import Pool
    y = np.asarray(target)
    if len(y) != len(features) or not np.isin(y, [0, 1]).all():
        raise ValueError("One binary outcome per match is required")
    labels = np.column_stack((y, 1 - y)).ravel()
    starts = np.arange(len(y)) * 2
    winners = starts + (1 - y).astype(int)
    losers = starts + y.astype(int)
    return Pool(paired_rows(features), label=labels,
                group_id=np.repeat(np.arange(len(y)), 2),
                pairs=np.column_stack((winners, losers)))


def ranker_probability(model, features: pd.DataFrame) -> np.ndarray:
    scores = np.asarray(model.predict(paired_rows(features)), dtype=float).reshape(-1, 2)
    return expit(scores[:, 0] - scores[:, 1])


def pair_losses(margins, target):
    """Устойчивый расчёт PairLogit на матч: softplus(-(2y-1)*(s_A-s_B))."""
    return np.logaddexp(0.0, -(2 * np.asarray(target) - 1) * np.asarray(margins))
