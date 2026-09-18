import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from src.modeling import swap_team_perspective
from src.pairwise import paired_rows, ranking_pool, pair_losses, ranker_probability
from src.research_design import FEATURE_GROUPS, columns_for, validate_groups


def test_canonical_partition():
    validate_groups()
    assert {g: len(v['features']) for g, v in FEATURE_GROUPS.items()} == {
        'individual': 4, 'team': 21, 'cohesion': 2, 'controls': 13}
    assert len(columns_for(FEATURE_GROUPS)) == 40


def test_ranking_is_one_pair_with_two_objects_per_match():
    x = pd.DataFrame({'diff_elo_pre': [100., -50.], 'bo3': [1., 1.]})
    rows = paired_rows(x)
    np.testing.assert_array_equal(rows.diff_elo_pre.to_numpy(), [100, -100, -50, 50])
    pool = ranking_pool(x, np.array([1, 0]))
    assert pool.num_row() == 4 and pool.num_pairs() == 2
    np.testing.assert_equal(pool.get_label(), [1, 0, 0, 1])
    groups = pool.get_group_id_hash()
    assert groups[0] == groups[1] and groups[2] == groups[3] and groups[0] != groups[2]
    with pytest.raises(ValueError):
        ranking_pool(x, np.array([2, 0]))


def test_pair_logit_equals_probability_logloss_and_gradient():
    m = np.array([-2., -.4, .1, 1., 3.])
    y = np.array([0, 1, 0, 1, 1])
    p = expit(m)
    np.testing.assert_allclose(pair_losses(m, y), -y*np.log(p)-(1-y)*np.log1p(-p))
    h = 1e-5
    numeric = (pair_losses(m+h, y)-pair_losses(m-h, y))/(2*h)
    np.testing.assert_allclose(numeric, p-y, atol=1e-9)
    assert np.isfinite(pair_losses(np.array([-1000., 1000.]), np.array([1, 0]))).all()


def test_scores_swap_complement_without_probability_averaging():
    class Score:
        def predict(self, x):
            return x.diff_elo_pre.to_numpy() / 100 + x.bo3.to_numpy() * 0.3
    x = pd.DataFrame({'diff_elo_pre': [100., -50., 0], 'bo3': [1., 1., 1.]})
    p = ranker_probability(Score(), x)
    q = ranker_probability(Score(), swap_team_perspective(x, x.columns))
    np.testing.assert_allclose(p+q, 1, atol=1e-15)
    assert p[2] == .5


def test_real_catboost_pair_training_smoke():
    from catboost import CatBoostRanker
    x = pd.DataFrame({'diff_elo_pre': np.tile([-2., -1., 1., 2.], 10)})
    y = (x.diff_elo_pre.to_numpy() > 0).astype(int)
    model = CatBoostRanker(loss_function='PairLogit', iterations=15, depth=2,
                          verbose=False, allow_writing_files=False, thread_count=2)
    model.fit(ranking_pool(x, y))
    assert np.mean((ranker_probability(model, x) >= .5) == y) == 1.
