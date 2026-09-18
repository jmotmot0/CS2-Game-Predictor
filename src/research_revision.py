"""Reproducible ranking comparison and retrained group-contribution experiment.

Run: python -m src.research_revision --phase compare
Then: python -m src.research_revision --phase groups
Nothing in the existing production data/model directory is overwritten.
"""
from __future__ import annotations
import argparse
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from src.modeling import (MODEL_FEATURES, augment_team_swap, chronological_masks,
    load_feature_dataset, metric_row, sha256_file, symmetrized_probability,
    tune_elo, library_versions, write_json)
from src.pairwise import ranking_pool, ranker_probability
from src.research_design import FEATURE_GROUPS, columns_for

DEFAULT_OUTPUT = Path('artifacts/research_revision_2026-09-30')
SEEDS = [42, 43, 44]
FAMILIES = ['team', 'individual', 'cohesion']


def point_loss(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    return -np.asarray(y)*np.log(p) - (1-np.asarray(y))*np.log1p(-p)


def weekly_interval(values, timestamps, *, samples=10000, seed=42):
    """Paired week-cluster bootstrap, conditional on already fitted models."""
    weeks = pd.to_datetime(timestamps, utc=True).dt.tz_localize(None).dt.to_period('W').astype(str)
    blocks = pd.DataFrame({'week': np.asarray(weeks), 'value': values}).groupby('week').value.agg(['sum', 'count'])
    if len(blocks) < 2:
        raise ValueError('At least two calendar weeks required for cluster bootstrap')
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(blocks), size=(samples, len(blocks)))
    sums, counts = blocks['sum'].to_numpy(), blocks['count'].to_numpy()
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {'estimate': float(np.mean(values)), 'ci95_low': float(np.quantile(means, .025)),
            'ci95_high': float(np.quantile(means, .975)), 'weeks': len(blocks)}


def protected_paths():
    paths = [Path('data/processed/features_dataset.csv')]
    paths += sorted(Path('data/interim/hltv_final_clean').glob('*.csv'))
    paths += [Path('artifacts')/f for f in ['catboost_model.cbm', 'feature_schema.json',
        'inference_state.json', 'test_predictions.csv', 'model_metrics.csv', 'experiment_summary.json']]
    return {str(p): sha256_file(p) for p in paths}


def load_data():
    frame = load_feature_dataset(Path('data/processed/features_dataset.csv'))
    masks = dict(zip(['train', 'validation', 'test'], chronological_masks(frame)))
    k, elo, e1, e2, _ = tune_elo(frame, masks['validation'])
    frame['diff_elo_pre'] = e1-e2
    x = frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    y = frame.team1_win.to_numpy(dtype=int)
    return frame, masks, x, y, k, elo


def fit(kind, x, y, masks, seed, output, key):
    params = dict(iterations=1000, depth=6, learning_rate=.04, l2_leaf_reg=5.,
                  random_seed=seed, thread_count=8, allow_writing_files=False, verbose=False)
    start = time.perf_counter()
    if kind == 'ranking':
        model = CatBoostRanker(**params, loss_function='PairLogit', eval_metric='PairLogit')
        model.fit(ranking_pool(x.loc[masks['train']], y[masks['train']]),
                  eval_set=ranking_pool(x.loc[masks['validation']], y[masks['validation']]),
                  early_stopping_rounds=100, use_best_model=True)
        predict = ranker_probability
    else:
        model = CatBoostClassifier(**params, loss_function='Logloss', eval_metric='Logloss')
        xt, yt = augment_team_swap(x.loc[masks['train']], y[masks['train']])
        xv, yv = augment_team_swap(x.loc[masks['validation']], y[masks['validation']])
        model.fit(xt, yt, eval_set=(xv, yv), early_stopping_rounds=100, use_best_model=True)
        predict = symmetrized_probability
    model.save_model(str(output / f'{key}.cbm'))
    prediction = {part: predict(model, x.loc[masks[part]]) for part in ['validation', 'test']}
    record = {'key': key, 'kind': kind, 'seed': seed, 'feature_count': x.shape[1],
              'features': list(x.columns), 'trees': int(model.tree_count_),
              'seconds': time.perf_counter()-start, 'params': model.get_all_params(),
              'metrics': {part: metric_row(y[masks[part]], prediction[part]) for part in prediction}}
    write_json(output / f'{key}.json', record)
    print(f'{key}: {model.tree_count_} trees; validation LL={record["metrics"]["validation"]["log_loss"]:.6f}; '
          f'test LL={record["metrics"]["test"]["log_loss"]:.6f}; {record["seconds"]:.1f}s', flush=True)
    return model, prediction, record


def compare(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'protocol.json').exists():
        raise FileExistsError('This run is immutable; use a new output directory to repeat it')
    frame, masks, x, y, k, elo = load_data()
    protocol = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'question': 'Каков предсказательный вклад индивидуальных характеристик игроков по сравнению с командной историей и сохранностью состава?',
        'prediction_time': 'После завершения предматчевого veto, до начала серии',
        'groups': FEATURE_GROUPS,
        'split': {'train_end_exclusive': '2025-07-01', 'validation_end_exclusive': '2026-01-01',
                  'counts': {part: int(mask.sum()) for part, mask in masks.items()}},
        'seeds': SEEDS, 'elo_k_selected_on_validation': k,
        'selection_rule': 'Compare mean validation match LogLoss across seeds 42/43/44. Choose ranking only if lower than classification. Main fit seed 42, other seeds are sensitivity checks, not an ensemble.',
        'ranking': 'CatBoostRanker PairLogit. Two contextual team-perspective objects per match, explicit single winner-loser pair. Same 40 columns/information as classifier. Probability sigmoid(score_A-score_B). Not a global team leaderboard.',
        'decision': 'p >= 0.5 chooses team1; exact ties reported separately; no threshold optimization',
        'classifier': 'CatBoostClassifier Logloss; mirrored training; average forward probability and complemented reverse probability at inference',
        'budget': 'Both: depth 6, lr .04, L2 5, <=1000 trees, validation early stopping 100, native objective-dependent CatBoost defaults saved per fit',
        'group_method': 'Retrain all 8 subsets of {T,I,S}, always retaining C. Same splits, hyperparameters, early stopping and seeds. Conditional loss increases and two-group Shapley loss allocation for I versus T+S. Not summed native feature importance.',
        'importance': 'Primary: actual leave-group-out retraining. Secondary: CatBoost LossFunctionChange on validation Pool, approximate feature removal from fixed trees, not retraining.',
        'uncertainty': '10000 paired calendar-week cluster bootstrap draws, conditional on fitted models; three seeds reported separately; two primary group intervals descriptive, not causal significance tests',
        'test_status': 'Historical test Jan-Apr 2026 has been inspected in earlier project stages. This is a retrospective comparison, not a new untouched confirmation set. Test not used for this run selection.',
        'limitations': ['No archived publication/end timestamps: point-in-time ordering by match start is not proof that every previous result was already published.',
                         'Roster overlap is a proxy for continuity, not a full measurement of synergy.',
                         'Player-group NaN pattern can carry information; complete-history cohort is checked separately.',
                         'HLTV ranks/lineups/veto are reconstructed from historical pages, not timestamped pre-match snapshots.'],
        'protected_hashes': protected_paths(), 'versions': library_versions(),
        'source_hashes': {str(p): sha256_file(p) for p in [Path(__file__), Path('src/pairwise.py'), Path('src/research_design.py')]},
    }
    write_json(output/'protocol.json', protocol)
    prediction_frames = {part: frame.loc[mask, ['match_id', 'match_datetime_utc', 'team1_id', 'team2_id', 'team1_win']].reset_index(drop=True)
                         for part, mask in masks.items() if part != 'train'}
    records = []
    for kind in ['classification', 'ranking']:
        for seed in SEEDS:
            key = f'{kind}_full_s{seed}'
            _, prediction, record = fit(kind, x, y, masks, seed, output, key)
            records.append(record)
            for part in prediction:
                prediction_frames[part][key] = prediction[part]
                prediction_frames[part].to_csv(output/f'{part}_predictions.csv', index=False)
    # Elo is the interpretable outcome-history baseline. Chance is the sanity baseline.
    for part, out in prediction_frames.items():
        out['chance'] = .5
        out['elo'] = elo[masks[part]]
    # Refit the linear comparison on the same historical split, with C chosen on validation.
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    xt, yt = augment_team_swap(x.loc[masks['train']], y[masks['train']])
    lr_fits = []
    for c in [.01, .1, 1., 10.]:
        lr = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                           LogisticRegression(C=c, max_iter=2000, random_state=42))
        lr.fit(xt, yt)
        vp = symmetrized_probability(lr, x.loc[masks['validation']])
        lr_fits.append((metric_row(y[masks['validation']], vp)['log_loss'], c, lr))
    _, c, lr = min(lr_fits, key=lambda t:t[0])
    for part, out in prediction_frames.items():
        out['logistic'] = symmetrized_probability(lr, x.loc[masks[part]])
        out.to_csv(output/f'{part}_predictions.csv', index=False)
    validation_mean = {kind: float(np.mean([r['metrics']['validation']['log_loss'] for r in records if r['kind']==kind])) for kind in ['classification', 'ranking']}
    selected = min(validation_mean, key=validation_mean.get)
    selection = {'selected_kind': selected, 'main_seed': 42, 'mean_validation_logloss': validation_mean,
                 'logistic_C': c, 'rule': protocol['selection_rule'], 'decision_made_without_test_selection': True}
    write_json(output/'selection.json', selection)
    rows = []
    for part, out in prediction_frames.items():
        for name in [*['chance', 'elo', 'logistic'], *[r['key'] for r in records]]:
            rows.append({'split':part, 'model':name, **metric_row(out.team1_win.to_numpy(), out[name].to_numpy()),
                         'n':len(out), 'exact_ties':int(out[name].eq(.5).sum())})
    pd.DataFrame(rows).to_csv(output/'comparison_metrics.csv', index=False)
    intervals = {}
    for part, out in prediction_frames.items():
        delta = point_loss(out.team1_win, out.classification_full_s42) - point_loss(out.team1_win, out.ranking_full_s42)
        intervals[part] = weekly_interval(delta, out.match_datetime_utc)
    write_json(output/'ranking_gain_intervals.json', intervals)
    if protected_paths() != protocol['protected_hashes']:
        raise RuntimeError('Protected source/artifact changed during the experiment')
    print(json.dumps(selection, ensure_ascii=False), flush=True)


def subset_name(groups):
    return 'C' + ''.join({'team':'T','individual':'I','cohesion':'S'}[g] for g in FAMILIES if g in groups)


def groups(output):
    protocol = json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    if protected_paths() != protocol['protected_hashes']:
        raise RuntimeError('Protected inputs changed since comparison')
    if (output/'group_contributions.csv').exists():
        raise FileExistsError('Completed group experiment is immutable')
    selected = json.loads((output/'selection.json').read_text(encoding='utf-8'))['selected_kind']
    frame, masks, x, y, _, _ = load_data()
    tables = {part: pd.read_csv(output/f'{part}_predictions.csv') for part in ['validation','test']}
    records = []
    for size in range(4):
        for subset in itertools.combinations(FAMILIES, size):
            name = subset_name(subset)
            columns = columns_for(['controls', *subset])
            for seed in SEEDS:
                key = f'{selected}_{name}_s{seed}'
                if len(subset)==3:
                    fullkey = f'{selected}_full_s{seed}'
                    for part in tables:
                        tables[part][key] = tables[part][fullkey]
                    record = json.loads((output/f'{fullkey}.json').read_text(encoding='utf-8'))
                else:
                    _, predictions, record = fit(selected, x[columns], y, masks, seed, output, key)
                    for part in tables:
                        tables[part][key] = predictions[part]
                records.append({'subset':name, 'seed':seed, 'feature_count':len(columns), 'trees':record['trees'],
                                **{f'{part}_{metric}':value for part, metrics in record['metrics'].items() for metric,value in metrics.items()}})
                for part in tables:
                    tables[part].to_csv(output/f'{part}_group_predictions.csv', index=False)
                pd.DataFrame(records).to_csv(output/'group_metrics.csv', index=False)
    contributions, allocations = [], []
    for part, table in tables.items():
        target = table.team1_win.to_numpy()
        for seed in SEEDS:
            losses = {name: point_loss(target, table[f'{selected}_{name}_s{seed}'])
                      for name in ['C','CT','CI','CS','CTI','CTS','CIS','CTIS']}
            for name, diff in {
                'individual_given_team_and_cohesion': losses['CTS']-losses['CTIS'],
                'team_and_cohesion_given_individual': losses['CI']-losses['CTIS'],
                'cohesion_given_team_and_individual': losses['CTI']-losses['CTIS'],
                'individual_over_controls': losses['C']-losses['CI'],
                'team_and_cohesion_over_controls': losses['C']-losses['CTS'],
            }.items():
                contributions.append({'split':part,'seed':seed,'contrast':name,**weekly_interval(diff, table.match_datetime_utc)})
            phi_i = .5*((losses['C']-losses['CI']) + (losses['CTS']-losses['CTIS']))
            phi_t = .5*((losses['C']-losses['CTS']) + (losses['CI']-losses['CTIS']))
            for name, diff in [('individual',phi_i),('team_and_cohesion',phi_t)]:
                allocations.append({'split':part,'seed':seed,'group':name,**weekly_interval(diff, table.match_datetime_utc)})
            np.testing.assert_allclose(phi_i+phi_t, losses['C']-losses['CTIS'], atol=1e-15)
    pd.DataFrame(contributions).to_csv(output/'group_contributions.csv', index=False)
    pd.DataFrame(allocations).to_csv(output/'group_shapley.csv', index=False)
    # A conditional sensitivity cohort: individual quality has no missing values.
    test = frame.loc[masks['test']].reset_index(drop=True)
    complete = test[columns_for(['individual'])].notna().all(axis=1)
    for side in ['team1', 'team2']:
        complete &= test[f'{side}_lineup_history_coverage'].ge(.8)
        complete &= test[f'{side}_matches_before'].ge(10)
    table = tables['test'].loc[complete]
    cohort = []
    for seed in SEEDS:
        delta = point_loss(table.team1_win, table[f'{selected}_CTS_s{seed}']) - point_loss(table.team1_win, table[f'{selected}_CTIS_s{seed}'])
        cohort.append({'seed':seed,'n':len(table),**weekly_interval(delta, table.match_datetime_utc)})
    write_json(output/'complete_history_sensitivity.json', {'rule':'Both coverage>=.8, both >=10 earlier matches, no missing individual features. Evaluate already fitted models; no cohort refit.', 'results':cohort})
    model = CatBoostRanker() if selected=='ranking' else CatBoostClassifier()
    model.load_model(str(output/f'{selected}_full_s42.cbm'))
    xv, yv = x.loc[masks['validation']], y[masks['validation']]
    if selected=='ranking':
        pool = ranking_pool(xv, yv)
    else:
        xa, ya = augment_team_swap(xv, yv)
        pool = Pool(xa, label=ya)
    importance = model.get_feature_importance(pool, type='LossFunctionChange', thread_count=8)
    pd.DataFrame({'feature':MODEL_FEATURES,'validation_LossFunctionChange':importance}).sort_values(
        'validation_LossFunctionChange',ascending=False).to_csv(output/'native_importance_secondary.csv',index=False)
    # Save raw features and predictions for transparent selection of a real example.
    example_columns = [c for c in frame if c in ['match_id','match_datetime_utc','team1_name','team2_name','team1_id','team2_id','team1_win']
                       or c.startswith(('team1_lineup_', 'team2_lineup_', 'team1_roster_', 'team2_roster_'))]
    candidates = frame.loc[masks['test'], example_columns].reset_index(drop=True)
    for name in ['CI','CTS','CTI','CTIS']:
        candidates[f'prob_{name}'] = tables['test'][f'{selected}_{name}_s42']
    candidates.to_csv(output/'example_candidates.csv',index=False)
    if protected_paths()!=protocol['protected_hashes']:
        raise RuntimeError('Protected inputs changed')
    print('Group contributions, Shapley allocation, sensitivity cohort and secondary importance saved.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['compare','groups'],required=True)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    (compare if args.phase=='compare' else groups)(args.output)


if __name__=='__main__':
    main()
