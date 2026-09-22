"""Supervisor revision: own-team PairLogit and player/team contribution.

Old data, models and experiments are immutable. Run --phase base, then extended.
The new research model is a pairwise ranker with 46 inputs by design, not an
assertion that ranking must outperform classification. The 40-input ranker is
a control for the cost/benefit of measuring the supervisor's new hypotheses.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker

from src.modeling import (MODEL_FEATURES, augment_team_swap, chronological_masks,
    load_feature_dataset, metric_row, sha256_file, symmetrized_probability,
    tune_elo, write_json)
from src.research_revision import point_loss, protected_paths, weekly_interval
from src.team_ranking import (TEAM_MODEL_FEATURES, team_ranking_pool,
    team_ranking_scores as team_scores, team_columns_for as group_columns,
    team_ranking_decision)

OUT = Path('artifacts/supervisor_revision_2026-09-30')
OLD = Path('artifacts/research_revision_2026-09-30')
SEEDS = [42, 43, 44]
EXTRA_I = ['lineup_player_rating_max', 'lineup_player_rating_min', 'lineup_player_rating_std']
EXTRA_S = ['roster_same_matches_90d', 'roster_consecutive_before', 'roster_pair_experience_90d']


def data():
    f = load_feature_dataset(Path('data/processed/features_dataset.csv'))
    masks = dict(zip(['train', 'validation', 'test'], chronological_masks(f)))
    k, elo, a, b, _ = tune_elo(f, masks['validation'])
    # Собственный Elo команды и разность классификатора используют одну историю.
    f['team1_elo_pre'], f['team2_elo_pre'], f['diff_elo_pre'] = a, b, a-b
    return f, masks, f.team1_win.to_numpy(int), k, elo


def prediction(model, frame, columns):
    from scipy.special import expit
    scores = team_scores(model, frame, columns)
    return expit(scores[:, 0]-scores[:, 1]), scores


def train_ranker(frame, y, masks, columns, seed, key, out):
    start = time.perf_counter()
    model = CatBoostRanker(iterations=1000, depth=6, learning_rate=.04,
        l2_leaf_reg=5., loss_function='PairLogit', eval_metric='PairLogit',
        custom_metric=['PairAccuracy'], random_seed=seed, thread_count=8,
        allow_writing_files=False, verbose=False)
    model.fit(team_ranking_pool(frame.loc[masks['train']], y[masks['train']], columns),
        eval_set=team_ranking_pool(frame.loc[masks['validation']], y[masks['validation']], columns),
        early_stopping_rounds=100, use_best_model=True)
    model.save_model(str(out/f'{key}.cbm'))
    predictions, scores, metrics = {}, {}, {}
    for part in ['validation', 'test']:
        predictions[part], scores[part] = prediction(model, frame.loc[masks[part]], columns)
        metrics[part] = metric_row(y[masks[part]], predictions[part])
        # Accuracy считается по порядку оценок, без подбора порога.
        decision, ties = team_ranking_decision(model, frame.loc[masks[part]], columns)
        metrics[part]['accuracy'] = float(np.mean(decision == y[masks[part]]))
        metrics[part]['exact_score_ties'] = int(ties.sum())
    record = {'key':key, 'kind':'own_team_ranking', 'seed':seed,
        'features':columns, 'feature_count':len(columns), 'trees':int(model.tree_count_),
        'seconds':time.perf_counter()-start, 'params':model.get_all_params(),
        'metrics':metrics}
    write_json(out/f'{key}.json', record)
    print(f'{key}: {model.tree_count_} trees, val accuracy {metrics["validation"]["accuracy"]:.6f}, '
          f'test accuracy {metrics["test"]["accuracy"]:.6f}, {record["seconds"]:.1f}s', flush=True)
    return model, predictions, scores, record


def table_frames(f, masks):
    return {part:f.loc[masks[part], ['match_id','match_datetime_utc','team1_id','team2_id','team1_win']].reset_index(drop=True)
            for part in ['validation','test']}


def rank_decisions_from_table(tab, column):
    p=tab[column].to_numpy()
    labels=(p>.5).astype(int)
    ties=p==.5
    labels[ties]=(tab.team1_id.to_numpy()[ties]<tab.team2_id.to_numpy()[ties]).astype(int)
    return labels


def base(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise FileExistsError('Use a fresh output directory for a new experiment')
    f, masks, y, k, elo = data()
    protocol = {'question':'Какой вклад дают индивидуальная сила игроков и командные характеристики, включая опыт совместной игры?',
        'training_end_exclusive':'2025-07-01','validation_end_exclusive':'2026-01-01',
        'main_method':'Pairwise ranking of two own-team feature vectors. No mirror augmentation, no negated difference vectors.',
        'objective':'PairLogit', 'winner':'argmax of the two raw scores; exact tie chooses the lower stable team ID and is counted separately',
        'prediction_time':'After pre-match veto, before first map',
        'seeds':SEEDS, 'split_counts':{p:int(m.sum()) for p,m in masks.items()},
        'selection_rule':'The primary research method is own-team ranking with 46 inputs, fixed to measure individual-strength distribution and shared-roster experience. Rank40 is a measured control, not a post-hoc reason to remove the research factors. Winner accuracy is the primary quality metric. Classifier remains a measured comparison. Main seed42 fixed before fits, no ranking superiority presumed.',
        'training':'Depth6, learning_rate .04, L2=5, max1000 trees, early stopping100 on validation PairLogit. The checkpoint uses the smooth training objective. Accuracy evaluates winner selection. The 46-input research representation is fixed before fitting; no post-hoc configuration selection.',
        'additional_individual':EXTRA_I, 'additional_joint_experience':EXTRA_S,
        'primary_importance':'Leave whole group out, retrain under the same protocol, evaluate paired match loss and accuracy changes. Individual I vs Team T+S, C always retained. Joint-experience S separately tested.',
        'secondary_importance':'LossFunctionChange of the selected own-team ranker on the full validation ranking Pool; approximate fixed-tree feature exclusion, not refitting.',
        'elo_k':k, 'protected_hashes':protected_paths(),
        'source_hashes':{str(p):sha256_file(p) for p in [Path(__file__),Path('src/team_ranking.py'),Path('src/research_team_features.py')]},
        'old_results_manifest_sha256':sha256_file(OLD/'results_manifest.json'),
        'test_status':'Retrospective test, previously examined. Not a new independent holdout.',
        'availability':'History sorted by match start time. End/publication timestamps not archived. Historical lineup/rank/veto reconstructed from pages.',
    }
    write_json(out/'protocol.json', protocol)
    tabs = table_frames(f, masks)
    for part, tab in tabs.items():
        old = pd.read_csv(OLD/f'{part}_predictions.csv')
        np.testing.assert_array_equal(tab.match_id, old.match_id)
        for col in ['elo','logistic',*[f'classification_full_s{s}' for s in SEEDS]]:
            tab[col] = old[col]
    for seed in SEEDS:
        key=f'ranking_40_s{seed}'
        _, pp, _, _=train_ranker(f, y, masks, list(TEAM_MODEL_FEATURES), seed, key, out)
        for part in tabs:
            tabs[part][key]=pp[part]
            tabs[part].to_csv(out/f'{part}_predictions.csv',index=False)
    if protected_paths()!=protocol['protected_hashes']:
        raise RuntimeError('Protected original inputs changed')


def extended(out):
    from src.feature_engineering import normalize_lineups, normalize_player_stats
    from src.research_team_features import build_research_team_features
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    if (out/'selection.json').exists():
        raise FileExistsError('Completed experiment is immutable')
    if protected_paths()!=protocol['protected_hashes']:
        raise RuntimeError('Protected inputs changed')
    f,masks,y,_,_=data()
    clean=Path('data/interim/hltv_final_clean')
    lineups=normalize_lineups(pd.read_csv(clean/'match_lineups.csv'),set(f.match_id.astype(int)))
    if lineups.empty:
        raise ValueError('No historical lineups survived normalization')
    stats=normalize_player_stats(pd.read_csv(clean/'map_player_stats.csv'),f)
    extra=build_research_team_features(f,lineups,stats)
    for side in ['team1','team2']:
        if extra[f'{side}_lineup_player_rating_max'].notna().mean()<.5:
            raise ValueError('Unexpectedly low complete-lineup player history coverage')
    f=f.merge(extra,on='match_id',how='left',validate='one_to_one')
    f.to_csv(out/'extended_features.csv',index=False)
    expanded=list(TEAM_MODEL_FEATURES)+EXTRA_I+EXTRA_S
    # Собственные признаки сохраняют уровни показателей, которые теряются в разностях.
    # Сравнивается весь подход, а не только функция потерь.
    tabs={p:pd.read_csv(out/f'{p}_predictions.csv') for p in ['validation','test']}
    for seed in SEEDS:
        key=f'ranking_46_s{seed}'
        _,pp,_,_=train_ranker(f,y,masks,expanded,seed,key,out)
        for p in tabs:
            tabs[p][key]=pp[p]
            tabs[p].to_csv(out/f'{p}_predictions.csv',index=False)
    means={n:{'accuracy':float(np.mean([json.loads((out/f'ranking_{n}_s{s}.json').read_text())['metrics']['validation']['accuracy'] for s in SEEDS])),
              'pair_loss':float(np.mean([json.loads((out/f'ranking_{n}_s{s}.json').read_text())['metrics']['validation']['log_loss'] for s in SEEDS]))}
           for n in [40,46]}
    selection={'kind':'own_team_ranking','feature_count':46,'main_seed':42,
        'configuration_validation':means,'rule':protocol['selection_rule'],
        'research_question_extension':46,
        'interpretation_configuration':46,
        'validation_accuracy_best_control':min([40,46],key=lambda n:(-means[n]['accuracy'],means[n]['pair_loss']))}
    # Набор из 46 входов включает показатели исследуемых индивидуальных различий
    # и совместного опыта. Вариант с 40 входами проверяет пользу их добавления.
    # Он не заменяет исследовательскую модель, в которой эти показатели измерены.
    # Сохраняем и результат сравнения конфигураций, и полную модель.
    write_json(out/'selection.json',selection)
    comp=[]
    for p,tab in tabs.items():
        for col in tab.columns[5:]:
            mm=metric_row(tab.team1_win.to_numpy(),tab[col].to_numpy())
            if col.startswith('ranking_'):
                mm.update(json.loads((out/f'{col}.json').read_text())['metrics'][p])
            comp.append({'split':p,'model':col,**mm,
                         'n':len(tab),'exact_ties':int(tab[col].eq(.5).sum())})
    pd.DataFrame(comp).to_csv(out/'comparison_metrics.csv',index=False)
    # Зафиксированный набор признаков включает проверяемые гипотезы.
    families={'I':group_columns(['individual'])+EXTRA_I,
              'T':group_columns(['team']), 'S':group_columns(['cohesion'])+EXTRA_S,
              'C':group_columns(['controls'])}
    groups={'C':families['C'],'CI':families['C']+families['I'],
            'CTS':families['C']+families['T']+families['S'],
            'CTI':families['C']+families['T']+families['I'],
            'CTIS':expanded}
    group_records=[]
    for subset,cols in groups.items():
        for seed in SEEDS:
            key=f'ranking_{subset}_s{seed}'
            if subset=='CTIS':
                record=json.loads((out/f'ranking_46_s{seed}.json').read_text())
                pp={p:tabs[p][f'ranking_46_s{seed}'].to_numpy() for p in tabs}
            else:
                _,pp,_,record=train_ranker(f,y,masks,cols,seed,key,out)
            group_records.append({'subset':subset,'seed':seed,'feature_count':len(cols),
                **{f'{p}_{m}':v for p,mm in record['metrics'].items() for m,v in mm.items()}})
            for p in tabs:
                tabs[p][key]=pp[p]
                tabs[p].to_csv(out/f'{p}_group_predictions.csv',index=False)
            pd.DataFrame(group_records).to_csv(out/'group_metrics.csv',index=False)
    contrasts=[]
    for p,tab in tabs.items():
        yy=tab.team1_win.to_numpy()
        for s in SEEDS:
            losses={g:point_loss(yy,tab[f'ranking_{g}_s{s}']) for g in groups}
            correct={}
            for g in groups:
                pp=tab[f'ranking_{g}_s{s}'].to_numpy()
                decisions=(pp>.5).astype(int)
                ties=pp==.5
                decisions[ties]=(tab.team1_id.to_numpy()[ties]<tab.team2_id.to_numpy()[ties]).astype(int)
                correct[g]=decisions==yy
            for name,without in [('individual','CTS'),('team','CI'),('joint_experience','CTI')]:
                contrasts.append({'split':p,'seed':s,'group':name,'without':without,
                    **weekly_interval(losses[without]-losses['CTIS'],tab.match_datetime_utc),
                    'accuracy_gain':float(np.mean(correct['CTIS'].astype(float)-correct[without].astype(float)))})
    pd.DataFrame(contrasts).to_csv(out/'group_contributions.csv',index=False)
    model=CatBoostRanker(); model.load_model(str(out/'ranking_46_s42.cbm'))
    pool=team_ranking_pool(f.loc[masks['validation']],y[masks['validation']],expanded)
    importance=model.get_feature_importance(pool,type='LossFunctionChange',thread_count=8)
    owner={col:group for group,cols in families.items() for col in cols}
    pd.DataFrame({'feature':expanded,'group':[owner[c] for c in expanded],
                  'validation_LossFunctionChange':importance}).sort_values(
                  'validation_LossFunctionChange',ascending=False).to_csv(out/'feature_importance.csv',index=False)
    example(f.loc[masks['test']].reset_index(drop=True),tabs['test'],out)
    differences={}
    for p,tab in tabs.items():
        differences[p]={
            'accuracy_rank46_minus_classification':weekly_interval(
                (rank_decisions_from_table(tab,'ranking_46_s42')==tab.team1_win).astype(float)-
                ((tab.classification_full_s42>=.5)==tab.team1_win).astype(float),tab.match_datetime_utc),
            'loss_classifier_minus_rank46':weekly_interval(
                point_loss(tab.team1_win,tab.classification_full_s42)-
                point_loss(tab.team1_win,tab.ranking_46_s42),tab.match_datetime_utc)}
    write_json(out/'comparison_intervals.json',differences)
    if protected_paths()!=protocol['protected_hashes']:
        raise RuntimeError('Protected inputs changed')
    write_json(out/'results_manifest.json',{'files':{p.name:sha256_file(p) for p in sorted(out.glob('*')) if p.is_file() and p.name!='results_manifest.json'}})
    print(json.dumps(selection,ensure_ascii=False),flush=True)


def example(frame, tab, out):
    # Правило не зависит от исхода: первый BO3 с полной историей, где преимущества
    # по сильнейшему игроку и среднему совместному опыту направлены противоположно.
    valid=frame.bo.eq(3)
    for side in ['team1','team2']:
        valid &= frame[f'{side}_lineup_history_coverage'].eq(1)
        valid &= frame[f'{side}_team_rank'].between(1,30)
        valid &= frame[f'{side}_roster_pair_experience_90d'].gt(0)
    dstar=frame.team1_lineup_player_rating_max-frame.team2_lineup_player_rating_max
    djoint=frame.team1_roster_pair_experience_90d-frame.team2_roster_pair_experience_90d
    valid &= dstar*djoint<0
    ids=np.flatnonzero(valid.to_numpy())
    if not len(ids):
        write_json(out/'illustrative_match.json',{'available':False,'reason':'No match meets the pre-specified descriptive rule'})
        return
    i=int(ids[0]); r=frame.iloc[i]
    fields=['lineup_player_rating_mean',*EXTRA_I,*EXTRA_S,'team_rank','winrate_last_10','roster_overlap_prev_ratio']
    record={'available':True,'rule':'First chronological test BO3 with both HLTV ranks1-30, full player history, positive pair history and strongest-player difference opposite joint-experience difference. Outcome never used in selection.',
        'match_id':int(r.match_id),'time':str(r.match_datetime_utc),'team1':str(r.team1),'team2':str(r.team2),
        'source_url':str(r.source_url),'winner':str(r.team1 if r.team1_win==1 else r.team2),
        'score':f'{int(r.team1_score)}:{int(r.team2_score)}',
        'team1_features':{c:float(r[f'team1_{c}']) for c in fields},
        'team2_features':{c:float(r[f'team2_{c}']) for c in fields},
        'probabilities_team1':{g:float(tab.iloc[i][f'ranking_{g}_s42']) for g in ['CI','CTS','CTI','CTIS']}}
    write_json(out/'illustrative_match.json',record)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['base','extended'],required=True)
    p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args()
    (base if args.phase=='base' else extended)(args.output)


if __name__=='__main__':
    main()
