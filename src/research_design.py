"""Canonical, disjoint feature families for the player-versus-team study.

These are observational predictive groups, not causal factors. Cohesion is
represented only by roster continuity; it is not a measurement of communication.
"""
from src.modeling import MODEL_FEATURES

FEATURE_GROUPS = {
    "individual": {
        "label": "Индивидуальные характеристики (I)",
        "features": {
            "diff_lineup_player_rating_mean": "Средний исторический рейтинг игроков текущего состава",
            "diff_lineup_player_adr_mean": "Средний исторический урон за раунд (ADR) игроков состава",
            "diff_lineup_player_kast_mean": "Средняя историческая доля полезных раундов (KAST) игроков",
            "diff_lineup_player_opening_diff_mean": "Средний исторический баланс первых убийств и смертей за карту",
        },
    },
    "team": {
        "label": "Командная история (T)",
        "features": {
            "diff_elo_pre": "Рейтинг Elo до матча",
            "diff_rank": "Обратная разность мест в рейтинге HLTV: rank(B) − rank(A)",
            "diff_days_since_last_match": "Число дней после предыдущего матча",
            "diff_activity_7d": "Число матчей за предыдущие 7 дней",
            "diff_activity_30d": "Число матчей за предыдущие 30 дней",
            "diff_activity_90d": "Число матчей за предыдущие 90 дней",
            "diff_overall_winrate": "Доля побед во всей доступной истории команды",
            "diff_winrate_last_5": "Доля побед в последних 5 матчах",
            "diff_winrate_last_10": "Доля побед в последних 10 матчах",
            "diff_winrate_last_20": "Доля побед в последних 20 матчах",
            "diff_win_streak": "Длина текущей серии побед",
            "diff_loss_streak": "Длина текущей серии поражений",
            "diff_avg_opp_elo_last_10": "Средний предматчевый Elo последних 10 соперников",
            "diff_h2h_wins_all": "Число побед в предыдущих личных встречах",
            "diff_h2h_wins_last5": "Число побед в последних 5 личных встречах",
            "diff_avg_map_wr_before": "Средняя историческая доля побед на картах из предматчевого veto",
            "diff_avg_map_ct_wr_before": "Средняя историческая доля выигранных раундов в защите на этих картах",
            "diff_avg_map_t_wr_before": "Средняя историческая доля выигранных раундов в атаке на этих картах",
            "diff_veto_pick_rate_before": "Историческая частота выбора соответствующих карт",
            "diff_veto_remove_rate_before": "Историческая частота исключения соответствующих карт",
            "diff_veto_leftover_rate_before": "Историческая частота оставления соответствующих карт",
        },
    },
    "cohesion": {
        "label": "Сохранность состава (S): ограниченный показатель сыгранности",
        "features": {
            "diff_roster_overlap_prev": "Число игроков, сохранившихся относительно предыдущего матча команды",
            "diff_roster_overlap_prev_ratio": "Доля сохранившихся игроков в текущем составе",
        },
    },
    "controls": {
        "label": "Контекст и полнота наблюдений (C)",
        "features": {
            "bo1": "Индикатор серии из одной карты",
            "bo3": "Индикатор серии до двух побед",
            "bo5": "Индикатор серии до трёх побед",
            "is_lan": "Индикатор очного матча",
            "is_online": "Индикатор онлайн-матча",
            "rank_available": "Оба места HLTV известны",
            "diff_matches_before": "Число предыдущих матчей в собранной истории",
            "diff_roster_size": "Число известных игроков текущего состава",
            "diff_lineup_history_coverage": "Доля игроков состава с историей не менее 5 карт",
            "diff_lineup_players_with_history": "Число игроков с историей не менее 5 карт",
            "diff_lineup_player_maps_played_mean": "Средняя глубина доступной истории игроков в картах",
            "diff_avg_map_count_before": "Среднее число предыдущих игр на картах из veto",
            "diff_series_maps_known": "Число выбранных до матча карт, для которых у команды есть история",
        },
    },
}


def columns_for(groups):
    chosen = {f for g in groups for f in FEATURE_GROUPS[g]["features"]}
    return [f for f in MODEL_FEATURES if f in chosen]


def validate_groups():
    flat = [f for group in FEATURE_GROUPS.values() for f in group["features"]]
    if len(flat) != len(set(flat)) or set(flat) != set(MODEL_FEATURES):
        raise ValueError("Research groups must partition the complete model allow-list")


validate_groups()
