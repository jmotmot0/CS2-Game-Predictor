# CS2 Match Outcome Prediction

Курсовой программный проект для оценки вероятности победы команды в
профессиональной серии Counter-Strike 2. Источник исторических данных — HLTV.

Проект реализует конвейер, воспроизводимый при наличии исходных данных
и указанных зависимостей:

```text
сбор HLTV → обогащение страниц матчей → очистка связанных таблиц
→ point-in-time признаки → train/validation/test → CatBoost → JSON-прогноз
```

## Что реализовано

- сбор завершённых матчей и подробных match/map pages;
- пять нормализованных таблиц: матчи, составы, veto, карты и статистика игроков;
- 40 разрешённых модели предматчевых признаков;
- одновременные Elo/H2H snapshot обеих команд до обновления результатом;
- зеркальное дополнение train-выборки и строго симметричный прогноз:
  `P(A, B) = 1 - P(B, A)`;
- player, roster и map-pool history только по более ранним сериям;
- `overall_winrate` считается по всей прошлой истории команды, а окна 5/10/20
  матчей остаются отдельными краткосрочными признаками;
- текущий map pool строится по `picked` и `left_over` из veto, включая
  несыгранную решающую карту; сценарий предполагает доступность veto до начала
  серии, а фактически сыгранные карты используются для обновления истории;
- хронологический split без случайного перемешивания;
- сравнение historical prior, Elo, LogisticRegression,
  HistGradientBoosting, CatBoost и Platt-калибровки;
- модельный артефакт, схема признаков, таблицы метрик и графики;
- быстрый CLI прогнозирования с информацией о полноте входных данных;
- расширяющийся walk-forward backtest по трём кварталам;
- мониторинг сдвига распределений признаков (PSI и доля пропусков);
- offline HTML-fixtures для парсера HLTV и CI-проверка каждого изменения;
- автоматические тесты временной границы и воспроизводимости.

## Установка

Рекомендуется Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Браузер Playwright нужен только для повторного обогащения страниц HLTV:

```powershell
python -m playwright install chromium
```

Готовые clean-таблицы, витрина и модель позволяют выполнять обучение и прогноз
без Playwright.

### Что доступно после клонирования

Репозиторий содержит исходный код, тесты, небольшие HTML-примеры для проверки
парсера и настройки CI. Данные HLTV, обученная модель, каталог `artifacts`,
личные документы для защиты и локальные профили браузера в Git не включены.

После установки зависимостей можно запустить `python -m pytest -q` без сети
и выгрузок HLTV. Проверка сохранённой модели будет пропущена, если её артефакты
отсутствуют. Для полного обучения нужны очищенные таблицы, а для быстрого
прогноза — `catboost_model.cbm`, `feature_schema.json` и `inference_state.json`
в каталоге `artifacts`. Они создаются при выполнении этапов ниже. Само
клонирование репозитория не загружает эти файлы и не воспроизводит метрики.

## Данные

Крупные CSV не хранятся в Git. Рабочий набор находится в
`data/interim/hltv_final_clean` и содержит:

| Таблица | Строк |
|---|---:|
| `matches_final.csv` | 25 333 |
| `match_lineups.csv` | 253 088 |
| `veto_steps.csv` | 169 704 |
| `match_maps.csv` | 67 863 |
| `map_player_stats.csv` | 521 679 |

Витрина `data/processed/features_dataset.csv` содержит 25 333 строки и 146
столбцов. В модель передаются только 40 колонок из явного `MODEL_FEATURES` в
`src/modeling.py`; target, счёт, названия и URL туда попасть не могут.

## Основной сценарий

### 1. Сбор списка матчей

```powershell
python -m src.load_hltv_matches --help
```

HLTV может ограничивать автоматические запросы, поэтому сбор поддерживает паузы,
повторные попытки и локальный профиль браузера.

### 2. Обогащение матчей

```powershell
python -m src.enrich_hltv_matches `
  --input data/raw/matches_raw.csv `
  --out-dir data/interim/hltv_enriched `
  --checkpoint-every 25 `
  --resume
```

Скрипт фиксирует только полностью разобранные матчи, атомарно обновляет CSV и
пишет ошибки в `failed_matches.csv`. При повторном запуске `--resume` загружает
контрольную точку, удаляет незавершённые дочерние строки и пропускает уже
готовые `match_id`. Для Chrome, запущенного с remote debugging, можно передать
`--cdp-url http://127.0.0.1:9222`; без него используется локальный постоянный
профиль Playwright.

### 3. Очистка связанных таблиц

```powershell
python -m src.build_clean_final_hltv_dataset `
  --raw-csv data/raw/matches_raw.csv `
  --enriched-dirs data/interim/hltv_enriched_latest_1000 data/interim/hltv_enriched_rest `
  --output-dir data/interim/hltv_final_clean `
  --cutoff-date 2023-10-16
```

### 4. Генерация исторических признаков

```powershell
python -m src.build_features `
  --clean-dir data/interim/hltv_final_clean `
  --output-csv data/processed/features_dataset.csv
```

Сначала для всех матчей с одним timestamp фиксируются состояния обеих команд,
и только затем их результаты обновляют историю. Изменение результата матча не
может изменить признаки этого же матча или более ранних наблюдений.

Ограничение: страницы собраны ретроспективно, время публикации составов и veto
и точное время завершения серий не сохранены. Более раннее начало другой
серии не гарантирует, что она уже закончилась. Поэтому порядок обновления
защищает от конкретных ошибок использования текущего исхода, но не доказывает
полное отсутствие утечек через доступность данных. Основные метрики относятся
к сценарию после veto с этими допущениями.

### 5. Обучение

```powershell
python -m src.train_models `
  --features data/processed/features_dataset.csv `
  --clean-dir data/interim/hltv_final_clean `
  --output-dir artifacts
```

Границы по умолчанию:

- train: раньше `2025-07-01` — 15 674 матча;
- validation: `2025-07-01`–`2025-12-31` — 5 421 матч;
- test: начиная с `2026-01-01` — 4 238 матчей.

Свежий полный запуск с seed 42 дал следующие test-метрики:

| Модель | Accuracy | ROC-AUC | LogLoss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Historical prior | 0.559 | 0.500 | 0.686 | 0.247 | 0.006 |
| Elo | 0.624 | 0.681 | 0.635 | 0.223 | 0.031 |
| Logistic regression | 0.667 | 0.723 | 0.605 | 0.209 | 0.020 |
| HistGradientBoosting | 0.669 | 0.726 | 0.604 | 0.209 | 0.017 |
| CatBoost | 0.668 | 0.727 | 0.602 | 0.208 | 0.014 |
| CatBoost + Platt | 0.668 | 0.727 | 0.602 | 0.208 | 0.016 |

CatBoost даёт лучший ROC-AUC, LogLoss и Brier, а также минимальный ECE среди
непостоянных моделей. У Historical prior ECE ниже, но ROC-AUC равен 0,5.
Наибольшую accuracy в этом запуске показывает HistGradientBoosting. Небольшие
различия между сильными моделями не доказывают статистически значимого
превосходства. Финальным артефактом
выбран CatBoost с симметричным усреднением двух порядков команд. Platt ухудшает
LogLoss, Brier и ECE относительно исходного CatBoost, поэтому остаётся только
контрольным экспериментом и не используется в CLI.

### 6. Проверка метрик и графики

```powershell
python -m src.evaluate_models --artifacts artifacts
python -m src.backtest_models
python -m src.monitor_drift
python -m src.audit_project
```

Первая команда повторно рассчитывает метрики из сохранённых test-предсказаний
и создаёт PNG в `artifacts/figures`. Walk-forward использует три непересекающихся
тестовых квартала и отдельное 90-дневное validation-окно перед каждым из них.
Обучающие истории пересекаются, команды повторяются, а последний квартал входит
в основной тест: полностью независимыми экспериментами эти проверки не являются.
Суммарный результат CatBoost на 9 096 прогнозах: Accuracy 0,668, ROC-AUC 0,727,
LogLoss 0,603, Brier 0,209. Мониторинг сравнивает свежий test-период с
последними 180 днями train и сообщает о PSI/изменениях пропусков; его
предупреждение является сигналом для наблюдения, а не подменой оценки качества.
Финальный аудит воспроизводит метрики, проверяет ключи таблиц, хронологию,
контрольные суммы, схему и точную симметрию модели.

### 7. Прогноз матча

```powershell
python -m src.predict_match `
  --team1-id 4608 `
  --team2-id 6667 `
  --match-time 2026-04-12T15:00:00Z `
  --bo 3 --lan `
  --team1-rank 8 --team2-rank 11 `
  --team1-players 14759,22673,18987,20127,9816 `
  --team2-players 429,18053,10394,22383,9960 `
  --maps mirage,nuke,ancient `
  --team1-picks mirage --team2-picks nuke `
  --team1-removes inferno --team2-removes dust2 `
  --decider ancient `
  --model-dir artifacts
```

Пример ответа:

```json
{
  "team1_win_probability": 0.7316705229,
  "team2_win_probability": 0.2683294771,
  "predicted_winner_team_id": 4608,
  "model": "catboost_v3_symmetric",
  "feature_coverage": 1.0,
  "features_available": 40,
  "features_total": 40,
  "warnings": [],
  "prediction_mode": "post_veto"
}
```

Если veto ещё неизвестен, можно передать допустимый набор карт вместо
`--maps`, например `--candidate-maps mirage,nuke,inferno,ancient`. CLI переберёт
все BO-размерные сочетания, вернёт среднюю вероятность и диапазон сценариев,
явно предупредив, что они считаются равновероятными.

`inference_state.json` сокращает вычисление snapshot с нескольких минут до
долей секунды. Флаг `--rebuild-history` оставлен для диагностического полного
пересчёта. Быстрый и полный пути проверены на равенство всех 40 входов модели.

## Артефакты

После обучения каталог `artifacts` содержит только необходимые результаты:

- `catboost_model.cbm` — обученная модель;
- `feature_schema.json` — порядок 40 признаков и параметры split;
- `inference_state.json` — компактное состояние истории для быстрого прогноза;
- `experiment_summary.json` — версии, параметры и результаты;
- `artifact_manifest.json` — размеры и SHA-256 исходных данных и артефактов;
- `audit_report.json` — результат полной проверки связности и воспроизводимости;
- `model_metrics.csv`, `feature_importance.csv`, `segment_metrics.csv`,
  `ablation_metrics.csv` — численные результаты;
- `test_predictions.csv` — проверяемые ответы на test-периоде;
- `walk_forward_metrics.csv`, `walk_forward_predictions.csv`,
  `walk_forward_summary.json` — диагностика устойчивости по времени;
- `drift_report.json` — мониторинг распределений 40 входов;
- `figures/` — графики оценки.

## Тесты

```powershell
python -m pytest -q
```

Тесты проверяют:

- одновременный snapshot обеих команд;
- запрет влияния текущего результата на текущие и прошлые признаки;
- матчи с одинаковым timestamp;
- отсутствие обновления map-history внутри серии;
- построение текущего map pool только из предматчевого veto, включая
  несыгранную decider-карту;
- ремонт противоречивых дат;
- хронологические границы train/validation/test;
- полное и непересекающееся покрытие 40 признаков группами абляции;
- симметрию разностных признаков;
- зеркальное дополнение train и точное дополнение вероятностей при перестановке команд;
- корректность вероятностей сохранённой модели;
- единицы времени и допустимый диапазон fast inference-state.
- различие общего winrate и окна последних 20 матчей;
- генерацию pre-veto сценариев карт;
- непересекающиеся train/validation/test-окна walk-forward;
- PSI на неизменном и сдвинутом распределениях;
- разбор сохранённых HTML-страниц матча и карты без обращения к сети.

Всего выполняется 31 автоматический тест, включая восстановление контрольной
точки обогащения без незавершённых дочерних строк.

## Структура

```text
src/
  load_hltv_matches.py              сбор списка матчей
  enrich_hltv_matches.py            lineups, veto, maps, player stats
  build_clean_final_hltv_dataset.py очистка и нормализация
  feature_engineering.py            point-in-time engine
  build_features.py                 CLI генерации витрины
  modeling.py                       признаки, split, метрики, Elo
  train_models.py                   обучение и артефакты
  evaluate_models.py                проверка результатов и графики
  backtest_models.py                расширяющийся walk-forward backtest
  monitor_drift.py                  контроль PSI и пропусков
  audit_project.py                  единая проверка данных и артефактов
  inference_state.py                быстрый snapshot для будущего матча
  build_inference_state.py          отдельная пересборка состояния
  predict_match.py                  JSON-прогноз
tests/                               автоматические проверки
plan.md                              план и результаты исходного аудита
```

Модель оценивает вероятность по публичной статистике и не является гарантией
исхода или финансовой рекомендацией.
