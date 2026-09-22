# Прогнозирование результатов матчей CS2

Курсовой проект на Python для прогнозирования победителя матча Counter-Strike 2 по статистике HLTV. В проекте реализован сбор данных, подготовка признаков, обучение моделей и получение прогноза через командную строку. Отдельная часть работы посвящена тому, какой вклад в прогноз дают показатели игроков, командная история и опыт совместной игры.

Основная модель использует CatBoostRanker. Для сравнения реализованы Elo, логистическая регрессия, CatBoostClassifier и линейное парное ранжирование. Данные обрабатываются с помощью pandas и NumPy, страницы HLTV разбираются через Beautiful Soup и Playwright.

## Установка

Понадобятся Git и Python 3.12. Команды ниже предназначены для PowerShell в Windows.

```powershell
git clone https://github.com/jmotmot0/CS2-Game-Predictor.git
cd CS2-Game-Predictor
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Пример запуска

В репозитории уже есть обученная модель и исторические признаки. Для проверки работы не нужно заново собирать данные или запускать обучение.

```powershell
.\.venv\Scripts\python.exe -m src.predict_team_ranking --verify
.\.venv\Scripts\python.exe -m src.predict_team_ranking --match-id 2389256
```

Первая команда проверяет целостность файлов модели. Вторая воспроизводит прогноз матча Ninjas in Pyjamas против Liquid по сохранённым предматчевым данным. Программа выводит JSON с оценками обеих команд, вероятностями и выбранным победителем. В этом примере прогнозируется победа Liquid с вероятностью около 53,32%.

## Работа со своими данными

Сбор списка матчей реализован в [load_hltv_matches.py](src/load_hltv_matches.py), подробной статистики - в [enrich_hltv_matches.py](src/enrich_hltv_matches.py). После очистки данные используются для расчёта истории команд и игроков. Параметры каждого модуля можно посмотреть через `--help`.

Для прогноза нового матча нужны очищенные таблицы `matches_final.csv`, `match_lineups.csv`, `match_maps.csv`, `map_player_stats.csv` и `veto_steps.csv` в папке `data/interim/hltv_final_clean/`. Исходные выгрузки и браузерные профили не загружаются в Git. Команде прогноза передаются время матча, ID команд и доступные предматчевые сведения о составах, рейтингах и выборе карт.

```powershell
.\.venv\Scripts\python.exe -m src.predict_team_ranking --help
```

Признаки рассчитываются по прошлым матчам, без результата прогнозируемой серии. Набор карт берётся из предматчевого выбора, а не из списка фактически сыгранных карт.

## Структура проекта

```text
src/                сбор данных, признаки, модели и прогноз
tests/              тесты и HTML-примеры для проверки парсеров
scripts/            запуск браузера и проверка пересборки признаков
artifacts/          обученные модели и результаты экспериментов
data/               локальные выгрузки и промежуточные таблицы
requirements.txt    зависимости проекта
```

Основная команда прогноза находится в [predict_team_ranking.py](src/predict_team_ranking.py), подготовка входов ранкера - в [team_ranking.py](src/team_ranking.py), расчёт исторических признаков - в [feature_engineering.py](src/feature_engineering.py) и [research_team_features.py](src/research_team_features.py).

Результаты сравнения моделей сохранены в [таблице экспериментов CatBoost](artifacts/supervisor_revision_2026-09-30/comparison_metrics.csv) и [таблице линейного ранжирования](artifacts/linear_ranking_2026-10-02/comparison_metrics.csv). Исследовательские расчёты находятся в [supervisor_experiment.py](src/supervisor_experiment.py) и [linear_team_ranking.py](src/linear_team_ranking.py).

## Тесты

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Тесты проверяют парсинг страниц, расчёт признаков, защиту от использования будущих данных и работу прогноза. Проверки также запускаются в GitHub Actions. В чистом клоне один тест старой модели пропускается, если её локальный файл отсутствует.
