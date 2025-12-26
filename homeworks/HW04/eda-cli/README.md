# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Тесты

```bash
uv run pytest -q
```
# eda-cli

Mini EDA tool with CLI interface.

## Commands

- `eda-cli overview <csv>` — quick quality flags
- `eda-cli report <csv> [OPTIONS]` — full markdown report with plots

## New `report` options

- `--max-hist-columns INT`: limit number of histograms (default: 5)
- `--top-k-categories INT`: top categories per bar plot (default: 10)
- `--title TEXT`: report title (default: "EDA Report")
- `--min-missing-share FLOAT`: threshold for missing-value warnings (default: 0.1)

## Example

```bash
uv run eda-cli report data/example.csv \
  --out-dir my_report \
  --max-hist-columns 3 \
  --top-k-categories 5 \
  --title "My Custom EDA" \
  --min-missing-share 0.2