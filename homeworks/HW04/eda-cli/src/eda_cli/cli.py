from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df)

    # 2. Качество в целом
    quality_flags = compute_quality_flags(summary, missing_df)

    # 3. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 4. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# EDA-отчёт\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write("См. файлы в папке `top_categories/`.\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        f.write("См. файлы `hist_*.png`.\n")

    # 5. Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")


if __name__ == "__main__":
    app()

# src/eda_cli/cli.py

import click
import pandas as pd
from pathlib import Path
from .core import compute_quality_flags, get_categorical_summary, get_numeric_summary
from .viz import plot_histograms, plot_category_bars

@click.group()
def cli():
    pass

@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
def overview(csv_path):
    df = pd.read_csv(csv_path)
    flags = compute_quality_flags(df)
    for k, v in flags.items():
        click.echo(f"{k}: {v}")

@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--out-dir", default="report", help="Output directory for report files.")
@click.option("--max-hist-columns", default=5, help="Max number of numeric columns to plot histograms for.")
@click.option("--top-k-categories", default=10, help="Number of top categories to show per categorical column.")
@click.option("--title", default="EDA Report", help="Title of the report.")
@click.option("--min-missing-share", default=0.1, help="Threshold for highlighting columns with many missing values.")
def report(csv_path, out_dir, max_hist_columns, top_k_categories, title, min_missing_share):
    df = pd.read_csv(csv_path)
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)

    # 1. Качество данных
    flags = compute_quality_flags(df, max_cardinality_threshold=50)  # используем threshold как в эвристике

    # Определяем "плохие" колонки по пропускам
    missing_by_col = df.isnull().mean()
    bad_missing_cols = missing_by_col[missing_by_col >= min_missing_share].index.tolist()

    # 2. Генерация отчёта
    report_lines = []
    report_lines.append(f"# {title}\n")
    report_lines.append("## Quality Flags\n")
    for k, v in flags.items():
        report_lines.append(f"- {k}: {v}")
    if bad_missing_cols:
        report_lines.append(f"\n## Columns with missing share ≥ {min_missing_share}")
        report_lines.append(", ".join(bad_missing_cols))

    # 3. Визуализации
    numeric_cols = df.select_dtypes(include='number').columns[:max_hist_columns]
    if len(numeric_cols) > 0:
        plot_histograms(df[numeric_cols], out_path / "histograms.png")
        report_lines.append("\n![Histograms](histograms.png)")

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        top_vals = df[col].value_counts().head(top_k_categories)
        plot_category_bars(top_vals, out_path / f"bar_{col}.png")
        report_lines.append(f"\n### {col}\n![{col}](bar_{col}.png)")

    # Сохранение
    with open(out_path / "report.md", "w") as f:
        f.write("\n".join(report_lines))

    click.echo(f"Report saved to {out_path}/report.md")