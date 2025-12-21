from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

import pandas as pd
from eda_cli.core import compute_quality_flags


def test_constant_columns_flag():
    df = pd.DataFrame({"A": [1, 1, 1], "B": [2, 3, 4]})
    flags = compute_quality_flags(df)
    assert flags["has_constant_columns"] is True


def test_no_constant_columns():
    df = pd.DataFrame({"X": [1, 2, 3], "Y": ["a", "b", "c"]})
    flags = compute_quality_flags(df)
    assert flags["has_constant_columns"] is False


def test_high_cardinality_categorical():
    df = pd.DataFrame({"cat": [f"cat_{i}" for i in range(100)]})
    flags = compute_quality_flags(df, max_cardinality_threshold=50)
    assert flags["has_high_cardinality_categoricals"] is True


def test_no_high_cardinality():
    df = pd.DataFrame({"cat": ["A", "B", "A"]})
    flags = compute_quality_flags(df, max_cardinality_threshold=10)
    assert flags["has_high_cardinality_categoricals"] is False


def test_suspicious_id_duplicates():
    df = pd.DataFrame({"user_id": [1, 2, 2, 4]})
    flags = compute_quality_flags(df)
    assert flags["has_suspicious_id_duplicates"] is True


def test_no_id_duplicates():
    df = pd.DataFrame({"order_id": [1, 2, 3, 4]})
    flags = compute_quality_flags(df)
    assert flags["has_suspicious_id_duplicates"] is False


def test_many_zero_values():
    df = pd.DataFrame({"mostly_zeros": [0, 0, 0, 0, 0, 1]})
    flags = compute_quality_flags(df)
    assert flags["has_many_zero_values"] is True


def test_missing_table_and_quality_flags():
    """Исправленный тест: работает с pd.DataFrame"""
    df = pd.DataFrame({
        "age": [25, None, 30, 35],
        "city": ["Moscow", "London", None, "Berlin"],
        "height": [175, 180, 178, 182]
    })
    flags = compute_quality_flags(df)
    assert flags["has_missing"] is True
   