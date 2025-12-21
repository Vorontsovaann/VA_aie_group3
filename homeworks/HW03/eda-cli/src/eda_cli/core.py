# src/eda_cli/core.py
import pandas as pd
from typing import Dict, Any, List
import numpy as np


def compute_quality_flags(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Вычисляет различные флаги качества данных.
    
    Args:
        df: DataFrame для анализа
        **kwargs: дополнительные параметры (например, пороги)
    
    Returns:
        Словарь с флагами качества и связанными метриками
    """
    flags = {
        "has_missing": False,
        "has_duplicate_rows": False,
        "has_constant_column": False,
        "has_high_cardinality_categories": False,
        "has_suspicious_id_duplicates": False,
        "has_many_zero_values": False,
        "quality_score": 1.0,
    }
    metrics = {}
    
    # 1. Проверка на пропуски
    missing_counts = df.isnull().sum()
    missing_shares = missing_counts / len(df)
    metrics["missing_shares"] = missing_shares.to_dict()
    flags["has_missing"] = missing_counts.any()
    
    # 2. Проверка на дубликаты строк
    duplicate_count = df.duplicated().sum()
    metrics["duplicate_rows_count"] = int(duplicate_count)
    flags["has_duplicate_rows"] = duplicate_count > 0
    
    # 3. НОВАЯ: Проверка на константные колонки
    constant_columns = []
    for col in df.columns:
        if df[col].nunique() == 1:
            constant_columns.append(col)
    flags["has_constant_column"] = len(constant_columns) > 0
    metrics["constant_columns"] = constant_columns
    
    # 4. НОВАЯ: Проверка на высокую кардинальность категориальных признаков
    # Получаем порог из kwargs или используем значение по умолчанию
    high_cardinality_threshold = kwargs.get('high_cardinality_threshold', 100)
    
    high_cardinality_cols = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_count = df[col].nunique()
        if unique_count > high_cardinality_threshold:
            high_cardinality_cols.append((col, unique_count))
    flags["has_high_cardinality_categories"] = len(high_cardinality_cols) > 0
    metrics["high_cardinality_columns"] = high_cardinality_cols
    
    # 5. НОВАЯ: Проверка дубликатов ID (предполагаем, что есть колонка с 'id' в названии)
    id_columns = [col for col in df.columns if 'id' in col.lower()]
    suspicious_id_cols = []
    
    for id_col in id_columns:
        duplicate_ids = df[id_col].duplicated().sum()
        if duplicate_ids > 0:
            suspicious_id_cols.append((id_col, duplicate_ids))
    
    flags["has_suspicious_id_duplicates"] = len(suspicious_id_cols) > 0
    metrics["suspicious_id_columns"] = suspicious_id_cols
    
    # 6. НОВАЯ: Проверка на много нулевых значений в числовых колонках
    # Получаем порог из kwargs или используем значение по умолчанию
    zero_threshold = kwargs.get('zero_threshold', 0.5)
    
    many_zero_cols = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        zero_count = (df[col] == 0).sum()
        zero_share = zero_count / len(df)
        if zero_share > zero_threshold:
            many_zero_cols.append((col, zero_share))
    
    flags["has_many_zero_values"] = len(many_zero_cols) > 0
    metrics["many_zero_columns"] = many_zero_cols
    
    # Расчёт интегрального показателя качества
    penalty = 0.0
    
    if flags["has_missing"]:
        # Штраф пропорционален максимальной доле пропусков
        max_missing_share = missing_shares.max()
        penalty += max_missing_share * 0.3
    
    if flags["has_duplicate_rows"]:
        duplicate_share = duplicate_count / len(df)
        penalty += min(duplicate_share * 0.5, 0.2)
    
    if flags["has_constant_column"]:
        penalty += 0.1 * len(constant_columns) / len(df.columns)
    
    if flags["has_high_cardinality_categories"]:
        penalty += 0.15
    
    if flags["has_suspicious_id_duplicates"]:
        penalty += 0.2
    
    if flags["has_many_zero_values"]:
        penalty += 0.1
    
    flags["quality_score"] = max(0.0, 1.0 - penalty)
    flags["metrics"] = metrics
    
    return flags