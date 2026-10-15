"""Strict chronological data partitioning without future-data leakage."""

from typing import Tuple
import pandas as pd
from shieldml.config import settings
from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.ingestion.split")


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = settings.TRAIN_RATIO,
    val_ratio: float = settings.VAL_RATIO,
    test_ratio: float = settings.TEST_RATIO,
    time_col: str = "timestamp",
    target_col: str = "is_fraud",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a dataset strictly chronologically into Train, Validation, and Test sets.

    Chronological splitting mirrors real-world production environments where models
    predict future transactions based exclusively on past transactions.

    Args:
        df: Input DataFrame.
        train_ratio: Fraction for training set (default 0.70).
        val_ratio: Fraction for validation set (default 0.15).
        test_ratio: Fraction for testing set (default 0.15).
        time_col: Column name containing datetime timestamps.
        target_col: Target binary column (e.g., 'is_fraud').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(f"Split ratios must sum to 1.0, got: {total_ratio}")

    # Ensure chronological order
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col).reset_index(drop=True)

    n = len(df)
    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end_idx].copy().reset_index(drop=True)
    val_df = df.iloc[train_end_idx:val_end_idx].copy().reset_index(drop=True)
    test_df = df.iloc[val_end_idx:].copy().reset_index(drop=True)

    # Verification of chronological bounds (zero temporal overlap)
    train_max = train_df[time_col].max()
    val_min = val_df[time_col].min()
    val_max = val_df[time_col].max()
    test_min = test_df[time_col].min()

    assert train_max <= val_min, f"Temporal leakage detected: train_max ({train_max}) > val_min ({val_min})"
    assert val_max <= test_min, f"Temporal leakage detected: val_max ({val_max}) > test_min ({test_min})"

    # Log partition details
    logger.info(f"Chronological Split Complete:")
    logger.info(f"  Train: {len(train_df)} rows ({train_df[time_col].min()} to {train_max}) | "
                f"Frauds: {train_df[target_col].sum() if target_col in train_df else 'N/A'}")
    logger.info(f"  Val:   {len(val_df)} rows ({val_min} to {val_max}) | "
                f"Frauds: {val_df[target_col].sum() if target_col in val_df else 'N/A'}")
    logger.info(f"  Test:  {len(test_df)} rows ({test_min} to {test_df[time_col].max()}) | "
                f"Frauds: {test_df[target_col].sum() if target_col in test_df else 'N/A'}")

    return train_df, val_df, test_df
