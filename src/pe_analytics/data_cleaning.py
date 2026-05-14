"""
data_cleaning.py
----------------
Cleans raw private equity portfolio company data.
Handles missing values, outliers, type coercion, and feature engineering.
"""

import pandas as pd
import numpy as np
from src.pe_analytics.logging_config import setup_logger

logger = setup_logger(__name__)


class DataCleaningError(Exception):
    """Raised when data cleaning fails due to invalid input."""


def load_portfolio_data(filepath: str) -> pd.DataFrame:
    """
    Load raw portfolio company data from a CSV file.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Raw DataFrame as loaded from disk.

    Raises:
        DataCleaningError: If file is not found or cannot be parsed.
    """
    logger.info(f"Loading data from {filepath}")

    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows and {df.shape[1]} columns")
        return df

    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise DataCleaningError(f"File not found: {filepath}")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise DataCleaningError(f"Failed to load data: {e}")


def clean_portfolio_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate raw PE portfolio company data.

    Steps:
        1. Validate input is not empty
        2. Coerce revenue and EBITDA to numeric
        3. Flag and fill missing values
        4. Winsorize outliers at 1st and 99th percentile
        5. Derive new features

    Args:
        df: Raw portfolio DataFrame with columns:
            company_id, company_name, revenue, ebitda,
            entry_date, sector, hold_period_years

    Returns:
        Cleaned DataFrame ready for modeling.

    Raises:
        DataCleaningError: If input DataFrame is empty or missing
        required columns.
    """
    logger.info("Starting data cleaning pipeline")

    # --- Step 1: Validate input ---
    if df.empty:
        logger.error("Input DataFrame is empty")
        raise DataCleaningError("Input DataFrame is empty")

    required_columns = [
        "company_id", "company_name", "revenue",
        "ebitda", "entry_date", "sector"
    ]
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise DataCleaningError(f"Missing required columns: {missing_cols}")

    # Work on a copy — never modify the original DataFrame
    df = df.copy()
    logger.info(f"Input shape: {df.shape}")

    # --- Step 2: Coerce numeric columns ---
    for col in ["revenue", "ebitda"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        logger.info(f"Coerced {col} to numeric")

    # --- Step 3: Handle missing values ---
    revenue_nulls = df["revenue"].isna().sum()
    ebitda_nulls = df["ebitda"].isna().sum()

    logger.info(f"Missing values — revenue: {revenue_nulls}, ebitda: {ebitda_nulls}")

    # Flag missing values before filling
    df["revenue_missing"] = df["revenue"].isna()
    df["ebitda_missing"] = df["ebitda"].isna()

    # Fill with median — robust to outliers
    df["revenue"] = df["revenue"].fillna(df["revenue"].median())
    df["ebitda"] = df["ebitda"].fillna(df["ebitda"].median())

    # --- Step 4: Winsorize outliers ---
    for col in ["revenue", "ebitda"]:
        low = df[col].quantile(0.01)
        high = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=low, upper=high)
        logger.info(f"Winsorized {col} to [{low:.1f}, {high:.1f}]")

    # --- Step 5: Derive features ---
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["ebitda_margin"] = (df["ebitda"] / df["revenue"]).round(4)
    df["months_held"] = (
        (pd.Timestamp.now() - df["entry_date"]) / pd.Timedelta(days=30)
    ).round(1)

    logger.info(f"Output shape: {df.shape}")
    logger.info("Data cleaning pipeline complete")

    return df
