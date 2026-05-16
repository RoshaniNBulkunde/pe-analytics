"""
test_data_cleaning.py
---------------------
Unit tests for the data_cleaning module.
Run with: pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pe_analytics.data_cleaning import clean_portfolio_data, DataCleaningError


# ----------------------------------------------------------------
# Fixtures — reusable test data
# ----------------------------------------------------------------

@pytest.fixture
def valid_df():
    """Minimal valid portfolio DataFrame for testing."""
    return pd.DataFrame({
        "company_id": [1, 2, 3],
        "company_name": ["Alpha", "Beta", "Gamma"],
        "revenue": [100.0, 200.0, 300.0],
        "ebitda": [20.0, 40.0, 60.0],
        "entry_date": ["2020-01-01", "2021-01-01", "2022-01-01"],
        "sector": ["Software", "Healthcare", "Manufacturing"]
    })


@pytest.fixture
def df_with_nulls():
    """DataFrame with missing revenue and ebitda values."""
    return pd.DataFrame({
        "company_id": [1, 2, 3],
        "company_name": ["Alpha", "Beta", "Gamma"],
        "revenue": [100.0, None, 300.0],
        "ebitda": [20.0, 40.0, None],
        "entry_date": ["2020-01-01", "2021-01-01", "2022-01-01"],
        "sector": ["Software", "Healthcare", "Manufacturing"]
    })


# ----------------------------------------------------------------
# Tests — happy path
# ----------------------------------------------------------------

def test_clean_returns_dataframe(valid_df):
    """clean_portfolio_data should return a DataFrame."""
    result = clean_portfolio_data(valid_df)
    assert isinstance(result, pd.DataFrame)


def test_clean_adds_ebitda_margin(valid_df):
    """clean_portfolio_data should add ebitda_margin column."""
    result = clean_portfolio_data(valid_df)
    assert "ebitda_margin" in result.columns


def test_clean_adds_months_held(valid_df):
    """clean_portfolio_data should add months_held column."""
    result = clean_portfolio_data(valid_df)
    assert "months_held" in result.columns


def test_clean_does_not_modify_original(valid_df):
    """clean_portfolio_data should not modify the input DataFrame."""
    original_shape = valid_df.shape
    clean_portfolio_data(valid_df)
    assert valid_df.shape == original_shape


def test_ebitda_margin_correct(valid_df):
    """ebitda_margin should equal ebitda / revenue."""
    result = clean_portfolio_data(valid_df)
    expected = (valid_df["ebitda"] / valid_df["revenue"]).round(4)
    pd.testing.assert_series_equal(
        result["ebitda_margin"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False
    )


# ----------------------------------------------------------------
# Tests — missing value handling
# ----------------------------------------------------------------

def test_missing_revenue_flagged(df_with_nulls):
    """Missing revenue values should be flagged in revenue_missing column."""
    result = clean_portfolio_data(df_with_nulls)
    assert "revenue_missing" in result.columns
    assert result["revenue_missing"].sum() == 1


def test_missing_ebitda_flagged(df_with_nulls):
    """Missing ebitda values should be flagged in ebitda_missing column."""
    result = clean_portfolio_data(df_with_nulls)
    assert "ebitda_missing" in result.columns
    assert result["ebitda_missing"].sum() == 1


def test_no_nulls_after_cleaning(df_with_nulls):
    """revenue and ebitda should have no nulls after cleaning."""
    result = clean_portfolio_data(df_with_nulls)
    assert result["revenue"].isna().sum() == 0
    assert result["ebitda"].isna().sum() == 0


# ----------------------------------------------------------------
# Tests — error handling
# ----------------------------------------------------------------

def test_empty_dataframe_raises_error():
    """Empty DataFrame should raise DataCleaningError."""
    with pytest.raises(DataCleaningError):
        clean_portfolio_data(pd.DataFrame())


def test_missing_columns_raises_error():
    """DataFrame missing required columns should raise DataCleaningError."""
    bad_df = pd.DataFrame({"revenue": [100.0], "ebitda": [20.0]})
    with pytest.raises(DataCleaningError):
        clean_portfolio_data(bad_df)


def test_string_revenue_coerced(valid_df):
    """String revenue values should be coerced to numeric."""
    valid_df["revenue"] = ["100.0", "200.0", "300.0"]
    result = clean_portfolio_data(valid_df)
    assert result["revenue"].dtype in [np.float64, np.float32]
