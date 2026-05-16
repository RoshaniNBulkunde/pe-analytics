"""
test_models.py
--------------
Unit tests for the models module.
Run with: pytest tests/
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pe_analytics.models import (
    generate_pe_data,
    prepare_features,
    train_and_evaluate,
    run_all_models,
    ModelTrainingError
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------

@pytest.fixture
def pe_df():
    """Small synthetic PE DataFrame for testing."""
    return generate_pe_data(n_companies=100, random_seed=42)


@pytest.fixture
def X_y(pe_df):
    """Feature matrix and target vector from PE data."""
    return prepare_features(pe_df)


# ----------------------------------------------------------------
# Tests — generate_pe_data()
# ----------------------------------------------------------------

def test_generate_returns_dataframe():
    """generate_pe_data should return a DataFrame."""
    df = generate_pe_data(n_companies=50)
    assert isinstance(df, pd.DataFrame)


def test_generate_correct_shape():
    """generate_pe_data should return correct number of rows."""
    df = generate_pe_data(n_companies=50)
    assert len(df) == 50


def test_generate_has_required_columns():
    """generate_pe_data should include all required columns."""
    df = generate_pe_data()
    required = [
        "revenue", "ebitda_margin", "revenue_growth",
        "hold_period", "sector", "exit_multiple"
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_generate_reproducible():
    """Same random seed should produce identical DataFrames."""
    df1 = generate_pe_data(random_seed=42)
    df2 = generate_pe_data(random_seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_different_seeds_differ():
    """Different seeds should produce different DataFrames."""
    df1 = generate_pe_data(random_seed=42)
    df2 = generate_pe_data(random_seed=99)
    assert not df1["revenue"].equals(df2["revenue"])


def test_exit_multiple_in_valid_range():
    """Exit multiples should be clipped between 3 and 20."""
    df = generate_pe_data()
    assert df["exit_multiple"].min() >= 3.0
    assert df["exit_multiple"].max() <= 20.0


def test_ebitda_margin_in_valid_range():
    """EBITDA margins should be between 0.02 and 0.45."""
    df = generate_pe_data()
    assert df["ebitda_margin"].min() >= 0.02
    assert df["ebitda_margin"].max() <= 0.45


# ----------------------------------------------------------------
# Tests — prepare_features()
# ----------------------------------------------------------------

def test_prepare_returns_numpy_arrays(pe_df):
    """prepare_features should return numpy arrays."""
    X, y = prepare_features(pe_df)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_prepare_correct_row_count(pe_df):
    """X and y should have same number of rows as input."""
    X, y = prepare_features(pe_df)
    assert X.shape[0] == len(pe_df)
    assert y.shape[0] == len(pe_df)


def test_prepare_no_nulls(pe_df):
    """Feature matrix should contain no NaN values."""
    X, y = prepare_features(pe_df)
    # Convert to float first to handle boolean columns from get_dummies
    assert not np.isnan(X.astype(float)).any()
    assert not np.isnan(y.astype(float)).any()


def test_prepare_raises_on_missing_columns():
    """Missing required columns should raise ModelTrainingError."""
    bad_df = pd.DataFrame({"revenue": [100.0], "sector": ["Software"]})
    with pytest.raises(ModelTrainingError):
        prepare_features(bad_df)


# ----------------------------------------------------------------
# Tests — train_and_evaluate()
# ----------------------------------------------------------------

def test_train_returns_dict(X_y):
    """train_and_evaluate should return a dictionary."""
    X, y = X_y
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    result = train_and_evaluate(X, y, "Ridge", model)
    assert isinstance(result, dict)


def test_train_has_required_keys(X_y):
    """Result dictionary should have all required keys."""
    X, y = X_y
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    result = train_and_evaluate(X, y, "Ridge", model)
    assert "model_name" in result
    assert "mean_r2" in result
    assert "std_r2" in result
    assert "mean_rmse" in result


def test_train_r2_in_valid_range(X_y):
    """R² should be between -1 and 1."""
    X, y = X_y
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    result = train_and_evaluate(X, y, "Ridge", model)
    assert -1.0 <= result["mean_r2"] <= 1.0


def test_train_rmse_positive(X_y):
    """RMSE should always be positive."""
    X, y = X_y
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    result = train_and_evaluate(X, y, "Ridge", model)
    assert result["mean_rmse"] > 0


# ----------------------------------------------------------------
# Tests — run_all_models()
# ----------------------------------------------------------------

def test_run_all_models_returns_dataframe(pe_df):
    """run_all_models should return a DataFrame."""
    results = run_all_models(pe_df)
    assert isinstance(results, pd.DataFrame)


def test_run_all_models_has_three_rows(pe_df):
    """run_all_models should return results for 3 models."""
    results = run_all_models(pe_df)
    assert len(results) == 3


def test_run_all_models_index_names(pe_df):
    """run_all_models should have correct model names as index."""
    results = run_all_models(pe_df)
    assert "Ridge" in results.index
    assert "Lasso" in results.index
    assert "XGBoost" in results.index


def test_run_all_models_r2_positive(pe_df):
    """All models should achieve positive R² on PE data."""
    results = run_all_models(pe_df)
    assert (results["mean_r2"] > 0).all()
