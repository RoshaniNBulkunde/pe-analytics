"""
models.py
---------
Trains and evaluates ML models to predict exit EV/EBITDA multiples
for PE portfolio companies.

Models: Ridge regression, Lasso regression, XGBoost.
Evaluation: TimeSeriesSplit cross-validation, R², RMSE.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from src.pe_analytics.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelTrainingError(Exception):
    """Raised when model training fails due to invalid input."""


def generate_pe_data(n_companies: int = 200, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic PE portfolio company data.

    Args:
        n_companies: Number of portfolio companies to generate.
        random_seed: Seed for reproducibility.

    Returns:
        DataFrame with features and target variable (exit_multiple).
    """
    logger.info(f"Generating synthetic PE data for {n_companies} companies")
    rng = np.random.default_rng(random_seed)

    # Features
    revenue = rng.lognormal(mean=4.0, sigma=0.8, size=n_companies)
    ebitda_margin = rng.normal(loc=0.18, scale=0.06, size=n_companies).clip(0.02, 0.45)
    revenue_growth = rng.normal(loc=0.12, scale=0.08, size=n_companies).clip(-0.2, 0.5)
    hold_period = rng.uniform(low=2.0, high=8.0, size=n_companies)
    sector = rng.choice(
        ["Software", "Healthcare", "Manufacturing", "Consumer", "Energy"],
        size=n_companies
    )

    # Target: exit multiple — driven by margin, growth, sector
    # Software companies command higher multiples
    sector_premium = np.where(sector == "Software", 2.5,
                     np.where(sector == "Healthcare", 1.5, 0.0))

    exit_multiple = (
        8.0
        + 10.0 * ebitda_margin
        + 8.0 * revenue_growth
        + 0.3 * np.log(revenue)
        + sector_premium
        + rng.normal(0, 0.8, size=n_companies)  # noise
    ).clip(3.0, 20.0)

    df = pd.DataFrame({
        "revenue": revenue.round(2),
        "ebitda_margin": ebitda_margin.round(4),
        "revenue_growth": revenue_growth.round(4),
        "hold_period": hold_period.round(1),
        "sector": sector,
        "exit_multiple": exit_multiple.round(2)
    })

    logger.info(f"Generated data shape: {df.shape}")
    logger.info(f"Exit multiple — mean: {df['exit_multiple'].mean():.2f}, "
                f"std: {df['exit_multiple'].std():.2f}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix X and target vector y.
    One-hot encodes sector column.

    Args:
        df: DataFrame from generate_pe_data().

    Returns:
        Tuple of (X, y) as numpy arrays.

    Raises:
        ModelTrainingError: If required columns are missing.
    """
    required = ["revenue", "ebitda_margin", "revenue_growth",
                "hold_period", "sector", "exit_multiple"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ModelTrainingError(f"Missing columns: {missing}")

    # One-hot encode sector
    df_encoded = pd.get_dummies(df, columns=["sector"], drop_first=True)

    X = df_encoded.drop(columns=["exit_multiple"]).values
    y = df_encoded["exit_multiple"].values

    logger.info(f"Feature matrix shape: {X.shape}")
    return X, y


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    model
) -> dict:
    """
    Train a model using TimeSeriesSplit cross-validation and return metrics.

    Args:
        X: Feature matrix.
        y: Target vector.
        model_name: Name of the model for logging.
        model: Sklearn-compatible model or pipeline.

    Returns:
        Dictionary with mean R², RMSE, and std of R².
    """
    logger.info(f"Training {model_name} with TimeSeriesSplit CV")

    tscv = TimeSeriesSplit(n_splits=5)

    cv_results = cross_validate(
        model, X, y,
        cv=tscv,
        scoring=["r2", "neg_root_mean_squared_error"],
        return_train_score=False
    )

    mean_r2 = cv_results["test_r2"].mean()
    std_r2 = cv_results["test_r2"].std()
    mean_rmse = -cv_results["test_neg_root_mean_squared_error"].mean()

    logger.info(f"{model_name} — R²: {mean_r2:.3f} (±{std_r2:.3f}), "
                f"RMSE: {mean_rmse:.3f}")

    return {
        "model_name": model_name,
        "mean_r2": round(mean_r2, 3),
        "std_r2": round(std_r2, 3),
        "mean_rmse": round(mean_rmse, 3)
    }


def run_all_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Ridge, Lasso, and XGBoost models and return comparison table.

    Args:
        df: Cleaned PE portfolio DataFrame.

    Returns:
        DataFrame comparing model performance metrics.

    Raises:
        ModelTrainingError: If feature preparation fails.
    """
    logger.info("Starting model comparison pipeline")

    X, y = prepare_features(df)

    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01))
        ]),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
    }

    results = []
    for name, model in models.items():
        try:
            result = train_and_evaluate(X, y, name, model)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
            raise ModelTrainingError(f"Failed to train {name}: {e}")

    results_df = pd.DataFrame(results).set_index("model_name")
    logger.info("Model comparison complete")
    logger.info(f"\n{results_df.to_string()}")

    return results_df
