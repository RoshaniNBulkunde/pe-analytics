"""
run_pipeline.py
---------------
End-to-end PE analytics pipeline.

Runs in sequence:
    1. Generate synthetic PE portfolio data
    2. Clean and validate the data
    3. Train and compare ML models (Ridge, Lasso, XGBoost)
    4. Extract financial metrics from sample company descriptions
       using LLM API (mock mode if no API key set)

Usage:
    From the pe-analytics root folder:
    python scripts/run_pipeline.py

Output:
    - Logs printed to console and saved to logs/ folder
    - Model comparison table printed to console
    - Extracted company metrics printed to console
"""

import sys
import os

# Add project root to path so src/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pe_analytics.logging_config import setup_logger
from src.pe_analytics.data_cleaning import clean_portfolio_data
from src.pe_analytics.models import generate_pe_data, run_all_models
from src.pe_analytics.llm_extraction import extract_batch

logger = setup_logger(__name__)


def run_pipeline() -> None:
    """
    Execute the full PE analytics pipeline end to end.

    Steps:
        1. Generate synthetic data
        2. Clean data
        3. Train and evaluate models
        4. Extract metrics from company descriptions

    Raises:
        SystemExit: If any pipeline stage fails critically.
    """
    logger.info("=" * 60)
    logger.info("PE Analytics Pipeline — Starting")
    logger.info("=" * 60)

    # ----------------------------------------------------------
    # STAGE 1: Generate synthetic PE portfolio data
    # ----------------------------------------------------------
    logger.info("STAGE 1: Generating portfolio data")
    try:
        raw_df = generate_pe_data(n_companies=200, random_seed=42)
        logger.info(f"Generated {len(raw_df)} portfolio companies")
    except Exception as e:
        logger.error(f"STAGE 1 FAILED: {e}")
        sys.exit(1)

    # ----------------------------------------------------------
    # STAGE 2: Clean and validate the data
    # ----------------------------------------------------------
    logger.info("STAGE 2: Cleaning portfolio data")
    try:
        # Add required columns for cleaning
        import pandas as pd
        import numpy as np

        raw_df["company_id"] = range(1, len(raw_df) + 1)
        raw_df["company_name"] = [f"Company_{i}" for i in raw_df["company_id"]]
        raw_df["entry_date"] = pd.date_range(
            start="2015-01-01",
            periods=len(raw_df),
            freq="W"
        )
        raw_df["ebitda"] = raw_df["revenue"] * raw_df["ebitda_margin"]

        cleaned_df = clean_portfolio_data(raw_df)
        logger.info(f"Cleaned data shape: {cleaned_df.shape}")
    except Exception as e:
        logger.error(f"STAGE 2 FAILED: {e}")
        sys.exit(1)

    # ----------------------------------------------------------
    # STAGE 3: Train and compare ML models
    # ----------------------------------------------------------
    logger.info("STAGE 3: Training ML models")
    try:
        results_df = run_all_models(raw_df.drop(
    columns=["company_id", "company_name", "entry_date", "ebitda",
             "revenue_missing", "ebitda_missing", "months_held"],
    errors="ignore"))

        logger.info("Model comparison results:")
        logger.info(f"\n{results_df.to_string()}")

        print("\n" + "=" * 50)
        print("MODEL COMPARISON RESULTS")
        print("=" * 50)
        print(results_df.to_string())
        print("=" * 50 + "\n")

        best_model = results_df["mean_r2"].idxmax()
        best_r2 = results_df["mean_r2"].max()
        logger.info(f"Best model: {best_model} with R²={best_r2:.3f}")

    except Exception as e:
        logger.error(f"STAGE 3 FAILED: {e}")
        sys.exit(1)

    # ----------------------------------------------------------
    # STAGE 4: LLM extraction from company descriptions
    # ----------------------------------------------------------
    logger.info("STAGE 4: Extracting metrics from company descriptions")

    company_descriptions = [
        """Alpha Software Inc is a B2B SaaS company serving mid-market
        manufacturers. In fiscal year 2025, the company reported revenue
        of $45.2 million, representing 18% year-over-year growth.
        EBITDA for the period was $9.1 million.""",

        """Beta Healthcare Solutions provides medical billing software
        to regional hospital networks. Annual recurring revenue reached
        $28 million with EBITDA margins of 24%. Revenue grew 31% vs
        prior year driven by 12 new hospital system wins.""",

        """Gamma Industrial supplies precision components to aerospace
        manufacturers. FY2025 revenue was $92M with EBITDA of $11M.
        Growth was flat year over year due to supply chain disruptions."""
    ]

    try:
        extracted = extract_batch(company_descriptions)

        print("\n" + "=" * 50)
        print("EXTRACTED COMPANY METRICS")
        print("=" * 50)
        for metrics in extracted:
            print(f"\nCompany:        {metrics.company_name}")
            print(f"Sector:         {metrics.sector}")
            print(f"Revenue ($M):   {metrics.revenue_usd_m}")
            print(f"EBITDA ($M):    {metrics.ebitda_usd_m}")
            print(f"Revenue Growth: {metrics.revenue_growth_pct}")
            print(f"Confidence:     {metrics.confidence}")
        print("=" * 50 + "\n")

    except Exception as e:
        logger.error(f"STAGE 4 FAILED: {e}")
        sys.exit(1)

    # ----------------------------------------------------------
    # PIPELINE COMPLETE
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PE Analytics Pipeline — Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()

