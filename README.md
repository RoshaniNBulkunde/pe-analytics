# pe-analytics

Private equity analytics toolkit — ML valuation models, 
data wrangling, and LLM-powered company analysis.

Built to demonstrate production-quality Python for 
quantitative finance applications.

---

## What this does

End-to-end pipeline that:

1. **Generates** synthetic PE portfolio company data 
   (200 companies, realistic revenue and margin distributions)
2. **Cleans** raw data using pandas — handles missing values, 
   outliers, type coercion, and feature engineering
3. **Models** exit EV/EBITDA multiples using Ridge, Lasso, 
   and XGBoost with TimeSeriesSplit cross-validation
4. **Extracts** structured financial metrics from unstructured 
   company descriptions using the OpenAI API

---

## Project structure
pe-analytics/
├── src/pe_analytics/
│   ├── logging_config.py    # Centralized logging setup
│   ├── data_cleaning.py     # pandas data cleaning pipeline
│   ├── models.py            # Ridge, Lasso, XGBoost training
│   └── llm_extraction.py   # OpenAI API extraction
├── tests/
│   └── test_data_cleaning.py # pytest unit tests
├── scripts/
│   └── run_pipeline.py      # End-to-end entry point
└── requirements.txt


---

## Quickstart

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/pe-analytics.git
cd pe-analytics

# Install dependencies
pip install -r requirements.txt

# Optional: add OpenAI API key for real LLM extraction
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run the full pipeline
python scripts/run_pipeline.py

# Run unit tests
pytest tests/ -v
```

---

## Model results

Predicting exit EV/EBITDA multiples on 200 synthetic 
PE portfolio companies using TimeSeriesSplit CV:

| Model   | R²    | RMSE  |
|---------|-------|-------|
| Ridge   | 0.735 | 0.842 |
| Lasso   | 0.737 | 0.839 |
| XGBoost | 0.676 | 0.929 |

Ridge and Lasso outperform XGBoost on this dataset — 
consistent with the predominantly linear data generating 
process and relatively small sample size (n=200).

---

## Code quality features

- **Logging** — timestamps, levels, module names to 
  console and daily log files
- **Error handling** — custom exceptions at every stage, 
  graceful batch recovery
- **Type hints** — all public functions fully annotated
- **Docstrings** — Google-style on all public functions
- **Unit tests** — 11 pytest tests covering happy path, 
  missing data, and error cases
- **No data leakage** — StandardScaler fitted inside 
  CV folds via sklearn Pipeline

---

## Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, xgboost
- openai, pydantic (for LLM extraction)
- pytest (for tests)

See `requirements.txt` for pinned versions.