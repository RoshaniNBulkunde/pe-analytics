# pe-analytics

> Predicting how much a private equity firm will get paid 
> when it sells a company — using that company's financial 
> characteristics.

Built by a PhD economist to demonstrate production-quality 
Python, ML modeling, and AI-assisted development for 
quantitative finance applications.

---

## The Business Problem

Private equity firms buy companies, grow them, then sell 
them 3–7 years later. The price they receive at exit is 
typically expressed as a multiple of EBITDA — for example, 
selling a company with $10M EBITDA at 12x means receiving 
$120M.

**The question this project answers:**

> "Given what we know about a company today — its revenue, 
> margins, growth rate, and sector — what exit multiple 
> should we expect when we sell it?"

Accurately predicting this number before buying a company 
informs billion-dollar investment decisions.

---

## What I Built

An end-to-end Python pipeline that:

1. **Generates** realistic synthetic PE portfolio data 
   (200 companies with revenue, EBITDA margins, growth 
   rates, sectors, and hold periods)

2. **Cleans** the raw data — handles missing values, 
   outliers, and type coercion using pandas

3. **Models** exit EV/EBITDA multiples using three ML 
   models and compares them honestly

4. **Extracts** structured financial metrics from 
   unstructured company descriptions using the OpenAI API

---

## The Models — and What I Found

I built three models deliberately, not randomly. Each 
teaches us something different:

| Model | R² | RMSE | Why I built it |
|-------|----|------|-----------------|
| Ridge | 0.735 | 0.842 | Interpretable baseline — tells us which features matter and by exactly how much |
| Lasso | 0.737 | 0.839 | Feature selection — automatically zeros out irrelevant predictors |
| XGBoost | 0.676 | 0.929 | Complex patterns — tests whether non-linear relationships improve predictions |

**The interesting finding:** Ridge outperformed XGBoost.

With 200 companies and mostly linear relationships in the 
data, the simpler model was more stable and more accurate. 
XGBoost needs thousands of examples to reliably find 
non-linear patterns — with 200 it overfits. This reinforces 
the practice of always establishing a linear baseline before 
adding complexity.

**What the model tells a PE client:**

> "For every additional percentage point of EBITDA margin, 
> we expect roughly 0.9x higher exit multiple, holding 
> everything else constant. Our model estimates a 12x 
> exit multiple for this company, with a 95% prediction 
> interval of 10.5x to 13.5x."

---

## How I Built This — AI-Assisted Development

This project was built using **Claude** as the primary 
AI coding tool, with the following workflow:

1. Describe the function needed — inputs, outputs, edge 
   cases, error handling, and logging requirements
2. Review every line of generated code carefully
3. Test individual functions in Spyder console
4. Debug issues and understand root causes
5. Commit only after understanding every line

**Two specific examples of catching AI errors during review:**
- Detected a duplicate handler bug in the logging setup 
  that would have caused every log message to print 
  multiple times in production
- Caught a data leakage issue in cross-validation design 
  where the scaler was being fit before the CV fold split, 
  leaking test data into training

AI accelerated the development. Understanding the code 
made it correct.

---

## Project Structure

## pe-analytics/
├── src/pe_analytics/
│   ├── logging_config.py    # Centralized logging —
│   │                        # imported by every module
│   ├── data_cleaning.py     # pandas cleaning pipeline —
│   │                        # missing values, outliers,
│   │                        # feature engineering
│   ├── models.py            # Ridge, Lasso, XGBoost
│   │                        # with TimeSeriesSplit CV
│   └── llm_extraction.py   # OpenAI API extraction
│                            # with Pydantic validation
├── tests/
│   ├── test_data_cleaning.py # 11 unit tests
│   └── test_models.py        # 20 unit tests
├── scripts/
│   └── run_pipeline.py      # Single entry point —
│                            # runs all 4 stages
└── README.md
---
## 

## Quickstart

```bash
# Clone the repo
git clone https://github.com/RoshaniNBulkunde/pe-analytics.git
cd pe-analytics

# Install dependencies
pip install -r requirements.txt

# Optional: add OpenAI API key for real LLM extraction
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your_key_here

# Run the full pipeline
python scripts/run_pipeline.py

# Run unit tests
pytest tests/ -v
```

Expected output:
---

## Code Quality

Every module demonstrates production Python standards:

- **Logging** — timestamps, levels, module names to 
  console and daily log files. No print() statements
- **Error handling** — custom exceptions 
  (DataCleaningError, ModelTrainingError, ExtractionError) 
  at every stage
- **Type hints** — all public functions fully annotated
- **Docstrings** — Google-style on all public functions
- **Unit tests** — 30 pytest tests, all passing
- **No data leakage** — StandardScaler fitted inside 
  CV folds via sklearn Pipeline
- **Reproducibility** — fixed random seeds throughout

---

## Key Technical Decisions Explained

**Why TimeSeriesSplit instead of random CV?**
Financial data has time order. Random cross-validation 
allows training on 2023 data and testing on 2020 data — 
impossible in real deployment. TimeSeriesSplit always 
trains on earlier data and tests on later data.

**Why Winsorize instead of deleting outliers?**
Deleting outliers loses data. A legitimate PE portfolio 
company might just be much larger than the rest. 
Winsorizing at the 1st/99th percentile keeps all 
companies in the dataset while stopping extreme values 
from dominating the model.

**Why flag missing values before filling them?**
Once you fill a missing value with the median, you lose 
the information that it was missing. In PE data, whether 
a company is missing financial data is itself a signal. 
The flag column preserves that information for the model.

**Why mock mode for the LLM extraction?**
The full pipeline should run without any external 
dependencies for testing and review. Mock mode activates 
automatically when no API key is set — zero code changes 
needed to switch between mock and real extraction.

---

## About

Built as part of preparation for quantitative finance 
and PE analytics roles. Demonstrates the intersection 
of economics intuition, ML engineering, and 
production-quality Python.

PhD in Economics · Python · sklearn · XGBoost · 
OpenAI API · pandas · pytest