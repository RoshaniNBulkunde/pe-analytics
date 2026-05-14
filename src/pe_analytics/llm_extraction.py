"""
llm_extraction.py
-----------------
Uses OpenAI API to extract structured financial metrics from
unstructured company descriptions.

If OPENAI_API_KEY is not set, runs in mock mode for testing.
"""

import os
import json
from typing import Optional
from pydantic import BaseModel, Field
from src.pe_analytics.logging_config import setup_logger

logger = setup_logger(__name__)


class CompanyMetrics(BaseModel):
    """
    Structured financial metrics extracted from unstructured text.
    All monetary values in USD millions.
    """
    company_name: str
    sector: str
    revenue_usd_m: Optional[float] = Field(
        None, description="Annual revenue in USD millions"
    )
    ebitda_usd_m: Optional[float] = Field(
        None, description="EBITDA in USD millions"
    )
    revenue_growth_pct: Optional[float] = Field(
        None, description="YoY revenue growth as decimal e.g. 0.15 for 15%"
    )
    confidence: str = Field(
        ..., description="Confidence level: high, medium, or low"
    )


class ExtractionError(Exception):
    """Raised when LLM extraction fails."""


def _mock_extraction(text: str) -> CompanyMetrics:
    """
    Returns a mock CompanyMetrics object for testing without API key.
    Simulates what the OpenAI API would return.

    Args:
        text: Company description text.

    Returns:
        Mock CompanyMetrics object.
    """
    logger.warning("Running in MOCK mode — no OpenAI API key found")
    logger.info(f"Would extract metrics from text: {text[:80]}...")

    # Return realistic mock data
    return CompanyMetrics(
        company_name="Alpha Software Inc",
        sector="Software",
        revenue_usd_m=45.2,
        ebitda_usd_m=9.1,
        revenue_growth_pct=0.18,
        confidence="high"
    )


def _real_extraction(text: str, api_key: str) -> CompanyMetrics:
    """
    Calls OpenAI API to extract financial metrics from text.

    Args:
        text: Company description text.
        api_key: OpenAI API key.

    Returns:
        Validated CompanyMetrics object.

    Raises:
        ExtractionError: If API call fails or response is invalid.
    """
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        system_prompt = """You are a financial data extractor for private equity analysis.
Extract ONLY information explicitly stated in the text.
If a value is not mentioned, set it to null.
Set confidence to:
  - high: all key metrics clearly stated
  - medium: some metrics inferred
  - low: significant uncertainty
Respond ONLY with valid JSON. No explanation, no markdown."""

        schema = CompanyMetrics.model_json_schema()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Schema: {json.dumps(schema)}\n\nText:\n{text}"
                }
            ]
        )

        usage = response.usage
        logger.info(
            f"OpenAI usage — prompt tokens: {usage.prompt_tokens}, "
            f"completion tokens: {usage.completion_tokens}"
        )

        raw = json.loads(response.choices[0].message.content)
        metrics = CompanyMetrics(**raw)
        logger.info(
            f"Extracted: {metrics.company_name} | "
            f"Revenue: ${metrics.revenue_usd_m}M | "
            f"Confidence: {metrics.confidence}"
        )
        return metrics

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        raise ExtractionError(f"Invalid JSON from OpenAI: {e}")

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise ExtractionError(f"API call failed: {e}")


def extract_company_metrics(text: str) -> CompanyMetrics:
    """
    Extract structured financial metrics from unstructured company text.
    Automatically uses mock mode if OPENAI_API_KEY is not set.

    Args:
        text: Unstructured text describing a company
              (e.g. earnings summary, CIM excerpt, news article).

    Returns:
        CompanyMetrics object with extracted financial data.

    Raises:
        ExtractionError: If extraction fails in real mode.
    """
    if not text or not text.strip():
        logger.error("Empty text passed to extract_company_metrics")
        raise ExtractionError("Input text cannot be empty")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return _mock_extraction(text)
    else:
        logger.info("OPENAI_API_KEY found — using real extraction")
        return _real_extraction(text, api_key)


def extract_batch(texts: list[str]) -> list[CompanyMetrics]:
    """
    Extract metrics from a list of company descriptions.
    Logs progress and skips failed extractions.

    Args:
        texts: List of company description strings.

    Returns:
        List of successfully extracted CompanyMetrics objects.
    """
    logger.info(f"Starting batch extraction for {len(texts)} companies")
    results = []

    for i, text in enumerate(texts):
        try:
            metrics = extract_company_metrics(text)
            results.append(metrics)
            logger.info(f"Processed {i + 1}/{len(texts)}: {metrics.company_name}")
        except ExtractionError as e:
            logger.warning(f"Skipping item {i + 1} due to error: {e}")
            continue

    logger.info(
        f"Batch complete — {len(results)}/{len(texts)} successfully extracted"
    )
    return results

