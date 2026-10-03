from __future__ import annotations

import json
import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


class GenAIInsightsClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "30"))

    def enabled(self) -> bool:
        return bool(self.api_key)

    def _build_prompt(self, summary: dict[str, Any]) -> str:
        return f"""
You are an assistant that turns product review analysis into executive customer insights.

Goal: produce concise, business-friendly summary based strictly on the aggregated review data below.

Important:
- Do not claim that the model performed the sentiment classification.
- The ML model classifies review sentiment locally.
- The LLM only interprets the aggregated feedback and produces AI-generated insights.
- Label all outputs clearly as AI-generated insights.

Aggregated review data:
{json.dumps(summary, indent=2, ensure_ascii=False)}

Return the following sections in plain text:
1. Customer Summary:
2. Positive Themes:
3. Negative Themes:
4. Frequently Mentioned Product Aspects:
5. Recommended Improvements:
6. Business Conclusion:
"""

    def generate_customer_insights(self, summary: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled():
            raise ValueError("LLM API key is required to generate AI insights. Add LLM_API_KEY to your environment or .env file.")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You generate concise, business-oriented summaries from aggregated customer feedback data. Clearly label all output as AI-generated insights."},
                {"role": "user", "content": self._build_prompt(summary)},
            ],
            "temperature": 0.4,
            "max_tokens": 500,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise TimeoutError("LLM request timed out while generating insights.")
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"LLM API request failed: {exc}") from exc

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise ValueError("Invalid LLM response format.") from exc

        if not content or not content.strip():
            raise ValueError("Empty response received from the LLM.")

        return {"ai_generated": True, "content": content.strip()}
