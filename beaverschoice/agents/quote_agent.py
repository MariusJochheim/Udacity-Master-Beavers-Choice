from __future__ import annotations

import re
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Sequence

from smolagents.agents import ToolCallingAgent
from smolagents.models import Model, OpenAIModel
from smolagents.tools import Tool

from beaverschoice.config import OPENAI_API_KEY, OPENAI_BASE_URL
from beaverschoice.db import get_engine
from beaverschoice.tooling import _normalize_date_input, _validate_nonempty_string, recent_quote_history


def _extract_terms(payload: Dict) -> List[str]:
    """Collect lightweight search terms from the request payload."""
    terms: list[str] = []

    request_text = payload.get("request", "")
    if isinstance(request_text, str):
        tokens = re.findall(r"[a-zA-Z]{4,}", request_text.lower())
        terms.extend(tokens[:5])

    for key in ("job_type", "order_size", "event_type"):
        value = payload.get(key)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                terms.append(cleaned)

    # Deduplicate while preserving order
    seen = set()
    unique_terms = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
    return unique_terms


def _size_factor(order_size: str | None) -> float:
    """Return a multiplier based on order size semantics."""
    if not isinstance(order_size, str):
        return 1.0
    normalized = order_size.strip().lower()
    if normalized == "large":
        return 0.9
    if normalized in {"medium", "medium-large"}:
        return 1.0
    if normalized == "small":
        return 1.05
    return 1.0


class QuotePricingTool(Tool):
    """Tool that calculates a structured quote using history and inventory availability."""

    name = "prepare_quote"
    description = "Compute a priced quote and availability estimate based on request text and metadata."
    inputs = {
        "request_text": {"type": "string", "description": "Customer request text to quote"},
        "as_of_date": {"type": "string", "description": "ISO date for the quote", "nullable": True},
        "order_size": {"type": "string", "description": "Optional order size indicator", "nullable": True},
        "job_type": {"type": "string", "description": "Optional job type descriptor", "nullable": True},
        "event_type": {"type": "string", "description": "Optional event type descriptor", "nullable": True},
    }
    output_type = "object"

    def __init__(self, engine=None, inventory_agent=None):
        super().__init__()
        self.engine = engine or get_engine()
        if inventory_agent is None:
            raise ValueError("QuotePricingTool requires an inventory agent for availability checks")
        self.inventory_agent = inventory_agent

    def _anchor_price(self, terms: Sequence[str]) -> tuple[float, List[Dict]]:
        """Pull recent quote history and return an average total and the raw rows used."""
        history = recent_quote_history(self.engine, terms, limit=3)
        if not history:
            return 120.0, []

        totals = [float(entry.total_amount) for entry in history]
        average = sum(totals) / len(totals)
        return average, [asdict(entry) for entry in history]

    def _run_inventory_agent(self, request_text: str, as_of_date: str) -> Dict:
        def _parse_candidate(value: Any) -> Dict | None:
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                text = value
                if "Observation:" in text:
                    text = text.split("Observation:", 1)[1].strip()
                stripped = text.strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    import ast

                    try:
                        parsed = ast.literal_eval(stripped)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        return None
            if isinstance(value, list):
                for item in value:
                    parsed = _parse_candidate(item)
                    if isinstance(parsed, dict):
                        return parsed
            return None

        availability = self.inventory_agent.run(request_text, additional_args={"as_of_date": as_of_date})
        parsed = _parse_candidate(availability)
        if parsed is None:
            # Fall back to wrapping the raw response so downstream logic can still proceed.
            return {"raw_response": availability}
        return parsed

    def _restock_markup(self, availability: Dict) -> float:
        if availability.get("restock_recommended"):
            return 0.05
        return 0.0

    def forward(
        self,
        request_text: str,
        as_of_date: str | None = None,
        order_size: str | None = None,
        job_type: str | None = None,
        event_type: str | None = None,
    ) -> Dict[str, Any]:
        request = _validate_nonempty_string(request_text, "request_text")
        as_of = _normalize_date_input(as_of_date or datetime.now(), "as_of_date")

        payload = {
            "request": request,
            "order_size": order_size,
            "job_type": job_type,
            "event_type": event_type,
        }
        search_terms = _extract_terms(payload)
        anchor_price, history_used = self._anchor_price(search_terms)

        availability = self._run_inventory_agent(request, as_of)
        if availability.get("error"):
            return {
                "error": availability["error"],
                "request": request,
                "as_of_date": as_of,
                "availability": availability,
                "metadata": {
                    key: value
                    for key, value in {
                        "job_type": job_type,
                        "order_size": order_size,
                        "event_type": event_type,
                    }.items()
                    if isinstance(value, str) and value.strip()
                },
            }
        size_factor = _size_factor(order_size)
        markup_rate = self._restock_markup(availability)
        adjusted_total = round(anchor_price * size_factor * (1 + markup_rate), 2)
        if adjusted_total < 5.0:
            adjusted_total = 5.0

        explanation_parts: list[str] = []
        if history_used:
            explanation_parts.append(f"Anchored to {len(history_used)} similar quotes averaging ${anchor_price:.2f}.")
        else:
            explanation_parts.append("No close history found; using baseline pricing.")
        explanation_parts.append(f"Applied order size factor {size_factor:.2f}.")
        if markup_rate > 0:
            explanation_parts.append("Added restock markup due to low inventory.")
        explanation = " ".join(explanation_parts)

        metadata = {
            key: value
            for key, value in {
                "job_type": job_type,
                "order_size": order_size,
                "event_type": event_type,
            }.items()
            if isinstance(value, str) and value.strip()
        }

        return {
            "request": request,
            "as_of_date": as_of,
            "metadata": metadata,
            "history_anchors": history_used,
            "availability": availability,
            "pricing": {
                "anchor_average": round(anchor_price, 2),
                "order_size_factor": size_factor,
                "restock_markup": markup_rate,
                "currency": "USD",
                "total": adjusted_total,
            },
            "quote_explanation": explanation,
        }


class FinalAnswerTool(Tool):
    """Return the prepared quote payload as the final answer."""

    name = "final_answer"
    description = "Return the prepared quote payload."
    inputs = {"answer": {"type": "any", "description": "Final response content"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        # Smolagents often wraps tool responses in text chunks; try to unwrap to the underlying dict.
        def _parse_text_payload(text: str) -> Any:
            if "Observation:" in text:
                text = text.split("Observation:", 1)[1].strip()
            stripped = text.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                import ast

                try:
                    return ast.literal_eval(stripped)
                except Exception:
                    return stripped
            return stripped

        if isinstance(answer, list):
            text_parts = []
            for item in answer:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(str(item["text"]))
                elif isinstance(item, str):
                    text_parts.append(item)
            if text_parts:
                parsed = [_parse_text_payload(part) for part in text_parts]
                for candidate in parsed:
                    if isinstance(candidate, dict):
                        return candidate
                return " ".join(str(part) for part in parsed)
        if isinstance(answer, dict) and "text" in answer:
            parsed = _parse_text_payload(str(answer["text"]))
            return parsed
        return answer


class QuoteAgent(ToolCallingAgent):
    """
    ToolCallingAgent that prepares quotes using historical anchors and inventory availability.
    """

    def __init__(self, engine=None, inventory_agent=None, model: Model | None = None, **kwargs):
        self.engine = engine or get_engine()
        if inventory_agent is None:
            raise ValueError("QuoteAgent requires an inventory agent for availability checks")
        if model is None:
            raise ValueError("An LLM model must be provided to QuoteAgent")

        self.inventory_agent = inventory_agent
        self.quote_tool = QuotePricingTool(self.engine, inventory_agent)
        final_tool = FinalAnswerTool()

        super().__init__(
            tools=[self.quote_tool, final_tool],
            model=model,
            add_base_tools=False,
            max_tool_threads=1,
            instructions=(
                "You are the Quote agent. Call prepare_quote exactly once using the customer's request text, "
                "as_of_date, order_size, job_type, and event_type when available. Pass the tool's JSON result "
                "directly to final_answer without rephrasing or summarizing."
            ),
            **kwargs,
        )

    def generate_quote(self, request_payload: Dict) -> Dict[str, Any]:
        """
        Build a quote for a customer request by delegating to the prepare_quote tool.

        Expected payload keys:
            - request: free-form text describing the need (required)
            - job_type, order_size, event_type: optional metadata strings
            - as_of_date: optional ISO8601 date string or datetime
        """
        if not isinstance(request_payload, dict):
            raise TypeError("request_payload must be a dictionary")

        request_text = _validate_nonempty_string(request_payload.get("request", ""), "request")
        as_of_date = _normalize_date_input(request_payload.get("as_of_date") or datetime.now(), "as_of_date")

        arguments = {
            "request_text": request_text,
            "as_of_date": as_of_date,
            "order_size": request_payload.get("order_size"),
            "job_type": request_payload.get("job_type"),
            "event_type": request_payload.get("event_type"),
        }

        # Call the pricing tool directly to avoid LLM paraphrasing that breaks downstream parsing.
        result = self.quote_tool.forward(**arguments)
        if isinstance(result, dict):
            return result
        if isinstance(result, str) and result.strip().startswith("{"):
            import ast

            try:
                parsed = ast.literal_eval(result)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {"raw_response": result, "request": request_text, "as_of_date": as_of_date}


def create_openai_quote_agent(
    engine=None,
    inventory_agent=None,
    model_id: str = "gpt-4o-mini",
    api_key: str | None = None,
    api_base: str | None = None,
    **kwargs,
) -> QuoteAgent:
    """
    Factory to build a QuoteAgent backed by an OpenAI-compatible endpoint.
    """
    api_key = api_key or OPENAI_API_KEY
    api_base = api_base or OPENAI_BASE_URL
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required to initialize the OpenAI model")

    model = OpenAIModel(model_id=model_id, api_key=api_key, api_base=api_base)
    return QuoteAgent(engine=engine, inventory_agent=inventory_agent, model=model, **kwargs)
