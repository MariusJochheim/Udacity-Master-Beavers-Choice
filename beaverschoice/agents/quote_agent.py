from __future__ import annotations

import re
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Sequence

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


class QuoteAgent:
    """
    Generate structured quotes using historical anchors and inventory availability.

    The agent relies on a companion inventory agent (ToolCallingAgent-compatible)
    to fetch availability and delivery estimates for the requested items.
    """

    def __init__(self, engine=None, inventory_agent=None):
        self.engine = engine or get_engine()
        if inventory_agent is None:
            raise ValueError("QuoteAgent requires an inventory agent for availability checks")
        self.inventory_agent = inventory_agent

    def _anchor_price(self, terms: Sequence[str]) -> tuple[float, List[Dict]]:
        """
        Pull recent quote history and return an average total and the raw rows used.
        """
        history = recent_quote_history(self.engine, terms, limit=3)
        if not history:
            return 120.0, []

        totals = [float(entry.total_amount) for entry in history]
        average = sum(totals) / len(totals)
        return average, [asdict(entry) for entry in history]

    def _run_inventory_agent(self, request_text: str, as_of_date: str) -> Dict:
        availability = self.inventory_agent.run(request_text, additional_args={"as_of_date": as_of_date})
        if not isinstance(availability, dict):
            raise ValueError("inventory agent response must be a dictionary")
        return availability

    def _restock_markup(self, availability: Dict) -> float:
        if availability.get("restock_recommended"):
            return 0.05
        return 0.0

    def generate_quote(self, request_payload: Dict) -> Dict:
        """
        Build a quote for a customer request.

        Expected payload keys:
            - request: free-form text describing the need (required)
            - job_type, order_size, event_type: optional metadata strings
            - as_of_date: optional ISO8601 date string or datetime
        """
        if not isinstance(request_payload, dict):
            raise TypeError("request_payload must be a dictionary")

        request_text = _validate_nonempty_string(request_payload.get("request", ""), "request")
        as_of_date = _normalize_date_input(request_payload.get("as_of_date") or datetime.now(), "as_of_date")

        order_size = request_payload.get("order_size")
        search_terms = _extract_terms(request_payload)
        anchor_price, history_used = self._anchor_price(search_terms)

        availability = self._run_inventory_agent(request_text, as_of_date)
        size_factor = _size_factor(order_size)
        markup_rate = self._restock_markup(availability)
        adjusted_total = round(anchor_price * size_factor * (1 + markup_rate), 2)

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
                "job_type": request_payload.get("job_type"),
                "order_size": request_payload.get("order_size"),
                "event_type": request_payload.get("event_type"),
            }.items()
            if isinstance(value, str) and value.strip()
        }

        return {
            "request": request_text,
            "as_of_date": as_of_date,
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
