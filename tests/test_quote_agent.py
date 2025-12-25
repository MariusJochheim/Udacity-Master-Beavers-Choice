import pandas as pd
import pytest

from beaverschoice.agents.quote_agent import QuoteAgent, _extract_terms, _size_factor
from beaverschoice.procurement import compute_procurement_decision
from beaverschoice.tooling import recent_quote_history
from beaverschoice.transactions import create_transaction


class FakeInventoryAgent:
    """Deterministic stand-in for the inventory agent."""

    def __init__(self, engine, as_of_date):
        self.engine = engine
        self.as_of_date = as_of_date

    def run(self, request_text, additional_args=None):
        date = (additional_args or {}).get("as_of_date") or self.as_of_date
        return compute_procurement_decision(self.engine, request_text, date).to_dict()


def test_quote_agent_returns_priced_quote_with_history(engine, start_date):
    inventory_agent = FakeInventoryAgent(engine, start_date)
    payload = {
        "request": "Looking for a bulk paper package for a festival event",
        "job_type": "event manager",
        "order_size": "large",
        "event_type": "festival",
        "as_of_date": start_date,
    }
    agent = QuoteAgent(engine, inventory_agent)
    quote = agent.generate_quote(payload)

    terms = _extract_terms(payload)
    anchors = recent_quote_history(engine, terms, limit=3)
    expected_anchor = (
        sum(entry.total_amount for entry in anchors) / len(anchors) if anchors else 120.0
    )

    assert quote["request"] == payload["request"].strip()
    assert quote["pricing"]["anchor_average"] == pytest.approx(expected_anchor)
    markup = quote["pricing"]["restock_markup"]
    expected_total = round(expected_anchor * _size_factor(payload["order_size"]) * (1 + markup), 2)
    assert quote["pricing"]["total"] == expected_total
    assert "Anchored" in quote["quote_explanation"] or "baseline" in quote["quote_explanation"]
    assert "estimated_delivery_date" in quote["availability"]


def test_quote_agent_applies_markup_when_restock_needed(engine, start_date):
    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
    item = inventory_df.iloc[0]
    sale_units = max(1, int(item["current_stock"] - item["min_stock_level"] + 1))
    sale_units = min(sale_units, int(item["current_stock"]))
    create_transaction(
        engine,
        item["item_name"],
        "sales",
        sale_units,
        float(item["unit_price"] * sale_units),
        start_date,
    )

    inventory_agent = FakeInventoryAgent(engine, start_date)
    payload = {"request": f"Order {item['item_name']} immediately", "order_size": "small", "as_of_date": start_date}
    agent = QuoteAgent(engine, inventory_agent)
    quote = agent.generate_quote(payload)

    assert quote["availability"]["restock_recommended"] is True
    assert quote["pricing"]["restock_markup"] == 0.05
    base = quote["pricing"]["anchor_average"]
    expected_total = round(base * _size_factor("small") * 1.05, 2)
    assert quote["pricing"]["total"] == expected_total
    assert "restock markup" in quote["quote_explanation"]
