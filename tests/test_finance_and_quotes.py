from datetime import timedelta

import pandas as pd
import pytest

from beaverschoice.config import START_DATE
from beaverschoice.finance import generate_financial_report
from beaverschoice.quotes import search_quote_history
from beaverschoice.transactions import create_transaction


def test_financial_report_reflects_sales_and_inventory(engine):
    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
    item = inventory_df.iloc[0]
    before_date = START_DATE.isoformat()
    after_date = (START_DATE + timedelta(days=1)).isoformat()

    sale_units = min(10, int(item["current_stock"] // 2))
    sale_price = sale_units * item["unit_price"]

    before_report = generate_financial_report(engine, before_date)
    create_transaction(engine, item["item_name"], "sales", sale_units, sale_price, after_date)
    after_report = generate_financial_report(engine, after_date)

    assert after_report["cash_balance"] == pytest.approx(before_report["cash_balance"] + sale_price)
    assert after_report["inventory_value"] == pytest.approx(
        before_report["inventory_value"] - sale_units * item["unit_price"]
    )
    assert after_report["total_assets"] == pytest.approx(
        after_report["cash_balance"] + after_report["inventory_value"]
    )


def test_search_quote_history_filters_by_terms(engine):
    results = search_quote_history(engine, ["bulk", "festival"], limit=10)

    assert results
    for entry in results:
        combined = f"{entry.get('original_request', '')} {entry.get('quote_explanation', '')}".lower()
        assert "bulk" in combined and "festival" in combined


def test_search_quote_history_defaults_to_limit(engine):
    results = search_quote_history(engine, [])

    assert len(results) == 5
