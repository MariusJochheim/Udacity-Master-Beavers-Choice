from datetime import timedelta

import pandas as pd
import pytest

from beaverschoice import tooling
from beaverschoice.config import START_DATE


def test_inventory_snapshot_returns_typed_items(engine):
    snapshot = tooling.inventory_snapshot(engine, START_DATE)

    assert snapshot.as_of_date.startswith("2025-01-01")
    assert snapshot.items
    for item in snapshot.items:
        assert isinstance(item, tooling.InventoryItem)
        assert item.item_name
        assert item.current_stock >= 0
        assert item.min_stock_level > 0


def test_recent_quote_history_validates_inputs(engine):
    entries = tooling.recent_quote_history(engine, ["bulk", "festival"], limit=3)

    assert 0 < len(entries) <= 3
    for entry in entries:
        assert isinstance(entry, tooling.QuoteHistoryEntry)
        combined = f"{entry.original_request} {entry.quote_explanation}".lower()
        assert "bulk" in combined and "festival" in combined

    with pytest.raises(ValueError):
        tooling.recent_quote_history(engine, ["bulk"], limit=0)

    with pytest.raises(TypeError):
        tooling.recent_quote_history(engine, "bulk")


def test_delivery_eta_returns_valid_date(engine):
    estimate = tooling.delivery_eta(START_DATE, 12)
    expected = (START_DATE + timedelta(days=1)).strftime("%Y-%m-%d")

    assert isinstance(estimate, tooling.DeliveryEstimate)
    assert estimate.estimated_delivery_date == expected

    with pytest.raises(ValueError):
        tooling.delivery_eta(START_DATE, 0)


def test_record_transaction_returns_receipt(engine):
    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
    item = inventory_df.iloc[0]
    as_of = START_DATE.isoformat()
    initial_stock = int(item["current_stock"])

    receipt = tooling.record_transaction(
        engine,
        item_name=item["item_name"],
        transaction_type="sales",
        quantity=1,
        total_price=float(item["unit_price"]),
        as_of_date=as_of,
    )

    assert isinstance(receipt, tooling.TransactionReceipt)
    assert receipt.transaction_id > 0
    assert receipt.resulting_stock == initial_stock - 1

    with pytest.raises(ValueError):
        tooling.record_transaction(engine, item["item_name"], "sales", 0, 0.0, as_of)


def test_summarize_financials_returns_dataclass(engine):
    report = tooling.summarize_financials(engine, START_DATE)

    assert isinstance(report, tooling.FinancialReport)
    assert report.inventory_summary
    assert report.top_selling_products
    assert report.total_assets == pytest.approx(report.cash_balance + report.inventory_value)
    assert all(isinstance(entry, tooling.InventoryValuation) for entry in report.inventory_summary)
