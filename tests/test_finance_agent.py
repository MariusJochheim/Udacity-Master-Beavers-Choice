from datetime import timedelta

import pandas as pd

from beaverschoice.agents.finance_agent import FinancialReportTool
from beaverschoice.config import START_DATE
from beaverschoice.transactions import create_transaction


def test_financial_report_tool_returns_summary(engine):
    tool = FinancialReportTool(engine)

    result = tool.forward(START_DATE.isoformat())
    report = result["report"]
    summary = result["summary"].lower()

    assert "summary" in result and "report" in result
    assert report["as_of_date"].startswith(START_DATE.strftime("%Y-%m-%d"))
    assert isinstance(report["cash_balance"], float)
    assert isinstance(report["inventory_value"], float)
    assert "status as of" in summary
    assert "cash" in summary and "inventory" in summary


def test_financial_snapshot_updates_after_transactions(engine):
    tool = FinancialReportTool(engine)
    before = tool.forward(START_DATE.isoformat())

    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
    item = inventory_df.iloc[0]
    sale_units = max(1, int(item["current_stock"] // 3))
    sale_price = sale_units * float(item["unit_price"])
    after_date = (START_DATE + timedelta(days=1)).isoformat()

    create_transaction(engine, item["item_name"], "sales", sale_units, sale_price, after_date)
    after = tool.forward(after_date)

    assert after["report"]["cash_balance"] > before["report"]["cash_balance"]
    assert after["report"]["inventory_value"] < before["report"]["inventory_value"]
    assert "top seller" in after["summary"].lower()
    assert after["summary"] != before["summary"]
