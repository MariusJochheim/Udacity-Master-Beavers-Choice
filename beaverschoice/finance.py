from datetime import datetime
from typing import Dict, Union

import pandas as pd
from sqlalchemy import Engine

from .db import get_engine
from .transactions import get_cash_balance, get_stock_level


def _normalize_date(as_of_date: Union[str, datetime]) -> str:
    if isinstance(as_of_date, datetime):
        return as_of_date.isoformat()
    return as_of_date


def generate_financial_report(engine: Engine | None, as_of_date: Union[str, datetime]) -> Dict:
    """Generate a financial report snapshot as of a date."""
    engine = engine or get_engine()
    as_of_date = _normalize_date(as_of_date)

    cash = get_cash_balance(engine, as_of_date)
    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
    inventory_value = 0.0
    inventory_summary = []

    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(engine, item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value
        inventory_summary.append(
            {
                "item_name": item["item_name"],
                "stock": stock,
                "unit_price": item["unit_price"],
                "value": item_value,
            }
        )

    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, engine, params={"date": as_of_date})

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_sales.to_dict(orient="records"),
    }
