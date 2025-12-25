from datetime import datetime
from typing import Dict, Union

import pandas as pd
from sqlalchemy import Engine

from .db import get_engine


def _normalize_date(as_of_date: Union[str, datetime]) -> str:
    """Return an ISO 8601 string for date inputs."""
    if isinstance(as_of_date, datetime):
        return as_of_date.isoformat()
    return as_of_date


def create_transaction(
    engine: Engine | None,
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    Record a transaction of type 'stock_orders' or 'sales'.

    Returns the inserted row id.
    """
    engine = engine or get_engine()
    date_str = _normalize_date(date)

    if transaction_type not in {"stock_orders", "sales"}:
        raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

    transaction = pd.DataFrame(
        [
            {
                "item_name": item_name,
                "transaction_type": transaction_type,
                "units": quantity,
                "price": price,
                "transaction_date": date_str,
            }
        ]
    )

    transaction.to_sql("transactions", engine, if_exists="append", index=False)
    result = pd.read_sql("SELECT last_insert_rowid() as id", engine)
    return int(result.iloc[0]["id"])


def get_all_inventory(engine: Engine | None, as_of_date: str) -> Dict[str, int]:
    """Return available inventory as of a specific date."""
    engine = engine or get_engine()

    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    result = pd.read_sql(query, engine, params={"as_of_date": as_of_date})
    return dict(zip(result["item_name"], result["stock"]))


def get_stock_level(
    engine: Engine | None, item_name: str, as_of_date: Union[str, datetime]
) -> pd.DataFrame:
    """Return current stock for a specific item as of a date."""
    engine = engine or get_engine()
    as_of_date = _normalize_date(as_of_date)

    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    return pd.read_sql(stock_query, engine, params={"item_name": item_name, "as_of_date": as_of_date})


def get_cash_balance(engine: Engine | None, as_of_date: Union[str, datetime]) -> float:
    """Compute net cash balance as of the given date."""
    engine = engine or get_engine()
    as_of_date = _normalize_date(as_of_date)

    try:
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            engine,
            params={"as_of_date": as_of_date},
        )

        if transactions.empty:
            return 0.0

        total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
        total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
        return float(total_sales - total_purchases)

    except Exception as exc:  # pragma: no cover - defensive print
        print(f"Error getting cash balance: {exc}")
        return 0.0
