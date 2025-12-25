import ast
import pandas as pd
from sqlalchemy import Engine, create_engine

from .config import DB_URL, DEFAULT_SEED, START_DATE, data_path
from .inventory import generate_sample_inventory, paper_supplies


def get_engine(url: str = DB_URL) -> Engine:
    """Create a SQLAlchemy engine."""
    return create_engine(url)


def init_database(engine: Engine | None = None, seed: int = DEFAULT_SEED) -> Engine:
    """
    Set up the SQLite database with base tables and seed data.

    This loads CSV data for quote requests/quotes, seeds inventory and transactions,
    and returns the engine used.
    """
    engine = engine or get_engine()

    try:
        transactions_schema = pd.DataFrame(
            {
                "id": [],
                "item_name": [],
                "transaction_type": [],
                "units": [],
                "price": [],
                "transaction_date": [],
            }
        )
        transactions_schema.to_sql("transactions", engine, if_exists="replace", index=False)

        initial_date = START_DATE.isoformat()

        quote_requests_df = pd.read_csv(data_path("quote_requests.csv"))
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", engine, if_exists="replace", index=False)

        quotes_df = pd.read_csv(data_path("quotes.csv"))
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        quotes_df = quotes_df[
            [
                "request_id",
                "total_amount",
                "quote_explanation",
                "order_date",
                "job_type",
                "order_size",
                "event_type",
            ]
        ]
        quotes_df.to_sql("quotes", engine, if_exists="replace", index=False)

        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        initial_transactions = [
            {
                "item_name": None,
                "transaction_type": "sales",
                "units": None,
                "price": 50000.0,
                "transaction_date": initial_date,
            }
        ]

        for _, item in inventory_df.iterrows():
            initial_transactions.append(
                {
                    "item_name": item["item_name"],
                    "transaction_type": "stock_orders",
                    "units": item["current_stock"],
                    "price": item["current_stock"] * item["unit_price"],
                    "transaction_date": initial_date,
                }
            )

        pd.DataFrame(initial_transactions).to_sql("transactions", engine, if_exists="append", index=False)
        inventory_df.to_sql("inventory", engine, if_exists="replace", index=False)

        return engine

    except Exception as exc:  # pragma: no cover - defensive print
        print(f"Error initializing database: {exc}")
        raise
