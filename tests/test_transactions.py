import pandas as pd
import pytest

from beaverschoice.config import START_DATE
from beaverschoice.inventory import generate_sample_inventory, paper_supplies
from beaverschoice.transactions import (
    create_transaction,
    get_all_inventory,
    get_cash_balance,
    get_stock_level,
)


def test_init_database_seeds_inventory_and_transactions(engine):
    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
    expected_inventory = generate_sample_inventory(paper_supplies)

    assert len(inventory_df) == len(expected_inventory)

    transactions_df = pd.read_sql("SELECT * FROM transactions", engine)
    sales_rows = transactions_df[transactions_df["transaction_type"] == "sales"]

    assert len(sales_rows) == 1
    assert sales_rows.iloc[0]["price"] == 50000.0
    assert (transactions_df["transaction_type"] == "stock_orders").sum() == len(inventory_df)


def test_create_transaction_and_stock_level_updates(engine):
    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
    item = inventory_df.iloc[0]
    as_of = START_DATE.isoformat()

    initial_stock = int(get_stock_level(engine, item["item_name"], as_of)["current_stock"].iloc[0])
    sale_units = max(1, initial_stock // 3)
    sale_price = sale_units * item["unit_price"]

    new_id = create_transaction(engine, item["item_name"], "sales", sale_units, sale_price, as_of)

    assert isinstance(new_id, int)
    updated_stock = int(get_stock_level(engine, item["item_name"], as_of)["current_stock"].iloc[0])
    assert updated_stock == initial_stock - sale_units


def test_create_transaction_rejects_invalid_type(engine):
    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
    item_name = inventory_df.iloc[0]["item_name"]

    with pytest.raises(ValueError):
        create_transaction(engine, item_name, "return", 1, 1.0, START_DATE.isoformat())


def test_get_all_inventory_matches_seeded_stock(engine):
    as_of = START_DATE.isoformat()
    inventory_snapshot = get_all_inventory(engine, as_of)
    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)

    assert set(inventory_snapshot.keys()) == set(inventory_df["item_name"])
    for _, row in inventory_df.iterrows():
        assert inventory_snapshot[row["item_name"]] == row["current_stock"]


def test_get_cash_balance_matches_transactions(engine):
    as_of = START_DATE.isoformat()
    transactions_df = pd.read_sql("SELECT * FROM transactions", engine)

    sales_total = transactions_df.loc[transactions_df["transaction_type"] == "sales", "price"].sum()
    purchase_total = transactions_df.loc[transactions_df["transaction_type"] == "stock_orders", "price"].sum()
    expected_cash = float(sales_total - purchase_total)

    assert get_cash_balance(engine, as_of) == expected_cash
