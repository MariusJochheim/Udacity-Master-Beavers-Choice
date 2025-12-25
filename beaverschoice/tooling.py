from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence, Union

import pandas as pd
from sqlalchemy import Engine

from .db import get_engine
from .finance import generate_financial_report
from .logistics import get_supplier_delivery_date
from .quotes import search_quote_history
from .transactions import create_transaction, get_stock_level


@dataclass
class InventoryItem:
    item_name: str
    category: str
    unit_price: float
    current_stock: int
    min_stock_level: int


@dataclass
class InventorySnapshot:
    as_of_date: str
    items: List[InventoryItem]


@dataclass
class QuoteHistoryEntry:
    original_request: str
    total_amount: float
    quote_explanation: str
    job_type: str
    order_size: str
    event_type: str
    order_date: str


@dataclass
class DeliveryEstimate:
    requested_on: str
    quantity: int
    estimated_delivery_date: str


@dataclass
class TransactionReceipt:
    transaction_id: int
    item_name: str
    transaction_type: str
    quantity: int
    price: float
    as_of_date: str
    resulting_stock: int


@dataclass
class InventoryValuation:
    item_name: str
    stock: float
    unit_price: float
    value: float


@dataclass
class TopProduct:
    item_name: str
    total_units: float
    total_revenue: float


@dataclass
class FinancialReport:
    as_of_date: str
    cash_balance: float
    inventory_value: float
    total_assets: float
    inventory_summary: List[InventoryValuation]
    top_selling_products: List[TopProduct]


def _ensure_engine(engine: Engine | None) -> Engine:
    return engine or get_engine()


def _normalize_date_input(date_input: Union[str, datetime], field_name: str = "date") -> str:
    """Validate and normalize date-like inputs to an ISO 8601 string."""
    if isinstance(date_input, datetime):
        return date_input.isoformat()

    if isinstance(date_input, str):
        try:
            parsed = datetime.fromisoformat(date_input)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(f"{date_input}T00:00:00")
            except ValueError as exc:
                raise ValueError(f"{field_name} must be ISO 8601 formatted") from exc
        return parsed.isoformat()

    raise TypeError(f"{field_name} must be a string or datetime")


def _validate_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return value


def _validate_nonempty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


def _validate_price(price: float) -> float:
    if not isinstance(price, (int, float)):
        raise TypeError("price must be numeric")
    price_f = float(price)
    if price_f < 0:
        raise ValueError("price cannot be negative")
    return price_f


def _coerce_search_terms(search_terms: Sequence[str]) -> list[str]:
    if isinstance(search_terms, str):
        raise TypeError("search_terms must be a sequence of strings, not a single string")
    cleaned_terms: list[str] = []
    for term in search_terms:
        if not isinstance(term, str):
            raise TypeError("each search term must be a string")
        stripped = term.strip()
        if stripped:
            cleaned_terms.append(stripped)
    return cleaned_terms


def inventory_snapshot(engine: Engine | None, as_of_date: Union[str, datetime]) -> InventorySnapshot:
    """
    Typed wrapper for inventory state as of a date.

    This uses transactions to compute current stock per item so the values stay
    consistent even after multiple stock or sales transactions.
    """
    engine = _ensure_engine(engine)
    as_of = _normalize_date_input(as_of_date, "as_of_date")
    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)

    if inventory_df.empty:
        raise ValueError("inventory table is empty; seed data before requesting a snapshot")

    items: list[InventoryItem] = []
    for row in inventory_df.itertuples(index=False):
        stock_df = get_stock_level(engine, row.item_name, as_of)
        stock_value = stock_df["current_stock"].iloc[0] if not stock_df.empty else 0
        items.append(
            InventoryItem(
                item_name=str(row.item_name),
                category=str(row.category),
                unit_price=float(row.unit_price),
                current_stock=int(stock_value or 0),
                min_stock_level=int(row.min_stock_level),
            )
        )

    return InventorySnapshot(as_of_date=as_of, items=items)


def recent_quote_history(
    engine: Engine | None, search_terms: Sequence[str], limit: int = 5
) -> List[QuoteHistoryEntry]:
    """Validated access to quote history with typed output."""
    engine = _ensure_engine(engine)
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    terms = _coerce_search_terms(search_terms)
    results = search_quote_history(engine, terms, limit)

    entries: list[QuoteHistoryEntry] = []
    for entry in results:
        required_keys = {
            "original_request",
            "total_amount",
            "quote_explanation",
            "job_type",
            "order_size",
            "event_type",
            "order_date",
        }
        missing = required_keys - set(entry.keys())
        if missing:
            raise ValueError(f"quote history row missing keys: {missing}")

        entries.append(
            QuoteHistoryEntry(
                original_request=str(entry["original_request"] or ""),
                total_amount=float(entry["total_amount"]),
                quote_explanation=str(entry["quote_explanation"] or ""),
                job_type=str(entry["job_type"] or ""),
                order_size=str(entry["order_size"] or ""),
                event_type=str(entry["event_type"] or ""),
                order_date=str(entry["order_date"]),
            )
        )

    return entries


def delivery_eta(order_date: Union[str, datetime], quantity: int) -> DeliveryEstimate:
    """Return a validated supplier delivery estimate."""
    normalized_date = _normalize_date_input(order_date, "order_date")
    qty = _validate_positive_int(quantity, "quantity")

    estimated = get_supplier_delivery_date(normalized_date, qty)
    try:
        # Validate date output format; only the date portion is expected.
        datetime.fromisoformat(estimated)
    except ValueError as exc:
        raise ValueError("delivery ETA could not be parsed") from exc

    return DeliveryEstimate(requested_on=normalized_date, quantity=qty, estimated_delivery_date=estimated)


def record_transaction(
    engine: Engine | None,
    item_name: str,
    transaction_type: str,
    quantity: int,
    total_price: float,
    as_of_date: Union[str, datetime],
) -> TransactionReceipt:
    """
    Create a transaction with validation and typed output.

    The function ensures the item exists in the inventory and returns the new
    stock level so agents can chain decisions confidently.
    """
    engine = _ensure_engine(engine)
    item_name = _validate_nonempty_string(item_name, "item_name")
    transaction_type = _validate_nonempty_string(transaction_type, "transaction_type")
    if transaction_type not in {"stock_orders", "sales"}:
        raise ValueError("transaction_type must be either 'stock_orders' or 'sales'")

    qty = _validate_positive_int(quantity, "quantity")
    price = _validate_price(total_price)
    as_of = _normalize_date_input(as_of_date, "as_of_date")

    inventory_row = pd.read_sql(
        "SELECT 1 FROM inventory WHERE item_name = :name LIMIT 1", engine, params={"name": item_name}
    )
    if inventory_row.empty:
        raise ValueError(f"item '{item_name}' is not in the inventory catalog")

    transaction_id = create_transaction(engine, item_name, transaction_type, qty, price, as_of)
    resulting_stock_df = get_stock_level(engine, item_name, as_of)
    resulting_stock = int(resulting_stock_df["current_stock"].iloc[0]) if not resulting_stock_df.empty else 0

    return TransactionReceipt(
        transaction_id=transaction_id,
        item_name=item_name,
        transaction_type=transaction_type,
        quantity=qty,
        price=price,
        as_of_date=as_of,
        resulting_stock=resulting_stock,
    )


def summarize_financials(engine: Engine | None, as_of_date: Union[str, datetime]) -> FinancialReport:
    """
    Typed wrapper around generate_financial_report with output validation.
    """
    engine = _ensure_engine(engine)
    as_of = _normalize_date_input(as_of_date, "as_of_date")
    report = generate_financial_report(engine, as_of)

    required_keys = {
        "as_of_date",
        "cash_balance",
        "inventory_value",
        "total_assets",
        "inventory_summary",
        "top_selling_products",
    }
    missing = required_keys - set(report.keys())
    if missing:
        raise ValueError(f"financial report missing keys: {missing}")

    inventory_summary = [
        InventoryValuation(
            item_name=str(entry.get("item_name", "")),
            stock=float(entry.get("stock", 0)),
            unit_price=float(entry.get("unit_price", 0.0)),
            value=float(entry.get("value", 0.0)),
        )
        for entry in report["inventory_summary"]
    ]

    top_selling_products = [
        TopProduct(
            item_name=str(entry.get("item_name", "")),
            total_units=float(entry.get("total_units") or 0),
            total_revenue=float(entry.get("total_revenue") or 0.0),
        )
        for entry in report["top_selling_products"]
    ]

    return FinancialReport(
        as_of_date=str(report["as_of_date"]),
        cash_balance=float(report["cash_balance"]),
        inventory_value=float(report["inventory_value"]),
        total_assets=float(report["total_assets"]),
        inventory_summary=inventory_summary,
        top_selling_products=top_selling_products,
    )
