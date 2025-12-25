from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
import re
from typing import Union

import pandas as pd
from sqlalchemy import Engine

from .db import get_engine
from .logistics import get_supplier_delivery_date
from .transactions import get_stock_level
from .tooling import _normalize_date_input, _validate_nonempty_string


@dataclass
class ProcurementDecision:
    request: str
    matched_item: str
    as_of_date: str
    current_stock: int
    min_stock_level: int
    restock_recommended: bool
    recommended_order_quantity: int
    estimated_delivery_date: str

    def to_dict(self) -> dict:
        return asdict(self)


def _resolve_inventory_item(engine: Engine, product_request: str) -> tuple[str, int]:
    """
    Map a free-form product request to a known inventory item and its min stock.

    The function prefers substring matches; if none are found it falls back to
    token overlap to pick the closest item name.
    """
    request_text = _validate_nonempty_string(product_request, "product_request")
    request_lower = request_text.lower()
    # Lightweight synonym mapping to improve matches for common paraphrases.
    synonyms = {
        "poster board": "Large poster paper (24x36 inches)",
        "poster paper": "Large poster paper (24x36 inches)",
        "banner": "Banner paper",
        "banner paper": "Banner paper",
        "rolls of banner": "Rolls of banner paper (36-inch width)",
        "streamer": "Crepe paper",
        "streamers": "Crepe paper",
        "balloon": "Crepe paper",
        "balloons": "Crepe paper",
        "flyer": "A4 paper",
        "flyers": "A4 paper",
        "napkin": "Paper plates",
        "napkins": "Paper plates",
    }
    for keyword, canonical in synonyms.items():
        if keyword in request_lower:
            df = pd.read_sql(
                "SELECT min_stock_level FROM inventory WHERE item_name = :name LIMIT 1",
                engine,
                params={"name": canonical},
            )
            if not df.empty:
                return canonical, int(df.iloc[0]["min_stock_level"])
    inventory_df = pd.read_sql("SELECT item_name, min_stock_level FROM inventory", engine)
    if inventory_df.empty:
        raise ValueError("inventory table is empty; seed data before requesting a match")

    for row in inventory_df.itertuples(index=False):
        if row.item_name.lower() in request_lower:
            return str(row.item_name), int(row.min_stock_level)

    tokens = set(re.findall(r"[a-z0-9]+", request_lower))
    best_row = None
    best_score = 0
    for row in inventory_df.itertuples(index=False):
        item_tokens = set(re.findall(r"[a-z0-9]+", row.item_name.lower()))
        overlap = len(tokens & item_tokens)
        if overlap > best_score:
            best_score = overlap
            best_row = row

    if best_row and best_score > 0:
        return str(best_row.item_name), int(best_row.min_stock_level)

    raise ValueError("could not match the product request to any inventory item")


def compute_procurement_decision(
    engine: Engine | None, product_request: str, as_of_date: Union[str, datetime]
) -> ProcurementDecision:
    """
    Evaluate a product request against inventory and return procurement guidance.

    The function reports current stock, whether to restock, a suggested order size,
    and an estimated delivery date based on supplier lead times.
    """
    engine = engine or get_engine()
    as_of = _normalize_date_input(as_of_date, "as_of_date")
    item_name, min_stock_level = _resolve_inventory_item(engine, product_request)

    stock_df = get_stock_level(engine, item_name, as_of)
    current_stock = int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0

    restock_needed = current_stock <= min_stock_level
    recommended_qty = 0
    if restock_needed:
        recommended_qty = max(min_stock_level * 2 - current_stock, min_stock_level)

    eta_qty = recommended_qty if recommended_qty > 0 else 1
    estimated_delivery = get_supplier_delivery_date(as_of, eta_qty)

    return ProcurementDecision(
        request=product_request,
        matched_item=item_name,
        as_of_date=as_of,
        current_stock=current_stock,
        min_stock_level=min_stock_level,
        restock_recommended=restock_needed,
        recommended_order_quantity=recommended_qty,
        estimated_delivery_date=estimated_delivery,
    )
