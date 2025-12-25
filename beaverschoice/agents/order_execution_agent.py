from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List
import re

import pandas as pd
from smolagents.agents import ToolCallingAgent
from smolagents.models import Model, OpenAIModel
from smolagents.tools import Tool

from beaverschoice.config import OPENAI_API_KEY, OPENAI_BASE_URL
from beaverschoice.db import get_engine
from beaverschoice.logistics import get_supplier_delivery_date
from beaverschoice.procurement import compute_procurement_decision
from beaverschoice.tooling import (
    TransactionReceipt,
    _normalize_date_input,
    _validate_nonempty_string,
    _validate_positive_int,
    _validate_price,
    record_transaction,
)
from beaverschoice.transactions import get_cash_balance


def _unit_price(engine, item_name: str) -> float:
    """Lookup the unit price for an inventory item."""
    df = pd.read_sql(
        "SELECT unit_price FROM inventory WHERE item_name = :name LIMIT 1",
        engine,
        params={"name": item_name},
    )
    if df.empty:
        raise ValueError(f"Item '{item_name}' not found in inventory")
    return float(df.iloc[0]["unit_price"])


def _receipt_to_dict(receipt: TransactionReceipt) -> Dict[str, Any]:
    """Convert a TransactionReceipt dataclass into a serializable dict."""
    return asdict(receipt)


def _parse_request_items(request_text: str, fallback_quantity: int) -> List[Dict[str, Any]]:
    """
    Extract individual item requests from free-form text.

    The parser is intentionally simple: it looks for leading quantities on lines and
    splits on common bullet markers, then falls back to inline quantity parsing. Non-quantity
    lines are ignored unless nothing was detected, in which case the full request is used.
    """
    items: list[Dict[str, Any]] = []
    line_pattern = re.compile(r"^(?P<qty>[0-9][0-9,]*)\\s*(?:[a-zA-Z]+)?\\s*(?:of\\s+)?(?P<item>.+)$")
    for raw_line in request_text.splitlines():
        line = raw_line.strip().lstrip("-•* ").strip()
        if not line:
            continue
        match = line_pattern.match(line)
        if match:
            qty = int(match.group("qty").replace(",", ""))
            item_desc = match.group("item").strip()
            if item_desc:
                items.append({"request_text": item_desc, "quantity": qty})
    # Fall back to inline parsing if no bullet-style quantities were found.
    if not items:
        inline_pattern = re.compile(r"(?P<qty>[0-9][0-9,]*)\\s+(?P<item>[a-zA-Z][^,;\\n]+)")
        for match in inline_pattern.finditer(request_text):
            qty = int(match.group("qty").replace(",", ""))
            item_desc = match.group("item").strip()
            if item_desc:
                items.append({"request_text": item_desc, "quantity": qty})

    if not items:
        items.append({"request_text": request_text.strip(), "quantity": fallback_quantity})

    return items


class ExecuteOrderTool(Tool):
    """Process a customer order, record the sale, and trigger replenishment if needed."""

    name = "execute_order"
    description = (
        "Place a customer order by recording a sales transaction, check for replenishment needs, "
        "and return the customer ETA."
    )
    inputs = {
        "request_text": {"type": "string", "description": "Customer request text to match inventory"},
        "quantity": {"type": "integer", "description": "Units to sell to the customer", "nullable": True},
        "as_of_date": {"type": "string", "description": "ISO date for the transaction", "nullable": True},
        "quoted_total": {"type": "number", "description": "Total price from a prior quote", "nullable": True},
    }
    output_type = "object"

    def __init__(self, engine=None):
        super().__init__()
        self.engine = engine or get_engine()

    def forward(
        self,
        request_text: str,
        quantity: int | None = None,
        as_of_date: str | None = None,
        quoted_total: float | None = None,
    ) -> Dict[str, Any]:
        request = _validate_nonempty_string(request_text, "request_text")
        qty = _validate_positive_int(quantity or 1, "quantity")
        as_of = _normalize_date_input(as_of_date or datetime.now(), "as_of_date")

        # Parse multiple item lines so we don't silently ignore parts of the request.
        parsed_items = _parse_request_items(request, qty)
        total_requested_qty = sum(item["quantity"] for item in parsed_items)
        # Track cash so we can avoid overspending on restocks.
        cash_available = get_cash_balance(self.engine, as_of)
        overall_sale = 0.0
        overall_restock_spend = 0.0
        item_results: list[dict[str, Any]] = []

        for item in parsed_items:
            try:
                decision = compute_procurement_decision(self.engine, item["request_text"], as_of)
            except ValueError as exc:
                item_results.append(
                    {
                        "request": item["request_text"],
                        "error": str(exc),
                        "requested_quantity": item["quantity"],
                    }
                )
                continue

            item_name = decision.matched_item
            item_qty = _validate_positive_int(item["quantity"], "quantity")
            unit_price = _unit_price(self.engine, item_name)
            available_stock = max(int(decision.current_stock), 0)

            ship_qty = min(item_qty, available_stock)
            backorder_qty = max(0, item_qty - ship_qty)

            sale_receipt = None
            sale_total = 0.0
            if ship_qty > 0:
                if quoted_total is not None and total_requested_qty > 0:
                    proportional_total = _validate_price(quoted_total) * (item_qty / total_requested_qty)
                    sale_total = proportional_total * (ship_qty / item_qty)
                else:
                    sale_total = unit_price * ship_qty

                sale_receipt = record_transaction(
                    self.engine,
                    item_name=item_name,
                    transaction_type="sales",
                    quantity=ship_qty,
                    total_price=sale_total,
                    as_of_date=as_of,
                )
                cash_available += sale_total
                overall_sale += sale_total

            post_stock = max(0, sale_receipt.resulting_stock if sale_receipt else available_stock)
            restock_payload = None
            final_stock = post_stock

            restock_qty = 0
            # Cover any backordered units and refill to a healthy buffer if low.
            if backorder_qty > 0:
                restock_qty = max(restock_qty, backorder_qty + decision.min_stock_level)
            if post_stock <= decision.min_stock_level:
                restock_qty = max(
                    restock_qty, max(decision.min_stock_level * 2 - post_stock, decision.min_stock_level)
                )

            restock_budget_blocked = False
            if restock_qty > 0:
                restock_price = unit_price * restock_qty
                if restock_price > cash_available:
                    affordable_units = int(cash_available // max(unit_price, 0.01))
                    if affordable_units <= 0:
                        restock_budget_blocked = True
                        restock_qty = 0
                    else:
                        restock_qty = affordable_units
                        restock_price = unit_price * restock_qty
                        restock_budget_blocked = True

                if restock_qty > 0:
                    restock_receipt = record_transaction(
                        self.engine,
                        item_name=item_name,
                        transaction_type="stock_orders",
                        quantity=restock_qty,
                        total_price=restock_price,
                        as_of_date=as_of,
                    )
                    restock_eta = get_supplier_delivery_date(as_of, restock_qty)
                    restock_payload = {
                        "receipt": _receipt_to_dict(restock_receipt),
                        "estimated_delivery_date": restock_eta,
                        "budget_limited": restock_budget_blocked,
                    }
                    final_stock = restock_receipt.resulting_stock
                    cash_available -= restock_price
                    overall_restock_spend += restock_price

            if sale_receipt:
                sale_receipt.resulting_stock = final_stock
            customer_eta = get_supplier_delivery_date(as_of, ship_qty if ship_qty > 0 else backorder_qty or 1)

            item_results.append(
                {
                    "request": item["request_text"],
                    "matched_item": item_name,
                    "requested_quantity": item_qty,
                    "shipped_quantity": ship_qty,
                    "backorder_quantity": backorder_qty,
                    "sale_price_total": round(sale_total, 2),
                    "sale_receipt": _receipt_to_dict(sale_receipt) if sale_receipt else None,
                    "restock": restock_payload,
                    "customer_eta": customer_eta,
                    "final_stock": final_stock,
                    "budget_blocked_restock": restock_budget_blocked,
                }
            )

        # Provide a backward-compatible summary for downstream formatting.
        first_item = next((item for item in item_results if not item.get("error")), item_results[0] if item_results else {})
        eta_values = [item.get("customer_eta") for item in item_results if item.get("customer_eta")]
        overall_eta = max(eta_values) if eta_values else None
        return {
            "request": request,
            "matched_item": first_item.get("matched_item"),
            "quantity": total_requested_qty,
            "items": item_results,
            "overall_sale_total": round(overall_sale, 2),
            "overall_restock_spend": round(overall_restock_spend, 2),
            "cash_balance_after": round(cash_available, 2),
            "customer_eta": overall_eta,
        }


class FinalAnswerTool(Tool):
    """Return the final answer payload for the order execution request."""

    name = "final_answer"
    description = "Return the final answer payload for the order execution request."
    inputs = {"answer": {"type": "any", "description": "Final answer content"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        # Smolagents sometimes wraps tool results in text chunks; unwrap when possible.
        def _parse_text_payload(text: str) -> Any:
            if "Observation:" in text:
                text = text.split("Observation:", 1)[1].strip()
            stripped = text.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                import ast

                try:
                    return ast.literal_eval(stripped)
                except Exception:
                    return stripped
            return stripped

        if isinstance(answer, list):
            text_parts = []
            for item in answer:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(str(item["text"]))
                elif isinstance(item, str):
                    text_parts.append(item)
            if text_parts:
                parsed = [_parse_text_payload(part) for part in text_parts]
                for candidate in parsed:
                    if isinstance(candidate, dict):
                        return candidate
                return " ".join(str(part) for part in parsed)

        if isinstance(answer, dict) and "text" in answer:
            return _parse_text_payload(str(answer["text"]))

        if isinstance(answer, str):
            parsed = _parse_text_payload(answer)
            return parsed

        return answer


class OrderExecutionAgent(ToolCallingAgent):
    """
    ToolCallingAgent that records sales, triggers replenishment via transactions, and shares customer ETA.
    """

    def __init__(self, engine=None, model: Model | None = None, **kwargs):
        self.engine = engine or get_engine()
        if model is None:
            raise ValueError("An LLM model must be provided to OrderExecutionAgent")

        execute_tool = ExecuteOrderTool(engine=self.engine)
        self.execute_tool = execute_tool
        final_tool = FinalAnswerTool()

        super().__init__(
            tools=[execute_tool, final_tool],
            model=model,
            add_base_tools=False,
            max_tool_threads=1,
            instructions=(
                "You are the Order Execution agent. Call the execute_order tool exactly once using the customer's "
                "request text, order quantity, as_of_date, and quoted_total when provided. Pass the tool's JSON "
                "result directly to final_answer without rephrasing."
            ),
            **kwargs,
        )


def create_openai_order_execution_agent(
    engine=None,
    model_id: str = "gpt-4o-mini",
    api_key: str | None = None,
    api_base: str | None = None,
    **kwargs,
) -> OrderExecutionAgent:
    """
    Factory to build a ToolCallingAgent backed by an OpenAI-compatible endpoint.
    """
    api_key = api_key or OPENAI_API_KEY
    api_base = api_base or OPENAI_BASE_URL
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required to initialize the OpenAI model")

    model = OpenAIModel(model_id=model_id, api_key=api_key, api_base=api_base)
    return OrderExecutionAgent(engine=engine, model=model, **kwargs)
