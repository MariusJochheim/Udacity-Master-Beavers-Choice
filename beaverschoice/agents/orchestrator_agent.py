from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from smolagents.agents import ToolCallingAgent
from smolagents.models import Model, OpenAIModel
from smolagents.tools import Tool

from beaverschoice.config import OPENAI_API_KEY, OPENAI_BASE_URL
from beaverschoice.agents.order_execution_agent import OrderExecutionAgent, create_openai_order_execution_agent
from beaverschoice.db import get_engine
from beaverschoice.tooling import _normalize_date_input, _validate_nonempty_string, summarize_financials


def _format_quote_message(quote: Dict[str, Any]) -> str:
    pricing = quote.get("pricing", {})
    total = pricing.get("total")
    currency = pricing.get("currency", "USD")
    availability = quote.get("availability", {}) or {}
    eta = availability.get("estimated_delivery_date")

    message_parts = []
    if isinstance(total, (int, float)):
        message_parts.append(f"Quote prepared: ${total:.2f} {currency}.")
    else:
        message_parts.append("Quote prepared.")
    if eta:
        message_parts.append(f"Estimated delivery {eta}.")
    if quote.get("quote_explanation"):
        message_parts.append(str(quote["quote_explanation"]))

    return " ".join(message_parts).strip()


def _format_order_message(order_result: Dict[str, Any]) -> str:
    item = order_result.get("matched_item") or order_result.get("request")
    sale_receipt = order_result.get("sale_receipt") or {}
    eta = order_result.get("customer_eta") or order_result.get("estimated_delivery_date")
    quantity = sale_receipt.get("quantity") or order_result.get("quantity") or order_result.get("recommended_order_quantity")
    price = sale_receipt.get("price") or order_result.get("sale_price_total")

    parts = [f"Order placed for {item}."]
    if quantity:
        parts.append(f"Quantity: {quantity}.")
    if isinstance(price, (int, float)):
        parts.append(f"Total ${float(price):.2f}.")
    if eta:
        parts.append(f"ETA {eta}.")
    return " ".join(parts).strip()


def _format_status_message(status_result: Dict[str, Any]) -> str:
    as_of_date = status_result.get("as_of_date")
    cash = status_result.get("cash_balance")
    inventory_value = status_result.get("inventory_value")

    base = f"Status as of {as_of_date}:"
    if isinstance(cash, (int, float)) and isinstance(inventory_value, (int, float)):
        base += f" Cash ${cash:.2f}, Inventory ${inventory_value:.2f}."
    return base


class FinanceStatusAgent:
    """Lightweight facade around financial reporting."""

    def __init__(self, engine=None):
        self.engine = engine or get_engine()

    def status_snapshot(self, as_of_date: str) -> Dict[str, Any]:
        report = summarize_financials(self.engine, as_of_date)
        return {
            "as_of_date": report.as_of_date,
            "cash_balance": report.cash_balance,
            "inventory_value": report.inventory_value,
            "top_selling_products": report.top_selling_products,
        }


class QuoteTool(Tool):
    """Tool that delegates to the quote agent and returns a customer-friendly summary."""

    name = "quote_request"
    description = "Generate a quote with pricing and delivery estimate for a customer request."
    inputs = {
        "request_text": {"type": "string", "description": "The customer's request text"},
        "as_of_date": {"type": "string", "description": "ISO date for the request", "nullable": True},
        "order_size": {"type": "string", "description": "Optional order size descriptor", "nullable": True},
        "job_type": {"type": "string", "description": "Optional job type", "nullable": True},
        "event_type": {"type": "string", "description": "Optional event type", "nullable": True},
    }
    output_type = "string"

    def __init__(self, quote_agent):
        super().__init__()
        if quote_agent is None:
            raise ValueError("quote_agent is required for QuoteTool")
        self.quote_agent = quote_agent

    def forward(
        self,
        request_text: str,
        as_of_date: str | None = None,
        order_size: str | None = None,
        job_type: str | None = None,
        event_type: str | None = None,
    ) -> str:
        request = _validate_nonempty_string(request_text, "request_text")
        payload: Dict[str, Any] = {
            "request": request,
            "as_of_date": _normalize_date_input(as_of_date or datetime.now(), "as_of_date"),
            "order_size": order_size,
            "job_type": job_type,
            "event_type": event_type,
        }
        quote = self.quote_agent.generate_quote(payload)
        return _format_quote_message(quote)


class OrderTool(Tool):
    """Tool that places an order through the OrderExecutionAgent and returns the delivery summary."""

    name = "order_request"
    description = "Place an order for the requested items and return delivery ETA."
    inputs = {
        "request_text": {"type": "string", "description": "The customer's request text"},
        "as_of_date": {"type": "string", "description": "ISO date for the request", "nullable": True},
        "quantity": {"type": "integer", "description": "Units to order", "nullable": True},
        "quoted_total": {"type": "number", "description": "Total from prior quote", "nullable": True},
    }
    output_type = "string"

    def __init__(self, order_agent: OrderExecutionAgent):
        super().__init__()
        self.order_agent = order_agent

    def forward(self, request_text: str, as_of_date: str | None = None, quantity: int | None = None, quoted_total: float | None = None) -> str:
        request = _validate_nonempty_string(request_text, "request_text")
        payload = {
            "request_text": request,
            "as_of_date": _normalize_date_input(as_of_date or datetime.now(), "as_of_date"),
            "quantity": quantity,
            "quoted_total": quoted_total,
        }
        order_result = self.order_agent.run(request, additional_args=payload)
        if not isinstance(order_result, dict):
            raise ValueError("order execution agent must return a dictionary payload")
        return _format_order_message(order_result)


class StatusTool(Tool):
    """Tool that returns a financial status snapshot."""

    name = "status_snapshot"
    description = "Return current financial and inventory status as of a date."
    inputs = {"as_of_date": {"type": "string", "description": "ISO date to evaluate the status", "nullable": True}}
    output_type = "string"

    def __init__(self, finance_agent: FinanceStatusAgent):
        super().__init__()
        self.finance_agent = finance_agent

    def forward(self, as_of_date: str | None = None) -> str:
        date = _normalize_date_input(as_of_date or datetime.now(), "as_of_date")
        status = self.finance_agent.status_snapshot(date)
        return _format_status_message(status)


class FinalAnswerTool(Tool):
    """Lightweight final answer tool to terminate the tool-calling loop."""

    name = "final_answer"
    description = "Return the final customer-facing response."
    inputs = {"answer": {"type": "any", "description": "Final response content"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        # Smolagents passes tool results as a list of text chunks; unwrap to a string for callers.
        if isinstance(answer, list):
            text_parts = []
            for item in answer:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(str(item["text"]))
            if text_parts:
                return " ".join(text_parts)
        if isinstance(answer, dict) and "text" in answer:
            return str(answer["text"])
        return answer


class OrchestratorAgent(ToolCallingAgent):
    """
    LLM-driven orchestrator that routes to quote/order/status tools via smolagents.

    The order path delegates to OrderExecutionAgent so sales are recorded and replenishment
    can be triggered when needed.
    """

    def __init__(
        self,
        engine=None,
        quote_agent=None,
        order_agent: OrderExecutionAgent | None = None,
        finance_agent: FinanceStatusAgent | None = None,
        model: Model | None = None,
        **kwargs,
    ):
        if model is None:
            raise ValueError("An LLM model must be provided to OrchestratorAgent")
        if quote_agent is None:
            raise ValueError("quote_agent is required for quoting flows")
        if order_agent is None:
            raise ValueError("order_agent is required for order execution flows")

        self.engine = engine or get_engine()
        self.quote_agent = quote_agent
        self.order_agent = order_agent
        self.finance_agent = finance_agent or FinanceStatusAgent(self.engine)

        quote_tool = QuoteTool(self.quote_agent)
        order_tool = OrderTool(self.order_agent)
        status_tool = StatusTool(self.finance_agent)
        final_tool = FinalAnswerTool()

        super().__init__(
            tools=[quote_tool, order_tool, status_tool, final_tool],
            model=model,
            add_base_tools=False,
            max_tool_threads=1,
            instructions=(
                "You are the Orchestrator agent. Identify if the user wants a quote, to place an order (record the sale "
                "and replenish if needed), or a financial/status update. Call exactly one of the tools: "
                "quote_request (for pricing/quotes), order_request (for executing the order and ETA), "
                "status_snapshot (for finance/inventory status). When available, pass through as_of_date, "
                "job_type, order_size, event_type, quantity, and quoted_total arguments to keep context. When done, "
                "call final_answer with the tool result as the answer."
            ),
            **kwargs,
        )


def create_openai_orchestrator_agent(
    quote_agent,
    engine=None,
    model_id: str = "gpt-4o-mini",
    api_key: str | None = None,
    api_base: str | None = None,
    order_agent: OrderExecutionAgent | None = None,
    finance_agent: FinanceStatusAgent | None = None,
    **kwargs,
) -> OrchestratorAgent:
    """
    Factory to build an orchestrator agent backed by an OpenAI-compatible endpoint.
    """
    api_key = api_key or OPENAI_API_KEY
    api_base = api_base or OPENAI_BASE_URL
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required to initialize the OpenAI model")

    model = OpenAIModel(model_id=model_id, api_key=api_key, api_base=api_base)
    order_agent = order_agent or create_openai_order_execution_agent(engine=engine, model_id=model_id, api_key=api_key, api_base=api_base)

    return OrchestratorAgent(
        engine=engine,
        quote_agent=quote_agent,
        order_agent=order_agent,
        finance_agent=finance_agent,
        model=model,
        **kwargs,
    )
