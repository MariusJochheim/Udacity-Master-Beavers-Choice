from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from smolagents.agents import ToolCallingAgent
from smolagents.models import Model, OpenAIModel
from smolagents.tools import Tool

from beaverschoice.config import OPENAI_API_KEY, OPENAI_BASE_URL
from beaverschoice.db import get_engine
from beaverschoice.finance import generate_financial_report
from beaverschoice.tooling import _normalize_date_input


def _format_financial_summary(report: Dict[str, Any]) -> str:
    """Build a concise, customer-friendly financial status line."""
    as_of_date = report.get("as_of_date")
    cash = report.get("cash_balance")
    inventory_value = report.get("inventory_value")
    total_assets = report.get("total_assets")
    top_products = report.get("top_selling_products") or []

    parts = [f"Status as of {as_of_date}:"]
    if isinstance(cash, (int, float)):
        parts.append(f"Cash ${float(cash):.2f}")
    if isinstance(inventory_value, (int, float)):
        parts.append(f"Inventory ${float(inventory_value):.2f}")
    if isinstance(total_assets, (int, float)):
        parts.append(f"Assets ${float(total_assets):.2f}")

    summary = ", ".join(parts)

    if top_products:
        top = top_products[0] or {}
        name = top.get("item_name") or "top seller"
        revenue = top.get("total_revenue")
        if isinstance(revenue, (int, float)):
            summary += f". Top seller: {name} (${float(revenue):.2f} revenue)"
        else:
            summary += f". Top seller: {name}"

    return summary


class FinancialReportTool(Tool):
    """Tool that returns a finance snapshot for the requested date."""

    name = "financial_snapshot"
    description = "Generate a finance status snapshot with cash, inventory, and top sellers for a date."
    inputs = {"as_of_date": {"type": "string", "description": "ISO date to evaluate the snapshot", "nullable": True}}
    output_type = "object"

    def __init__(self, engine=None):
        super().__init__()
        self.engine = engine or get_engine()

    def forward(self, as_of_date: str | None = None) -> Dict[str, Any]:
        snapshot_date = _normalize_date_input(as_of_date or datetime.now(), "as_of_date")
        report = generate_financial_report(self.engine, snapshot_date)
        summary = _format_financial_summary(report)
        return {"summary": summary, "report": report}


class FinalAnswerTool(Tool):
    """Return the finance snapshot response as the final agent output."""

    name = "final_answer"
    description = "Return the final finance snapshot response."
    inputs = {"answer": {"type": "any", "description": "Final response content"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
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


class FinanceAgent(ToolCallingAgent):
    """ToolCallingAgent that reports finance status snapshots."""

    def __init__(self, engine=None, model: Model | None = None, **kwargs):
        self.engine = engine or get_engine()
        if model is None:
            raise ValueError("An LLM model must be provided to FinanceAgent")

        snapshot_tool = FinancialReportTool(self.engine)
        final_tool = FinalAnswerTool()

        super().__init__(
            tools=[snapshot_tool, final_tool],
            model=model,
            add_base_tools=False,
            max_tool_threads=1,
            instructions=(
                "You are the Finance agent. Call financial_snapshot once to generate a finance/status update using "
                "generate_financial_report, then return the result via final_answer."
            ),
            **kwargs,
        )


def create_openai_finance_agent(
    engine=None,
    model_id: str = "gpt-4o-mini",
    api_key: str | None = None,
    api_base: str | None = None,
    **kwargs,
) -> FinanceAgent:
    """
    Factory to build a FinanceAgent backed by an OpenAI-compatible endpoint.
    """
    api_key = api_key or OPENAI_API_KEY
    api_base = api_base or OPENAI_BASE_URL
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required to initialize the OpenAI model")

    model = OpenAIModel(model_id=model_id, api_key=api_key, api_base=api_base)
    return FinanceAgent(engine=engine, model=model, **kwargs)
