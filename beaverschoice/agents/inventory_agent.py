from __future__ import annotations

from datetime import datetime
from typing import Any

from smolagents.agents import ToolCallingAgent
from smolagents.models import Model, OpenAIModel
from smolagents.tools import Tool

from beaverschoice.config import OPENAI_API_KEY, OPENAI_BASE_URL
from beaverschoice.db import get_engine
from beaverschoice.procurement import compute_procurement_decision
from beaverschoice.tooling import _normalize_date_input
from beaverschoice.transactions import get_all_inventory


class ProcurementTool(Tool):
    name = "inventory_procurement"
    description = (
        "Check inventory against a product request and recommend restock quantities with delivery estimates."
    )
    inputs = {
        "product_request": {"type": "string", "description": "Natural language description of the requested product"},
        "as_of_date": {
            "type": "string",
            "description": "ISO 8601 date representing when the request is made",
            "nullable": True,
        },
    }
    output_type = "object"

    def __init__(self, engine=None):
        super().__init__()
        self.engine = engine or get_engine()

    def forward(self, product_request: str, as_of_date: str | None = None) -> dict:
        try:
            decision = compute_procurement_decision(self.engine, product_request, as_of_date or datetime.now().isoformat())
            return decision.to_dict()
        except ValueError as exc:
            # Return structured error instead of raising so callers can gracefully respond.
            return {
                "error": str(exc),
                "request": product_request,
                "as_of_date": as_of_date or datetime.now().isoformat(),
            }


class InventorySnapshotTool(Tool):
    """Tool that returns current inventory by item using get_all_inventory."""

    name = "inventory_snapshot"
    description = "Return available inventory quantities per item as of a date."
    inputs = {
        "as_of_date": {"type": "string", "description": "ISO 8601 date representing when to check inventory", "nullable": True}
    }
    output_type = "object"

    def __init__(self, engine=None):
        super().__init__()
        self.engine = engine or get_engine()

    def forward(self, as_of_date: str | None = None) -> dict:
        date = _normalize_date_input(as_of_date or datetime.now(), "as_of_date")
        inventory = get_all_inventory(self.engine, date)
        return {"as_of_date": date, "inventory": inventory}


class FinalAnswerTool(Tool):
    """Lightweight final answer tool to avoid pulling in the full base toolset."""

    name = "final_answer"
    description = "Return the final answer payload for the inventory procurement request."
    inputs = {"answer": {"type": "any", "description": "Final answer content"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        # Smolagents can wrap tool results in text chunks; unwrap and prefer dict payloads.
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
            parsed = _parse_text_payload(str(answer["text"]))
            return parsed

        return answer


class InventoryProcurementAgent(ToolCallingAgent):
    """
    ToolCallingAgent that relies on an LLM (or compatible Model) to decide tool calls.

    The agent keeps only the procurement tool available and instructs the model to call
    it to produce the final answer.
    """

    def __init__(self, engine=None, model: Model | None = None, **kwargs):
        self.engine = engine or get_engine()
        if model is None:
            raise ValueError("An LLM model must be provided to InventoryProcurementAgent")
        procurement_tool = ProcurementTool(engine=self.engine)
        snapshot_tool = InventorySnapshotTool(engine=self.engine)
        final_answer_tool = FinalAnswerTool()
        super().__init__(
            tools=[procurement_tool, snapshot_tool, final_answer_tool],
            model=model,
            add_base_tools=False,
            max_tool_threads=1,
            instructions=(
                "You are the Inventory & Procurement agent. For product requests, call inventory_procurement exactly once "
                "with the customer request text and as_of_date. If the user asks for inventory levels by item, call "
                "inventory_snapshot. Pass any tool result directly to final_answer without summaries or rephrasing."
            ),
            **kwargs,
        )


def create_openai_inventory_agent(
    engine=None,
    model_id: str = "gpt-4o-mini",
    api_key: str | None = None,
    api_base: str | None = None,
    **kwargs,
) -> InventoryProcurementAgent:
    """
    Factory to build a ToolCallingAgent backed by an OpenAI-compatible endpoint.

    Credentials are pulled from environment variables OPENAI_API_KEY and OPENAI_BASE_URL
    unless explicitly provided.
    """
    api_key = api_key or OPENAI_API_KEY
    api_base = api_base or OPENAI_BASE_URL
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required to initialize the OpenAI model")

    model = OpenAIModel(model_id=model_id, api_key=api_key, api_base=api_base)
    return InventoryProcurementAgent(engine=engine, model=model, **kwargs)
