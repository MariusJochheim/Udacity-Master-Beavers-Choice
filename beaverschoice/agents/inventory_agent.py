from __future__ import annotations

from datetime import datetime
from typing import Any

from smolagents.agents import ToolCallingAgent
from smolagents.models import Model, OpenAIModel
from smolagents.tools import Tool

from beaverschoice.config import OPENAI_API_KEY, OPENAI_BASE_URL
from beaverschoice.db import get_engine
from beaverschoice.procurement import compute_procurement_decision


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
        decision = compute_procurement_decision(self.engine, product_request, as_of_date or datetime.now().isoformat())
        return decision.to_dict()


class FinalAnswerTool(Tool):
    """Lightweight final answer tool to avoid pulling in the full base toolset."""

    name = "final_answer"
    description = "Return the final answer payload for the inventory procurement request."
    inputs = {"answer": {"type": "any", "description": "Final answer content"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
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
        final_answer_tool = FinalAnswerTool()
        super().__init__(
            tools=[procurement_tool, final_answer_tool],
            model=model,
            add_base_tools=False,
            max_tool_threads=1,
            instructions=(
                "You are the Inventory & Procurement agent. Always call the inventory_procurement tool with the "
                "customer request text and date (as_of_date). Respond only with the tool result."
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
