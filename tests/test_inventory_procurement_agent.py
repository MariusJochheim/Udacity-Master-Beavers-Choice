import pandas as pd
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole, Model

from beaverschoice.agents.inventory_agent import InventoryProcurementAgent
from beaverschoice.logistics import get_supplier_delivery_date
from beaverschoice.procurement import compute_procurement_decision
from beaverschoice.transactions import create_transaction, get_stock_level


class FakeProcurementModel(Model):
    """Deterministic model that calls the procurement tool, then returns a final answer on the next turn."""

    def __init__(self, engine, request: str, as_of_date: str):
        super().__init__()
        self.request = request
        self.as_of_date = as_of_date
        self.precomputed = compute_procurement_decision(engine, request, as_of_date).to_dict()
        self._called_once = False

    def generate(self, messages, **kwargs):
        if not self._called_once:
            self._called_once = True
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(
                        name="inventory_procurement",
                        arguments={"product_request": self.request, "as_of_date": self.as_of_date},
                    ),
                    id="call_1",
                    type="function",
                )
            ]
        else:
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(
                        name="final_answer",
                        arguments={"answer": self.precomputed},
                    ),
                    id="call_2",
                    type="function",
                )
            ]

        return ChatMessage(role=MessageRole.ASSISTANT, tool_calls=tool_calls)


def test_inventory_procurement_agent_reports_stock_and_eta(engine, start_date):
    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
    item = inventory_df.iloc[0]
    request = f"Need {item['item_name']} for the office"

    model = FakeProcurementModel(engine, request, start_date)
    agent = InventoryProcurementAgent(engine, model=model)
    decision = agent.run(request, additional_args={"as_of_date": start_date})

    assert decision["matched_item"] == item["item_name"]
    expected_stock = get_stock_level(engine, item["item_name"], start_date)["current_stock"].iloc[0]
    assert decision["current_stock"] == expected_stock
    eta_basis = decision["recommended_order_quantity"] if decision["recommended_order_quantity"] > 0 else 1
    assert decision["estimated_delivery_date"] == get_supplier_delivery_date(start_date, eta_basis)


def test_inventory_procurement_agent_recommends_restock_when_low(engine, start_date):
    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
    item = inventory_df.iloc[0]
    sale_units = max(1, int(item["current_stock"] - item["min_stock_level"] + 1))
    sale_units = min(sale_units, int(item["current_stock"]))
    create_transaction(
        engine,
        item["item_name"],
        "sales",
        sale_units,
        float(item["unit_price"] * sale_units),
        start_date,
    )

    model = FakeProcurementModel(engine, f"Requesting {item['item_name']}", start_date)
    agent = InventoryProcurementAgent(engine, model=model)
    decision = agent.run(f"Requesting {item['item_name']}", additional_args={"as_of_date": start_date})

    assert decision["restock_recommended"] is True
    remaining_stock = get_stock_level(engine, item["item_name"], start_date)["current_stock"].iloc[0]
    expected_order = max(int(item["min_stock_level"] * 2 - remaining_stock), int(item["min_stock_level"]))
    assert decision["recommended_order_quantity"] == expected_order
    assert decision["estimated_delivery_date"] == get_supplier_delivery_date(start_date, expected_order)
