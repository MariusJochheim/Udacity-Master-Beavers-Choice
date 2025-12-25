import pandas as pd
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole, Model

from beaverschoice.agents.order_execution_agent import OrderExecutionAgent
from beaverschoice.agents.quote_agent import QuoteAgent
from beaverschoice.logistics import get_supplier_delivery_date
from beaverschoice.procurement import compute_procurement_decision
from beaverschoice.transactions import get_stock_level


class FakeOrderModel(Model):
    """Deterministic model that calls execute_order, then echoes the tool output."""

    def __init__(self, tool_args: dict):
        super().__init__()
        self.tool_args = tool_args
        self._step = 0

    def generate(self, messages, **kwargs):
        if self._step == 0:
            self._step += 1
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(name="execute_order", arguments=self.tool_args),
                    id="call_1",
                    type="function",
                )
            ]
        else:
            self._step += 1
            tool_result = None
            for message in reversed(messages):
                role = getattr(message, "role", None)
                if role in {getattr(MessageRole, "TOOL", None), getattr(MessageRole, "TOOL_RESPONSE", None)}:
                    tool_result = message.content
                    break
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(name="final_answer", arguments={"answer": tool_result}),
                    id="call_2",
                    type="function",
                )
            ]

        return ChatMessage(role=MessageRole.ASSISTANT, tool_calls=tool_calls)


class InlineInventoryAgent:
    """Minimal inventory agent for quotes that delegates to procurement decisions directly."""

    def __init__(self, engine, as_of_date: str):
        self.engine = engine
        self.as_of_date = as_of_date

    def run(self, request_text, additional_args=None):
        date = (additional_args or {}).get("as_of_date") or self.as_of_date
        return compute_procurement_decision(self.engine, request_text, date).to_dict()


class FakeQuoteModel(Model):
    """Deterministic model that calls prepare_quote then echoes the tool output."""

    def __init__(self, tool_args: dict):
        super().__init__()
        self.tool_args = tool_args
        self._step = 0

    def generate(self, messages, **kwargs):
        if self._step == 0:
            self._step += 1
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(name="prepare_quote", arguments=self.tool_args),
                    id="call_1",
                    type="function",
                )
            ]
        else:
            self._step += 1
            tool_result = None
            for message in reversed(messages):
                role = getattr(message, "role", None)
                if role in {getattr(MessageRole, "TOOL", None), getattr(MessageRole, "TOOL_RESPONSE", None)}:
                    tool_result = message.content
                    break
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(name="final_answer", arguments={"answer": tool_result}),
                    id="call_2",
                    type="function",
                )
            ]

        return ChatMessage(role=MessageRole.ASSISTANT, tool_calls=tool_calls)


def test_order_execution_records_sale_and_replenishes(engine, start_date):
    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
    item = inventory_df.iloc[0]
    sale_units = max(1, int(item["current_stock"] - item["min_stock_level"] + 1))
    sale_units = min(sale_units, int(item["current_stock"]))
    request_text = f"Order {item['item_name']} ASAP"

    model = FakeOrderModel({"request_text": request_text, "quantity": sale_units, "as_of_date": start_date})
    agent = OrderExecutionAgent(engine, model=model)
    result = agent.run(request_text, additional_args={"as_of_date": start_date})

    assert result["sale_receipt"]["transaction_type"] == "sales"
    assert result["sale_receipt"]["quantity"] == sale_units
    resulting_stock = get_stock_level(engine, item["item_name"], start_date)["current_stock"].iloc[0]
    assert result["sale_receipt"]["resulting_stock"] == resulting_stock

    assert result["restock"] is not None
    restock_qty = result["restock"]["receipt"]["quantity"]
    assert result["restock"]["receipt"]["transaction_type"] == "stock_orders"
    assert result["restock"]["estimated_delivery_date"] == get_supplier_delivery_date(start_date, restock_qty)
    assert result["customer_eta"] == get_supplier_delivery_date(start_date, sale_units)


def test_quote_to_order_integration(engine, start_date):
    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
    item = inventory_df.iloc[0]
    request_text = f"Need {item['item_name']} for a client event"
    sale_units = max(2, int(item["current_stock"] - item["min_stock_level"] + 2))
    sale_units = min(sale_units, int(item["current_stock"]))

    inventory_agent = InlineInventoryAgent(engine, start_date)
    quote_payload = {"request": request_text, "as_of_date": start_date, "order_size": "medium"}
    quote_model = FakeQuoteModel(
        {
            "request_text": quote_payload["request"],
            "as_of_date": quote_payload["as_of_date"],
            "order_size": quote_payload["order_size"],
            "job_type": quote_payload.get("job_type"),
            "event_type": quote_payload.get("event_type"),
        }
    )
    quote_agent = QuoteAgent(engine, inventory_agent, model=quote_model)
    quote = quote_agent.generate_quote(quote_payload)

    model = FakeOrderModel(
        {
            "request_text": request_text,
            "quantity": sale_units,
            "as_of_date": start_date,
            "quoted_total": quote["pricing"]["total"],
        }
    )
    order_agent = OrderExecutionAgent(engine, model=model)
    order_result = order_agent.run(request_text, additional_args={"as_of_date": start_date})

    assert order_result["matched_item"] == quote["availability"]["matched_item"]
    assert order_result["sale_receipt"]["price"] == quote["pricing"]["total"]
    assert order_result["sale_receipt"]["quantity"] == sale_units
    assert order_result["customer_eta"] == get_supplier_delivery_date(start_date, sale_units)

    if order_result["restock"]:
        restock_qty = order_result["restock"]["receipt"]["quantity"]
        assert order_result["restock"]["estimated_delivery_date"] == get_supplier_delivery_date(start_date, restock_qty)
