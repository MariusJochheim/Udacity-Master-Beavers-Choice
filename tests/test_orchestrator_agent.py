from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole, Model

from beaverschoice.agents.orchestrator_agent import OrchestratorAgent


class FakeQuoteAgent:
    def __init__(self):
        self.called_with = None

    def generate_quote(self, payload):
        self.called_with = payload
        return {
            "request": payload["request"],
            "as_of_date": payload["as_of_date"],
            "pricing": {"total": 111.11, "currency": "USD"},
            "availability": {"estimated_delivery_date": "2025-01-05"},
            "quote_explanation": "Test quote explanation",
        }


class FakeOrderExecutionAgent:
    def __init__(self):
        self.called_with = None
        self.called_additional_args = None

    def run(self, request_text, additional_args=None):
        self.called_with = request_text
        self.called_additional_args = additional_args or {}
        return {
            "request": request_text,
            "matched_item": "A4 paper",
            "customer_eta": "2025-01-08",
            "sale_receipt": {
                "transaction_id": 1,
                "item_name": "A4 paper",
                "transaction_type": "sales",
                "quantity": 250,
                "price": 125.0,
                "as_of_date": self.called_additional_args.get("as_of_date"),
                "resulting_stock": 100,
            },
        }


class FakeFinanceAgent:
    def __init__(self):
        self.called_with = None

    def status_snapshot(self, as_of_date):
        self.called_with = as_of_date
        return {"as_of_date": as_of_date, "cash_balance": 2500.0, "inventory_value": 1800.0}


class FakeOrchestratorModel(Model):
    """
    Deterministic model that calls a preselected tool, then returns a final_answer with the tool output.
    """

    def __init__(self, tool_name: str, tool_args: dict):
        super().__init__()
        self.tool_name = tool_name
        self.tool_args = tool_args
        self._step = 0

    def generate(self, messages, **kwargs):
        if self._step == 0:
            self._step += 1
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(name=self.tool_name, arguments=self.tool_args),
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
            if tool_result is None and messages:
                tool_result = getattr(messages[-1], "content", None)
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(name="final_answer", arguments={"answer": tool_result}),
                    id="call_2",
                    type="function",
                )
            ]

        return ChatMessage(role=MessageRole.ASSISTANT, tool_calls=tool_calls)


def test_orchestrator_llm_routes_quote(engine, start_date):
    quote_agent = FakeQuoteAgent()
    order_agent = FakeOrderExecutionAgent()
    finance_agent = FakeFinanceAgent()

    model = FakeOrchestratorModel(
        "quote_request",
        {
            "request_text": "Quote needed for office paper",
            "as_of_date": start_date,
            "order_size": "large",
            "job_type": "office manager",
            "event_type": "conference",
        },
    )
    orchestrator = OrchestratorAgent(
        engine,
        quote_agent=quote_agent,
        order_agent=order_agent,
        finance_agent=finance_agent,
        model=model,
    )

    response = orchestrator.run("Quote needed for office paper", additional_args={"as_of_date": start_date})

    assert quote_agent.called_with["request"] == "Quote needed for office paper"
    assert quote_agent.called_with["order_size"] == "large"
    assert "Quote prepared" in response
    assert "Estimated delivery" in response


def test_orchestrator_llm_routes_order_and_status(engine, start_date):
    quote_agent = FakeQuoteAgent()
    order_agent = FakeOrderExecutionAgent()
    finance_agent = FakeFinanceAgent()

    order_model = FakeOrchestratorModel(
        "order_request",
        {"request_text": "Order paper cups for delivery", "as_of_date": start_date},
    )
    order_orchestrator = OrchestratorAgent(
        engine,
        quote_agent=quote_agent,
        order_agent=order_agent,
        finance_agent=finance_agent,
        model=order_model,
    )

    order_response = order_orchestrator.run("Order paper cups for delivery", additional_args={"as_of_date": start_date})
    assert order_agent.called_with == "Order paper cups for delivery"
    assert order_agent.called_additional_args["request_text"] == "Order paper cups for delivery"
    assert order_agent.called_additional_args["as_of_date"] == start_date
    assert "Order placed" in order_response
    assert "ETA" in order_response

    status_model = FakeOrchestratorModel("status_snapshot", {"as_of_date": start_date})
    status_orchestrator = OrchestratorAgent(
        engine,
        quote_agent=quote_agent,
        order_agent=order_agent,
        finance_agent=finance_agent,
        model=status_model,
    )

    status_response = status_orchestrator.run("Status report for finance", additional_args={"as_of_date": start_date})
    assert finance_agent.called_with == start_date
    assert "Status as of" in status_response
    assert "$" in status_response
