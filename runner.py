import logging
import os
import time

import pandas as pd

from beaverschoice.agents.finance_agent import create_openai_finance_agent
from beaverschoice.agents.inventory_agent import create_openai_inventory_agent
from beaverschoice.agents.order_execution_agent import create_openai_order_execution_agent
from beaverschoice.agents.orchestrator_agent import OrchestratorAgent, create_openai_orchestrator_agent
from beaverschoice.agents.quote_agent import create_openai_quote_agent
from beaverschoice.config import data_path
from beaverschoice.db import get_engine, init_database
from beaverschoice.finance import generate_financial_report

logger = logging.getLogger(__name__)


def _setup_logging():
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # Surface smolagents tool calls when SMOLAGENTS_TRACE is set.
    smol_logger = logging.getLogger("smolagents")
    smol_logger.setLevel(logging.DEBUG if os.environ.get("SMOLAGENTS_TRACE") == "1" else logging.INFO)


def initialize_agents(engine=None, model_id: str = "gpt-4o-mini") -> dict:
    """
    Build the full agent stack and return them for reuse across flows.
    """
    engine = engine or get_engine()
    inventory_agent = create_openai_inventory_agent(engine=engine, model_id=model_id)
    quote_agent = create_openai_quote_agent(engine=engine, inventory_agent=inventory_agent, model_id=model_id)
    order_agent = create_openai_order_execution_agent(engine=engine, model_id=model_id)
    finance_agent = create_openai_finance_agent(engine=engine, model_id=model_id)
    orchestrator = create_openai_orchestrator_agent(
        quote_agent=quote_agent,
        engine=engine,
        model_id=model_id,
        order_agent=order_agent,
        finance_agent=finance_agent,
    )

    logger.info("Agents ready: inventory, quote, order execution, finance, orchestrator.")
    return {
        "engine": engine,
        "inventory_agent": inventory_agent,
        "quote_agent": quote_agent,
        "order_agent": order_agent,
        "finance_agent": finance_agent,
        "orchestrator": orchestrator,
    }


def handle_request(orchestrator: OrchestratorAgent, request: str, additional_args: dict | None = None) -> str:
    """
    Route a request through the orchestrator so the LLM can select the right tool.
    """
    additional_args = additional_args or {}
    logger.info("Routing through orchestrator: %s", request.replace("\n", " ")[:120])
    return orchestrator.run(request, additional_args=additional_args)



def run_test_scenarios(engine=None):
    _setup_logging()
    engine = engine or get_engine()

    print("Initializing Database...")
    init_database(engine)
    logger.info("Building agents for test scenarios...")
    agents = initialize_agents(engine)
    orchestrator = agents["orchestrator"]
    try:
        quote_requests_sample = pd.read_csv(data_path("quote_requests_sample.csv"))
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as exc:
        print(f"FATAL: Error loading test data: {exc}")
        return []

    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(engine, initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        request_with_date = f"{row['request']} (Date of request: {request_date})"
        additional_args = {
            "as_of_date": request_date,
            "order_size": row.get("need_size"),
            "job_type": row.get("job"),
            "event_type": row.get("event"),
        }
        response = handle_request(orchestrator, request_with_date, additional_args)

        report = generate_financial_report(engine, request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(engine, final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    run_test_scenarios()
