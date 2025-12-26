# README

BeaversChoice implements a five-agent, text-only workflow for the fictional Munder Difflin Paper Company. An orchestrator routes each customer turn to quoting, ordering, or finance tools; specialized agents keep pricing, procurement, fulfillment, and reporting logic isolated while sharing a common SQLite store for inventory and transactions.

## Solution Overview
- **Agents (smolagents ToolCallingAgent):** Orchestrator fronts all requests; Quote agent prices using recent quotes plus live availability; Inventory/Procurement agent maps free-form asks to catalog items and recommends restocks; Order Execution agent records sales, triggers budget-aware replenishment, and returns ETAs; Finance agent snapshots cash/inventory and top sellers.
- **Data + state:** `init_database` seeds `munder_difflin.db` from `data/` CSVs, loads initial cash/inventory via transactions, and adds historical quotes/requests used for anchoring prices.
- **Pricing + procurement:** Quotes anchor to similar historical requests, adjust for order size, and add markup when stock is low. Procurement matches paraphrased items (synonyms + token overlap), checks against min stock, and surfaces delivery dates from simple supplier lead times.
- **Fulfillment:** Orders parse multi-line item lists, ship what is available, create sales transactions, then restock to cover backorders while respecting current cash. Customer-facing summaries include shipped/backordered quantities and ETAs.
- **Reporting:** Finance snapshots compute cash, inventory value, total assets, and top sellers; the orchestrator returns concise status lines. A high-level architecture diagram lives in `documentation/diagram.png`.

## Repository Layout
- `runner.py`: Entry point that seeds the DB, builds agents, and runs the sample scenario loop.
- `beaverschoice/`: Core package (agents, procurement, logistics, transactions, finance, tooling, config).
- `data/`: CSV inputs for quote history and request scenarios.
- `documentation/diagram.(png|mmd)`: Architecture diagram.
- `test_results.csv`: Generated after running the sample scenarios.

## Setup
1) Python 3.8+ (tested with pandas/SQLAlchemy 2.x and smolagents 1.23).
2) Install deps: `pip install -r requirements.txt`
3) Configure API access (OpenAI-compatible):
   - Copy `example.env` to `.env` and set `OPENAI_API_KEY` and `OPENAI_BASE_URL` (defaults to Udacity’s proxy).
4) Optional: set `SMOLAGENTS_TRACE=1` to see tool-calling traces in logs.

## Running the Sample Scenarios
1) From this folder: `python runner.py`
2) The script will:
   - Initialize/overwrite `munder_difflin.db` with seeded data.
   - Build all agents (default model `gpt-4o-mini`, override via `initialize_agents(model_id=...)`).
   - Replay `data/quote_requests_sample.csv` in date order, printing responses and writing `test_results.csv` with cash/inventory after each request.

## Using the Agents Programmatically
```python
from runner import initialize_agents, handle_request

agents = initialize_agents(model_id="gpt-4o-mini")
orchestrator = agents["orchestrator"]
response = handle_request(
    orchestrator,
    "Need 300 flyers for a campus fair (Date of request: 2025-01-05)",
    {"as_of_date": "2025-01-05", "order_size": "large", "event_type": "fair"},
)
print(response)
```

## Tests
- Fast unit tests (no live LLM calls) live in `tests/`. Run `pytest` to exercise procurement, quoting, order execution, and finance helpers against the seeded SQLite DB.

## Solution Notes
- Evaluation on the provided 20-scenario script shows consistent ETAs/backorder messaging and cash/inventory tracking; large restocks can drain cash (see `REPORT.md` for detailed observations and improvement ideas).
- Orchestrator keeps routing deterministic under the five-agent limit, while narrow tool surfaces prevent LLM paraphrasing from breaking structured outputs.
