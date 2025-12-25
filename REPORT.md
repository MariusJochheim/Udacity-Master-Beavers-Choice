# Multi-Agent System Report

## Project Summary
This project implements a text-first, five-agent stack for the fictional Munder Difflin Paper Company. An orchestrator fronts customer interactions and routes each turn to specialized agents: a Quote agent that prices requests using recent quote anchors plus live availability, an Inventory & Procurement agent that validates stock and recommends restocks, an Order Execution agent that records sales and triggers replenishment, and a Finance agent that snapshots cash and inventory. The system runs against a SQLite operational store and exposes simple tools so LLM calls stay structured and deterministic.

## Agent Workflow & Architecture

![Agent Communication Diagram](/documentation/diagram.png)

- **Customer → Orchestrator:** The orchestrator (smolagents ToolCallingAgent) inspects each request and chooses exactly one tool path—`quote_request`, `order_request`, or `status_snapshot`—passing along context (dates, job/event metadata, quoted totals, quantities) to preserve continuity.
- **Quoting path:** The Quote agent’s `prepare_quote` tool pulls recent similar quotes to anchor pricing, asks the Inventory agent for availability, applies order-size/markup rules, and returns a structured payload that the orchestrator formats for the customer. Direct tool calls are used to avoid LLM paraphrasing that could break downstream parsing.
- **Inventory & Procurement:** The Inventory agent centralizes stock checks and restock logic via `inventory_procurement` and `inventory_snapshot`, keeping the orchestration narrow (only one entry point for availability decisions) and ensuring replenishment metadata (min levels, supplier ETA) is consistent.
- **Ordering path:** The Order Execution agent parses multi-line requests, records sales transactions, and triggers budget-aware restocks, returning customer ETAs and backorder details. The orchestrator delegates here whenever the intent is fulfillment rather than pricing.
- **Finance/status:** For `status_snapshot`, the orchestrator calls the Finance agent, which wraps `generate_financial_report` to produce concise cash and inventory summaries.
- **Design rationale:** A single orchestrator keeps tool routing deterministic and under the 5-agent limit, while specialized agents encapsulate domain rules (pricing anchors, procurement thresholds, transaction recording). Tool-only instruction sets limit hallucinated actions and preserve structured outputs needed by downstream steps. The diagram in `documentation/diagram.mmd` mirrors this flow: customer → orchestrator → quote/order/status agents, each backed by narrow tool surfaces (inventory, procurement, transaction recording, supplier ETA, finance reports) with responses looped back to the customer.

## Evaluation Results (test_results.csv)
- Ran 20 scripted scenarios; 90% of responses include ETAs/delivery commitments and 60% include explicit dollar totals, showing consistent customer-facing clarity.
- Inventory value grew from ~$4.9k to ~$10.4k while serving orders, indicating the system reinvests in stock as it fulfills demand; peak inventory ($22.1k) at request 16 coincides with a major restock event.
- Cash declined from ~$45.1k to ~$16.6k because of aggressive replenishment (largest single drop of ~$32k at request 16), demonstrating that the replenishment logic prioritizes coverage but still tracks budgets.
- Backorders are explicitly surfaced in 12 of 20 responses (e.g., requests 7–9, 14–19), showing the agents can partially fulfill orders while communicating remaining gaps and triggering restocks.
- Multi-line, multi-item requests are parsed and executed with itemized confirmations (e.g., requests 1, 7, 8, 9), evidencing robust request parsing in the order execution path.

## Suggestions for Improvement
- Tighten replenishment guardrails: the drop to ~$4.8k cash at request 16 suggests restocks can overdraw working cash. Add thresholds (e.g., minimum cash reserve, tiered restock sizes) and blend quote margins to maintain liquidity.
- Strengthen quote-to-order linkage: only ~55% of responses surface totals; wire quoted totals directly into order execution and require explicit line-item pricing so customers see consistent numbers from quote through fulfillment.
- Reduce backorder churn: use demand forecasting or supplier lead-time preferences to pre-emptively stage inventory for frequently requested items, aiming to cut the 12/20 backorder occurrences and shorten ETAs on large runs.
