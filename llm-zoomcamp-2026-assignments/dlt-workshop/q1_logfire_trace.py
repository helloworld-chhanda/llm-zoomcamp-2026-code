"""
dlt workshop homework - Q1
Instrument the Pydantic AI FAQ agent with Logfire and run the Q1 query.

Requires LOGFIRE_TOKEN in .env (write token from https://logfire.dev/dashboard).

Run:
    uv run python q1_logfire_trace.py
"""

from dotenv import load_dotenv

load_dotenv()

import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

from agent import faq_agent, SearchDeps
from ingest import build_index, load_faq_data

QUESTION = "How do I run Ollama locally?"


def main():
    documents = load_faq_data()
    index = build_index(documents)
    deps = SearchDeps(index=index)

    result = faq_agent.run_sync(QUESTION, deps=deps)

    print("\n--- Answer ---")
    print(result.output)
    print("\nCheck your Logfire project dashboard for the trace produced by this run.")
    print("Count the spans (agent run + LLM calls + tool calls) to answer Q1.")


if __name__ == "__main__":
    main()
