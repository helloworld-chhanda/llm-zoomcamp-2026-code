"""
dlt workshop homework - Q3
Sum gen_ai.usage.input_tokens across all LLM call spans in the loaded traces.

Run (after logfire_pipeline.py has populated logfire.duckdb):
    uv run python q3_token_usage.py
"""

import duckdb


def main():
    conn = duckdb.connect("logfire.duckdb")

    rows = conn.execute(
        """
        SELECT span_name, attributes__gen_ai_usage_input_tokens, attributes__gen_ai_usage_output_tokens
        FROM agent_traces.records
        ORDER BY start_timestamp
        """
    ).fetchall()

    print("span_name | input_tokens | output_tokens")
    for span_name, inp, outp in rows:
        print(span_name, "|", inp, "|", outp)

    total = conn.execute(
        "SELECT SUM(attributes__gen_ai_usage_input_tokens) FROM agent_traces.records"
    ).fetchone()[0]

    print(f"\nTotal input tokens across all LLM calls: {total}")
    conn.close()


if __name__ == "__main__":
    main()
