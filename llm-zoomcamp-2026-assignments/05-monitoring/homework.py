"""
Homework 05 - Monitoring (OpenTelemetry)
LLM Zoomcamp 2026

Run:
    uv run python homework.py
"""

import os
import sqlite3

import pandas as pd

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

# Tracer provider must be set up before importing `starter`, so any
# instrumentation created at import time is backed by our provider.
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")

from starter import rag  # noqa: E402  (must come after tracer provider setup)

# The homework spec's recommended "gpt-5.4-mini" isn't a real released model;
# use a real current OpenAI model instead.
rag.model = "gpt-4o-mini"

QUERY = "How does the agentic loop keep calling the model until it stops?"

INPUT_PRICE_PER_MILLION = 0.15
OUTPUT_PRICE_PER_MILLION = 0.60


class RAGTraced:
    """Wraps a RAGBase instance so rag(), search(), and llm() each
    produce their own OTel span."""

    def __init__(self, rag_base, tracer):
        self._rag = rag_base
        self._tracer = tracer

    def search(self, query, num_results=5):
        with self._tracer.start_as_current_span("search") as span:
            results = self._rag.search(query, num_results=num_results)
            span.set_attribute("num_results", len(results))
            return results

    def llm(self, prompt):
        with self._tracer.start_as_current_span("llm") as span:
            response = self._rag.llm(prompt)
            usage = response.usage

            span.set_attribute("input_tokens", usage.input_tokens)
            span.set_attribute("output_tokens", usage.output_tokens)

            input_cost = (usage.input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
            output_cost = (usage.output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
            cost = input_cost + output_cost
            span.set_attribute("cost", cost)

            return response

    def build_prompt(self, query, search_results):
        return self._rag.build_prompt(query, search_results)

    def rag(self, query):
        with self._tracer.start_as_current_span("rag") as span:
            search_results = self.search(query)
            prompt = self.build_prompt(query, search_results)
            response = self.llm(prompt)
            return response.output_text


traced_rag = RAGTraced(rag, tracer)

# ---------------------------------------------------------------------------
# Q1 & Q2 & Q3: run once with the console exporter, inspect the printed spans
# ---------------------------------------------------------------------------
print("=" * 70)
print("Q1-Q3: Running traced RAG with ConsoleSpanExporter")
print("=" * 70)

answer = traced_rag.rag(QUERY)
provider.force_flush()

print("\nAnswer:", answer[:200], "...")
print(
    "\n(Inspect the ReadableSpan dicts printed above to answer Q1 [span count], "
    "Q2 [llm span's input_tokens attribute], and Q3 [search vs llm span duration])"
)

# ---------------------------------------------------------------------------
# Q4, Q5, Q6: SQLiteSpanExporter
# ---------------------------------------------------------------------------


class SQLiteSpanExporter(SpanExporter):

    def __init__(self, db_path="traces.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
            """
        )
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self, timeout_millis=30000):
        return True


db_path = "traces.db"
if os.path.exists(db_path):
    os.remove(db_path)

sqlite_provider = TracerProvider()
sqlite_provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter(db_path)))
sqlite_tracer = sqlite_provider.get_tracer("llm-zoomcamp")
sqlite_traced_rag = RAGTraced(rag, sqlite_tracer)

print("\n" + "=" * 70)
print("Q4: Re-running the query with the SQLite exporter")
print("=" * 70)

sqlite_traced_rag.rag(QUERY)
sqlite_provider.force_flush()

conn = sqlite3.connect(db_path)
span_names = pd.read_sql("SELECT DISTINCT name FROM spans ORDER BY name", conn)["name"].tolist()
conn.close()
print(f"Span names in the spans table: {span_names}")

# ---------------------------------------------------------------------------
# Q5: total duration per span type (excluding rag), from ONE run's data
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Q5: Total duration by span name (excluding 'rag')")
print("=" * 70)

conn = sqlite3.connect(db_path)
df_spans = pd.read_sql("SELECT * FROM spans", conn)
conn.close()

df_spans["duration_ms"] = (df_spans["end_time"] - df_spans["start_time"]) / 1_000_000

duration_by_name = (
    df_spans[df_spans["name"] != "rag"]
    .groupby("name")["duration_ms"]
    .sum()
    .sort_values(ascending=False)
)
print(duration_by_name)
q5_top_span = duration_by_name.idxmax()
print(f"\nSpan type with most total time: {q5_top_span}")

# ---------------------------------------------------------------------------
# Q6: token stability - run the same query 3 more times (4 total in the db)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Q6: Running the same query 3 more times for token-stability check")
print("=" * 70)

for _ in range(3):
    sqlite_traced_rag.rag(QUERY)
sqlite_provider.force_flush()

conn = sqlite3.connect(db_path)
df_llm = pd.read_sql(
    "SELECT input_tokens FROM spans WHERE name = 'llm' ORDER BY start_time", conn
)
conn.close()

tokens = df_llm["input_tokens"].tolist()
print(f"Input tokens across {len(tokens)} runs: {tokens}")

min_tok, max_tok = min(tokens), max(tokens)
variance_pct = ((max_tok - min_tok) / min_tok) * 100 if min_tok else 0.0
print(f"Min: {min_tok}, Max: {max_tok}, variance: {variance_pct:.2f}%")

if variance_pct == 0:
    q6_answer = "They're identical"
elif variance_pct <= 10:
    q6_answer = "Within 10% of each other"
elif variance_pct <= 50:
    q6_answer = "Within 50% of each other"
else:
    q6_answer = "They vary more than 50%"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Q1-Q3: see console span output above")
print(f"Q4. Span names in SQLite: {span_names}")
print(f"Q5. Span type with most total time (excl. rag): {q5_top_span}")
print(f"Q6. Input token variance across 4 runs: {variance_pct:.2f}% -> {q6_answer}")
