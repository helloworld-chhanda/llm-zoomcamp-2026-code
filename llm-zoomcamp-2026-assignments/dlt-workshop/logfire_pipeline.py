"""
dlt workshop homework - Q2
Pull Pydantic Logfire trace data into DuckDB with dlt.

Requires LOGFIRE_READ_TOKEN in .env (read token from https://logfire.dev/dashboard).

Run:
    uv run python logfire_pipeline.py
"""

import os

import dlt
import requests
from dotenv import load_dotenv

load_dotenv()

LOGFIRE_QUERY_URL = "https://logfire-us.pydantic.dev/v1/query"


@dlt.resource(name="records", write_disposition="replace")
def logfire_records(read_token=dlt.secrets.value):
    """All span records currently in the Logfire project.

    The Logfire query API returns data column-oriented
    ({"columns": [{"name", "datatype", "values": [...]}]}), so we
    reshape it into row dicts before yielding. The nested JSON columns
    (attributes, otel_events, otel_links, otel_resource_attributes,
    otel_scope_attributes, attributes_json_schema) are passed through
    as-is so dlt's normalizer can explode them into child tables.
    """
    response = requests.get(
        LOGFIRE_QUERY_URL,
        params={"sql": "SELECT * FROM records"},
        headers={"Authorization": f"Bearer {read_token}"},
    )
    response.raise_for_status()
    columns = response.json()["columns"]

    row_count = len(columns[0]["values"]) if columns else 0
    for i in range(row_count):
        yield {col["name"]: col["values"][i] for col in columns}


def load():
    pipeline = dlt.pipeline(
        pipeline_name="logfire",
        destination="duckdb",
        dataset_name="agent_traces",
    )

    read_token = os.environ["LOGFIRE_READ_TOKEN"]
    load_info = pipeline.run(logfire_records(read_token=read_token))

    print(load_info)
    print(pipeline.last_trace.last_normalize_info)

    return pipeline


if __name__ == "__main__":
    load()
