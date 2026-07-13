"""
Homework 04 - Evaluation
LLM Zoomcamp 2026

Run:
    uv run python homework.py
"""

import json

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch
from sentence_transformers import SentenceTransformer

from evaluation_utils import llm_structured

load_dotenv()
openai_client = OpenAI()

# ---------------------------------------------------------------------------
# Load the 72 lesson pages
# ---------------------------------------------------------------------------
print("Loading course lesson pages...")
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]
print(f"Loaded {len(documents)} pages")

# ---------------------------------------------------------------------------
# Q1. Generating questions - average input tokens for the first 3 pages
# ---------------------------------------------------------------------------
print("\n=== Q1: Generating questions ===")

data_gen_instructions = """
You emulate a student who is taking our LLM course.
You are given one lesson page from the course.
Formulate 5 questions this student might ask that are answered by this page.

Rules:
- The page should contain the answer to each question.
- Make the questions complete and not too short.
- Use as few words as possible from the page; don't copy its phrasing.
- The questions should resemble how people actually ask things online:
  not too formal, not too short, not too long.
- Ask about the content of the lesson, not about its formatting or filename.
""".strip()


class Questions(BaseModel):
    questions: list[str]


q1_filenames = [
    "01-agentic-rag/lessons/01-intro.md",
    "01-agentic-rag/lessons/02-environment.md",
    "01-agentic-rag/lessons/03-rag.md",
]

doc_by_filename = {doc["filename"]: doc for doc in documents}

input_tokens = []

for filename in q1_filenames:
    doc = doc_by_filename[filename]
    user_prompt = json.dumps({"filename": doc["filename"], "content": doc["content"]})

    result, usage = llm_structured(
        openai_client,
        data_gen_instructions,
        user_prompt,
        Questions,
        model="gpt-4o-mini",
    )

    print(f"{filename}: {usage.input_tokens} input tokens, {len(result.questions)} questions")
    input_tokens.append(usage.input_tokens)

avg_input_tokens = sum(input_tokens) / len(input_tokens)
print(f"Average input tokens: {avg_input_tokens:.1f}")

# ---------------------------------------------------------------------------
# Load the full ground truth (360 questions)
# ---------------------------------------------------------------------------
df_ground_truth = pd.read_csv("ground-truth.csv")
ground_truth = df_ground_truth.to_dict(orient="records")
print(f"\nLoaded {len(ground_truth)} ground truth questions")

# ---------------------------------------------------------------------------
# Chunk the documents
# ---------------------------------------------------------------------------
chunks = chunk_documents(documents, size=2000, step=1000)
print(f"Created {len(chunks)} chunks")

# ---------------------------------------------------------------------------
# Build text and vector search
# ---------------------------------------------------------------------------
print("\nBuilding text index...")
text_index = Index(text_fields=["content"], keyword_fields=["filename"])
text_index.fit(chunks)


def text_search(query, num_results=5):
    return text_index.search(query, num_results=num_results)


print("Building vector index...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_vectors = embedding_model.encode([c["content"] for c in chunks])

vector_index = VectorSearch(keyword_fields=["filename"])
vector_index.fit(chunk_vectors, chunks)


def vector_search(query, num_results=5):
    query_vector = embedding_model.encode(query)
    return vector_index.search(query_vector, num_results=num_results)


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def hybrid_search(query, k=60):
    text_results = text_search(query, num_results=10)
    vector_results = vector_search(query, num_results=10)
    return rrf([text_results, vector_results], k=k)


# ---------------------------------------------------------------------------
# Q2. First result with text search
# ---------------------------------------------------------------------------
print("\n=== Q2: First result with text search ===")

q = ground_truth[0]["question"]
print(f"Question: {q}")

text_results = text_search(q)
q2_filename = text_results[0]["filename"]
print(f"First text search result filename: {q2_filename}")

# ---------------------------------------------------------------------------
# Q3. First result with vector search
# ---------------------------------------------------------------------------
print("\n=== Q3: First result with vector search ===")

vector_results = vector_search(q)
q3_filename = vector_results[0]["filename"]
print(f"First vector search result filename: {q3_filename}")
print(f"(question generated from: {ground_truth[0]['filename']})")

# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def compute_relevance(q, search_function):
    filename = q["filename"]
    results = search_function(q["question"])

    relevance = []
    for d in results:
        relevance.append(int(d["filename"] == filename))

    return relevance


def compute_relevance_total(ground_truth, search_function):
    relevance_total = []

    for q in ground_truth:
        relevance = compute_relevance(q, search_function)
        relevance_total.append(relevance)

    return relevance_total


def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if 1 in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)


def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == 1:
                total_score = total_score + 1 / (rank + 1)
                break

    return total_score / len(relevance_total)


def evaluate(ground_truth, search_function):
    relevance_total = compute_relevance_total(ground_truth, search_function)

    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }


# ---------------------------------------------------------------------------
# Q4. Evaluating text search
# ---------------------------------------------------------------------------
print("\n=== Q4: Evaluating text search ===")
text_metrics = evaluate(ground_truth, text_search)
print(f"Text search: {text_metrics}")

# ---------------------------------------------------------------------------
# Q5. Evaluating vector search
# ---------------------------------------------------------------------------
print("\n=== Q5: Evaluating vector search ===")
vector_metrics = evaluate(ground_truth, vector_search)
print(f"Vector search: {vector_metrics}")

# ---------------------------------------------------------------------------
# Q6. Tuning hybrid search
# ---------------------------------------------------------------------------
print("\n=== Q6: Tuning hybrid search (RRF k) ===")

hybrid_results = {}

for k in [1, 50, 100, 200]:
    metrics = evaluate(
        ground_truth,
        lambda query, k=k: hybrid_search(query, k=k),
    )
    hybrid_results[k] = metrics
    print(f"k={k}: {metrics}")

best_k = max(hybrid_results, key=lambda k: (hybrid_results[k]["mrr"], -k))
print(f"Best k: {best_k}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("SUMMARY OF ANSWERS")
print("=" * 50)
print(f"Q1. Average input tokens: {avg_input_tokens:.1f}")
print(f"Q2. First text search result filename: {q2_filename}")
print(f"Q3. First vector search result filename: {q3_filename}")
print(f"Q4. Text search hit rate: {text_metrics['hit_rate']:.4f}")
print(f"Q5. Vector search MRR: {vector_metrics['mrr']:.4f}")
print(f"Q6. Best RRF k: {best_k}")
