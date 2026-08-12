from typing import List, Dict

def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)

def run_evaluation(test_queries: List[Dict], retrieve_fn, k_values=(5, 10, 20)):
    """
    test_queries: [{"query": str, "relevant_ids": [chunk_id, ...]}, ...]
    retrieve_fn: your retrieve_query_results function
    relevant_ids must be Pinecone vector "id" values (the "<title>-chunk-<i>" strings),
    labeled by you manually for a handful of queries you know the answer to.
    """
    results = {k: {"precision": [], "recall": []} for k in k_values}

    for case in test_queries:
        matches = retrieve_fn(case["query"])
        retrieved_ids = [m["id"] for m in matches]

        for k in k_values:
            p = precision_at_k(retrieved_ids, case["relevant_ids"], k)
            r = recall_at_k(retrieved_ids, case["relevant_ids"], k)
            results[k]["precision"].append(p)
            results[k]["recall"].append(r)

    summary = {}
    for k in k_values:
        n = len(results[k]["precision"]) or 1
        summary[k] = {
            "precision@k": round(sum(results[k]["precision"]) / n, 4),
            "recall@k": round(sum(results[k]["recall"]) / n, 4),
        }
    return summary


# ── Fill this in with real labeled queries before running ──
TEST_QUERIES = [
    {
        "query": "example question about an uploaded PDF",
        "relevant_ids": ["book-title-chunk-3", "book-title-chunk-4"],  # get these from print(match['id']) during a manual query
    },
]

if __name__ == "__main__":
    from main import retrieve_query_results  # adjust import to your filename
    summary = run_evaluation(TEST_QUERIES, retrieve_query_results)
    for k, metrics in summary.items():
        print(f"k={k}: precision@k={metrics['precision@k']}, recall@k={metrics['recall@k']}")