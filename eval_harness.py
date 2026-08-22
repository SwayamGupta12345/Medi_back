from typing import List, Dict


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for rid in top_k if rid in relevant_set)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for rid in top_k if rid in relevant_set)
    return hits / len(relevant_ids)
def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Returns reciprocal rank of the first relevant result.

    #1 -> 1.0
    #2 -> 0.5
    #3 -> 0.333
    #5 -> 0.2
    """

    relevant_set = set(relevant_ids)

    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant_set:
            return 1.0 / rank

    return 0.0


def dcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Binary relevance:
    relevant = 1
    irrelevant = 0
    """

    relevant_set = set(relevant_ids)

    score = 0.0

    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant_set:
            score += 1.0 / __import__("math").log2(rank + 1)

    return score


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0

    actual_dcg = dcg_at_k(retrieved_ids, relevant_ids, k)

    # Ideal ranking:
    # all relevant chunks appear first
    ideal_count = min(len(relevant_ids), k)

    ideal_dcg = sum(
        1.0 / __import__("math").log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg

def run_evaluation(test_queries: List[Dict], retrieve_fn, k_values=(1,5, 10, 20)):
    """
    test_queries: [{"query": str, "relevant_ids": [chunk_id, ...]}, ...]
    retrieve_fn: your retrieve_query_results function
    relevant_ids must be Pinecone vector "id" values (the "<title>-chunk-<i>" strings),
    labeled by you manually for a handful of queries you know the answer to.
    """
    results = {
        k: {
            "precision": [],
            "recall": [],
            "mrr": [],
            "ndcg": []
        }
        for k in k_values
    }

    for case in test_queries:
        query = case["query"]
        matches = retrieve_fn(case["query"])
        retrieved_ids = [m["id"] for m in matches]

        for k in k_values:
            p = precision_at_k(retrieved_ids, case["relevant_ids"], k)
            r = recall_at_k(retrieved_ids, case["relevant_ids"], k)
            rr = reciprocal_rank(
                retrieved_ids,
                case["relevant_ids"],
                k
            )

            ndcg = ndcg_at_k(
                retrieved_ids,
                case["relevant_ids"]    ,
                k
            )
            results[k]["precision"].append(p)
            results[k]["recall"].append(r)
            results[k]["mrr"].append(rr)
            results[k]["ndcg"].append(ndcg)
           
    summary = {}
    for k in k_values:
        n = len(results[k]["precision"]) or 1
        summary[k] = {
            "precision@k": round(
                sum(results[k]["precision"]) / n,
                4
            ),

            "recall@k": round(
                sum(results[k]["recall"]) / n,
                4
            ),

            "mrr@k": round(
                sum(results[k]["mrr"]) / n,
                4
            ),

            "ndcg@k": round(
                sum(results[k]["ndcg"]) / n,
                4
            )
        }
    return summary

P1 = "aloe-vera-a-potential-herb-and-its-medicinal-importance"
P2 = "aloe-vera-aloe-barbadensis-miller-phytochemicals-and-its-pharmacological-activity-compendium-review"

TEST_QUERIES = [
    # ---------- Paper 1: A Potential Herb and its Medicinal Importance ----------
    {"query": "What family does Aloe vera belong to according to this paper?",
     "relevant_ids": [f"{P1}-chunk-52"]},
    {"query": "What are the botanical names listed for Aloe vera?",
     "relevant_ids": [f"{P2}-chunk-14", f"{P2}-chunk-13"]},
    {"query": "How many of the twenty-two essential amino acids does Aloe vera contain?",
     "relevant_ids": [f"{P1}-chunk-11"]},
    {"query": "What percentage of dermatologically valuable extracts worldwide utilize Aloe vera?",
     "relevant_ids": [f"{P1}-chunk-23"]},
    {"query": "What condition is Aloe vera not recommended for during pregnancy?",
     "relevant_ids": [f"{P2}-chunk-44"]},
    {"query": "What sugar found in Aloe vera can inhibit HIV-1?",
     "relevant_ids": [f"{P1}-chunk-46"]},
    {"query": "By what percentage was virus reproduction reduced by Aloe Vera in the 1991 Molecular Biotherapy study?",
     "relevant_ids": [f"{P1}-chunk-46"]},
    {"query": "How many ounces of pure Aloe vera gel were given to HIV positive patients, and how many times daily?",
     "relevant_ids": [f"{P1}-chunk-47", f"{P1}-chunk-48"]},
    {"query": "Over how many days did the HIV positive patient study run?",
     "relevant_ids": [f"{P1}-chunk-48"]},
    {"query": "What are the side effects of using Aloe vera listed in this paper?",
     "relevant_ids": [f"{P2}-chunk-44", f"{P1}-chunk-44"]},
    {"query": "What conditions should someone avoid taking Aloe vera internally for?",
     "relevant_ids": [f"{P2}-chunk-44", f"{P1}-chunk-44"]},
    {"query": "What is Aloe vera believed to do for coronary heart disease risk factors?",
     "relevant_ids": [f"{P1}-chunk-29"]},
    {"query": "What compound in Aloe vera has an anti-tumour effect?",
     "relevant_ids": [f"{P2}-chunk-42"]},
    {"query": "What are lactates and salicylates in Aloe vera known for?",
     "relevant_ids": [f"{P1}-chunk-14"]},
    {"query": "Who is credited with conquering the island of Socotra due to its Aloe vera growth?",
     "relevant_ids": [f"{P1}-chunk-19"]},
    {"query": "How many of the thirteen recognised vitamins does Aloe vera contain?",
     "relevant_ids": [f"{P1}-chunk-13"]},
    {"query": "What historical military use of Aloe vera is mentioned regarding radiation protection?",
     "relevant_ids": [f"{P1}-chunk-50", f"{P1}-chunk-49"]},
    {"query": "What two substances does Aloe vera protect against radiation damage according to the Hoshi University research?",
     "relevant_ids": [f"{P1}-chunk-51"]},
    {"query": "What skin condition can Aloe vera lotions help treat according to the Seborrheic Dermatitis section?",
     "relevant_ids": [f"{P1}-chunk-43"]},
    {"query": "What is the only proven benefit of internal Aloe vera use according to the conclusion?",
     "relevant_ids": [f"{P1}-chunk-56"]},

    # ---------- Paper 2: Aloe Vera (Aloe Barbadensis Miller), Phytochemicals ----------
    {"query": "What is the DOI of this review?",
     "relevant_ids": [f"{P2}-chunk-0", f"{P2}-chunk-6"]},
    {"query": "In which journal and issue was this review published?",
     "relevant_ids": [f"{P2}-chunk-6"]},
    {"query": "What are the Hindi names for Aloe Vera mentioned in the plant profile?",
     "relevant_ids": [f"{P2}-chunk-13"]},
    {"query": "What is the Unani name for Aloe Vera?",
     "relevant_ids": [f"{P2}-chunk-13"]},
    {"query": "What is the biological classification division, subdivision, and class of Aloe Vera?",
     "relevant_ids": [f"{P2}-chunk-14"]},
    {"query": "How many species of aloe are cultivated industrially today, and which two are most prevalent?",
     "relevant_ids": [f"{P2}-chunk-19"]},
    {"query": "What are the three isomeric compounds that make up crystalline Aaloin?",
     "relevant_ids": [f"{P2}-chunk-23"]},
    {"query": "What compound is responsible for the purgative activity in the peripheral part of the Aloe Vera leaf skin?",
     "relevant_ids": [f"{P2}-chunk-25"]},
    {"query": "What was compared against Aloe Vera tooth gel in the dentistry study, and what was the finding?",
     "relevant_ids": [f"{P2}-chunk-26"]},
    {"query": "What standard medicine was Aloe Vera's anti-ulcer activity compared to in the indomethacin-induced ulcer study?",
     "relevant_ids": [f"{P2}-chunk-58", f"{P2}-chunk-29"]},
    {"query": "What virus was the antiviral activity of Aloe Vera tested against in the Iranian study?",
     "relevant_ids": [f"{P2}-chunk-38", f"{P2}-chunk-60"]},
    {"query": "What is Alprogen's mechanism of action on mast cells?",
     "relevant_ids": [f"{P2}-chunk-45", f"{P2}-chunk-62"]},
    {"query": "What is the nitrogen content of Aloe Vera per 100g?",
     "relevant_ids": [f"{P1}-chunk-11"]},
    {"query": "What is the calcium content of Aloe Vera per 100g?",
     "relevant_ids": [f"{P2}-chunk-9"]},
    {"query": "What is the total carbohydrate percentage in Aloe Vera?",
     "relevant_ids": [f"{P2}-chunk-47"]},
    {"query": "What is the protein content of Aloe Vera per gram?",
     "relevant_ids": [f"{P1}-chunk-11"]},
    {"query": "What compound was isolated in the anti-inflammatory study that decreased prostaglandin E2 generation?",
     "relevant_ids": [f"{P2}-chunk-31"]},
    {"query": "What cells did acemannan prevent Aloe Vera gel from adhering to in the anti-bacterial study?",
     "relevant_ids": [f"{P2}-chunk-32"]},
    {"query": "What genus and family does Aloe Vera belong to, and what other plants is it related to?",
     "relevant_ids": [f"{P1}-chunk-52"]},
    {"query": "Where is Aloe Vera geographically indigenous to, and where is it found in India?",
     "relevant_ids": [f"{P2}-chunk-16", f"{P2}-chunk-19"]},
]

if __name__ == "__main__":
    from converted import retrieve_query_results  # adjust import to your filename
    summary = run_evaluation(TEST_QUERIES, retrieve_query_results, k_values=(1, 5, 10, 20))
    for k, metrics in summary.items():

        print(
            f"K={k}: "
            f"Precision={metrics['precision@k']}, "
            f"Recall={metrics['recall@k']}, "
            f"MRR={metrics['mrr@k']}, "
            f"NDCG={metrics['ndcg@k']}"
        )
        
        
#         e Vera (Aloe Barbadensis Miller), Phytochemicals and Its Pharmacological Activity Compendium Review
# Enhanced query: Answer the following question in detail: Where is Aloe Vera geographically indigenous to, and where is it found in India?
# Query vector generated: 3072
# Keywords: ['answer', 'following', 'question', 'aloe', 'vera', 'geographically', 'indigenous', 'india']
# Dense matches: 50
# Final matches: 20
# MATCH: aloe-vera-aloe-barbadensis-miller-phytochemicals-and-its-pharmacological-activity-compendium-review-chunk-16 0.713882565 0.032522 Aloe Vera (Aloe Barbadensis Miller), Phytochemicals and Its Pharmacological Activity Compendium Review
# MATCH: aloe-vera-aloe-barbadensis-miller-phytochemicals-and-its-pharmacological-activity-compendium-review-chunk-19 0.719062865 0.031319 Aloe Vera (Aloe Barbadensis Miller), Phytochemicals and Its Pharmacological Activity Compendium Review
# MATCH: aloe-vera-a-potential-herb-and-its-medicinal-importance-chunk-19 0.688430846 0.030835 Aloe vera  A Potential Herb and its Medicinal Importance
# MATCH: aloe-vera-a-potential-herb-and-its-medicinal-importance-chunk-18 0.683685362 0.030366 Aloe vera  A Potential Herb and its Medicinal Importance
# MATCH: aloe-vera-a-potential-herb-and-its-medicinal-importance-chunk-53 0.692489684 0.030331 Aloe vera  A Potential Herb and its Medicinal Importance
# K=1: Precision=0.95, Recall=0.825, MRR=0.95, NDCG=0.95
# K=5: Precision=0.25, Recall=1.0, MRR=0.975, NDCG=0.9815
# K=10: Precision=0.125, Recall=1.0, MRR=0.975, NDCG=0.9815
# K=20: Precision=0.0625, Recall=1.0, MRR=0.975, NDCG=0.9815