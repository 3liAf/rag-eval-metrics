"""
rag-eval-metrics: Zero-dependency evaluation metrics for RAG retrieval pipelines.
Computes Hit Rate @ K, Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG),
and Reciprocal Rank Fusion (RRF).
"""

from typing import List, Sequence, Set, Tuple, Dict
import math


def hit_rate_at_k(retrieved_ids: Sequence[str], ground_truth_ids: Set[str], k: int) -> float:
    """Returns 1.0 if at least one ground-truth ID appears in top-k retrieved IDs, else 0.0."""
    if k <= 0 or not ground_truth_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return 1.0 if any(item in ground_truth_ids for item in top_k) else 0.0


def reciprocal_rank(retrieved_ids: Sequence[str], ground_truth_ids: Set[str]) -> float:
    """Computes Reciprocal Rank (1 / rank) of the first relevant retrieved document."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(queries_retrieved: Sequence[Sequence[str]], queries_ground_truth: Sequence[Set[str]]) -> float:
    """Computes Mean Reciprocal Rank (MRR) across all queries."""
    if not queries_retrieved or len(queries_retrieved) != len(queries_ground_truth):
        return 0.0
    scores = [reciprocal_rank(ret, gt) for ret, gt in zip(queries_retrieved, queries_ground_truth)]
    return sum(scores) / len(scores)


def dcg_at_k(retrieved_ids: Sequence[str], ground_truth_ids: Set[str], k: int) -> float:
    """Computes Discounted Cumulative Gain at rank K with binary relevance."""
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if doc_id in ground_truth_ids else 0.0
        if rank == 1:
            dcg += rel
        else:
            dcg += rel / math.log2(rank)
    return dcg


def ndcg_at_k(retrieved_ids: Sequence[str], ground_truth_ids: Set[str], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain (NDCG) at rank K."""
    if not ground_truth_ids or k <= 0:
        return 0.0
    actual_dcg = dcg_at_k(retrieved_ids, ground_truth_ids, k)
    ideal_relevance = [1.0] * min(len(ground_truth_ids), k)
    idcg = ideal_relevance[0] if ideal_relevance else 0.0
    for rank in range(2, len(ideal_relevance) + 1):
        idcg += 1.0 / math.log2(rank)
    return (actual_dcg / idcg) if idcg > 0.0 else 0.0


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Computes Reciprocal Rank Fusion (RRF) across multiple ranked lists.
    Formula: score(d) = sum(1 / (k + rank_m(d))) for all lists where d appears.
    Returns sorted list of (doc_id, score) tuples in descending order.
    """
    scores: Dict[str, float] = {}
    for rank_list in rankings:
        for rank, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
