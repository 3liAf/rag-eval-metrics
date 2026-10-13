# rag-eval-metrics

Zero-dependency Python utility for evaluating Retrieval-Augmented Generation (RAG) retrieval accuracy.

## Metrics Supported

- **Hit Rate @ K**: Binary hit rate indicating whether any ground-truth document was retrieved in the top $K$ results.
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of the first relevant document across queries.
- **NDCG @ K**: Normalized Discounted Cumulative Gain penalizing relevant results appearing further down the list.

## Usage

```python
from rageval import hit_rate_at_k, mean_reciprocal_rank, ndcg_at_k

retrieved_docs = ["chunk_30", "chunk_12", "chunk_99"]
ground_truth = {"chunk_12"}

hit = hit_rate_at_k(retrieved_docs, ground_truth, k=3)
ndcg = ndcg_at_k(retrieved_docs, ground_truth, k=3)

print(f"Hit Rate @ 3: {hit}")
print(f"NDCG @ 3:     {ndcg:.4f}")
```

## Running Tests

```bash
python -m unittest tests/test_evaluator.py
```

## License

MIT License. See [LICENSE](LICENSE) for details.