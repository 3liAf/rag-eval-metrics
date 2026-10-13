import unittest
from rageval.evaluator import hit_rate_at_k, reciprocal_rank, mean_reciprocal_rank, ndcg_at_k

class TestRagEvalMetrics(unittest.TestCase):
    def test_hit_rate(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        ground_truth = {"doc3"}
        self.assertEqual(hit_rate_at_k(retrieved, ground_truth, k=2), 0.0)
        self.assertEqual(hit_rate_at_k(retrieved, ground_truth, k=3), 1.0)

    def test_mrr(self):
        queries_ret = [["docA", "docB"], ["docX", "docY"]]
        queries_gt = [{"docB"}, {"docX"}]
        # Query 1: rank 2 -> RR = 0.5. Query 2: rank 1 -> RR = 1.0. MRR = 0.75
        self.assertAlmostEqual(mean_reciprocal_rank(queries_ret, queries_gt), 0.75)

    def test_ndcg(self):
        retrieved = ["doc1", "doc2", "doc3"]
        ground_truth = {"doc1"}
        # Perfect retrieval at rank 1 -> NDCG = 1.0
        self.assertAlmostEqual(ndcg_at_k(retrieved, ground_truth, k=3), 1.0)

if __name__ == "__main__":
    unittest.main()