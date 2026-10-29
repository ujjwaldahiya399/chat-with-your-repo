"""Deterministic unit tests for Week 1 retrieval evaluation utilities.

Run: python -m unittest discover -s evals -p 'test_week1_eval.py' -v
Uses the virtual environment's installed benchmark dependencies, but does not
load the embedding model, access a database, or call Groq.
"""
import json
import sys
import unittest
from pathlib import Path

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))

from run_rag_benchmark import metrics, rrf, tokenize, summarize  # noqa: E402


class TestRetrievalMetrics(unittest.TestCase):
    def test_all_relevant_at_first_and_third_rank(self):
        row = metrics(["a", "x", "b", "y", "z"], ["a", "b"])
        self.assertAlmostEqual(row['precision@5'], 0.4)
        self.assertAlmostEqual(row['recall@5'], 1.0)
        self.assertAlmostEqual(row['mrr_contribution@5'], 1.0)
        self.assertEqual(row['top1_hit'], 1)

    def test_first_relevant_at_fourth_rank(self):
        row = metrics(["x", "y", "z", "target", "other"], ["target"])
        self.assertAlmostEqual(row['precision@5'], 0.2)
        self.assertAlmostEqual(row['recall@5'], 1.0)
        self.assertAlmostEqual(row['mrr_contribution@5'], 0.25)
        self.assertEqual(row['top1_hit'], 0)

    def test_relevant_missing_returns_zero(self):
        row = metrics(["a", "b", "c"], ["target"])
        self.assertEqual(row['precision@5'], 0)
        self.assertEqual(row['recall@5'], 0)
        self.assertEqual(row['mrr_contribution@5'], 0)
        self.assertEqual(row['top1_hit'], 0)

    def test_empty_ground_truth_not_silently_scored(self):
        with self.assertRaises(ValueError):
            metrics(["a"], [])

    def test_identifier_tokenization(self):
        self.assertEqual(tokenize('Check GROQ_API_KEY in app.py'),
                         ['check', 'groq_api_key', 'in', 'app', 'py'])

    def test_rrf_uses_rank_positions(self):
        fused = rrf(['a', 'b'], ['b', 'c'])
        self.assertEqual(fused, ['b', 'a', 'c'])

    def test_multi_chunk_recall_and_partial_coverage(self):
        row = metrics(["first", "x", "y", "z", "u"], ["first", "missing"])
        self.assertEqual(row['precision@5'], 0.2)
        self.assertEqual(row['recall@5'], 0.5)
        self.assertEqual(row['top1_hit'], 1)

    def test_summary_averages_per_question(self):
        def report(p, r, rr, top1):
            return {'precision@5': p, 'recall@5': r,
                    'mrr_contribution@5': rr, 'top1_hit': top1}
        questions = [
            {'answerable': True, 'results': {
                method: {'metrics': report(0.2, 1, 1, 1)}
                for method in ['Semantic', 'BM25', 'Hybrid_RRF']}},
            {'answerable': True, 'results': {
                method: {'metrics': report(0, 0, 0, 0)}
                for method in ['Semantic', 'BM25', 'Hybrid_RRF']}},
            {'answerable': False, 'results': {
                method: {'metrics': None}
                for method in ['Semantic', 'BM25', 'Hybrid_RRF']}},
        ]
        summary = summarize(questions)
        self.assertEqual(summary['Semantic']['recall@5'], 0.5)
        self.assertEqual(summary['BM25']['mrr@5'], 0.5)
        self.assertEqual(summary['Hybrid_RRF']['top1_accuracy'], 0.5)

    def test_gold_labels_exist_in_uploaded_manifest(self):
        import re
        manifest = (EVALS / 'chunk_manifest.md').read_text(encoding='utf-8')
        known = set(re.findall(r'(?m)^CHUNK ID:\s*(\S+)\s*$', manifest))
        self.assertEqual(len(known), 32)
        golden = json.loads((EVALS / 'golden_set_rag.json').read_text())
        for q in golden:
            self.assertTrue(set(q['relevant_chunks']) <= known, q['id'])

    def test_benchmark_dataset_structure(self):
        golden = json.loads((EVALS / 'golden_set_rag.json').read_text(encoding='utf-8'))
        self.assertEqual(len(golden), 25)
        self.assertEqual(sum(q['answerable'] for q in golden), 23)
        self.assertEqual(len({q['id'] for q in golden}), 25)
        for q in golden:
            self.assertEqual(bool(q['relevant_chunks']), q['answerable'])


if __name__ == '__main__':
    unittest.main()
