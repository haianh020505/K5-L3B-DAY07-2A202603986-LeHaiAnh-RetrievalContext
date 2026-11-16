"""Offline check of retrieval metric bookkeeping."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval_retrieval import evaluate


def chunk(cid, text):
    return {'id': cid, 'text': text, 'source_id': 'doc', 'source': 'doc.pdf', 'page': 1, 'metadata': {}}


class Fixed:
    def __init__(self, chunks): self.chunks = chunks; self.filters = []
    def search(self, query, *, filters=None, top_k=3):
        self.filters.append(filters)
        return [{**c, 'score': 1.0, 'rank': r + 1, 'strategy': 'fixed'} for r, c in enumerate(self.chunks[:top_k])]


Q = [{'id': 'q1', 'split': 'dev', 'question': 'x?', 'expected_answer': 'y', 'filter': {'chapter': 4},
      'gold': [{'source_id': 'doc', 'page': 1, 'evidence': 'GOLD'}]}]


class EvalTests(unittest.TestCase):
    def test_hit_at_rank_two_counts_hit3_not_hit1(self):
        r = Fixed([chunk('a', 'nothing'), chunk('b', 'has GOLD'), chunk('c', 'z')])
        t = evaluate(r, Q, top_k=3, budget=10_000)[0]
        self.assertEqual((t['hit1'], t['hit3'], t['context_hit']), (False, True, True))
        self.assertEqual(r.filters, [{'chapter': 4}])

    def test_small_budget_drops_evidence_from_context(self):
        r = Fixed([chunk('a', 'x' * 500), chunk('b', 'has GOLD')])
        t = evaluate(r, Q, top_k=3, budget=540)[0]  # block a = 22-char header + 500 chars; b no longer fits
        self.assertEqual((t['hit3'], t['context_hit']), (True, False))


if __name__ == '__main__':
    unittest.main()
