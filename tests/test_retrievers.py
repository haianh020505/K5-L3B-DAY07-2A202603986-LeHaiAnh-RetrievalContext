"""Offline checks for lexical, fusion and rerank logic (dense needs the real Chroma + API)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrievers import BM25Retriever, HybridRetriever, RerankRetriever

KEYS = {'id', 'text', 'source_id', 'source', 'page', 'metadata', 'score', 'rank', 'strategy'}


def chunk(cid, text, chapter=4):
    return {'id': cid, 'text': text, 'source_id': 'doc', 'source': 'doc.pdf', 'page': 1,
            'metadata': {'chapter': chapter}}


CHUNKS = [chunk('a', 'The while loop repeats while a condition is true.', 5),
          chunk('b', 'An elif branch is checked when the if condition is false.', 4),
          chunk('c', 'A function returns a value with the return statement.', 6),
          chunk('d', 'Nested loops: the outer loop controls the inner loop.', 5)]


class Fixed:
    def __init__(self, ids): self.ids = ids
    def search(self, query, *, filters=None, top_k=3):
        return [{**chunk(i, i), 'score': 1.0, 'rank': r + 1, 'strategy': 'fixed'}
                for r, i in enumerate(self.ids[:top_k])]


class RetrieverTests(unittest.TestCase):
    def test_bm25_ranks_exact_term_first(self):
        res = BM25Retriever(CHUNKS).search('elif branch', filters=None, top_k=3)
        self.assertEqual(res[0]['id'], 'b')
        self.assertEqual([r['rank'] for r in res], list(range(1, len(res) + 1)))

    def test_bm25_respects_chapter_filter(self):
        res = BM25Retriever(CHUNKS).search('loop', filters={'chapter': 5}, top_k=3)
        self.assertTrue(res)
        self.assertTrue(all(r['metadata']['chapter'] == 5 for r in res))
        self.assertEqual(BM25Retriever(CHUNKS).search('elif', filters={'chapter': 6}, top_k=3), [])

    def test_rrf_fuses_two_rankings(self):
        res = HybridRetriever(Fixed(['a', 'b']), Fixed(['b', 'c']), k=60).search('q', filters=None, top_k=3)
        self.assertEqual([r['id'] for r in res], ['b', 'a', 'c'])
        self.assertAlmostEqual(res[0]['score'], 1 / 61 + 1 / 62)

    def test_results_have_required_keys(self):
        for res in (BM25Retriever(CHUNKS).search('loop', filters=None, top_k=2),
                    HybridRetriever(Fixed(['a']), Fixed(['b'])).search('q', filters=None, top_k=2)):
            for r in res:
                self.assertTrue(KEYS <= set(r), KEYS - set(r))

    def test_rerank_keeps_only_known_ids_then_fills(self):
        fake_chat = lambda messages, **kw: {'content': '["c", "zzz", "a"]'}
        res = RerankRetriever(Fixed(['a', 'b', 'c']), candidates=3, chat=fake_chat).search('q', filters=None, top_k=3)
        self.assertEqual([r['id'] for r in res], ['c', 'a', 'b'])
        self.assertEqual([r['rank'] for r in res], [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
