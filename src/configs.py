"""Corpora and experiment configurations (one place, shared by eval_retrieval and run_bench)."""
import json
import os
from functools import cache
from pathlib import Path

from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever

ROOT = Path(__file__).resolve().parents[1]
CORPORA = {
    'day7': {'data': ROOT / 'data' / 'day7', 'collection': 'day06_B',
             'chroma': Path(os.getenv('DAY6_CHROMA_PATH', ROOT.parent / 'K5-L3B-DAY06-LeHaiAnh-2A202603986-DataIngestion' / 'starter' / 'chroma_db'))},
    'aiindex': {'data': ROOT / 'bonus' / 'data', 'collection': 'aiindex', 'chroma': ROOT / 'chroma_db'},  # built by src.bonus_index
}

BASE = {'corpus': 'day7', 'top_k': 3, 'budget': 800, 'max_searches': 3}
CONFIGS = {
    'dense': {**BASE, 'method': 'dense', 'hypothesis': 'Baseline: Day 6 dense retrieval (index B).'},
    'bm25': {**BASE, 'method': 'bm25', 'hypothesis': 'Exact terms (elif, range, continue) may beat embeddings on keyword-heavy questions.'},
    'hybrid': {**BASE, 'method': 'hybrid', 'hypothesis': 'RRF of dense + BM25 recovers misses of either one.'},
    'hybrid_b2400': {**BASE, 'method': 'hybrid', 'budget': 2400,
                     'hypothesis': 'Index-B chunks reach 800 chars, so an 800-char budget keeps ~1 chunk; 2400 keeps all top-3.'},
    'hybrid_rerank': {**BASE, 'method': 'rerank', 'budget': 2400,
                      'hypothesis': 'LLM reorders 10 hybrid candidates -> evidence moves into top-1/top-3.'},
    # bonus (AI Index 2026): table/chart chunks are long and multi-hop needs two facts -> more chunks, bigger budget
    'ai_hybrid': {'corpus': 'aiindex', 'method': 'hybrid', 'top_k': 5, 'budget': 4000, 'max_searches': 3,
                  'hypothesis': 'Hybrid best on Day 7 dev; top-5 + 4000 chars fit two table/chart chunks.'},
    'ai_rerank': {'corpus': 'aiindex', 'method': 'rerank', 'top_k': 5, 'budget': 4000, 'max_searches': 3,
                  'hypothesis': 'Rerank was 20/20 hit@1 on Day 7 dev; should lift the right figure page to the top.'},
}


def load(corpus: str, name: str):
    return json.loads((CORPORA[corpus]['data'] / name).read_text(encoding='utf-8'))


@cache
def _parts(corpus: str):
    c = CORPORA[corpus]
    return DenseRetriever(c['chroma'], c['collection']), BM25Retriever(load(corpus, 'chunks.json'))


def build_retriever(cfg: dict):
    dense, bm25 = _parts(cfg['corpus'])
    hybrid = HybridRetriever(dense, bm25)
    return {'dense': dense, 'bm25': bm25, 'hybrid': hybrid, 'rerank': RerankRetriever(hybrid)}[cfg['method']]
