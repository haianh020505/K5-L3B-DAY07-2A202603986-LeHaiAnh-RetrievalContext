"""Retrieval-only benchmark: Hit@1, Evidence Hit@3, Context Evidence Rate on one split.

Run: python -m src.eval_retrieval --config hybrid --split dev
Only answerable questions count (unanswerable ones are excluded from the denominator).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from . import openrouter
from .configs import CONFIGS, ROOT, build_retriever, load

sys.path.insert(0, str(ROOT / 'starter'))
from baseline import hit, pack_context  # noqa: E402


def evaluate(retriever, questions: list[dict], *, top_k: int, budget: int) -> list[dict]:
    traces = []
    for q in questions:
        start, cost0 = time.perf_counter(), openrouter.USAGE['cost_usd']
        results = retriever.search(q['question'], filters=q.get('filter'), top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000
        context, ids = pack_context(results, budget=budget)
        marked = [{**r, 'hit': hit(r, q['gold'])} for r in results]
        traces.append({'query_id': q['id'], 'question': q['question'], 'filter': q.get('filter'),
                       'results': [{k: r[k] for k in ('rank', 'id', 'page', 'score', 'hit')} for r in marked],
                       'context_ids': ids, 'context_chars': len(context), 'elapsed_ms': round(elapsed, 1),
                       'cost_usd': openrouter.USAGE['cost_usd'] - cost0,
                       'hit1': bool(marked and marked[0]['hit']), 'hit3': any(r['hit'] for r in marked[:3]),
                       'context_hit': any(r['hit'] and r['id'] in ids for r in marked)})
    return traces


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, choices=CONFIGS)
    ap.add_argument('--split', required=True, choices=('dev', 'test'))
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    questions = [q for q in load(cfg['corpus'], 'questions.json') if q['split'] == args.split]
    traces = evaluate(build_retriever(cfg), questions, top_k=cfg['top_k'], budget=cfg['budget'])
    n = len(traces)
    summary = {'config': args.config, **cfg, 'split': args.split, 'n': n,
               **{m: f"{sum(t[m] for t in traces)}/{n}" for m in ('hit1', 'hit3', 'context_hit')},
               'avg_context_chars': round(statistics.mean(t['context_chars'] for t in traces)),
               'median_ms': statistics.median(t['elapsed_ms'] for t in traces),
               'cost_usd': round(sum(t['cost_usd'] for t in traces), 6)}
    out = ROOT / ('bonus/results' if cfg['corpus'] != 'day7' else 'results') / 'retrieval' / f'{args.config}-{args.split}'
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in (('summary.json', summary), ('traces.json', traces)):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('config', 'split', 'hit1', 'hit3', 'context_hit', 'avg_context_chars', 'median_ms', 'cost_usd')}))


if __name__ == '__main__':
    main()
