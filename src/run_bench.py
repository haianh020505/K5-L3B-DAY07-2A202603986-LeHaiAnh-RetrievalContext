"""Run the real agent on every case (answerable + unanswerable) of one split; save traces.

Run: python -m src.run_bench --config hybrid_b2400 --split dev [--only q01,u01]
No gold enters the agent: only id, question and the user-supplied filter are read.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from . import openrouter
from .configs import CONFIGS, ROOT, build_retriever, load
from .controller import OpenRouterController

sys.path.insert(0, str(ROOT / 'starter'))
from agentic import run_agent  # noqa: E402


def run_case(case: dict, controller, retriever, cfg: dict) -> dict:
    start, cost0 = time.perf_counter(), openrouter.USAGE['cost_usd']
    trace = {'query_id': case['id'], 'split': case['split'], 'question': case['question'],
             'filter': case.get('filter'), 'model': getattr(controller, 'model', None)}
    try:
        result = run_agent(case['question'], controller, retriever, filters=case.get('filter'),
                           max_searches=cfg['max_searches'], top_k=cfg['top_k'], budget=cfg['budget'])
        trace.update(result)
    except Exception as exc:  # provider/vector-store failure: recorded, never replaced by a fake answer
        trace.update({'status': 'error', 'error': f'{type(exc).__name__}: {exc}', 'answer': None,
                      'citations': [], 'events': []})
    trace.update({'n_searches': sum(e.get('event') == 'search' for e in trace['events']),
                  'decisions': controller.log, 'elapsed_ms': round((time.perf_counter() - start) * 1000),
                  'cost_usd': openrouter.USAGE['cost_usd'] - cost0})
    return trace


def run_resumable(cases: list[dict], run_one, partial: Path) -> list[dict]:
    """Save after every case; on restart reuse finished cases from `partial`, re-run errored/missing ones."""
    done = {t['query_id']: t for t in json.loads(partial.read_text(encoding='utf-8'))} if partial.exists() else {}
    done = {k: t for k, t in done.items() if t['status'] != 'error'}
    for case in cases:
        if case['id'] not in done:
            done[case['id']] = run_one(case)
            partial.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return [done[c['id']] for c in cases]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, choices=CONFIGS)
    ap.add_argument('--split', required=True, choices=('dev', 'test'))
    ap.add_argument('--only', help='comma-separated case ids (smoke runs)')
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    cases = [c for c in load(cfg['corpus'], 'cases.json') if c['split'] == args.split]
    if args.only:
        cases = [c for c in cases if c['id'] in args.only.split(',')]
    retriever = build_retriever(cfg)
    out = ROOT / ('bonus/results' if cfg['corpus'] != 'day7' else 'results') / 'agent' / f'{args.config}-{args.split}'
    out.mkdir(parents=True, exist_ok=True)
    name = 'traces.json' if not args.only else 'smoke_traces.json'

    def run_one(case):
        t = run_case(case, OpenRouterController(max_searches=cfg['max_searches']), retriever, cfg)
        print(f"{t['query_id']}: {t['status']} searches={t['n_searches']} cites={t['citations']} {t['elapsed_ms']}ms")
        return t

    partial = out / name.replace('.json', '.partial.json')
    traces = run_resumable(cases, run_one, partial)
    (out / name).write_text(json.dumps(traces, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    partial.unlink()
    status = {s: sum(t['status'] == s for t in traces) for s in ('answered', 'abstain', 'error')}
    print(json.dumps({'config': args.config, 'split': args.split, **status,
                      'cost_usd': round(sum(t['cost_usd'] for t in traces), 5)}))


if __name__ == '__main__':
    main()
