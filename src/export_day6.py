"""Export Day 6 chunks (Chroma) + benchmark to the Day 7 JSON contract (data/README.md).

Run: python -m src.export_day6 --day6 "<Day 6 repo>" [--out data/day7]
Day 6 test questions were already inspected during Day 6, so they move to dev here;
the Day 7 test split comes from data/day7/new_test.json (written fresh for Day 7).
"""
import argparse
import json
from pathlib import Path


def convert_question(q: dict) -> dict:
    gold = [{'source_id': q['gold_source_id'], 'page': q['gold_page'], 'evidence': q['evidence_quote']}]
    gold += [{'source_id': q['gold_source_id'], 'page': g['page'], 'evidence': g['evidence_quote']}
             for g in q.get('alt_gold', [])]
    out = {'id': q['id'], 'split': q['split'], 'question': q['question'],
           'expected_answer': q['expected_answer'], 'gold': gold}
    if q.get('filter'):
        out['filter'] = q['filter']
    return out


def export_chunks(chroma_path: Path, collection: str = 'day06_B') -> list[dict]:
    import chromadb
    from .retrievers import chunk_from_chroma
    col = chromadb.PersistentClient(path=str(chroma_path)).get_collection(collection)
    res = col.get(include=['documents', 'metadatas'])
    chunks = [chunk_from_chroma(*row) for row in zip(res['ids'], res['documents'], res['metadatas'])]
    return sorted(chunks, key=lambda c: (c['page'], c['id']))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--day6', required=True, type=Path, help='Day 6 repo root')
    ap.add_argument('--out', default=Path('data/day7'), type=Path)
    args = ap.parse_args()
    chunks = export_chunks(args.day6 / 'starter' / 'chroma_db')
    bench = json.loads((args.day6 / 'benchmark' / 'questions.json').read_text(encoding='utf-8'))
    questions = [{**convert_question(q), 'split': 'dev'} for q in bench['questions']]
    extra = []
    for name in ('new_test.json', 'unanswerable.json'):
        path = args.out / name
        if path.exists():
            extra += json.loads(path.read_text(encoding='utf-8'))
    questions += [q for q in extra if q.get('answerable', True)]
    cases = [{'id': q['id'], 'split': q['split'], 'question': q['question'], 'answerable': True,
              **({'filter': q['filter']} if q.get('filter') else {})} for q in questions]
    cases += [{k: q[k] for k in ('id', 'split', 'question', 'answerable', 'expected_behavior', 'absence_reason')}
              for q in extra if q.get('answerable') is False]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (('chunks.json', chunks), ('questions.json', questions), ('cases.json', cases)):
        (args.out / name).write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    splits = {s: sum(c['split'] == s and c['answerable'] for c in cases) for s in ('dev', 'test')}
    print(f'{len(chunks)} chunks; answerable {splits}; unanswerable {sum(not c["answerable"] for c in cases)}')


if __name__ == '__main__':
    main()
