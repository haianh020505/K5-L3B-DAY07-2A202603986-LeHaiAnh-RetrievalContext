"""Resumable bench: an interrupted run keeps finished cases and only re-runs missing/errored ones."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.run_bench import run_resumable


class ResumeTests(unittest.TestCase):
    def test_skips_finished_cases_and_reruns_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / 'traces.partial.json'
            partial.write_text(json.dumps([{'query_id': 'q1', 'status': 'answered'},
                                           {'query_id': 'q2', 'status': 'error'}]), encoding='utf-8')
            ran = []

            def run_one(case):
                ran.append(case['id'])
                return {'query_id': case['id'], 'status': 'abstain'}

            traces = run_resumable([{'id': 'q1'}, {'id': 'q2'}, {'id': 'q3'}], run_one, partial)
            self.assertEqual(ran, ['q2', 'q3'])
            self.assertEqual([(t['query_id'], t['status']) for t in traces],
                             [('q1', 'answered'), ('q2', 'abstain'), ('q3', 'abstain')])
            saved = json.loads(partial.read_text(encoding='utf-8'))
            self.assertEqual(sorted(t['query_id'] for t in saved), ['q1', 'q2', 'q3'])


if __name__ == '__main__':
    unittest.main()
