"""Offline checks for the Day 6 -> Day 7 conversion."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.export_day6 import convert_question

Q = {'id': 'q01', 'split': 'dev', 'type': 'direct', 'question': 'Q?', 'expected_answer': 'A',
     'gold_source_id': 'openstax-python', 'gold_page': 97, 'evidence_quote': 'False: 0',
     'alt_gold': [{'page': 96, 'evidence_quote': 'x'}], 'filter': None}


class ExportTests(unittest.TestCase):
    def test_convert_question_merges_alt_gold(self):
        out = convert_question(Q)
        self.assertEqual(out['gold'], [
            {'source_id': 'openstax-python', 'page': 97, 'evidence': 'False: 0'},
            {'source_id': 'openstax-python', 'page': 96, 'evidence': 'x'}])
        self.assertEqual(out['expected_answer'], 'A')

    def test_convert_question_drops_null_filter(self):
        self.assertNotIn('filter', convert_question(Q))
        self.assertEqual(convert_question({**Q, 'filter': {'chapter': 4}})['filter'], {'chapter': 4})


if __name__ == '__main__':
    unittest.main()
