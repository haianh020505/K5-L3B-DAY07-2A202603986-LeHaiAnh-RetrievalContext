"""Offline checks: fake chat payloads, no network. They do not measure model quality."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.controller import OpenRouterController
from src.run_bench import run_case

IDS = ['openstax-python-p111-c02', 'openstax-python-p111-c03']
HISTORY = [{'tool': 'search', 'queries': ['elif'], 'context': '[openstax-python-p111-c02 | x | page 111]\n...',
            'context_ids': IDS}]


class FakeChat:
    def __init__(self, *replies): self.replies, self.calls = list(replies), []
    def __call__(self, messages, **kw):
        self.calls.append((messages, kw)); return self.replies.pop(0)


def chunk(cid):
    return {'id': cid, 'text': 'Use elif.', 'source_id': 'd', 'source': 'd.pdf', 'page': 1, 'metadata': {}, 'score': 1}


class Retriever:
    def search(self, query, *, filters=None, top_k=3): return [chunk(IDS[0])]


class ControllerTests(unittest.TestCase):
    def test_decide_maps_tool_call_to_search(self):
        chat = FakeChat({'content': None, 'tool_calls': [
            {'id': 't1', 'function': {'name': 'search_documents', 'arguments': '{"query": "elif vs if"}'}}]})
        action = OpenRouterController(chat=chat).decide('Q?', [])
        self.assertEqual(action, {'action': 'search', 'query': 'elif vs if'})
        self.assertEqual(chat.calls[0][1]['tool_choice'], 'required')  # first turn must search

    def test_decide_parses_final_answer_without_tool_call(self):
        chat = FakeChat({'content': f'Only one branch runs [{IDS[0]}, {IDS[1]}].'})
        action = OpenRouterController(chat=chat).decide('Q?', HISTORY)
        self.assertEqual(action['action'], 'answer')
        self.assertEqual(action['citations'], IDS)
        self.assertTrue(all(f'[{i}]' in action['answer'] for i in IDS))

    def test_lenticular_brackets_become_citations(self):
        # gpt-oss on Groq cites with 【id】 (seen on q06); the starter only accepts [id]
        chat = FakeChat({'content': f'Only one branch runs【{IDS[0]}】【{IDS[1]}】'})
        action = OpenRouterController(chat=chat).decide('Q?', HISTORY)
        self.assertEqual(action['citations'], IDS)
        self.assertTrue(all(f'[{i}]' in action['answer'] for i in IDS))

    def test_context_header_copied_as_citation_is_reduced_to_id(self):
        # dev q07/q08/q16 (gpt-oss): the model cites the whole context header "[id | source | page N]"
        chat = FakeChat({'content': f'Ternary operator [{IDS[0]} | starter/data/x.pdf | page 119]'})
        action = OpenRouterController(chat=chat).decide('Q?', HISTORY)
        self.assertEqual(action['citations'], [IDS[0]])
        self.assertIn(f'[{IDS[0]}]', action['answer'])

    def test_abstain_reply_has_no_citations(self):
        action = OpenRouterController(chat=FakeChat({'content': 'Không đủ bằng chứng.'})).decide('Q?', HISTORY)
        self.assertEqual(action, {'action': 'answer', 'answer': 'Không đủ bằng chứng.', 'citations': []})

    def test_run_bench_records_error_and_continues(self):
        cfg = {'top_k': 3, 'budget': 800, 'max_searches': 2}
        boom = OpenRouterController(chat=FakeChat())  # empty replies -> IndexError, like a provider failure
        bad = run_case({'id': 'q1', 'split': 'dev', 'question': 'Q?'}, boom, Retriever(), cfg)
        self.assertEqual(bad['status'], 'error')
        self.assertIn('error', bad)
        ok_ctrl = OpenRouterController(chat=FakeChat(
            {'tool_calls': [{'id': 't', 'function': {'name': 'search_documents', 'arguments': '{"query": "elif"}'}}]},
            {'content': f'Use elif [{IDS[0]}].'}))
        ok = run_case({'id': 'q2', 'split': 'dev', 'question': 'Q?'}, ok_ctrl, Retriever(), cfg)
        self.assertEqual((ok['status'], ok['citations'], ok['n_searches']), ('answered', [IDS[0]], 1))
        self.assertEqual(len(ok['decisions']), 2)


if __name__ == '__main__':
    unittest.main()
