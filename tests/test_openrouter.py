"""Query-embedding cache keeps every config on identical query vectors (provider output drifts slightly)."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import openrouter


class CacheTests(unittest.TestCase):
    def test_embed_query_uses_disk_cache(self):
        calls = []

        def fake_post(path, body, timeout=180):
            calls.append(body['input'])
            return {'data': [{'index': 0, 'embedding': [0.1, 0.2]}]}

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(openrouter, 'post', fake_post):
            cache = Path(tmp) / 'q.json'
            self.assertEqual(openrouter.embed_query('hello', cache=cache), [0.1, 0.2])
            self.assertEqual(openrouter.embed_query('hello', cache=cache), [0.1, 0.2])
            self.assertEqual(len(calls), 1)
            self.assertTrue(cache.exists())


class ProviderTests(unittest.TestCase):
    def test_groq_prefix_routes_chat_to_groq(self):
        seen = {}

        def fake_post(path, body, timeout=180, **kw):
            seen.update(path=path, body=body, **kw)
            return {'choices': [{'message': {'content': 'ok'}}]}

        with mock.patch.object(openrouter, 'post', fake_post):
            self.assertEqual(openrouter.chat([{'role': 'user', 'content': 'hi'}], model='groq/openai/gpt-oss-120b'),
                             {'content': 'ok'})
        self.assertEqual(seen['body']['model'], 'openai/gpt-oss-120b')
        self.assertEqual((seen['base'], seen['key_env']), (openrouter.GROQ_URL, 'GROQ_API_KEY'))
        self.assertNotIn('usage', seen['body'])  # OpenRouter-only field

    def test_gemini_prefix_routes_chat_to_gemini(self):
        seen = {}

        def fake_post(path, body, timeout=180, **kw):
            seen.update(body=body, **kw)
            return {'choices': [{'message': {'content': 'ok'}}]}

        with mock.patch.object(openrouter, 'post', fake_post):
            openrouter.chat([{'role': 'user', 'content': 'hi'}], model='gemini/gemini-3.5-flash-lite')
        self.assertEqual(seen['body']['model'], 'gemini-3.5-flash-lite')
        self.assertEqual((seen['base'], seen['key_env']), (openrouter.GEMINI_URL, 'GEMINI_API_KEY'))

    def test_429_waits_retry_after(self):
        import io
        import urllib.error
        err = urllib.error.HTTPError('u', 429, 'rate', {'retry-after': '7'}, io.BytesIO(b''))
        ok = mock.MagicMock()
        ok.__enter__.return_value = io.BytesIO(b'{"data": []}')
        with mock.patch.dict('os.environ', {'OPENROUTER_API_KEY': 'k'}), \
                mock.patch('urllib.request.urlopen', side_effect=[err, ok]), \
                mock.patch.object(openrouter.time, 'sleep') as sleep:
            openrouter.post('embeddings', {})
        sleep.assert_called_once_with(7.0)


if __name__ == '__main__':
    unittest.main()
