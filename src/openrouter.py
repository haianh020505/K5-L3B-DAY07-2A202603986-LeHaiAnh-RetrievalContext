"""Minimal OpenAI-compatible client: OpenRouter for embeddings; "groq/<id>" and "gemini/<id>" models go to those APIs.

Errors raise; there is never a mock fallback. USAGE accumulates tokens and reported cost.
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
load_dotenv(ROOT / 'starter' / '.env')

BASE_URL = 'https://openrouter.ai/api/v1/'
GROQ_URL = 'https://api.groq.com/openai/v1/'
GEMINI_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/'
PROVIDERS = {'groq/': (GROQ_URL, 'GROQ_API_KEY'), 'gemini/': (GEMINI_URL, 'GEMINI_API_KEY')}  # free tiers, no cost reported
CHAT_MODEL = os.getenv('CHAT_MODEL', 'groq/openai/gpt-oss-120b')  # agent + rerank (Groq free tier)
VISION_MODEL = os.getenv('VISION_MODEL', 'gemini/gemini-3.5-flash-lite')  # bonus page transcription
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'openai/text-embedding-3-small')
USAGE = {'requests': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'cost_usd': 0.0}


def post(path: str, body: dict, timeout: int = 180, *, base: str = BASE_URL,
         key_env: str = 'OPENROUTER_API_KEY') -> dict:
    key = os.getenv(key_env)
    if not key:
        raise RuntimeError(f'{key_env} missing (starter/.env)')
    # explicit User-Agent: Groq's edge rejects the default "Python-urllib" agent
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={
        'Authorization': f'Bearer {key}', 'Content-Type': 'application/json', 'User-Agent': 'day7-rag-lab'})
    for attempt in range(6):
        wait = 2 ** attempt
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.load(resp)
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise RuntimeError(f'{base} HTTP {exc.code}: {exc.read()[:300]!r}') from exc
            wait = float((exc.headers or {}).get('retry-after') or wait)  # Groq free tier: 8K tokens/min
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 5:
                raise RuntimeError(f'{base} network error: {exc}') from exc
        time.sleep(wait)
    if 'error' in payload:
        raise RuntimeError(f"OpenRouter error: {str(payload['error'])[:300]}")
    usage = payload.get('usage') or {}
    USAGE['requests'] += 1
    USAGE['prompt_tokens'] += usage.get('prompt_tokens', 0) or 0
    USAGE['completion_tokens'] += usage.get('completion_tokens', 0) or 0
    USAGE['cost_usd'] += usage.get('cost', 0) or 0
    return payload


def embed(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    data = post('embeddings', {'model': model, 'input': texts})['data']
    return [d['embedding'] for d in sorted(data, key=lambda d: d['index'])]


QUERY_CACHE = ROOT / 'results' / 'query_embeddings.json'


def embed_query(text: str, model: str = EMBEDDING_MODEL, cache: Path = QUERY_CACHE) -> list[float]:
    """Provider vectors drift ~1e-4 between calls and once changed a dense top-20, so every
    config/re-run reuses the first vector stored on disk (keyed by model + text)."""
    store = json.loads(cache.read_text(encoding='utf-8')) if cache.exists() else {}
    key = f'{model}\n{text}'
    if key not in store:
        store[key] = embed([text], model)[0]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(store), encoding='utf-8')
    return store[key]


def chat(messages: list[dict], *, model: str = CHAT_MODEL, **kwargs) -> dict:
    """Return the assistant message dict (content and optional tool_calls)."""
    body = {'model': model, 'messages': messages, 'temperature': 0, **kwargs}
    prefix = next((p for p in PROVIDERS if model.startswith(p)), None)
    if prefix:  # Groq/Gemini report no cost: USAGE counts their tokens, cost stays 0
        body['model'] = model.removeprefix(prefix)
        base, key_env = PROVIDERS[prefix]
        payload = post('chat/completions', body, base=base, key_env=key_env)
    else:
        payload = post('chat/completions', {**body, 'usage': {'include': True}})
    choices = payload.get('choices')
    if not choices:
        raise RuntimeError(f'{model} returned no choices')
    return choices[0]['message']
