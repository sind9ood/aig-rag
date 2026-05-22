import json
import os
import ssl
from typing import Any
from urllib import error, request

import certifi


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def has_openrouter_api_key():
    return bool(os.getenv("OPENROUTER_API_KEY"))


def chat_completion(
    messages,
    *,
    model = None,
    temperature = 0,
    timeout = 120,
    max_tokens = None,
    referer = "https://localhost",
    title = "aig-rag",
):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY. Add it to your .env file.")

    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": referer,
        "X-Title": title,
    }

    req = request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    context = ssl.create_default_context(cafile=certifi.where())

    try:
        with request.urlopen(req, timeout=timeout, context=context) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenRouter request failed: {exc.code} {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    payload_obj = json.loads(body)
    return payload_obj["choices"][0]["message"]["content"]
