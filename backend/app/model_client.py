"""Shared model calls: native Claude/OpenAI APIs and local multilingual embeddings."""

import json
from functools import lru_cache
from threading import Lock

import httpx
import numpy as np

from .config import get_settings
from .schemas import ANSWER_JSON_SCHEMA

_embedding_lock = Lock()


class ModelError(RuntimeError):
    """Safe, user-facing provider or configuration failure."""


def _post(url: str, headers: dict, body: dict) -> dict:
    """Send a model API request and convert network or provider failures into safe ModelError messages."""
    try:
        response = httpx.post(url, headers=headers, json=body,
                              timeout=httpx.Timeout(75, connect=15))
    except httpx.HTTPError as exc:
        raise ModelError("Cannot reach the model API. Check MODEL_BASE_URL and your connection.") from exc
    _check_status(response)
    try:
        return response.json()
    except ValueError as exc:
        raise ModelError("Model API returned an invalid response.") from exc


def _check_status(response):
    """Keep provider errors safe without exposing response bodies or credentials."""
    if response.status_code != 200:
        categories = {401: "API key rejected", 403: "access denied", 404: "model or endpoint not found",
                      429: "quota or rate limit reached", 529: "provider temporarily overloaded"}
        reason = categories.get(response.status_code, "provider request failed")
        raise ModelError(f"Model API: {reason} (HTTP {response.status_code}).")


@lru_cache(maxsize=2)
def _embedding_model(name: str, cache_dir: str):
    """Load and cache a FastEmbed model for the optional local embedding backend."""
    from fastembed import TextEmbedding
    return TextEmbedding(name, cache_dir=cache_dir, threads=2)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts in input order and normalize vectors using the configured backend."""
    if not texts:
        return []
    settings = get_settings()
    if settings.embedding_backend == "openai":
        if not settings.model_api_key:
            raise ModelError("Set MODEL_API_KEY to enable OpenAI embeddings.")
        if settings.model_provider != "openai":
            raise ModelError("OpenAI embeddings require an OpenAI API key; use local embeddings with Claude.")
        if settings.model_api_key.startswith("sk-ant-"):
            raise ModelError("Replace the Claude key in .env with your OpenAI API key.")
        vectors = []
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            data = _post(settings.model_base_url + "/embeddings",
                         {"Authorization": "Bearer " + settings.model_api_key},
                         {"model": settings.embedding_model, "input": batch, "encoding_format": "float"})
            records = sorted(data.get("data", []), key=lambda item: item["index"])
            if len(records) != len(batch):
                raise ModelError("Embedding API returned the wrong number of vectors.")
            vectors.extend(item["embedding"] for item in records)
        return _normalize(vectors)
    # Short overlapping windows keep long Chinese clauses inside the model's
    # 512-token input limit. Mean pooling retains information at the chunk's end.
    windows, owners = [], []
    for owner, text in enumerate(texts):
        for start in range(0, max(len(text), 1), 250):
            windows.append(text[start:start + 300] or " ")
            owners.append(owner)
    with _embedding_lock:
        model = _embedding_model(settings.embedding_model, str(settings.data_dir / "models"))
        vectors = np.asarray(list(model.embed(windows, batch_size=32)), dtype=np.float32)
    owners = np.asarray(owners)
    return _normalize([vectors[owners == owner].mean(axis=0) for owner in range(len(texts))])


def _normalize(vectors) -> list[list[float]]:
    """Validate vectors and normalize their lengths for cosine similarity via dot products."""
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or np.any(norms == 0):
        raise ModelError("Embedding model returned invalid vectors.")
    return (matrix / norms).tolist()


def generate_answer(messages: list[dict[str, str]]) -> str:
    """Generate claims using the grounded-answer schema."""
    return generate_json(messages, ANSWER_JSON_SCHEMA, "grounded_answer", max_tokens=2200)


def stream_answer(messages: list[dict[str, str]]):
    """Yield OpenAI JSON text deltas; require a complete, normally finished stream."""
    settings = get_settings()
    if settings.model_provider != "openai":
        # The optional Claude path retains its existing buffered implementation.
        yield generate_answer(messages)
        return
    if not settings.model_api_key:
        raise ModelError("Set MODEL_API_KEY in .env to enable generated answers.")
    if settings.model_api_key.startswith("sk-ant-"):
        raise ModelError("Replace the Claude key in .env with your OpenAI API key.")
    body = {"model": settings.chat_model, "messages": messages, "stream": True,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "grounded_answer", "strict": True, "schema": ANSWER_JSON_SCHEMA}},
            "temperature": 0, "max_tokens": 2200}
    finished, done, has_text = False, False, False
    try:
        with httpx.stream("POST", settings.model_base_url + "/chat/completions",
                          headers={"Authorization": "Bearer " + settings.model_api_key},
                          json=body, timeout=httpx.Timeout(75, connect=15)) as response:
            _check_status(response)
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    done = True
                    break
                try:
                    data = json.loads(raw)
                    if data.get("error"):
                        raise ModelError("Model API returned a streaming error.")
                    choices = data["choices"]
                    if not choices:
                        continue  # Optional usage-only record.
                    choice = choices[0]
                    delta = choice["delta"]
                    if delta.get("refusal"):
                        raise ModelError("The model declined this answer. Please rephrase your question.")
                    content = delta.get("content")
                    if content is not None:
                        if not isinstance(content, str) or finished:
                            raise ValueError("Unexpected stream content")
                        if content:
                            has_text = True
                            yield content
                    reason = choice.get("finish_reason")
                    if reason == "length":
                        raise ModelError("The answer was truncated. Please ask a narrower question.")
                    if reason is not None:
                        if reason != "stop":
                            raise ModelError("The model could not complete the answer. Please rephrase your question.")
                        finished = True
                except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
                    raise ModelError("OpenAI returned an unexpected stream format.") from exc
    except httpx.HTTPError as exc:
        raise ModelError("Cannot reach the model API. Check MODEL_BASE_URL and your connection.") from exc
    if not done or not finished or not has_text:
        raise ModelError("The model stream ended before the answer was complete. Please try again.")


def generate_json(messages: list[dict[str, str]], schema: dict, name: str, *, max_tokens: int) -> str:
    """Call the configured chat model with a task-specific JSON schema and token limit."""
    settings = get_settings()
    if not settings.model_api_key:
        raise ModelError("Set MODEL_API_KEY in .env to enable generated answers.")
    if settings.model_provider == "openai":
        if settings.model_api_key.startswith("sk-ant-"):
            raise ModelError("Replace the Claude key in .env with your OpenAI API key.")
        data = _post(settings.model_base_url + "/chat/completions",
                     {"Authorization": "Bearer " + settings.model_api_key},
                     {"model": settings.chat_model, "messages": messages,
                      "response_format": {"type": "json_schema", "json_schema": {
                          "name": name, "strict": True, "schema": schema}},
                      "temperature": 0, "max_tokens": max_tokens})
        try:
            choice = data["choices"][0]
            if choice["finish_reason"] == "length":
                raise ModelError("The answer was truncated. Please ask a narrower question.")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ModelError("The model returned no answer. Please rephrase your question.")
            return content
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError("OpenAI returned an unexpected answer format.") from exc
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    conversation = [m for m in messages if m["role"] in {"user", "assistant"}]
    base = settings.model_base_url
    data = _post(base + ("/messages" if base.endswith("/v1") else "/v1/messages"),
                 {"x-api-key": settings.model_api_key, "anthropic-version": "2023-06-01"},
                 {"model": settings.chat_model, "max_tokens": max_tokens,
                  "system": system, "messages": conversation})
    try:
        if data.get("stop_reason") == "max_tokens":
            raise ModelError("The answer was truncated. Please ask a narrower question.")
        text = "\n".join(block["text"] for block in data["content"] if block.get("type") == "text")
        if not text.strip():
            raise ModelError("Claude returned no text answer.")
        return text
    except (KeyError, TypeError) as exc:
        raise ModelError("Claude returned an unexpected answer format.") from exc
