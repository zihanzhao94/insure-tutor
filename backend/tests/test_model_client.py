import httpx
import pytest
from app import model_client
from app.model_client import ModelError


def test_openai_payload_keeps_key_server_side_and_constrains_output(monkeypatch):
    captured = {}
    def post(url, headers, body):
        captured.update(url=url, headers=headers, body=body)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"status":"insufficient_evidence","claims":[]}'}}]}
    monkeypatch.setattr(model_client, '_post', post)
    model_client.generate_answer([{'role':'system','content':'Answer using JSON.'}, {'role':'user','content':'insurance?'}])
    assert captured['url'].endswith('/chat/completions')
    assert captured['headers']['Authorization'] == 'Bearer test-key'
    assert 'test-key' not in str(captured['body'])
    assert captured['body']['response_format']['json_schema']['strict'] is True


def test_provider_errors_are_sanitized(monkeypatch):
    monkeypatch.setattr(httpx, 'post', lambda *a, **kw: httpx.Response(401, json={'error': 'sensitive provider response'}))
    with pytest.raises(ModelError, match='API key rejected') as exc:
        model_client._post('https://example.invalid', {}, {})
    assert 'sensitive' not in str(exc.value)


def test_claude_key_is_never_sent_to_openai(monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY', 'sk-ant-test-only')
    with pytest.raises(ModelError, match='Replace the Claude key'):
        model_client.embed_texts(['insurance text'])
    with pytest.raises(ModelError, match='Replace the Claude key'):
        model_client.generate_answer([])


def test_embedding_results_are_reordered_and_normalized(monkeypatch):
    monkeypatch.setattr(model_client, '_post', lambda *a: {'data': [
        {'index': 1, 'embedding': [0., 2.]}, {'index': 0, 'embedding': [3., 0.]}]})
    assert model_client.embed_texts(['first passage', 'second passage']) == [[1., 0.], [0., 1.]]


def test_null_or_truncated_answer_is_a_safe_model_error(monkeypatch):
    for finish, content in [('stop', None), ('length', 'partial')]:
        monkeypatch.setattr(model_client, '_post', lambda *a: {'choices': [{'finish_reason': finish, 'message': {'content': content}}]})
        with pytest.raises(ModelError):
            model_client.generate_answer([])
