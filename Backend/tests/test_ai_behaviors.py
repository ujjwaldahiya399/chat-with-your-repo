import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from groq import GroqError

import main


def ask_with_mocked_answer(monkeypatch, answer):
    model = MagicMock()
    model.encode.return_value.tolist.return_value = [[0.1, 0.2]]

    collection = MagicMock()
    collection.query.return_value = {
        "documents": [["The answer is in this chunk."]],
        "metadatas": [[{
            "file_path": "Backend/main.py",
            "start_line": 1,
        }]],
    }

    root = MagicMock()
    root.start_observation.return_value = MagicMock()
    langfuse = MagicMock()
    langfuse.start_observation.return_value = root

    main.app.state.model = model
    main.app.state.collection = collection
    main.app.state.api_key = "test-key"
    main.app.state.langfuse = langfuse

    def mocked_query_stream(*args):
        if isinstance(answer, Exception):
            raise answer
        yield answer

    monkeypatch.setattr(main, "query_stream", mocked_query_stream)

    response = TestClient(main.app).post(
        "/ask",
        json={"question": "What is the answer?"},
    )
    return response.text


def test_valid_citation_returns_sources(monkeypatch):
    stream = ask_with_mocked_answer(monkeypatch, "The answer (Source 1).")

    assert "event: sources" in stream
    assert "event: error" not in stream


def test_nonexistent_citation_reports_error(monkeypatch):
    stream = ask_with_mocked_answer(monkeypatch, "The answer (Source 2).")

    assert "event: error" in stream
    assert "event: sources" not in stream


def test_out_of_scope_answer_preserves_exact_fallback(monkeypatch):
    stream = ask_with_mocked_answer(monkeypatch, "I don't know.")

    tokens = [
        json.loads(frame.split("data: ", 1)[1])
        for frame in stream.split("\n\n")
        if frame.startswith("event: token\n")
    ]

    assert tokens == ["I don't know."]
    assert "event: sources" in stream
    assert "event: error" not in stream


def test_provider_error_returns_sse_error(monkeypatch):
    stream = ask_with_mocked_answer(
        monkeypatch,
        GroqError("Mocked provider failure"),
    )

    assert "event: error" in stream
    assert "event: sources" not in stream


def test_invalid_source_inside_group_reports_error(monkeypatch):
    stream = ask_with_mocked_answer(
        monkeypatch, "The answer (Source 1, Source 2)."
    )

    assert "event: error" in stream
    assert "event: sources" not in stream


def test_invalid_group_with_narrow_spaces_reports_error(monkeypatch):
    stream = ask_with_mocked_answer(
        monkeypatch, "The answer (Source\u202f1, Source\u202f2)."
    )

    assert "event: error" in stream
    assert "event: sources" not in stream
