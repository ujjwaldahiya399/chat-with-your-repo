from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from main import AskRequest, ask


class FakeEmbedding:
    def tolist(self):
        return [[0.1, 0.2]]


class FakeModel:
    def encode(self, texts):
        return FakeEmbedding()


class FakeCollection:
    def __init__(self):
        self.where = None

    def query(self, *, query_embeddings, n_results, where):
        self.where = where
        return {
            "documents": [["Relevant repository content"]],
            "metadatas": [[{
                "file_path": "Backend/main.py",
                "start_line": 1,
            }]],
        }


class FakeObservation:
    def start_observation(self, **kwargs):
        return FakeObservation()

    def update(self, **kwargs):
        pass

    def end(self):
        pass


class FakeLangfuse:
    def start_observation(self, **kwargs):
        return FakeObservation()

    def flush(self):
        pass


def make_request(collection):
    state = SimpleNamespace(
        model=FakeModel(),
        collection=collection,
        api_key="test-key",
        langfuse=FakeLangfuse(),
    )
    return SimpleNamespace(
        app=SimpleNamespace(state=state)
    )


def test_ask_normalizes_dot_git_before_chroma_query():
    collection = FakeCollection()

    response = ask(
        AskRequest(
            question="How does repository scoping work?",
            top_k=5,
            repo_url="https://github.com/example/project.git",
        ),
        make_request(collection),
    )

    assert isinstance(response, StreamingResponse)
    assert collection.where == {
        "repo_url": "https://github.com/example/project"
    }


def test_ask_rejects_non_github_repository_url():
    with pytest.raises(HTTPException) as error:
        ask(
            AskRequest(
                question="Question",
                repo_url="https://gitlab.com/example/project",
            ),
            SimpleNamespace(),
        )

    assert error.value.status_code == 400