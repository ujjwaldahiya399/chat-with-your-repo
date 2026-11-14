from types import SimpleNamespace
from unittest.mock import MagicMock

import embed_repo
import main


def test_repository_embedding_reuses_supplied_model(tmp_path, monkeypatch):
    (tmp_path / "first.py").write_text("print('first')\n")
    (tmp_path / "second.py").write_text("print('second')\n")

    supplied_model = object()
    models_used = []

    def fake_embed_file(file_path, *, model, **kwargs):
        models_used.append(model)

    collection = MagicMock()
    collection.count.return_value = 2
    client = MagicMock()
    client.get_or_create_collection.return_value = collection

    monkeypatch.setattr(embed_repo, "embed_file", fake_embed_file)
    monkeypatch.setattr(embed_repo.chromadb, "PersistentClient", lambda path: client)

    files_embedded, total = embed_repo.embed_repo(str(tmp_path), model=supplied_model)

    assert files_embedded == 2
    assert total == 2
    assert models_used == [supplied_model, supplied_model]


def test_api_passes_loaded_model_to_repository_embedding(monkeypatch):
    supplied_model = object()
    received = {}

    def fake_embed_repo_from_url(github_url, *, branch, model):
        received.update(url=github_url, branch=branch, model=model)
        return 2, 5

    monkeypatch.setattr(main, "embed_repo_from_url", fake_embed_repo_from_url)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(model=supplied_model))
    )

    result = main.embed_repo_endpoint(
        main.EmbedRepoRequest(github_url="https://github.com/example/project"),
        request,
    )

    assert result.files_embedded == 2
    assert result.total_chunks_in_collection == 5
    assert received == {
        "url": "https://github.com/example/project",
        "branch": "main",
        "model": supplied_model,
    }
