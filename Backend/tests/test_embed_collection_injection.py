from unittest.mock import MagicMock

import embed


def test_embed_file_uses_injected_collection(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "sample.py"
    source.write_text(
        "value = 1\n",
        encoding="utf-8",
    )

    model = MagicMock()
    model.encode.return_value.tolist.return_value = [
        [0.1, 0.2]
    ]

    collection = MagicMock()
    collection.count.return_value = 1

    def fail_if_persistent_client_is_opened(*args, **kwargs):
        raise AssertionError(
            "PersistentClient must not be opened "
            "when a collection is supplied."
        )

    monkeypatch.setattr(
        embed.chromadb,
        "PersistentClient",
        fail_if_persistent_client_is_opened,
    )

    repository_url = "https://github.com/example/project"
    revision = "abc123"

    embed.embed_file(
        source,
        model=model,
        repo_root=tmp_path,
        repo_url=repository_url,
        branch=revision,
        collection=collection,
    )

    collection.delete.assert_called_once_with(
        where={
            "$and": [
                {"file_path": "sample.py"},
                {"repo_url": repository_url},
            ]
        }
    )

    upsert = collection.upsert.call_args.kwargs

    assert upsert["ids"] == [
        f"{repository_url}@{revision}:sample.py:1"
    ]
    assert upsert["documents"] == ["value = 1\n"]
    assert upsert["metadatas"] == [
        {
            "file_path": "sample.py",
            "repo_url": repository_url,
            "branch": revision,
            "start_line": 1,
        }
    ]
    assert upsert["embeddings"] == [[0.1, 0.2]]