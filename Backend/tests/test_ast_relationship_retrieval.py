from ast_relationship_retrieval import expand_ast_relationships


class FakeCollection:
    def __init__(self, ids, documents, metadatas):
        self.result = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }

    def get(self, **kwargs):
        return self.result


def metadata(file_path, start_line, repo_url="https://example.test/repo"):
    return {
        "file_path": file_path,
        "start_line": start_line,
        "repo_url": repo_url,
        "branch": "main",
    }


def query_result(ids, documents, metadatas):
    return {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
    }


def test_direct_relationship_displaces_later_semantic_chunk():
    ids = ["neuron", "value", "readme"]
    documents = [
        "class Neuron(Module):\n    def call(self, x):\n        return Value(x)\n",
        "class Value:\n    pass\n",
        "documentation",
    ]
    metadatas = [
        metadata("micrograd/nn.py", 13),
        metadata("micrograd/engine.py", 2),
        metadata("README.md", 1),
    ]
    collection = FakeCollection(ids, documents, metadatas)

    selected_documents, _, provenance = expand_ast_relationships(
        collection,
        query_result(
            ["neuron", "readme"],
            [documents[0], documents[2]],
            [metadatas[0], metadatas[2]],
        ),
        final_k=2,
    )

    assert selected_documents == [documents[0], documents[1]]
    assert provenance[1]["selection_origin"] == "ast_relationship"
    assert provenance[1]["relationship_depth"] == 1


def test_relationship_expansion_follows_two_hops():
    ids = ["mlp", "layer", "neuron", "readme"]
    documents = [
        "class MLP:\n    def build(self):\n        return Layer()\n",
        "class Layer:\n    def build(self):\n        return Neuron()\n",
        "class Neuron:\n    pass\n",
        "documentation",
    ]
    metadatas = [
        metadata("micrograd/nn.py", 45),
        metadata("micrograd/nn.py", 30),
        metadata("micrograd/nn.py", 13),
        metadata("README.md", 1),
    ]
    collection = FakeCollection(ids, documents, metadatas)

    selected_documents, _, provenance = expand_ast_relationships(
        collection,
        query_result(
            ["mlp", "readme", "neuron"],
            [documents[0], documents[3], documents[2]],
            [metadatas[0], metadatas[3], metadatas[2]],
        ),
        final_k=3,
    )

    assert selected_documents == documents[:3]
    assert [row["relationship_depth"] for row in provenance] == [0, 1, 2]


def test_relationships_do_not_cross_repository_scope():
    ids = ["seed", "other-value", "same-value"]
    documents = [
        "class Neuron:\n    def call(self):\n        return Value(1)\n",
        "class Value:\n    pass\n",
        "class Value:\n    pass\n",
    ]
    metadatas = [
        metadata("nn.py", 1, repo_url="https://example.test/one"),
        metadata("engine.py", 1, repo_url="https://example.test/two"),
        metadata("engine.py", 1, repo_url="https://example.test/one"),
    ]
    collection = FakeCollection(ids, documents, metadatas)

    _, selected_metadatas, _ = expand_ast_relationships(
        collection,
        query_result(["seed"], [documents[0]], [metadatas[0]]),
        final_k=2,
    )

    assert [row["repo_url"] for row in selected_metadatas] == [
        "https://example.test/one",
        "https://example.test/one",
    ]


def test_unparseable_python_preserves_semantic_results():
    ids = ["broken", "readme"]
    documents = ["    indented = True", "documentation"]
    metadatas = [metadata("broken.py", 1), metadata("README.md", 1)]
    collection = FakeCollection(ids, documents, metadatas)

    selected_documents, selected_metadatas, provenance = expand_ast_relationships(
        collection,
        query_result(ids, documents, metadatas),
        final_k=2,
    )

    assert selected_documents == documents
    assert selected_metadatas == metadatas
    assert [row["selection_origin"] for row in provenance] == [
        "semantic",
        "semantic",
    ]
