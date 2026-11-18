"""Expand semantic code retrieval through direct Python AST relationships."""

import ast


DEFAULT_MAX_RELATION_DEPTH = 2


def _chunk_id(metadata):
    repo_url = metadata.get("repo_url")
    branch = metadata.get("branch")
    prefix = f"{repo_url}@{branch}:" if repo_url else ""
    return f"{prefix}{metadata['file_path']}:{metadata['start_line']}"


def _scope(metadata):
    return (
        metadata.get("repo_url") or "__local__",
        metadata.get("branch") or "",
    )


def _parse_python(document, metadata):
    if not metadata.get("file_path", "").endswith(".py"):
        return None

    try:
        return ast.parse(document)
    except (SyntaxError, TypeError, ValueError):
        return None


def _semantic_fallback(documents, metadatas, final_k):
    count = min(final_k, len(documents), len(metadatas))
    selections = [
        {
            "selection_origin": "semantic",
            "semantic_rank": index,
            "relationship_depth": 0,
        }
        for index in range(1, count + 1)
    ]
    return documents[:count], metadatas[:count], selections


def _all_collection_records(collection, where):
    get_arguments = {"include": ["documents", "metadatas"]}
    if where is not None:
        get_arguments["where"] = where

    try:
        result = collection.get(**get_arguments)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None

    if not isinstance(result, dict):
        return None

    ids = result.get("ids")
    documents = result.get("documents")
    metadatas = result.get("metadatas")
    if not all(isinstance(values, list) for values in (ids, documents, metadatas)):
        return None
    if not (len(ids) == len(documents) == len(metadatas)):
        return None

    return {
        chunk_id: {"document": document, "metadata": metadata}
        for chunk_id, document, metadata in zip(ids, documents, metadatas)
    }


def _build_relationship_graph(records):
    parsed_trees = {}
    definitions = {}

    for chunk_id, record in records.items():
        tree = _parse_python(record["document"], record["metadata"])
        if tree is None:
            continue

        parsed_trees[chunk_id] = tree
        record_scope = _scope(record["metadata"])
        for node in tree.body:
            if isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                definitions.setdefault((record_scope, node.name), []).append(
                    chunk_id
                )

    graph = {chunk_id: [] for chunk_id in records}
    for chunk_id, tree in parsed_trees.items():
        record_scope = _scope(records[chunk_id]["metadata"])
        references = sorted(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and (record_scope, node.id) in definitions
            ),
            key=lambda node: (node.lineno, node.col_offset, node.id),
        )

        for reference in references:
            targets = sorted(definitions[(record_scope, reference.id)])
            for target in targets:
                if target != chunk_id and target not in graph[chunk_id]:
                    graph[chunk_id].append(target)

    return graph


def _select_chunks(semantic_ids, graph, final_k, max_relation_depth):
    selected = []
    selection = {}

    for semantic_rank, seed in enumerate(semantic_ids, start=1):
        if len(selected) >= final_k:
            break

        if seed not in selection:
            selected.append(seed)
            selection[seed] = {
                "selection_origin": "semantic",
                "semantic_rank": semantic_rank,
                "relationship_depth": 0,
            }

        queue = [(seed, 0)]
        queue_index = 0
        while queue_index < len(queue) and len(selected) < final_k:
            current, depth = queue[queue_index]
            queue_index += 1

            if depth >= max_relation_depth:
                continue

            for target in graph.get(current, []):
                if target in selection:
                    continue

                selected.append(target)
                selection[target] = {
                    "selection_origin": "ast_relationship",
                    "semantic_rank": None,
                    "relationship_depth": depth + 1,
                }
                queue.append((target, depth + 1))

                if len(selected) >= final_k:
                    break

    return selected, selection


def expand_ast_relationships(
    collection,
    query_result,
    *,
    where=None,
    final_k=5,
    max_relation_depth=DEFAULT_MAX_RELATION_DEPTH,
):
    """Return final documents, metadata, and selection provenance.

    Semantic results remain the fallback for collections without complete
    documents, metadata, or IDs. Python chunks that cannot be parsed are
    simply excluded from relationship discovery.
    """
    documents = query_result["documents"][0]
    metadatas = query_result["metadatas"][0]
    if not documents or final_k <= 0:
        return _semantic_fallback(documents, metadatas, final_k)

    records = _all_collection_records(collection, where)
    if records is None:
        return _semantic_fallback(documents, metadatas, final_k)

    nested_ids = query_result.get("ids")
    if nested_ids and nested_ids[0]:
        semantic_ids = nested_ids[0]
    else:
        semantic_ids = [_chunk_id(metadata) for metadata in metadatas]

    if not (len(semantic_ids) == len(documents) == len(metadatas)):
        return _semantic_fallback(documents, metadatas, final_k)

    for chunk_id, document, metadata in zip(
        semantic_ids,
        documents,
        metadatas,
    ):
        records.setdefault(
            chunk_id,
            {"document": document, "metadata": metadata},
        )

    graph = _build_relationship_graph(records)
    selected_ids, selections = _select_chunks(
        semantic_ids,
        graph,
        min(final_k, len(records)),
        max_relation_depth,
    )

    if not selected_ids:
        return _semantic_fallback(documents, metadatas, final_k)

    selected_documents = [records[chunk_id]["document"] for chunk_id in selected_ids]
    selected_metadatas = [records[chunk_id]["metadata"] for chunk_id in selected_ids]
    selected_provenance = [selections[chunk_id] for chunk_id in selected_ids]
    return selected_documents, selected_metadatas, selected_provenance
