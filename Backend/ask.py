import sys
import argparse

import chromadb
from sentence_transformers import SentenceTransformer

from embed import build_github_url, CHROMA_DB_PATH, COLLECTION_NAME, MODEL_NAME
from query import load_api_key, safe_query_stream


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    # Optional — same $and-free single-condition `where` scoping as main.py's
    # /ask; omitted, this searches the whole collection (old local-file
    # chunks with no repo_url included) same as before this change.
    parser.add_argument("--repo-url", default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # No more re-embedding here: ask.py used to take a single file_path and
    # re-embed it via embed_file() on every call, which made sense when
    # there was only ever one local file in play. Now that whole repos are
    # pre-embedded via embed_repo_from_url() (and single files via
    # embed_file() directly), there's no single file to re-embed — ask.py
    # is now a pure query against whatever's already in the collection,
    # matching main.py's /ask.
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    question_embedding = model.encode([args.question]).tolist()
    where = {"repo_url": args.repo_url} if args.repo_url else None
    # Capture the full query result (not just ["documents"][0]) so metadata
    # (file_path/start_line) can be read from the same call instead of a second query.
    query_result = collection.query(
        query_embeddings=question_embedding, n_results=args.top_k, where=where
    )
    retrieved_chunks = query_result["documents"][0]
    retrieved_metadatas = query_result["metadatas"][0]

    if not retrieved_chunks:
        print("Error: no chunks found in the collection", file=sys.stderr)
        sys.exit(1)

    # Prefix each chunk with a "[Source N: file:line]" header (1-indexed, matching
    # retrieval order) so the model can cite "(Source N)" and we can print a
    # source list afterward that lines up with what was actually retrieved.
    labeled_chunks = [
        f"[Source {i}: {meta['file_path']}:{meta['start_line']}]\n{content}"
        for i, (content, meta) in enumerate(zip(retrieved_chunks, retrieved_metadatas), start=1)
    ]
    combined_chunk = "\n\n".join(labeled_chunks)

    api_key = load_api_key()

    # Print tokens as they arrive instead of waiting for the full answer —
    # end=""/flush=True makes each token appear in the terminal immediately.
    for token in safe_query_stream(combined_chunk, args.question, api_key):
        print(token, end="", flush=True)
    print()

    # Sources are NOT streamed — retrieval already finished before the model
    # said a word, and this list is built directly from retrieved_metadatas,
    # not parsed from the model's answer, since an inline "(Source N)"
    # citation could hallucinate a file/line even when retrieval was correct.
    print("\nSources:")
    for i, meta in enumerate(retrieved_metadatas, start=1):
        line = f"  [{i}] {meta['file_path']}:{meta['start_line']}"
        github_url = build_github_url(meta)
        if github_url:
            line += f"  ({github_url})"
        print(line)
